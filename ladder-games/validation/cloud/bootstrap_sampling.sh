#!/bin/bash
# Sampling-validation box: play N local-server games and ship the per-game
# sampling record for auditing. Adapted from valuenet/cloud/bootstrap_duels.sh,
# which is the proven recipe for "build the engine wheel from source, stand up
# PS servers, run foul-play games, upload, self-terminate".
#
# Env prepended by launch_sampling.sh (EC2 user-data):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
#   S3_BUCKET     pokebot-valuenet-389825051723
#   S3_CODE       prefix holding sampling_code.tar.gz (default sampling)
#   S3_PREFIX     results prefix, unique per run
#   N_GAMES       games this box plays (default 100)
#   NGROUPS        parallel groups = PS servers (default nproc/16, min 1)
#   CONC          concurrent games PER GROUP (default 8)
#   BASE_PORT     first PS port (default 8000)
#   MS            ms per decision (default 100 -- this exercises the SAMPLER,
#                 which does identical work per decision at any think time)
#   WORLDS/POOL   sampled worlds and pool workers per game (default 8 / 2)
#
# WHY THE WHEEL IS BUILT FROM SOURCE: the run measures this engine's sampler; a
# stale wheel measures a different bot. Same reasoning as the duel fleet.
set -uo pipefail
exec > /root/run.log 2>&1

# cloud-init runs user-data with a MINIMAL environment: HOME is not set, and
# under `set -u` the first "$HOME/.cargo" reference aborts the box one minute
# in (canary-215103 died exactly there). Set it before anything reads it.
export HOME="${HOME:-/root}"
export USER="${USER:-root}"

: "${S3_CODE:=sampling}"
: "${N_GAMES:=100}"
# 12, not 8: a game's 2 pool workers idle during protocol round-trips, so 8
# lanes/group left the 1000-game box at load 48/64. Oversubscribing ~1.5x hides
# the I/O gaps (measured 75% -> ~full utilization).
: "${CONC:=12}"
: "${BASE_PORT:=8000}"
: "${MS:=100}"
: "${WORLDS:=8}"
: "${POOL:=2}"
NPROC=$(nproc)
# NOT named GROUPS: that is a special BUILT-IN bash array holding the user's
# group IDs, so `${GROUPS:-}` is already non-empty (root's primary GID, "0")
# and every guard of the form `[ -z "${GROUPS:-}" ]` silently never fires.
# canary-215700 died on `division by 0` for exactly this reason -- the earlier
# "min 1" fix was correct code that never ran.
# One PS server per ~16 vCPU, but never zero.
if [ -z "${NGROUPS:-}" ]; then
  NGROUPS=$(( NPROC / 16 ))
  [ "$NGROUPS" -lt 1 ] && NGROUPS=1
fi
NODE_VER=v22.14.0
ROOT=/opt
VENV=/opt/venv
PY="$VENV/bin/python3"

DONE_OK=0
MAIN_PID=$BASHPID
finish() {
  rc=$?
  trap - ERR EXIT
  # bash runs an inherited EXIT trap in EVERY forked subshell, including the
  # ones $( ) creates -- without this guard a plain `echo "$(f)"` would run
  # finish inside the substitution and shut the box down. Guard on BASHPID:
  # $$ stays the parent's inside a subshell.
  [ "$BASHPID" = "$MAIN_PID" ] || exit 0
  echo "=== finish rc=$rc done_ok=$DONE_OK"
  aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" || true
  # per-group and PS-server logs too: run.log alone could not say WHY a box
  # produced zero games (canary-215700), which cost a whole extra round-trip.
  for f in /root/grp*.log /root/ps*.log; do
    [ -f "$f" ] && aws s3 cp "$f" "s3://$S3_BUCKET/$S3_PREFIX/$(basename "$f")" || true
  done
  [ "$DONE_OK" = "1" ] && aws s3 cp /dev/null "s3://$S3_BUCKET/$S3_PREFIX/DONE" || true
  shutdown -h now
}
trap finish EXIT

# Heartbeat: run.log lands in S3 every 60s, so progress is visible mid-run and
# a wedged box is diagnosable WITHOUT waiting for exit or SSHing in. Four
# canary failures were each a blind ~15-minute round trip for want of this.
( while sleep 60; do
    aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" >/dev/null 2>&1
  done ) &
HEARTBEAT_PID=$!

echo "=== box: nproc=$NPROC groups=$NGROUPS conc=$CONC n_games=$N_GAMES ms=$MS"

# ---------------------------------------------------------------- toolchain
dnf install -y -q gcc gcc-c++ python3.11 python3.11-devel tar gzip xz || exit 1

cd /opt
# arch-aware: Graviton boxes are ~20% cheaper per effective core than any x86
# in eu-north (2026-08-21 scan) and this URL was the only x86-hardcoded step
NODE_ARCH=x64
[ "$(uname -m)" = "aarch64" ] && NODE_ARCH=arm64
curl -fsSL -o node.tar.xz "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-$NODE_ARCH.tar.xz" || exit 1
tar xf node.tar.xz && rm -f node.tar.xz
rm -rf /opt/node && mv "node-$NODE_VER-linux-$NODE_ARCH" /opt/node
export PATH=/opt/node/bin:$PATH
node --version || exit 1

# ---------------------------------------------------------------- code
aws s3 cp "s3://$S3_BUCKET/$S3_CODE/sampling_code.tar.gz" /opt/code.tar.gz || exit 1
tar xzf /opt/code.tar.gz -C /opt || exit 1
# The bundle is packed on macOS. Extracting it here can materialise AppleDouble
# sidecars (./._items.ts and friends) carrying raw xattr bytes; esbuild then
# tries to PARSE them and `node build` dies with 370+ 'Unexpected "\x00"'
# errors (canary-215252). They are never real source, so delete them.
find /opt -name '._*' -type f -delete 2>/dev/null
find /opt -name '.DS_Store' -type f -delete 2>/dev/null
for d in foul-play pokemon-showdown poke-engine ladder-games valuenet; do
  [ -d "/opt/$d" ] || { echo "tarball missing $d"; ls /opt; exit 1; }
done

# The harness sources $FP_ROOT/.env. The real one holds ladder credentials and
# is deliberately NOT in the tarball; a local --no-security server accepts any
# name, so synthesize the few vars the scripts actually read.
cat > /opt/.env << ENVEOF
FP_PYTHON=$PY
FP_REPO=/opt/foul-play
PYTHONIOENCODING=utf-8
PS_USERNAME=boxbot
PS_PASSWORD=
ENVEOF

# ---------------------------------------------------------------- build cache
# ~13 of a 22-minute box's life is COMPILING (rustup, poke-engine from source,
# npm ci, node build) and only ~9 is playing, so caching the build outputs cuts
# cost per datapoint further than any instance swap.
#
# The key is derived from the BUILD INPUTS ONLY -- the engine sources, the
# Python requirements, PS's lockfile, the node version and the machine arch --
# so editing an auditor or the sampler does NOT invalidate it, while touching
# anything the wheel is built from does. That distinction is the whole safety
# property: the duel fleet's rule is that a stale wheel silently measures a
# different engine than the one being shipped, and a cache keyed on the tarball
# checksum would be safe but would miss on every code edit, while one keyed on
# nothing would be fast and wrong.
BUILD_KEY=$( {
  find /opt/poke-engine -type f \( -name '*.rs' -o -name 'Cargo.toml' \
       -o -name 'Cargo.lock' -o -name 'pyproject.toml' \) -print0 \
    | sort -z | xargs -0 sha256sum 2>/dev/null
  sha256sum /opt/foul-play/requirements.txt 2>/dev/null
  sha256sum /opt/pokemon-showdown/package-lock.json 2>/dev/null
  echo "$NODE_VER $(uname -m) $(python3.11 -V 2>&1)"
} | sha256sum | cut -c1-20 )
CACHE_URL="s3://$S3_BUCKET/sampling/cache/build-$BUILD_KEY.tar.gz"
echo "=== build key $BUILD_KEY"

CACHE_HIT=0
if aws s3 ls "$CACHE_URL" >/dev/null 2>&1; then
  echo "=== cache HIT, restoring $CACHE_URL"
  if aws s3 cp "$CACHE_URL" /opt/build_cache.tar.gz && \
     tar xzf /opt/build_cache.tar.gz -C / ; then
    # Never TRUST the cache: prove the wheel imports and PS actually built.
    # A corrupt or half-uploaded blob must fall back to a real build, not run.
    if "$VENV/bin/python3" -c "import poke_engine" 2>/dev/null \
       && [ -d /opt/pokemon-showdown/dist ]; then
      CACHE_HIT=1; echo "=== cache verified"
    else
      echo "=== cache FAILED verification, rebuilding"
      rm -rf "$VENV" /opt/pokemon-showdown/dist
    fi
  fi
  rm -f /opt/build_cache.tar.gz
fi

# ---------------------------------------------------------------- PS build
cd /opt/pokemon-showdown
[ -f config/config.js ] || cp config/config-example.js config/config.js
printf '\nexports.repl = false;\nexports.logchat = false;\n' >> config/config.js
if [ "$CACHE_HIT" = "0" ]; then
  npm ci --no-audit --no-fund --silent || npm install --no-audit --no-fund --silent || exit 1
  node build || exit 1  # ONCE: K servers with --skip-build would race on dist/
fi

# ---------------------------------------------------------------- engine wheel
if [ "$CACHE_HIT" = "0" ]; then
  # Rust is ONLY needed to compile the wheel, so it is installed here rather
  # than with the base toolchain: a cache hit skips ~2 minutes of rustup.
  if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal || exit 1
  fi
  source "$HOME/.cargo/env"
  python3.11 -m venv "$VENV" || exit 1
  "$VENV/bin/pip3" install -q --upgrade pip maturin || exit 1
  "$VENV/bin/pip3" install --no-cache-dir /opt/poke-engine/poke-engine-py \
    --config-settings="build-args=--features poke-engine/terastallization --no-default-features" || exit 1
  grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
  "$VENV/bin/pip3" install -q -r /tmp/req.txt || exit 1
fi
"$PY" -c "import poke_engine; print('wheel ok', poke_engine.__file__)" || exit 1

# Publish for the next box. Only after the wheel has been proven to import, so
# a broken build is never cached. Absolute paths inside the venv are fine: /opt
# is fixed on every box.
if [ "$CACHE_HIT" = "0" ]; then
  echo "=== publishing build cache $CACHE_URL"
  tar czf /opt/build_cache.tar.gz \
    /opt/venv /opt/pokemon-showdown/node_modules /opt/pokemon-showdown/dist \
    2>/dev/null && aws s3 cp /opt/build_cache.tar.gz "$CACHE_URL" || \
    echo "  (cache publish failed -- not fatal)"
  rm -f /opt/build_cache.tar.gz
fi
ln -sfn "$VENV" /opt/foul-play/.venv   # run_validation_batch.sh audits via $ROOT/foul-play/.venv

export FP_ROOT=/opt
export FP_PYTHON="$PY"
export FP_ARCHIVE_WORKSPACE=/opt
export PS_ROOT=/opt/pokemon-showdown
export FP_LOCAL_LOGIN=1

# ---------------------------------------------------------------- PS servers
for g in $(seq 0 $((NGROUPS-1))); do
  PORT=$((BASE_PORT+g))
  ( cd /opt/pokemon-showdown && PORT=$PORT node dist/server/index.js $PORT --no-security > /root/ps$g.log 2>&1 ) &
done
for g in $(seq 0 $((NGROUPS-1))); do
  PORT=$((BASE_PORT+g)); OK=0
  for _ in $(seq 60); do
    (echo > /dev/tcp/127.0.0.1/$PORT) >/dev/null 2>&1 && { OK=1; break; }
    sleep 1
  done
  [ "$OK" = "1" ] || { echo "PS server on $PORT never came up"; tail -40 /root/ps$g.log; exit 1; }
  echo "  PS server up on $PORT"
done

# ---------------------------------------------------------------- play
PER=$(( (N_GAMES + NGROUPS - 1) / NGROUPS ))
mkdir -p /opt/out
for g in $(seq 0 $((NGROUPS-1))); do
  PORT=$((BASE_PORT+g))
  (
    export VG_URI="ws://localhost:$PORT/showdown/websocket"
    export VG_BATCH_DIR="/opt/out/grp$g"
    # PS caps usernames at 18 chars, and "synthopp"+7-char nonce+"x"+3-digit
    # game number is 19: run10k grp10/11 silently no-op'd every game past 99.
    # Short prefixes + a <=4-char nonce keep the longest name ("o"+4+"x"+4)
    # at 10 chars with room to spare.
    export VG_BOT_PREFIX=b VG_OPP_PREFIX=o
    export VG_NONCE="${g}$(date +%s | tail -c 3)"
    export VG_WORLDS="$WORLDS" VG_POOL="$POOL" RG_SEARCH_MS="$MS"
    export VG_NO_AUDIT=1          # audit runs once, over the MERGED set
    bash /opt/ladder-games/validation/run_validation_batch.sh "$PER" "$CONC"
  ) > /root/grp$g.log 2>&1 &
  GRP_PIDS="${GRP_PIDS:-} $!"
done
# Wait on the GROUP pids ONLY. A bare `wait` also waits for the PS server
# processes, which never exit -- canary-220151 finished all 20 games in 35s and
# then hung at this line for 17 minutes until an SSH post-mortem found it.
for p in $GRP_PIDS; do wait "$p"; done
echo "=== all groups finished"

# ---------------------------------------------------------------- audit ON BOX
# The 64 cores are idle once the games end, and shipping 264MB to audit locally
# was ~8 of the 13 minutes of run-222332. Audit here, ship the text reports
# plus artifacts for FLAGGED games only; FULL_SHIP=1 restores the old behavior.
export FP_ROOT=/opt
AUDIT_FLAGGED=""
AUDIT_PIDS=""
for g in $(seq 0 $((NGROUPS-1))); do
  ( "$PY" /opt/ladder-games/validation/audit_batch.py /opt/out/grp$g \
      > /opt/out/audit_grp$g.txt 2>&1 ) &
  AUDIT_PIDS="$AUDIT_PIDS $!"
done
# explicit PIDs -- a bare wait (or jobs -rp) would include the heartbeat loop,
# which never exits: the same hang class as the PS-server wait bug.
for pid in $AUDIT_PIDS; do wait "$pid" || true; done
for g in $(seq 0 $((NGROUPS-1))); do
  aws s3 cp /opt/out/audit_grp$g.txt "s3://$S3_BUCKET/$S3_PREFIX/audit_grp$g.txt" || true
  FLAGGED=$(awk '/^g[0-9]+ /{if ($4>0) print "grp'"$g"'/"$1}' /opt/out/audit_grp$g.txt)
  # TRUTH-ORPHAN games pass every hard check by construction, so flagged-only
  # shipping dropped exactly the evidence the orphan bug needs (run25k g61
  # Baxcalibur, 8 straight ceil=1.00 decisions, artifacts lost with the box).
  ORPHANED=$(awk '/^## TRUTH ORPHANS/,/^$/' /opt/out/audit_grp$g.txt \
    | grep -oE "^   g[0-9]+ " | tr -d ' ' | sort -u | sed "s|^|grp$g/|")
  AUDIT_FLAGGED="$AUDIT_FLAGGED $FLAGGED $ORPHANED"
done
echo "=== flagged games:${AUDIT_FLAGGED:- none}"

# ---------------------------------------------------------------- ship
# Only the files the auditors read, so the upload is megabytes not gigabytes.
cd /opt/out
if [ "${FULL_SHIP:-0}" = "1" ] || [ -z "${AUDIT_FLAGGED// /}" ]; then
  # everything on FULL_SHIP; on a fully-clean run ship ONE game as a spot-check
  # sample (an empty tarball would be indistinguishable from a broken one)
  LIST=$(find . -type d -name 'g*' | head -$( [ "${FULL_SHIP:-0}" = "1" ] && echo 100000 || echo 1 ))
else
  LIST=$(for f in $AUDIT_FLAGGED; do echo "./$f"; done)
fi
echo "$LIST" | while read -r d; do
  [ -n "$d" ] && find "$d" -type f \( -name worlds.jsonl -o -name truth.json \
    -o -name protocol.log -o -name search.log -o -name battle.log \
    -o -name EXCLUDED \) -print
done | tar czf /opt/sampling_results.tar.gz -T - || exit 1
ls -lh /opt/sampling_results.tar.gz
aws s3 cp /opt/sampling_results.tar.gz "s3://$S3_BUCKET/$S3_PREFIX/results.tar.gz" || exit 1
NGOT=$(find /opt/out -name worlds.jsonl | wc -l)
echo "=== shipped $NGOT games (wanted $N_GAMES)"
[ "$NGOT" -gt 0 ] && DONE_OK=1
