#!/bin/bash
# Box side of one value-mining round. Runs as EC2 user-data on ONE spot box,
# builds the engine from source, plays the round, pushes results, terminates.
#
# Env prepended by launch_mining.sh (via the user-data stub):
#   AWS_*                credentials + region
#   S3_BUCKET, S3_PREFIX results at s3://$S3_BUCKET/$S3_PREFIX/results/$TAG.tar.gz
#   TAG                  round tag (mine_value.py --tag; names the work dir)
#   GAMES, MS, SEED_BASE mine_value.py --games / --ms / --seed-base
#   MINE_ARGS            extra flags appended verbatim (e.g. "--screen-n 10")
#   HARD_TIMEOUT_S       wall-clock kill switch, armed before anything else
#   CODE_KEY             s3 key of the pack_mining_code.sh tarball
#
# Layout: the tarball extracts to /opt/pfb, which becomes mine_value.py's ROOT.
# The venv MUST live at $ROOT/foul-play/.venv and leaf_prof at
# $ROOT/poke-engine/target/release/leaf_prof -- mine_value.py derives both from
# its own path and re-execs itself with that interpreter for every subprocess.
set -Euo pipefail
export HOME="${HOME:-/root}"   # EC2 user-data runs with NO HOME; under `set -u`
                               # `source $HOME/.cargo/env` would abort silently.
cd /root
exec > /root/boot.log 2>&1

S3_PREFIX="${S3_PREFIX:-mining}"
S3="s3://$S3_BUCKET/$S3_PREFIX"
TAG="${TAG:-awsmine}"
GAMES="${GAMES:-10}"
MS="${MS:-4500}"
SEED_BASE="${SEED_BASE:-101}"
MINE_ARGS="${MINE_ARGS:-}"
HARD_TIMEOUT_S="${HARD_TIMEOUT_S:-7200}"
CODE_KEY="${CODE_KEY:-mining/code.tar.gz}"
ROOT=/opt/pfb

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
push_logs() {
  aws s3 cp /root/boot.log "$S3/logs/$TAG.boot.log" >/dev/null 2>&1 || true
  if [ -s /root/run.log ]; then
    aws s3 cp /root/run.log "$S3/logs/$TAG.run.log" >/dev/null 2>&1 || true
  fi
  return 0
}

# HARD TIMEOUT armed before a single package is installed. A hung dnf, a stuck
# cargo, a playout that never ends: all of them cost at most this much money.
( sleep "$HARD_TIMEOUT_S"
  echo "HARD TIMEOUT ${HARD_TIMEOUT_S}s -- pushing partial logs and terminating" >> /root/boot.log
  push_logs
  aws s3 cp /root/boot.log "$S3/results/$TAG.TIMEOUT.log" || true
  shutdown -h now ) &
WATCHDOG=$!

die() {
  say "FATAL: $*"
  push_logs
  aws s3 cp /root/boot.log "$S3/results/$TAG.FAILED.log" || true
  shutdown -h now
  exit 1
}
# Fail loud on anything not explicitly guarded with `|| die`.
trap 'die "unhandled error at line $LINENO"' ERR

say "boot $(hostname) nproc=$(nproc) TAG=$TAG GAMES=$GAMES MS=$MS SEED_BASE=$SEED_BASE timeout=${HARD_TIMEOUT_S}s"
aws s3 cp /root/boot.log "$S3/logs/$TAG.boot.log" || true

# Sync the log from the start: the toolchain+build phase is ~25 blind minutes
# otherwise, and a bootstrap that died on line 1 of the build looks exactly like
# one still compiling.
( while true; do sleep 60; push_logs; done ) &
LOGSYNC=$!

# SPOT RECLAIM WATCHER (Sally 2026-08-17): EC2 posts a 2-minute warning at the
# metadata endpoint before reclaiming a spot box. Poll every 30s; on notice,
# push logs and a RECLAIMED marker naming the in-flight work so the supervisor
# relaunches EXACTLY the lost seed range and nothing else.
( TOK=""
  while true; do
    sleep 30
    TOK=$(curl -sf -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 120"           http://169.254.169.254/latest/api/token 2>/dev/null) || continue
    ACT=$(curl -sf -H "X-aws-ec2-metadata-token: $TOK"           http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)
    if [ -n "$ACT" ]; then
      echo "$(date -u +%H:%M:%SZ) SPOT RECLAIM NOTICE: $ACT" >> /root/boot.log
      cp /root/progress.json /root/reclaim.json 2>/dev/null || echo "{}" > /root/reclaim.json
      aws s3 cp /root/reclaim.json "$S3/results/$TAG.RECLAIMED.json" || true
      push_logs
      exit 0
    fi
  done ) &
RECLAIMWATCH=$!

# ---------------------------------------------------------------- toolchain
dnf install -y -q gcc gcc-c++ python3.11 python3.11-devel tar gzip || die "dnf"
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal || die "rustup"
source "$HOME/.cargo/env"
say "toolchain ready ($(cargo --version))"

# ---------------------------------------------------------------- payload
mkdir -p "$ROOT"
aws s3 cp "s3://$S3_BUCKET/$CODE_KEY" /root/code.tar.gz || die "code fetch"
tar xzf /root/code.tar.gz -C "$ROOT" || die "code untar"
for f in "$ROOT/corrections/mine_value.py" \
         "$ROOT/valuenet/sprt/run_duels.py" \
         "$ROOT/valuenet/nets_v8b/v8b_h2.bin" "$ROOT/valuenet/nets_v8b/v8b_h2.constants.json" \
         "$ROOT/valuenet/nets_v8b/v8b_s1.bin" "$ROOT/valuenet/nets_v8b/v8b_s1.constants.json" \
         "$ROOT/valuenet/nets_v8c/v8c_h1g.bin" "$ROOT/valuenet/nets_v8c/v8c_h1g.constants.json"; do
  [ -s "$f" ] || die "payload missing $f"
done
say "payload extracted to $ROOT"

# ---------------------------------------------------------------- engine
# Provenance-safe build cache: artifacts are keyed by the sha256 of the FULL
# engine source + requirements + python minor + feature flags, so a cache hit
# is bit-provenance-equivalent to building from source (a stale wheel scoring
# a different encoder is structurally impossible — any change moves the key).
# First box of a generation builds and uploads; every later box pulls in ~60s.
VENV="$ROOT/foul-play/.venv"
FEATS="terastallization"
SRCHASH=$( (find "$ROOT/poke-engine/src" "$ROOT/poke-engine/poke-engine-py" \
              "$ROOT/poke-engine/Cargo.toml" -type f -print0 | sort -z | \
              xargs -0 sha256sum; cat "$ROOT/foul-play/requirements.txt"; \
              python3.11 -V; echo "$FEATS") | sha256sum | cut -c1-16 )
ART="$S3/artifacts/eng-$SRCHASH.tar.gz"
say "engine artifact key: eng-$SRCHASH"
if aws s3 cp "$ART" /root/eng.tar.gz --quiet 2>/dev/null; then
  tar xzf /root/eng.tar.gz -C "$ROOT" || die "artifact untar"
  "$VENV/bin/python" -c "import poke_engine; print('wheel ok (cached)')" || die "cached wheel import"
  [ -x "$ROOT/poke-engine/target/release/leaf_prof" ] || die "cached leaf_prof missing"
  say "engine from artifact cache (~60s vs ~13min build)"
else
  python3.11 -m venv "$VENV" || die "venv"
  "$VENV/bin/pip" install -q --upgrade pip maturin || die "pip/maturin"
  "$VENV/bin/pip" install --no-cache-dir "$ROOT/poke-engine/poke-engine-py" \
    --config-settings="build-args=--features poke-engine/$FEATS --no-default-features" \
    || die "wheel build"
  grep -v poke-engine "$ROOT/foul-play/requirements.txt" > /tmp/req.txt
  "$VENV/bin/pip" install -q -r /tmp/req.txt || die "foul-play deps"
  "$VENV/bin/python" -c "import poke_engine; print('wheel ok')" || die "wheel import"
  say "wheel built"
  cd "$ROOT/poke-engine"
  cargo build --release --bin leaf_prof --features "$FEATS" --no-default-features \
    || die "leaf_prof build"
  [ -x "$ROOT/poke-engine/target/release/leaf_prof" ] || die "leaf_prof missing after build"
  say "leaf_prof built"
  cd "$ROOT"
  tar czf /root/eng.tar.gz -C "$ROOT" foul-play/.venv poke-engine/target/release/leaf_prof \
    && aws s3 cp /root/eng.tar.gz "$ART" --quiet \
    && say "engine artifact uploaded for the fleet" || say "artifact upload skipped (non-fatal)"
fi

# Preflight the two pieces mine_value.py cannot recover from: the team generator
# (falls back to a DIFFERENT sampler if the vendored sets.json is missing) and
# leaf_prof's logits mode (the whole eval column).
cd "$ROOT"
"$VENV/bin/python" - <<'PYEOF' || die "ps_teams preflight"
import sys
sys.path.insert(0, "/opt/pfb/foul-play")
from fp.search import _ps_team_loop as L, ps_teams
print("sets.json:", L.PS_SETS_JSON, len(L.RANDOM_SETS), "species")
ps_teams.seed(1)
print("team ok:", [m["speciesId"] for m in ps_teams.random_team()])
PYEOF
say "preflight ok"

# ---------------------------------------------------------------- mine
# MINE_CONCURRENT = vCPUs: every child is a single-threaded search, so the box
# is saturated only at nproc-wide fan-out.
MINE_CONCURRENT="$(nproc)"
export MINE_CONCURRENT
# MODE=confirm: instead of playing games, pull a candidate shard from S3 and
# pair-confirm it (CAND_KEY = s3 key of the shard jsonl; SHARD_START/COUNT
# optional slice). Results land in the same tarball path as a mining round.
if [ "${MODE:-mine}" = "gen" ]; then
  # V9 CORPUS GENERATION (Sally 2026-08-17): chunked so a spot reclaim loses
  # at most ~half a chunk. Each chunk plays GEN_CHUNK games, phase-harvests,
  # labels with v8c_s1 @ 2000 iters, and uploads its shard immediately.
  say "gen mode: $GEN_CHUNKS chunks x $GEN_CHUNK games, MS=$MS, seed-base $SEED_BASE"
  RC=0
  for ci in $(seq 0 $((GEN_CHUNKS - 1))); do
    SB=$((SEED_BASE + ci * GEN_CHUNK))
    CT="$TAG-c$ci"
    echo "{\"tag\": \"$TAG\", \"chunk\": $ci, \"seed_base\": $SB, \"games\": $GEN_CHUNK}" > /root/progress.json
    aws s3 cp /root/progress.json "$S3/genv9/$TAG/progress.json" --quiet || true
    say "chunk $ci: seeds $SB..$((SB + GEN_CHUNK - 1))"
    set +e
    MINE_CONCURRENT="$(nproc)" MINE_MAX_STEPS=300 "$VENV/bin/python" "$ROOT/corrections/mine_value.py" gen \
      --games "$GEN_CHUNK" --ms "$MS" --tag "$CT" --seed-base "$SB" \
      --label-bin "$ROOT/valuenet/nets_v8c/v8c_s1.bin" \
      >> /root/run.log 2>&1
    r=$?
    set -e
    if [ $r -eq 0 ] && [ -s "$ROOT/corrections/_mine_work/$CT/shard.jsonl.gz" ]; then
      aws s3 cp "$ROOT/corrections/_mine_work/$CT/shard.jsonl.gz" \
        "$S3/genv9/$TAG/chunk_$ci.jsonl.gz" --quiet || say "chunk $ci upload FAILED"
      aws s3 cp "$ROOT/corrections/_mine_work/$CT/gen_stats.json" \
        "$S3/genv9/$TAG/chunk_$ci.stats.json" --quiet || true
      rm -rf "$ROOT/corrections/_mine_work/$CT"
      say "chunk $ci uploaded"
    else
      RC=$r; say "chunk $ci FAILED rc=$r"
    fi
    push_logs
  done
  mkdir -p "$ROOT/corrections/_mine_work/$TAG"
  echo "{\"rc\": $RC}" > "$ROOT/corrections/_mine_work/$TAG/done.json"
  say "gen loop exited rc=$RC"
  tail -20 /root/run.log
elif [ "${MODE:-mine}" = "audit" ]; then
  # Per-turn evaluator audit of an archived game shipped inside the payload.
  # AUDIT_GAME is the repo-relative game dir; N/MS are playouts per decision
  # and search ms per step.
  say "audit mode: N=${N:-20} MS=${MS} ARMS=${ARMS:-0} OPPS=${OPPS:-argmax} COLLAPSED=${COLLAPSED:-0}"
  mkdir -p "$ROOT/corrections/_mine_work/$TAG"
  RC=0
  for g in $AUDIT_GAME; do
    for opp in ${OPPS:-argmax}; do
      base="$(basename "$g")"
      say "auditing $base opp=$opp"
      set +e
      AUDIT_CONCURRENT="$(nproc)" "$VENV/bin/python" "$ROOT/corrections/audit_game.py" \
        "$ROOT/$g" --n "${N:-20}" --ms "$MS" --workers "$(nproc)" \
        --arms "${ARMS:-0}" --opp "$opp" \
        $( [ "${COLLAPSED:-0}" = "1" ] && echo --collapsed-only ) \
        --net "$ROOT/valuenet/nets_v8c/${AUDIT_NET:-v8c_hz18}.bin" \
        --out "$ROOT/corrections/_mine_work/$TAG/$base.$opp.json" \
        >> /root/run.log 2>&1
      r=$?; [ $r -ne 0 ] && RC=$r
      set -e
      push_logs
    done
  done
  say "audit loop exited rc=$RC"
  tail -30 /root/run.log
elif [ "${MODE:-mine}" = "confirm" ]; then
  say "confirm mode: shard $CAND_KEY [${SHARD_START:-0}+${SHARD_COUNT:-all}]"
  aws s3 cp "s3://$S3_BUCKET/$CAND_KEY" /root/cands.jsonl --quiet || \
    curl -sf -o /root/cands.jsonl "$CAND_URL"
  mkdir -p "$ROOT/corrections/_mine_work/$TAG"
  set +e
  "$VENV/bin/python" "$ROOT/corrections/mine_value.py" confirm-pairs \
    --states /root/cands.jsonl \
    --out "$ROOT/corrections/_mine_work/$TAG/confirmed.jsonl" \
    --start "${SHARD_START:-0}" --count "${SHARD_COUNT:-0}" $MINE_ARGS \
    > /root/run.log 2>&1
  RC=$?
  set -e
  say "confirm-pairs exited rc=$RC"
  tail -20 /root/run.log
else
say "mining: $GAMES games, ${MS}ms/decision, MINE_CONCURRENT=$MINE_CONCURRENT"
set +e
"$VENV/bin/python" "$ROOT/corrections/mine_value.py" run \
  --games "$GAMES" --ms "$MS" --tag "$TAG" --seed-base "$SEED_BASE" $MINE_ARGS \
  > /root/run.log 2>&1
RC=$?
set -e
say "mine_value.py exited rc=$RC"
tail -40 /root/run.log
fi

# ---------------------------------------------------------------- results
WORK="$ROOT/corrections/_mine_work/$TAG"
[ -d "$WORK" ] || die "no work dir $WORK (rc=$RC)"
cp /root/run.log "$WORK/run.log"
cp /root/boot.log "$WORK/boot.log"
tar czf "/root/$TAG.tar.gz" -C "$ROOT/corrections/_mine_work" "$TAG" || die "tar results"
aws s3 cp "/root/$TAG.tar.gz" "$S3/results/$TAG.tar.gz" || die "result upload"
push_logs
kill $LOGSYNC $WATCHDOG $RECLAIMWATCH 2>/dev/null || true
say "DONE rc=$RC -> $S3/results/$TAG.tar.gz"
aws s3 cp /root/boot.log "$S3/logs/$TAG.boot.log" || true
shutdown -h now
