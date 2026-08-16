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
         "$ROOT/valuenet/nets_v8b/v8b_s1.bin" "$ROOT/valuenet/nets_v8b/v8b_s1.constants.json"; do
  [ -s "$f" ] || die "payload missing $f"
done
say "payload extracted to $ROOT"

# ---------------------------------------------------------------- engine, FROM SOURCE
# Never a prebuilt wheel: the v8 encoder lives in this tree, and a stale wheel
# would either panic on load or, worse, score a DIFFERENT encoder than the net
# was certified against. Same feature set for the wheel and for leaf_prof, so
# the eval pass and the playouts agree by construction.
VENV="$ROOT/foul-play/.venv"
python3.11 -m venv "$VENV" || die "venv"
"$VENV/bin/pip" install -q --upgrade pip maturin || die "pip/maturin"
"$VENV/bin/pip" install --no-cache-dir "$ROOT/poke-engine/poke-engine-py" \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features" \
  || die "wheel build"
grep -v poke-engine "$ROOT/foul-play/requirements.txt" > /tmp/req.txt
"$VENV/bin/pip" install -q -r /tmp/req.txt || die "foul-play deps"
"$VENV/bin/python" -c "import poke_engine; print('wheel ok')" || die "wheel import"
say "wheel built"

cd "$ROOT/poke-engine"
cargo build --release --bin leaf_prof --features terastallization --no-default-features \
  || die "leaf_prof build"
[ -x "$ROOT/poke-engine/target/release/leaf_prof" ] || die "leaf_prof missing after build"
say "leaf_prof built"

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
say "mining: $GAMES games, ${MS}ms/decision, MINE_CONCURRENT=$MINE_CONCURRENT"
set +e
"$VENV/bin/python" "$ROOT/corrections/mine_value.py" run \
  --games "$GAMES" --ms "$MS" --tag "$TAG" --seed-base "$SEED_BASE" $MINE_ARGS \
  > /root/run.log 2>&1
RC=$?
set -e
say "mine_value.py exited rc=$RC"
tail -40 /root/run.log

# ---------------------------------------------------------------- results
WORK="$ROOT/corrections/_mine_work/$TAG"
[ -d "$WORK" ] || die "no work dir $WORK (rc=$RC)"
cp /root/run.log "$WORK/run.log"
cp /root/boot.log "$WORK/boot.log"
tar czf "/root/$TAG.tar.gz" -C "$ROOT/corrections/_mine_work" "$TAG" || die "tar results"
aws s3 cp "/root/$TAG.tar.gz" "$S3/results/$TAG.tar.gz" || die "result upload"
push_logs
kill $LOGSYNC $WATCHDOG 2>/dev/null || true
say "DONE rc=$RC -> $S3/results/$TAG.tar.gz"
aws s3 cp /root/boot.log "$S3/logs/$TAG.boot.log" || true
shutdown -h now
