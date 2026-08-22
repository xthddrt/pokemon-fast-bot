#!/bin/bash
# ONE local-server validation game: the v10 bot plays a PS-generated random
# battle against a random-move opponent, and we keep BOTH the per-world
# sampling record and the opponent's true team.
#
#   bash ladder-games/validation/run_validation_game.sh [out_dir]
#
# Requires a local Showdown server already running with --no-security:
#   node pokemon-showdown/dist/server/index.js --no-security
#
# Everything matches the ladder config in ladder-games/run_game.sh except the
# search budget (RG_SEARCH_MS, default 100ms) -- the point is to exercise the
# SAMPLER, which does the same work per decision regardless of think time.
set -euo pipefail
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
set -a; . "$ROOT/.env"; set +a

OUT="${1:-$ROOT/ladder-games/validation/runs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
URI="${VG_URI:-ws://localhost:8000/showdown/websocket}"
FORMAT="${VG_FORMAT:-gen9randombattleblitz}"
BOT="${VG_BOT:-v10bot}"
OPP="${VG_OPP:-synthopp}"
MS="${RG_SEARCH_MS:-100}"

T0=$(python3 -c "import time;print(time.time())")
_lap(){ python3 -c "import time,sys;print('  [t+%.1fs] '%(time.time()-float(sys.argv[1]))+sys.argv[2])" "$T0" "$1"; }
echo "=== validation game -> $OUT"
echo "    format=$FORMAT  worlds=${VG_WORLDS:-8} pool=${VG_POOL:-8}  search=${MS}ms  bot=$BOT vs $OPP"

# 1. opponent first: it must be logged in and waiting before the challenge.
"$FP_PYTHON" "$ROOT/ladder-games/validation/synth_opponent.py" \
  --uri "$URI" --name "$OPP" --out "$OUT/truth.json" --seed "${VG_SEED:-1}" \
  > "$OUT/opponent.log" 2>&1 &
OPP_PID=$!
_lap "opponent launched"
trap 'kill $OPP_PID 2>/dev/null || true' EXIT
for _ in $(seq 40); do grep -q "logged in as" "$OUT/opponent.log" 2>/dev/null && break; sleep 0.25; done
grep -q "logged in as" "$OUT/opponent.log" || { echo "opponent never logged in:"; cat "$OUT/opponent.log"; exit 1; }
_lap "opponent ready"

# 2. the bot. Same phantom + gate config as the ladder runner.
# Local server has no throttle: drop the 0.62s/message pacing and the 3s
# post-login settle that exist for play.pokemonshowdown.com. Gated on the
# address inside websocket_client.py, so it cannot affect a real ladder run.
export FP_LOCAL_NO_PACE=1
export PE_PHANTOM_MODE="${RG_PHANTOM_MODE:-soft}"
export PE_PHANTOM_ALPHA="${RG_PHANTOM_ALPHA:-0}"
export PE_PHANTOM_SELF_AS_SEEN="${RG_PHANTOM_SELF_AS_SEEN:-0.2}"
cd "$ROOT/foul-play"
set +e
"$FP_PYTHON" run.py \
  --websocket-uri "$URI" \
  --ps-username "$BOT" --ps-password "" \
  --bot-mode challenge_user --user-to-challenge "$OPP" \
  --pokemon-format "$FORMAT" --run-count 1 \
  --search-time-ms "$MS" --first-turn-search-time-ms "$MS" \
  --search-parallelism "${VG_WORLDS:-8}" --search-pool-workers "${VG_POOL:-8}" --search-threads 1 \
  --nn-weights "../$(cat "$ROOT/valuenet/PRODUCTION_NET")" \
  --selection-argmax-only --tera-gate-q-margin 0.01 --tera-gate-visit-frac 0.25 \
  --endgame-playout-gate 0 \
  --log-level INFO --log-to-file > "$OUT/bot.stdout" 2>&1
RC=$?
_lap "bot finished (startup+play)"
set -e
echo "bot exited rc=$RC"
wait $OPP_PID 2>/dev/null || true

# 3. keep the raw battle log and split out the per-world sampling record.
# Match the log by OPPONENT NAME, not `ls -t | head -1`: foul-play writes
# logs/battle-<tag>_<opponent>.log, and under concurrency the newest file
# belongs to whichever slot happened to finish last, cross-contaminating runs.
LOG=$(ls -t "$ROOT/foul-play/logs"/*_"$OPP".log 2>/dev/null | head -1 || true)
[ -n "$LOG" ] && cp "$LOG" "$OUT/battle.log" || echo "WARN: no battle log found"
"$FP_PYTHON" - "$OUT" << 'PY'
import json, os, re, sys
d = sys.argv[1]
src = os.path.join(d, "battle.log")
if not os.path.isfile(src):
    raise SystemExit("no battle.log to split")
sys.path.insert(0, os.path.join(os.environ.get("FP_ROOT", "/Users/sallyliu/pokemon-fast-bot"), "ladder-games"))
import archive_game as A
A.segment_log(open(src, errors="replace").read(), d)
n = sum(1 for _ in open(os.path.join(d, "worlds.jsonl")))
print("split: %d world rows" % n)
PY
_lap "archived+split"
echo "=== done: $OUT"
ls -la "$OUT"
