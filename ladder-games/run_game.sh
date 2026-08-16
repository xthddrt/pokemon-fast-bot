#!/bin/bash
# Play exactly ONE ranked gen9randombattleblitz ladder game with the canonical
# champion config + tera gate, then archive it into ladder-games/. See README.md.
set -euo pipefail

ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
set -a; . "$ROOT/.env"; set +a

# Overridable per run: RG_WORLDS (searched worlds), RG_POOL (parallel search
# processes; worlds/pool = waves), RG_FIRST_TURN_MS, RG_SEARCH_MS.
FLAGS="--search-time-ms ${RG_SEARCH_MS:-4500} \
--first-turn-search-time-ms ${RG_FIRST_TURN_MS:-14000} \
--search-parallelism ${RG_WORLDS:-8} --search-pool-workers ${RG_POOL:-8} \
--search-threads 1 \
--nn-weights "${RG_NN_WEIGHTS:-../valuenet/nets_v8c/v8c_h1g.bin}" \
--selection-argmax-only --tera-gate-per-mon 0.0015 --tera-gate-visit-frac 0.3333 \
--tera-gate-opp-tera-bonus 0.003"

START=$(date +%s)
cd "$ROOT/foul-play"
# RG_USERNAME switches account (default .env's PS_USERNAME); RG_PASSWORD
# unset = shared PS_PASSWORD, explicitly empty = unregistered guest login.
U="${RG_USERNAME:-$PS_USERNAME}"
P="${RG_PASSWORD-$PS_PASSWORD}"
PASS_ARGS=()
[ -n "$P" ] && PASS_ARGS=(--ps-password "$P")
"$FP_PYTHON" run.py \
  --websocket-uri "$FP_WEBSOCKET_URI" \
  --ps-username "$U" "${PASS_ARGS[@]}" \
  --bot-mode "$FP_BOT_MODE" --pokemon-format "${RG_FORMAT:-gen9randombattleblitz}" \
  --run-count 1 \
  $FLAGS \
  --save-replay always --log-level INFO --log-to-file

# Archive whatever was played (never let a bot crash skip archiving entirely:
# the trap would be nice, but a failed game with no battle log has nothing to
# archive anyway).
python3 "$ROOT/ladder-games/archive_game.py" --since "$START" --flags "$FLAGS"
