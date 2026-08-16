#!/bin/bash
# Run N simultaneous ladder games on ONE account (multi-battle containment via
# FP_CLAIM_DIR). Each slot is a separate foul-play process with its own log
# subdir; battles are claimed exactly-once via atomic files. Slots stagger
# their searches slightly so the "already searching" rejection is rare.
# Usage: run_parallel.sh [N] — defaults to 2. Per-slot config via RG_* envs
# (same as run_game.sh; defaults here: 8 worlds, 4-process pool, flat 4500ms).
set -euo pipefail

N="${1:-2}"
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
CLAIM_DIR="$(mktemp -d /tmp/fp-claims.XXXXXX)"
START=$(date +%s)
echo "claim dir: $CLAIM_DIR"

set -a; . "$ROOT/.env"; set +a
FLAGS="--search-time-ms ${RG_SEARCH_MS:-4500} \
--first-turn-search-time-ms ${RG_FIRST_TURN_MS:-4500} \
--search-parallelism ${RG_WORLDS:-8} --search-pool-workers ${RG_POOL:-4} \
--search-threads 1 \
--nn-weights "${RG_NN_WEIGHTS:-../valuenet/nets_v8b/v8b_h2.bin}" \
--selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.3333"

cd "$ROOT/foul-play"
pids=()
for i in $(seq 1 "$N"); do
  # Per-slot account override: RG_USERNAME_<i> (+ RG_PASSWORD_<i>, empty =
  # unregistered guest login). Defaults to the .env account for every slot,
  # which is the same-account multi-battle mode (claims arbitrate).
  U_VAR="RG_USERNAME_$i"; P_VAR="RG_PASSWORD_$i"
  U="${!U_VAR:-$PS_USERNAME}"
  # per-slot password: unset = shared PS_PASSWORD (registered accounts),
  # explicitly empty = unregistered guest login
  if [ "${!P_VAR+set}" = set ]; then P="${!P_VAR}"; else P="$PS_PASSWORD"; fi
  PASS_ARGS=()
  [ -n "$P" ] && PASS_ARGS=(--ps-password "$P")
  FP_CLAIM_DIR="$CLAIM_DIR" FP_LOG_SUBDIR="slot$i" \
  "$FP_PYTHON" run.py \
    --websocket-uri "$FP_WEBSOCKET_URI" \
    --ps-username "$U" "${PASS_ARGS[@]}" \
    --bot-mode "$FP_BOT_MODE" --pokemon-format gen9randombattleblitz \
    --run-count 1 \
    $FLAGS \
    --save-replay always --log-level INFO --log-to-file \
    > "/tmp/fp-slot$i.log" 2>&1 &
  pids+=($!)
  sleep 3
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=$?
done

python3 "$ROOT/ladder-games/archive_game.py" --since "$START" --flags "$FLAGS parallel=$N"
exit $rc
