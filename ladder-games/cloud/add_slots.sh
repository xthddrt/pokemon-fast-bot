#!/bin/bash
# Hot-add N slot loops for a given FORMAT to an already-running fleet box,
# alongside the existing slots. Shares the box's claim dir + relay tunnel.
# Each slot: game ends -> archive -> search again (continuous). Runs until
# /opt/DRAIN. Usage (on the box):
#   add_slots.sh <account> <format> <n_slots> <slot_offset>
set -uo pipefail
ACCOUNT="$1"; FORMAT="$2"; N="$3"; OFF="${4:-100}"
S3_BUCKET=pokebot-valuenet-389825051723
S3_PREFIX=$(cat /opt/S3_PREFIX 2>/dev/null || echo "ladder-fleet/fleet1/box-unknown")
slug=$(echo "$ACCOUNT" | tr -c 'A-Za-z0-9' '_')

run_slot() {
  local slot="$1" g=0
  cd /opt/foul-play
  while [ ! -f /opt/DRAIN ]; do
    g=$((g + 1)); local start=$(date +%s)
    FP_CLAIM_DIR=/opt/claims FP_LOG_SUBDIR="${slug}_${FORMAT}_s${slot}" \
    /opt/venv/bin/python run.py \
      --websocket-uri "ws://127.0.0.1:8765/showdown/websocket" \
      --ps-username "$ACCOUNT" --ps-password "$PS_PASSWORD" \
      --bot-mode search_ladder --pokemon-format "$FORMAT" \
      --run-count 1 \
      --search-time-ms 4000 --first-turn-search-time-ms 4000 \
      --search-parallelism 8 --search-pool-workers 1 --search-threads 1 \
      --nn-weights ../valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
      --selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.25 \
      --save-replay always --log-level INFO --log-to-file \
      > "/opt/fleet-logs/${slug}_${FORMAT}_s${slot}_g${g}.log" 2>&1 || true
    flock /opt/archive/.lock env \
      FP_ARCHIVE_WORKSPACE=/opt FP_ARCHIVE_DIR=/opt/archive \
      FP_ARCHIVE_LOGS=/opt/foul-play/logs FP_ARCHIVE_NO_LOSSES=1 \
      /opt/venv/bin/python /opt/ladder-games/archive_game.py \
        --since "$start" --flags "fleet $ACCOUNT $FORMAT s$slot" || true
  done
}

for i in $(seq 1 "$N"); do
  run_slot "$((OFF + i))" &
  sleep 3
done
echo "added $N $FORMAT slots for $ACCOUNT (pids: $(jobs -p | tr '\n' ' '))"
