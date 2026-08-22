#!/bin/bash
# R64 LADDER MODEL (Sally 2026-08-22): one ranked blitz game with big remote
# compute. The Mac keeps the Showdown websocket (residential IP -- Showdown
# rejects datacenter logins); ALL search + EPG playouts run on the r64 box.
#
#   bash ladder-games/cloud/launch_search_box.sh   # once, ~3 min
#   bash ladder-games/run_game_r64.sh              # per game
#   bash ladder-games/cloud/stop_search_box.sh     # when done ($0.27/hr idle)
#
# Spec vs the local champion config: same clocks (4500ms/turn, 14000ms first),
# same net/gates/mixing -- but 64 sampled worlds all searched concurrently on
# the box (one wave, full 4500ms each), EPG armed with 16 worlds and its
# playouts fanned across all 64 cores. Local pool stays 8 as the breaker
# fallback (any remote failure finishes the game at local strength).
set -euo pipefail
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
URL_FILE="$ROOT/ladder-games/cloud/searchbox_ip.txt"
[ -f "$URL_FILE" ] || { echo "no search box recorded -- run ladder-games/cloud/launch_search_box.sh first"; exit 1; }
URL=$(cat "$URL_FILE")
curl -sf -m 4 "$URL/health" >/dev/null || { echo "search box at $URL not healthy -- relaunch it"; exit 1; }

export FP_EPG_MAX_WORLDS=16
RG_WORLDS=64 RG_POOL=8 RG_EPG=1 RG_REMOTE_URL="$URL" \
  exec bash "$ROOT/ladder-games/run_game.sh"
