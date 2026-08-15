#!/bin/bash
# Play exactly one gen9randombattle ladder game on the official Showdown
# server as the account configured in .env (sibling of this script).
# DEBUG logs are written to foul-play/logs/, replays are always saved.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$DIR/.env"; set +a

cd "$FP_REPO"
exec "$FP_PYTHON" run.py \
  --websocket-uri "$FP_WEBSOCKET_URI" \
  --ps-username "$PS_USERNAME" \
  --ps-password "$PS_PASSWORD" \
  --bot-mode "$FP_BOT_MODE" \
  --pokemon-format "$FP_POKEMON_FORMAT" \
  --run-count "$FP_RUN_COUNT" \
  --search-time-ms "$FP_SEARCH_TIME_MS" \
  --first-turn-search-time-ms "$FP_FIRST_TURN_SEARCH_TIME_MS" \
  --search-parallelism "$FP_SEARCH_PARALLELISM" \
  --search-pool-workers "$FP_SEARCH_POOL_WORKERS" \
  --search-threads "$FP_SEARCH_THREADS" \
  --save-replay "$FP_SAVE_REPLAY" \
  --log-level "$FP_LOG_LEVEL" \
  --log-to-file
