#!/bin/bash
# Play exactly ONE ranked gen9randombattleblitz ladder game with the canonical
# champion config + tera gate, then archive it into ladder-games/. See README.md.
set -euo pipefail

ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
set -a; . "$ROOT/.env"; set +a

# Overridable per run: RG_WORLDS (searched worlds), RG_POOL (parallel search
# processes; worlds/pool = waves), RG_FIRST_TURN_MS, RG_SEARCH_MS.
# PHANTOM opponent-model (Sally 2026-08-17, alpha corrected 2026-08-19): soft
# mode, alpha 0 — sampled never-revealed OPPONENT mons get NO exploration bonus
# (full Q still counts, so a phantom line that demonstrably scores still wins).
# SELF_AS_SEEN 0.2 = the weight the modelled opponent puts on lines using OUR
# not-yet-revealed switch. These were 0.5/0.5 at one point and the comment was
# not updated; the values below are authoritative.
# Net: v9c0 (Sally 2026-08-17: 4M-row single net, fresh-holdout true error
# -21% vs v8c_s1, duel +36 Elo [+6,+67]; sidecar carries tau=1.0 UCB=0.0422).
export PE_PHANTOM_MODE="${RG_PHANTOM_MODE:-soft}"
export PE_PHANTOM_ALPHA="${RG_PHANTOM_ALPHA:-0}"
export PE_PHANTOM_SELF_AS_SEEN="${RG_PHANTOM_SELF_AS_SEEN:-0.2}"
FLAGS="--search-time-ms ${RG_SEARCH_MS:-4500} \
--first-turn-search-time-ms ${RG_FIRST_TURN_MS:-14000} \
--search-parallelism ${RG_WORLDS:-8} --search-pool-workers ${RG_POOL:-8} \
--search-threads 1 \
--nn-weights "${RG_NN_WEIGHTS:-../$(cat "$ROOT/valuenet/PRODUCTION_NET")}" \
--selection-argmax-only --tera-gate-q-margin 0.01 --tera-gate-visit-frac 0.25 \
--endgame-playout-gate ${RG_EPG:-0}"
# RG_REMOTE_URL: route searches + EPG playouts to a worker box (search_server.py);
# the websocket stays here (Showdown rejects datacenter logins). Any remote
# failure trips a breaker and the bot finishes the game at local strength.
[ -n "${RG_REMOTE_URL:-}" ] && FLAGS="$FLAGS --search-remote-url $RG_REMOTE_URL"

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
