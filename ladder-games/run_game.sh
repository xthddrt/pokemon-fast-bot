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
# Net: v10c0 (Sally 2026-08-20: 4M-row fresh v10 corpus, mirror-trained, seed 1).
# MIXING (Sally 2026-08-21, default ON, RG_MIX=0 reverts to pure argmax):
# sample among candidates with visit >= 25% of argmax AND avg score >= the
# argmax's, weighted by visit*score. Anti-readability: back-to-back losses were
# 7- and 15-turn single-move loops into opponent lines predicted at 80%+.
# Round-robin 39,312 games: +12.1 Elo vs v9c0, +45.8 vs v8c_s1, transitivity checked.
# Better evaluator on neutral data too (bench_v1 0.02709 vs v9c0 0.02942).
# Sidecar carries tau=1.0 UCB=0.02 -- the 12k-iteration plateau [0.019,0.042]
# transferred to ladder budget via lnN (ladder c behaves like 1.258x the tested c).
export PE_PHANTOM_MODE="${RG_PHANTOM_MODE:-soft}"
export PE_PHANTOM_ALPHA="${RG_PHANTOM_ALPHA:-0}"
export PE_PHANTOM_SELF_AS_SEEN="${RG_PHANTOM_SELF_AS_SEEN:-0.2}"
FLAGS="--search-time-ms ${RG_SEARCH_MS:-4500} \
--first-turn-search-time-ms ${RG_FIRST_TURN_MS:-14000} \
--search-parallelism ${RG_WORLDS:-8} --search-pool-workers ${RG_POOL:-8} \
--search-threads 1 \
--nn-weights "${RG_NN_WEIGHTS:-../$(cat "$ROOT/valuenet/PRODUCTION_NET")}" \
--selection-argmax-only $([ "${RG_MIX:-1}" != "0" ] && printf %s --selection-mix) --tera-gate-q-margin 0.01 --tera-gate-visit-frac 0.25 \
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
ARCHIVED=$(python3 "$ROOT/ladder-games/archive_game.py" --since "$START" --flags "$FLAGS")
echo "$ARCHIVED"

# Turn table for every game just archived. The dir is parsed from
# archive_game.py's own "-> <dir>" line rather than relying on turn_table.py's
# most-recent-game default: run_parallel.sh archives several games in ONE call,
# and the default would print the same game N times. RG_NO_TABLE=1 skips it.
# Analysis must never fail the run -- the game is already safely archived.
if [ -z "${RG_NO_TABLE:-}" ]; then
  printf '%s\n' "$ARCHIVED" | sed -n 's/^archived .* -> //p' | while IFS= read -r d; do
    echo
    python3 "$ROOT/ladder-games/analysis/turn_table.py" --timing "$d" \
      || echo "(turn table failed for $d)"
  done
fi
