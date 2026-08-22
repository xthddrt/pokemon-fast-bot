#!/bin/bash
# Run N validation games concurrently against the local PS server, then audit
# every one of them.
#
#   bash ladder-games/validation/run_validation_batch.sh [N] [CONCURRENCY]
#
# Each slot gets its OWN bot and opponent name, which is what makes the run
# parallel-safe: foul-play writes logs/battle-<tag>_<opponent>.log, so a unique
# opponent name is how a slot finds its own log instead of whichever finished
# last.
#
# Concurrency and pool size are deliberately modest. Each game asks for 8
# WORLDS (the audit unit) but only VG_POOL processes to run them, in waves --
# 8 worlds at 100ms is 0.8s of work per decision however it is split. Default
# 4 slots x 2 workers keeps the machine at roughly half its 8 cores.
set -uo pipefail
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
N="${1:-20}"
CONC="${2:-4}"
STAMP=$(date +%Y%m%d-%H%M%S)
# VG_BATCH_DIR/VG_NONCE let several groups run CONCURRENTLY against separate PS
# servers (one node process saturates around 8-12 lanes), which is how the EC2
# fleet shards. Without them two groups started in the same second collide on
# both the output dir and the account-name nonce.
BATCH="${VG_BATCH_DIR:-$ROOT/ladder-games/validation/runs/batch-$STAMP}"
# per-run nonce: account names must never collide with a previous (possibly
# killed) run's, or its dangling challenges get accepted by this run's bots
NONCE="${VG_NONCE:-$(date +%H%M%S)}"
mkdir -p "$BATCH"
export VG_POOL="${VG_POOL:-2}"
export VG_WORLDS="${VG_WORLDS:-8}"

echo "=== batch of $N games, $CONC at a time, pool=$VG_POOL -> $BATCH"
START=$(date +%s)
i=0
while [ "$i" -lt "$N" ]; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 0.2; done
  i=$((i+1))
  (
    VG_BOT="${VG_BOT_PREFIX:-v10bot}${NONCE}x$i" VG_OPP="${VG_OPP_PREFIX:-synthopp}${NONCE}x$i" VG_SEED="$((1000+i))" \
      bash "$ROOT/ladder-games/validation/run_validation_game.sh" "$BATCH/g$i" \
      > "$BATCH/g$i.launch.log" 2>&1
    # A log holding two battle tags means this slot captured someone else's
    # game, and the auditor would grade one battle's worlds against the other's
    # truth (184 spurious wrong_moves in batch-20260820-205240). Names are
    # nonced per run so a KILLED batch cannot leak its dangling challenges into
    # the next one; this is the belt-and-braces check that it worked.
    NTAG=$(grep -o "battle-[a-z0-9]*-[0-9]*" "$BATCH/g$i/protocol.log" 2>/dev/null | sort -u | wc -l | tr -d ' ')
    if [ "${NTAG:-0}" -gt 1 ]; then
      echo "  g$i CROSS-TALK: $NTAG battles in one log -- excluded" > "$BATCH/g$i/EXCLUDED"
    fi
    echo "  g$i done ($(grep -c 'Choice:' "$BATCH/g$i/search.log" 2>/dev/null || echo 0) decisions)${NTAG:+$([ "${NTAG:-0}" -gt 1 ] && echo ' [CROSS-TALK]')}"
  ) &
done
wait
echo "=== all games finished in $(( $(date +%s)-START ))s"
[ "${VG_NO_AUDIT:-0}" = "1" ] || \
  "$ROOT/foul-play/.venv/bin/python" "$ROOT/ladder-games/validation/audit_batch.py" "$BATCH"
