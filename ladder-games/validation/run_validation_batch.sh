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
BATCH="$ROOT/ladder-games/validation/runs/batch-$STAMP"
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
    VG_BOT="v10bot$i" VG_OPP="synthopp$i" VG_SEED="$((1000+i))" \
      bash "$ROOT/ladder-games/validation/run_validation_game.sh" "$BATCH/g$i" \
      > "$BATCH/g$i.launch.log" 2>&1
    echo "  g$i done ($(grep -c 'Choice:' "$BATCH/g$i/search.log" 2>/dev/null || echo 0) decisions)"
  ) &
done
wait
echo "=== all games finished in $(( $(date +%s)-START ))s"
"$ROOT/foul-play/.venv/bin/python" "$ROOT/ladder-games/validation/audit_batch.py" "$BATCH"
