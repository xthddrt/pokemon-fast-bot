#!/bin/bash
# Launch the plc2 label fleet: NBOX contiguous slices of plc2/positions.jsonl,
# each on its own spot box, all installing the PINNED 0.0.59 wheel.
# Reuses launch_playout.sh (UD_TEMPLATE override) so the slice/resume/reclaim
# contract is exactly plc1's.
#
#   WHEEL_SHA=<sha> ./launch_plc2fleet.sh [nbox] [count] [type] [tagprefix] [firstbox]
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
NBOX="${1:-20}"
COUNT="${2:-50000}"
TYPE="${3:-c7a.16xlarge}"
PFX="${4:-v}"
FIRST="${5:-0}"
: "${WHEEL_SHA:?pinned wheel sha256}"
# 2a and 2b are ~$0.77/h spot; 2c is ~$1.15. Alternate the two cheap AZs so a
# single-AZ capacity wall cannot stall the whole fleet.
AZS=(subnet-0c80748d73536ffb2 subnet-01b4c6f3171ce3a80)

for ((k=FIRST; k<FIRST+NBOX; k++)); do
  TAG=$(printf '%s%02d' "$PFX" "$k")
  START=$((k * COUNT))
  SUBNET=${AZS[$((k % 2))]} BUNDLE=/dev/null SKIP_BUNDLE=1 POSRUN=plc2 \
    TAG=$TAG START=$START COUNT=$COUNT N=10 ITERS=2000 W=64 WATCHDOG=10800 \
    WHEEL_SHA="$WHEEL_SHA" UD_TEMPLATE=$ROOT/evallab/cloud/userdata_plc2label.sh \
    "$ROOT/evallab/cloud/launch_playout.sh" plc2 "$TYPE" \
    || echo "LAUNCH FAILED $TAG"
done
