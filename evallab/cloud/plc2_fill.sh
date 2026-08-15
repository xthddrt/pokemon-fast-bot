#!/bin/bash
# plc2 wave-2 filler. The account's spot ceiling is ~480 vCPU, so the fleet
# cannot be launched in one shot; this retries the remaining slices until each
# one lands, which packs the ceiling as wave-1 boxes self-terminate.
# Slices tile 400000..999872 exactly; wave 1 owns 0..400000.
set -u
ROOT=/Users/sallyliu/pokemon-fast-bot
SHA=38e0c11ba765519dc55dbc7e26ab7dc34540b8b6241f7eb98547e51f8f655213
AZS=(subnet-0c80748d73536ffb2 subnet-01b4c6f3171ce3a80)
BASE=400000
SLICE=85696
NSLICE=7
pending=$(seq 0 $((NSLICE - 1)))

while [ -n "$pending" ]; do
  still=""
  for k in $pending; do
    TAG=$(printf 'z%02d' $((k + 8)))
    START=$((BASE + k * SLICE))
    OUT=$(SUBNET=${AZS[$((k % 2))]} BUNDLE=/dev/null SKIP_BUNDLE=1 POSRUN=plc2 \
      TAG=$TAG START=$START COUNT=$SLICE N=10 ITERS=2000 W=64 WATCHDOG=12600 \
      WHEEL_SHA="$SHA" UD_TEMPLATE=$ROOT/evallab/cloud/userdata_plc2label.sh \
      "$ROOT/evallab/cloud/launch_playout.sh" plc2 c7a.16xlarge 2>&1)
    if echo "$OUT" | grep -q INSTANCE; then
      echo "$(date -u +%FT%TZ) LAUNCHED $TAG start=$START $(echo "$OUT" | grep INSTANCE)"
    else
      still="$still $k"
    fi
  done
  pending=$(echo $still)
  [ -n "$pending" ] && { echo "$(date -u +%FT%TZ) waiting, pending:$pending"; sleep 60; }
done
echo "$(date -u +%FT%TZ) ALL WAVE-2 SLICES LAUNCHED"
