#!/bin/bash
# plc2 requeue supervisor. A slice can die two ways: a spot reclaim, or the
# reachable engine panic ("Invalid rest_turns value: -1") that pre-patch code
# turned into a whole-box kill. Either way the box writes FINISHED.<tag> with
# FAILED and its partial labels are already in S3, so relaunching the SAME
# tag/start/count resumes and costs only the positions in flight.
# Exits when every active slice reports OK.
set -u
ROOT=/Users/sallyliu/pokemon-fast-bot
S3=s3://pokebot-valuenet-389825051723/evallab/plc2
SHA=38e0c11ba765519dc55dbc7e26ab7dc34540b8b6241f7eb98547e51f8f655213
AWS=$(command -v aws)

# tag start count type workers
SLICES="
w06 300000 50000 c7a.8xlarge 32
z08 400000 85696 c7a.16xlarge 64
z09 485696 85696 c7a.16xlarge 64
z10 571392 85696 c7a.16xlarge 64
z11 657088 85696 c7a.16xlarge 64
z12 742784 85696 c7a.16xlarge 64
z13 828480 85696 c7a.16xlarge 64
z14 914176 85696 c7a.16xlarge 64
"
AZS=(subnet-0c80748d73536ffb2 subnet-01b4c6f3171ce3a80)

while true; do
  allok=1; i=0
  while read -r TAG START COUNT TYPE W; do
    [ -z "${TAG:-}" ] && continue
    i=$((i + 1))
    ST=$($AWS s3 cp $S3/FINISHED.$TAG - 2>/dev/null | tr -d '\n')
    if [ "$ST" = "OK" ]; then continue; fi
    allok=0
    [ -z "$ST" ] && continue                      # still running
    echo "$(date -u +%FT%TZ) REQUEUE $TAG ($ST) start=$START count=$COUNT"
    $AWS s3 rm $S3/FINISHED.$TAG --only-show-errors 2>/dev/null
    OUT=$(SUBNET=${AZS[$((i % 2))]} BUNDLE=/dev/null SKIP_BUNDLE=1 POSRUN=plc2 \
      TAG=$TAG START=$START COUNT=$COUNT N=10 ITERS=2000 W=$W WATCHDOG=12600 \
      WHEEL_SHA="$SHA" UD_TEMPLATE=$ROOT/evallab/cloud/userdata_plc2label.sh \
      "$ROOT/evallab/cloud/launch_playout.sh" plc2 "$TYPE" 2>&1)
    if echo "$OUT" | grep -q INSTANCE; then
      echo "$(date -u +%FT%TZ) RELAUNCHED $TAG $(echo "$OUT" | grep INSTANCE)"
    else
      echo "$(date -u +%FT%TZ) RELAUNCH BLOCKED $TAG (no capacity, will retry)"
    fi
  done <<< "$SLICES"
  [ "$allok" = "1" ] && { echo "$(date -u +%FT%TZ) ALL SLICES OK"; break; }
  sleep 90
done
