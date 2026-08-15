#!/bin/bash
# Supervisor for the CONSOLIDATED fleet: exactly one box tagged fleet-all
# (all 5 accounts, 1 slot each, max 5 concurrent games). Relaunches on spot
# reclaim, tracks S3 game count, emits heartbeats + stall alarms.
set -uo pipefail
AWS=/Users/sallyliu/.awscli-venv/bin/aws
REGION=us-east-2
BUCKET=pokebot-valuenet-389825051723
F="s3://$BUCKET/ladder-fleet/fleet1"
AMI_ID="${AMI_ID:-ami-0287b59d88f3fe375}"
ROOT=/Users/sallyliu/pokemon-fast-bot

count_games() {
  local n
  for _ in 1 2 3; do
    n=$($AWS s3 ls "$F/box-all/archive/games/" 2>/dev/null | grep -c PRE) && { echo "$n"; return; }
    sleep 2
  done
  echo "-1"
}

tick=0; last_total=-1; stall_ticks=0
echo "supervisor_all up: one box, 5 accounts x 1 slot, AMI=$AMI_ID"
while true; do
  tick=$((tick+1))
  st=$($AWS ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=fleet-all" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[-1].Instances[-1].State.Name' --output text 2>/dev/null)
  [ -z "$st" ] && st="none"
  if [ "$st" != "running" ] && [ "$st" != "pending" ]; then
    echo "RELAUNCH: fleet-all was '$st' -> launching replacement"
    $AWS s3 rm "$F/box-all/DRAINED" >/dev/null 2>&1 || true
    $AWS s3 rm "$F/box-all/READY" >/dev/null 2>&1 || true
    $AWS s3 rm "$F/box-all/archive/reclaim_notice.json" >/dev/null 2>&1 || true
    AMI_ID="$AMI_ID" bash "$ROOT/ladder-games/cloud/launch_consolidated.sh" >/dev/null 2>&1 \
      && echo "RELAUNCH OK" || echo "RELAUNCH FAILED"
  fi
  g=$(count_games)
  if [ $((tick % 15)) -eq 0 ]; then
    echo "heartbeat: ~$g games archived on S3 box-all (tick $tick)"
  fi
  if [ $((tick % 5)) -eq 0 ] && [ "$g" != "-1" ] && [ "$tick" -ge 15 ]; then
    if [ "$g" = "$last_total" ]; then
      stall_ticks=$((stall_ticks+1))
      if [ "$stall_ticks" -ge 3 ]; then
        echo "STALL: box-all stuck at $g games for ~15 min — check the box"
        stall_ticks=0
      fi
    else
      stall_ticks=0
    fi
    last_total=$g
  fi
  sleep 60
done
