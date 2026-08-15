#!/bin/bash
# Watchdog free-run: min-elo scheduler + box supervisor. Events on stdout.
# Stop cleanly: touch $SP/STOP_FREERUN (finishes current game, drains box).
AWS=/Users/sallyliu/.awscli-venv/bin/aws
B=pokebot-valuenet-389825051723
P=s3://$B/ladder-fleet/v7warm2
SP="$(cd "$(dirname "$0")" && pwd)"
N=$(date +%s)
settle() {  # wait for any in-flight game to archive before first pick
  local PREV=$(cat "$SP/.v7warm2_index_count" 2>/dev/null || echo 0) T0=$(date +%s)
  while [ $(( $(date +%s) - T0 )) -lt 1200 ]; do
    CUR=$( ($AWS s3 cp $P/archive/index.jsonl - 2>/dev/null || true) | grep -c . )
    [ "$CUR" -le "$PREV" ] || { python3 "$SP/import_new.py"; return; }
    $AWS s3 ls $P/ 2>/dev/null | grep -qE 'DONE|RECLAIMED|FAILED' && return
    sleep 45
  done
}
ensure_box() {
  while true; do
    ls=$($AWS s3 ls $P/ 2>/dev/null || true)
    ST=$($AWS ec2 describe-instances --region us-east-2 \
      --filters "Name=tag:Name,Values=fleet-v7game1" "Name=instance-state-name,Values=running,pending" \
      --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | tr -d '\n')
    if [ -n "$ST" ] && echo "$ls" | grep -qw READY && ! echo "$ls" | grep -qE 'RECLAIMED|FAILED' && ! echo "$ls" | grep -qw DONE; then
      return
    fi
    if [ -z "$ST" ] || echo "$ls" | grep -qE 'RECLAIMED|FAILED' || echo "$ls" | grep -qw DONE; then
      echo "BOX DOWN ($(echo "$ls" | grep -oE 'RECLAIMED|FAILED|DONE' | head -1 || echo gone)) — relaunching replacement 32-vCPU spot box"
      ID=$(bash "$SP/launch_warm.sh" 2>/dev/null | tail -1)
      echo "replacement launched: $ID (build ~5 min, tunnel follows tag)"
      T0=$(date +%s)
      while [ $(( $(date +%s) - T0 )) -lt 1200 ]; do
        $AWS s3 ls $P/READY >/dev/null 2>&1 && { echo "replacement READY"; return; }
        $AWS s3 ls $P/FAILED.log >/dev/null 2>&1 && { echo "replacement FAILED — trying again"; break; }
        sleep 30
      done
    else
      sleep 30   # box pending/building
    fi
  done
}
settle
while true; do
  [ -f "$SP/STOP_FREERUN" ] && { echo "STOP requested — draining box"; date -u +%FT%TZ | $AWS s3 cp - $P/DRAIN --quiet; exit 0; }
  ensure_box
  PICK=$(python3 "$SP/elo_pick.py" pick 2>/dev/null)
  ACCT=$(echo "$PICK" | python3 -c "import json,sys; print(json.load(sys.stdin)['lowest'])" 2>/dev/null)
  ELO=$(echo "$PICK" | python3 -c "import json,sys; print(json.load(sys.stdin)['table'][0]['elo'])" 2>/dev/null)
  [ -n "$ACCT" ] || { echo "elo pick failed — retry in 60s"; sleep 60; continue; }
  N=$((N+1))
  echo "NEXT UP: $ACCT ($ELO) — game q$N queuing"
  printf 'ACCOUNT=%s\nFORMAT=gen9randombattle\nFIRST_MS=14500\n' "$ACCT" | $AWS s3 cp - "$P/trigger/q$N" --quiet
  PREV=$(cat "$SP/.v7warm2_index_count" 2>/dev/null || echo 0)
  T0=$(date +%s); DEAD=0
  while true; do
    sleep 45
    CUR=$( ($AWS s3 cp $P/archive/index.jsonl - 2>/dev/null || true) | grep -c . )
    [ "$CUR" -gt "$PREV" ] && break
    ls=$($AWS s3 ls $P/ 2>/dev/null || true)
    echo "$ls" | grep -qE 'RECLAIMED|FAILED' && { echo "box lost mid-cycle (in-flight game, if any, is an infra loss)"; DEAD=1; break; }
    echo "$ls" | grep -qw DONE && { echo "box idled out mid-cycle"; DEAD=1; break; }
    [ $(( $(date +%s) - T0 )) -gt 2700 ] && { echo "NO GAME in 45 min — halting for inspection"; exit 1; }
  done
  [ "$DEAD" = 1 ] && continue
  python3 "$SP/import_new.py"
  sleep 8
  python3 "$SP/elo_pick.py" log "$ACCT" | python3 -c "import json,sys; d=json.load(sys.stdin); print('ELO NOW: %s = %.1f' % (d['account'], d['elo']))" 2>/dev/null || echo "elo log failed (non-fatal)"
done
