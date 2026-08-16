#!/bin/bash
# Fleet box boot (runs on the BAKED AMI — venv/wheel/deps preinstalled, so
# boot-to-first-search is ~90s). N accounts x PAR continuous slot loops: each
# slot re-queues the moment ITS game ends; GAMES_PER_ACCOUNT=0 = run until
# DRAIN (spot notice or manual /opt/DRAIN). Websocket via the Mac relay
# tunnel on 127.0.0.1:8765.
# Prepended: AWS creds/region, S3_BUCKET, S3_PREFIX, PS_PASSWORD,
# ACCOUNTS (comma-separated), GAMES_PER_ACCOUNT, PAR, SEARCH_MS
set -uo pipefail
exec > /root/run.log 2>&1

DONE_OK=0
finish() {
  rc=$?
  trap - ERR EXIT
  if [ "$DONE_OK" = "1" ]; then exit 0; fi
  echo "FAILED rc=$rc"
  cat /root/run.log > /dev/console 2>/dev/null || true
  aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/FAILED.log" || true
  shutdown -h now
}
trap finish ERR EXIT
set -eE
export HOME=/root

cd /opt
# refresh code over the baked snapshot (seconds; venv untouched). Code lives
# at the shared fleet prefix; S3_PREFIX is per-box (per-account) for markers,
# archive and rawlogs.
aws s3 cp "s3://$S3_BUCKET/${S3_CODE:-$S3_PREFIX}/code.tar.gz" . && tar xzf code.tar.gz
mkdir -p /opt/claims /opt/fleet-logs /opt/archive/games /opt/foul-play/logs

# ---- spot notice watcher: drain + record in-flight + tight sync until death
(
  while true; do
    TOKEN=$(curl -s -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
      http://169.254.169.254/latest/api/token || true)
    ACTION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/spot/instance-action || true)
    if echo "$ACTION" | grep -q '"action"'; then
      date -u +%FT%TZ > /opt/DRAIN
      ls /opt/claims > /opt/archive/reclaim_inflight.txt 2>/dev/null || true
      ls /opt/archive/games > /opt/archive/reclaim_archived.txt 2>/dev/null || true
      echo "$ACTION" > /opt/archive/reclaim_notice.json
      while true; do
        aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
        aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
        sleep 10
      done
    fi
    sleep 5
  done
) &

# ---- steady sync
( while true; do
    sleep 30
    aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
    aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
  done ) &

touch /opt/READY && aws s3 cp /opt/READY "s3://$S3_BUCKET/$S3_PREFIX/READY"
echo "waiting for relay tunnel on 127.0.0.1:8765..."
for i in $(seq 1 180); do
  if /opt/venv/bin/python -c "
import socket; s=socket.socket(); s.settimeout(2)
s.connect(('127.0.0.1', 8765)); s.close()" 2>/dev/null; then
    echo "tunnel is up"; break
  fi
  [ "$i" -lt 180 ] || { echo "tunnel never appeared"; exit 1; }
  sleep 10
done

run_slot() {
  local account="$1" slot="$2" per_slot="$3" fmt="${4:-gen9randombattleblitz}" g=0
  local slug=$(echo "$account" | tr -c 'A-Za-z0-9' '_')
  cd /opt/foul-play
  # /opt/STOP_<slug> retires ONE account (Sally 2026-08-09: play until 2400 Elo,
  # then stop that account) while the others keep laddering; /opt/DRAIN still
  # stops the whole box. Checked at the top of the loop, so the account always
  # FINISHES and archives its current game before retiring — never a forfeit.
  while { [ "$per_slot" -eq 0 ] || [ "$g" -lt "$per_slot" ]; } \
        && [ ! -f /opt/DRAIN ] && [ ! -f "/opt/STOP_${slug}" ]; do
    g=$((g + 1))
    local start=$(date +%s)
    FP_CLAIM_DIR=/opt/claims FP_LOG_SUBDIR="${slug}_${fmt}_s${slot}" \
    /opt/venv/bin/python run.py \
      --websocket-uri "ws://127.0.0.1:8765/showdown/websocket" \
      --ps-username "$account" --ps-password "$PS_PASSWORD" \
      --bot-mode search_ladder --pokemon-format "$fmt" \
      --run-count 1 \
      --search-time-ms "$SEARCH_MS" --first-turn-search-time-ms "$SEARCH_MS" \
      --search-parallelism 8 --search-pool-workers 8 --search-threads 1 \
      --nn-weights ../valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
      --selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.25 \
      --save-replay always --log-level INFO --log-to-file \
      > "/opt/fleet-logs/${slug}_s${slot}_g${g}.log" 2>&1 || true
    flock /opt/archive/.lock env \
      FP_ARCHIVE_WORKSPACE=/opt FP_ARCHIVE_DIR=/opt/archive \
      FP_ARCHIVE_LOGS=/opt/foul-play/logs FP_ARCHIVE_NO_LOSSES=1 \
      /opt/venv/bin/python /opt/ladder-games/archive_game.py \
        --since "$start" --flags "fleet $account s$slot" || true
    echo "$account slot$slot: game $g done"
  done
}

PER_SLOT=0
if [ "$GAMES_PER_ACCOUNT" -gt 0 ]; then
  PER_SLOT=$(( (GAMES_PER_ACCOUNT + PAR - 1) / PAR ))
fi
# Regular randbats only (Sally 2026-08-09): slower games keep aggregate
# battle-starts under Showdown's per-IP 12/3min cap on the shared relay IP.
# 6 slots = Showdown's per-account concurrent-battle ceiling.
FORMATS="${FORMATS:-gen9randombattle:6}"
pids=()
IFS=',' read -ra ACCTS <<< "$ACCOUNTS"
for acct in "${ACCTS[@]}"; do
  IFS=',' read -ra FSPECS <<< "$FORMATS"
  for spec in "${FSPECS[@]}"; do
    fmt="${spec%%:*}"; count="${spec##*:}"
    for s in $(seq 1 "$count"); do
      run_slot "$acct" "$s" "$PER_SLOT" "$fmt" &
      pids+=($!)
      # 15s stagger: 6 near-simultaneous logins+searches overflow both the
      # per-connection chat throttle and the shared IP's 12-preps/3min budget
      sleep 15
    done
  done
done
for p in "${pids[@]}"; do wait "$p" || true; done

aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
touch /opt/DRAINED && aws s3 cp /opt/DRAINED "s3://$S3_BUCKET/$S3_PREFIX/DRAINED"
aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" || true
DONE_OK=1
shutdown -h now
