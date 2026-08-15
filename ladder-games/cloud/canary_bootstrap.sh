#!/bin/bash
# AWS spot canary: ladder games on 2 accounts, 2 parallel games each, until
# GAMES_PER_ACCOUNT games per account. Per game: 8 worlds x 1 core in 8 waves
# of SEARCH_MS/8 each. Spot-notice drain: no NEW games after the 2-minute
# reclaim warning; in-flight tags recorded to S3 so unarchived games get
# tagged spot_reclaimed at reconciliation (excluded from training + register).
#
# Prepended by the launcher: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_DEFAULT_REGION, S3_BUCKET, S3_PREFIX, PS_PASSWORD, ACCOUNT_1, ACCOUNT_2,
# GAMES_PER_ACCOUNT, PAR, SEARCH_MS
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

# cloud-init runs user-data with an EMPTY $HOME (CLOUD_PLAYBOOK 2.2): pin it
# before rustup so cargo lands in /root/.cargo, and source that path directly.
export HOME=/root

dnf install -y gcc gcc-c++ python3.11 python3.11-devel tar gzip
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source /root/.cargo/env

cd /opt
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/code.tar.gz" . && tar xzf code.tar.gz

python3.11 -m venv venv
./venv/bin/pip install -q maturin numpy
./venv/bin/pip install ./poke-engine/poke-engine-py \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
grep -v poke-engine foul-play/requirements.txt > /tmp/req.txt
./venv/bin/pip install -q -r /tmp/req.txt

mkdir -p /opt/claims /opt/fleet-logs /opt/archive/games /opt/foul-play/logs

# ---- spot notice watcher: DRAIN + record in-flight + tight sync until death
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

# ---- steady sync every 30s
( while true; do
    sleep 30
    aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
    aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
  done ) &

run_account() {
  local account="$1" total="$2" done=0
  local slug=$(echo "$account" | tr -c 'A-Za-z0-9' '_')
  cd /opt/foul-play
  while [ "$done" -lt "$total" ] && [ ! -f /opt/DRAIN ]; do
    local round_start=$(date +%s)
    local pids=()
    for slot in $(seq 1 "$PAR"); do
      [ $((done + ${#pids[@]})) -lt "$total" ] || break
      [ ! -f /opt/DRAIN ] || break
      FP_CLAIM_DIR=/opt/claims FP_LOG_SUBDIR="${slug}_s${slot}" \
      /opt/venv/bin/python run.py \
        --websocket-uri "wss://sim3.psim.us/showdown/websocket" \
        --ps-username "$account" --ps-password "$PS_PASSWORD" \
        --bot-mode search_ladder --pokemon-format gen9randombattleblitz \
        --run-count 1 \
        --search-time-ms "$SEARCH_MS" --first-turn-search-time-ms "$SEARCH_MS" \
        --search-parallelism 8 --search-pool-workers 1 --search-threads 1 \
        --nn-weights ../valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
        --selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.3333 \
        --save-replay always --log-level INFO --log-to-file \
        > "/opt/fleet-logs/${slug}_s${slot}_g$((done + slot)).log" 2>&1 &
      pids+=($!)
      sleep 2
    done
    [ ${#pids[@]} -gt 0 ] || break
    for p in "${pids[@]}"; do wait "$p" || true; done
    done=$((done + ${#pids[@]}))
    FP_ARCHIVE_WORKSPACE=/opt FP_ARCHIVE_DIR=/opt/archive \
    FP_ARCHIVE_LOGS=/opt/foul-play/logs FP_ARCHIVE_NO_LOSSES=1 \
      /opt/venv/bin/python /opt/ladder-games/archive_game.py \
        --since "$round_start" --flags "aws-canary $account" || true
    echo "$account: $done/$total done"
  done
}

run_account "$ACCOUNT_1" "$GAMES_PER_ACCOUNT" &
A1=$!
run_account "$ACCOUNT_2" "$GAMES_PER_ACCOUNT" &
A2=$!
wait "$A1" || true
wait "$A2" || true

aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
touch /opt/DONE && aws s3 cp /opt/DONE "s3://$S3_BUCKET/$S3_PREFIX/DONE"
aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" || true
DONE_OK=1
shutdown -h now
