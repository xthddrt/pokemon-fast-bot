#!/bin/bash
# WARM ladder box: build once, then idle on an S3 trigger loop.
# Each trigger token = play ONE game (v7_puct, 16w x 2t, first FIRST_MS then
# SEARCH_MS), archive, sync, return to idle. Auto-shutdown after IDLE_MAX s
# without a trigger, or on s3://$S3_PREFIX/DRAIN.
# Prepended: AWS creds/region, S3_BUCKET, S3_PREFIX, PS_PASSWORD, ACCOUNT,
# SEARCH_MS, FIRST_MS, WORLDS, POOL, THREADS, IDLE_MAX
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

PE_NN_WEIGHTS=/opt/valuenet/m4_artifacts/valuenet_v7_puct.bin \
  ./venv/bin/python -c "from poke_engine import engine_config; c=engine_config(); print(c); assert 'nn_active=true' in c"

mkdir -p /opt/claims /opt/fleet-logs /opt/archive/games /opt/foul-play/logs

# spot-notice watcher: 2-min reclaim warning -> local DRAIN (no new games),
# RECLAIMED marker for the Mac supervisor, tight sync until death
(
  while true; do
    TOKEN=$(curl -s -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
      http://169.254.169.254/latest/api/token || true)
    ACTION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/spot/instance-action || true)
    if echo "$ACTION" | grep -q '"action"'; then
      date -u +%FT%TZ > /opt/DRAIN
      echo "$ACTION" > /opt/archive/reclaim_notice.json
      date -u +%FT%TZ | aws s3 cp - "s3://$S3_BUCKET/$S3_PREFIX/RECLAIMED" || true
      while true; do
        aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
        aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
        sleep 10
      done
    fi
    sleep 5
  done
) &

( while true; do
    sleep 30
    aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
    aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
  done ) &

echo "waiting for relay tunnel on 127.0.0.1:8765..."
T0=$(date +%s)
until (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -lt 900 ] || { echo "tunnel never came up"; false; }
  sleep 5
done
exec 3>&- 2>/dev/null || true
echo "tunnel up"

touch /opt/READY && aws s3 cp /opt/READY "s3://$S3_BUCKET/$S3_PREFIX/READY"

G=0
LAST_TRIGGER=$(date +%s)
while true; do
  if [ -f /opt/DRAIN ]; then
    echo "local DRAIN (spot notice) — no new games"; break
  fi
  if aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/DRAIN" >/dev/null 2>&1; then
    echo "DRAIN marker — shutting down"; break
  fi
  if [ $(( $(date +%s) - LAST_TRIGGER )) -ge "$IDLE_MAX" ]; then
    echo "idle ${IDLE_MAX}s — auto shutdown"; break
  fi
  TOK=$( (aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/trigger/" 2>/dev/null || true) | awk '{print $4}' | head -1)
  if [ -z "$TOK" ]; then sleep 5; continue; fi
  aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/trigger/$TOK" /tmp/tok --quiet || true
  aws s3 rm "s3://$S3_BUCKET/$S3_PREFIX/trigger/$TOK" --quiet || true
  T_ACCOUNT=$(sed -n 's/^ACCOUNT=//p' /tmp/tok); T_ACCOUNT="${T_ACCOUNT:-$ACCOUNT}"
  T_FORMAT=$(sed -n 's/^FORMAT=//p' /tmp/tok); T_FORMAT="${T_FORMAT:-gen9randombattle}"
  T_FIRST=$(sed -n 's/^FIRST_MS=//p' /tmp/tok); T_FIRST="${T_FIRST:-$FIRST_MS}"
  slug=$(echo "$T_ACCOUNT" | tr -c 'A-Za-z0-9' '_')
  LAST_TRIGGER=$(date +%s)
  G=$((G + 1))
  echo "=== trigger $TOK -> game $G acct=$T_ACCOUNT fmt=$T_FORMAT first=$T_FIRST ($(date -u +%FT%TZ))"
  cd /opt/foul-play
  START=$(date +%s)
  FP_CLAIM_DIR=/opt/claims FP_LOG_SUBDIR="${slug}_v7warm" \
  /opt/venv/bin/python run.py \
    --websocket-uri "ws://127.0.0.1:8765/showdown/websocket" \
    --ps-username "$T_ACCOUNT" --ps-password "$PS_PASSWORD" \
    --bot-mode search_ladder --pokemon-format "$T_FORMAT" \
    --run-count 1 \
    --search-time-ms "$SEARCH_MS" --first-turn-search-time-ms "$T_FIRST" \
    --search-parallelism "$WORLDS" --search-pool-workers "$POOL" \
    --search-threads "$THREADS" \
    --nn-weights ../valuenet/m4_artifacts/valuenet_v7_puct.bin \
    --selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.3333 \
    --save-replay always --log-level INFO --log-to-file \
    > "/opt/fleet-logs/${slug}_v7warm_g${G}.log" 2>&1 || true
  FP_ARCHIVE_WORKSPACE=/opt FP_ARCHIVE_DIR=/opt/archive \
  FP_ARCHIVE_LOGS=/opt/foul-play/logs FP_ARCHIVE_NO_LOSSES=1 \
    /opt/venv/bin/python /opt/ladder-games/archive_game.py \
      --since "$START" --flags "aws-v7warm $T_ACCOUNT $T_FORMAT 16w x 2t first${T_FIRST} ms${SEARCH_MS}" || true
  aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
  aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" --quiet || true
  echo "game $G archived; back to idle"
done

aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
touch /opt/DONE && aws s3 cp /opt/DONE "s3://$S3_BUCKET/$S3_PREFIX/DONE"
aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" || true
DONE_OK=1
shutdown -h now
