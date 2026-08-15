#!/bin/bash
# Relay-architecture ladder canary: the FULL bot (state, world sampling,
# search) runs here on AWS; the Showdown websocket goes through an SSH
# reverse tunnel to the Mac's relay (127.0.0.1:8765 -> Mac -> sim3.psim.us),
# so logins originate from the residential IP. 1 core per game: 8 worlds in
# 8 sequential waves of SEARCH_MS/8 each. Spot drain: no new games on the
# 2-minute notice; in-flight tags recorded for spot_reclaimed tagging.
#
# Prepended by launcher: AWS creds/region, S3_BUCKET, S3_PREFIX, PS_PASSWORD,
# ACCOUNT_1, ACCOUNT_2, GAMES_PER_ACCOUNT, PAR, SEARCH_MS
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
# Prebuilt wheel saves ~5 min/box (CLOUD_PLAYBOOK 4.1). PINNED by exact name:
# republish + bump this name whenever poke-engine source changes — a stale
# wheel is silent (conformance-gate discipline). Falls back to source build.
WHEEL_NAME="poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl"  # built 2026-08-08, post-F1
if aws s3 cp "s3://$S3_BUCKET/wheels/$WHEEL_NAME" /tmp/; then
  ./venv/bin/pip install -q "/tmp/$WHEEL_NAME"
else
  ./venv/bin/pip install ./poke-engine/poke-engine-py \
    --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
fi
grep -v poke-engine foul-play/requirements.txt > /tmp/req.txt
./venv/bin/pip install -q -r /tmp/req.txt

mkdir -p /opt/claims /opt/fleet-logs /opt/archive/games /opt/foul-play/logs

# ---- spot notice watcher (drain + record in-flight + tight sync)
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

# ---- signal ready-for-tunnel, then wait for the Mac's reverse tunnel
touch /opt/READY && aws s3 cp /opt/READY "s3://$S3_BUCKET/$S3_PREFIX/READY"
echo "waiting for relay tunnel on 127.0.0.1:8765..."
for i in $(seq 1 180); do
  if ./venv/bin/python -c "
import socket; s=socket.socket(); s.settimeout(2)
s.connect(('127.0.0.1', 8765)); s.close()" 2>/dev/null; then
    echo "tunnel is up"
    break
  fi
  [ "$i" -lt 180 ] || { echo "tunnel never appeared"; exit 1; }
  sleep 10
done

# Each SLOT is an independent loop: the moment ITS game ends it archives and
# searches again (no round barrier — a fast game never idles behind a slow
# sibling; the per-account search serialization is handled by the client's
# search watchdog). GAMES_PER_SLOT=0 means run until DRAIN.
run_slot() {
  local account="$1" slot="$2" per_slot="$3" g=0
  local slug=$(echo "$account" | tr -c 'A-Za-z0-9' '_')
  cd /opt/foul-play
  while { [ "$per_slot" -eq 0 ] || [ "$g" -lt "$per_slot" ]; } && [ ! -f /opt/DRAIN ]; do
    g=$((g + 1))
    local start=$(date +%s)
    FP_CLAIM_DIR=/opt/claims FP_LOG_SUBDIR="${slug}_s${slot}" \
    /opt/venv/bin/python run.py \
      --websocket-uri "ws://127.0.0.1:8765/showdown/websocket" \
      --ps-username "$account" --ps-password "$PS_PASSWORD" \
      --bot-mode search_ladder --pokemon-format gen9randombattleblitz \
      --run-count 1 \
      --search-time-ms "$SEARCH_MS" --first-turn-search-time-ms "$SEARCH_MS" \
      --search-parallelism 8 --search-pool-workers 1 --search-threads 1 \
      --nn-weights ../valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
      --selection-argmax-only --tera-gate-per-mon 0.001 --tera-gate-visit-frac 0.3333 \
      --save-replay always --log-level INFO --log-to-file \
      > "/opt/fleet-logs/${slug}_s${slot}_g${g}.log" 2>&1 || true
    # per-game archive; flock serializes concurrent slots' index writes
    flock /opt/archive/.lock env \
      FP_ARCHIVE_WORKSPACE=/opt FP_ARCHIVE_DIR=/opt/archive \
      FP_ARCHIVE_LOGS=/opt/foul-play/logs FP_ARCHIVE_NO_LOSSES=1 \
      /opt/venv/bin/python /opt/ladder-games/archive_game.py \
        --since "$start" --flags "aws-relay $account s$slot" || true
    echo "$account slot$slot: $g/${per_slot:-inf} done"
  done
}

PER_SLOT=$(( (GAMES_PER_ACCOUNT + PAR - 1) / PAR ))
[ "$GAMES_PER_ACCOUNT" -eq 0 ] && PER_SLOT=0
pids=()
for acct in "$ACCOUNT_1" "$ACCOUNT_2"; do
  for s in $(seq 1 "$PAR"); do
    run_slot "$acct" "$s" "$PER_SLOT" &
    pids+=($!)
    sleep 3
  done
done
for p in "${pids[@]}"; do wait "$p" || true; done

aws s3 sync /opt/archive "s3://$S3_BUCKET/$S3_PREFIX/archive/" --quiet || true
aws s3 sync /opt/foul-play/logs "s3://$S3_BUCKET/$S3_PREFIX/rawlogs/" --quiet || true
touch /opt/DONE && aws s3 cp /opt/DONE "s3://$S3_BUCKET/$S3_PREFIX/DONE"
aws s3 cp /root/run.log "s3://$S3_BUCKET/$S3_PREFIX/run.log" || true
DONE_OK=1
shutdown -h now
