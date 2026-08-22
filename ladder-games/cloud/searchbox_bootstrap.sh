#!/bin/bash
# r64 search box: remote MCTS worker for the big-compute ladder config.
# Ultra-light by design -- search_server.py needs only the engine wheel, the
# champion net + sidecar, and stdlib. No foul-play checkout, no node, no PS.
# Runs until stop_search_box.sh terminates it (spot, terminate-on-shutdown).
set -uo pipefail
exec > /root/run.log 2>&1
export HOME="${HOME:-/root}"
dnf install -y -q python3.11 tar gzip || exit 1
aws s3 cp "s3://$S3_BUCKET/sampling/cache/wheel-arm-winhunt-v1.tar.gz" /opt/wc.tar.gz || exit 1
tar xzf /opt/wc.tar.gz -C / || exit 1
/opt/venv/bin/python -c "import poke_engine" || exit 1
mkdir -p /opt/nets
aws s3 cp "s3://$S3_BUCKET/nets/v10c0.bin" /opt/nets/v10c0.bin || exit 1
aws s3 cp "s3://$S3_BUCKET/nets/v10c0.constants.json" /opt/nets/v10c0.constants.json || exit 1
aws s3 cp "s3://$S3_BUCKET/nets/search_server.py" /opt/search_server.py || exit 1
export PE_NN_WEIGHTS=/opt/nets/v10c0.bin
# server default workers = max(nproc, 64): one wave for 64 worlds on this box
exec /opt/venv/bin/python /opt/search_server.py --port 8000
