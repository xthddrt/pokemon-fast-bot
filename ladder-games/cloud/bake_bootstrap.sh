#!/bin/bash
# AMI bake: install everything a fleet box needs (packages, venv, pinned
# engine wheel, python deps, code snapshot), verify the engine loads the net,
# scrub cloud-init state (so fleet user-data runs fresh and no bake
# credentials persist), then signal READY and idle for imaging.
# Prepended: AWS creds/region, S3_BUCKET, S3_PREFIX, WHEEL_NAME
set -uo pipefail
exec > /root/bake.log 2>&1

fail() {
  aws s3 cp /root/bake.log "s3://$S3_BUCKET/$S3_PREFIX/BAKE_FAILED.log" || true
  cat /root/bake.log > /dev/console 2>/dev/null || true
  exit 1
}
trap fail ERR
set -eE
export HOME=/root

dnf install -y python3.11 python3.11-devel tar gzip flock util-linux || dnf install -y python3.11 python3.11-devel tar gzip util-linux

cd /opt
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/code.tar.gz" . && tar xzf code.tar.gz

python3.11 -m venv venv
./venv/bin/pip install -q numpy
aws s3 cp "s3://$S3_BUCKET/wheels/$WHEEL_NAME" /tmp/
./venv/bin/pip install -q "/tmp/$WHEEL_NAME"
grep -v poke-engine foul-play/requirements.txt > /tmp/req.txt
./venv/bin/pip install -q -r /tmp/req.txt

# fail-loud verification: the baked venv must run the champion net
PE_NN_WEIGHTS=/opt/valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
  ./venv/bin/python -c "from poke_engine import engine_config; c=engine_config(); print(c); assert 'nn_active=true' in c"

# scrub cloud-init so (a) fleet user-data executes on next boot of the AMI,
# (b) the bake's credentials do not persist in the image
cloud-init clean --logs || rm -rf /var/lib/cloud/instances /var/lib/cloud/instance /var/log/cloud-init*.log

touch /opt/BAKE_READY
aws s3 cp /opt/BAKE_READY "s3://$S3_BUCKET/$S3_PREFIX/BAKE_READY"
echo "bake complete; idling for imaging"
