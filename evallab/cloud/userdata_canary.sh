#!/bin/bash
# evallab END-TO-END TRAINING CANARY box. One job, then self-terminate.
#
# Stages, in this order, each uploading its report the moment it finishes so a
# later failure never costs the earlier numbers:
#   fetch  -- the whole plc1 cache + a 1 % block sample of the pretrain cache
#   train  -- pretrain (lambda-return) -> fine-tune (label_p) -> holdout eval
#   bench  -- fwd+bwd throughput vs thread count, for the full-run projection
#   verify -- 20 plc1 + 20 pre1 rows re-encoded from the raw state, bit-compared
#
# Structure copied verbatim from userdata_enc.sh (the proven one). Differences:
# torch is installed (CPU wheel index), and OMP_NUM_THREADS is NOT pinned to 1,
# because here the box IS the trainer.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WHEEL__ __WATCHDOG__
#   __BLOCKS__ __BLOCK_ROWS__ __PRE_EPOCHS__ __FT_EPOCHS__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/canary

mkdir -p /root/.aws /opt/work /opt/wheel
cat > /root/.aws/credentials <<'CRED'
[default]
aws_access_key_id = __AKID__
aws_secret_access_key = __ASEC__
CRED
chmod 600 /root/.aws/credentials
printf '[default]\nregion = us-east-2\n' > /root/.aws/config

AWS=$(command -v aws || echo /usr/bin/aws)
status(){ echo "$(date -u +%FT%TZ) $*" >> /opt/STATUS; $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null; }
push(){ $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --only-show-errors 2>/dev/null; }

cat > /opt/uploader.sh <<UP
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2
while true; do
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors 2>/dev/null
  [ -f /opt/STATUS ] && $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null
  sleep 60
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

nohup bash -c "sleep __WATCHDOG__; $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED.$TAG; } || { echo "FAILED rc=$rc" > /opt/FINISHED.$TAG; }
  $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED.$TAG $DEST/FINISHED.$TAG --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc --all) tag=$TAG"
dnf install -y -q gcc python3.11 python3.11-devel tar gzip

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/vt_canary.py
test -f /opt/valuenet/lossless_encoder.py

python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
$AWS s3 cp __WHEEL__ /opt/wheel/ --only-show-errors
/opt/venv/bin/pip3 install -q /opt/wheel/*.whl
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy boto3
/opt/venv/bin/pip3 install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
/opt/venv/bin/python -c "import poke_engine, numpy, boto3, torch; print('deps ok', numpy.__version__, torch.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
W=$(nproc --all)
cd /opt/evallab
status "deps ready, vcpu=$W torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"

status "STAGE fetch"
/opt/venv/bin/python vt_canary.py fetch /opt/work --blocks __BLOCKS__ --block-rows __BLOCK_ROWS__
push; status "fetch done: $(du -sh /opt/work | cut -f1)"

status "STAGE train"
/opt/venv/bin/python vt_canary.py train /opt/work --pre-epochs __PRE_EPOCHS__ --ft-epochs __FT_EPOCHS__
push; status "train done"

status "STAGE bench"
/opt/venv/bin/python vt_canary.py bench /opt/work
push; status "bench done"

status "STAGE verify"
/opt/venv/bin/python vt_canary.py verify /opt/work
push; status "verify done"

$AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --only-show-errors
$AWS s3 cp /opt/work/ckpt_finetune.pt $DEST/$TAG/ckpt_finetune.pt --only-show-errors || true
status "CANARY DONE"
DONE_OK=1
finish
