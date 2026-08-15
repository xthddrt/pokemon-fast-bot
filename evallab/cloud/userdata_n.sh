#!/bin/bash
# evallab ENGINE-NATIVE-WIDTH training box. One job, then self-terminate.
#
# Structure copied verbatim from userdata_fast.sh (the proven one). Only the
# script (vt_n.py), the width (128/256) and the publish prefix (nets_v8n) differ,
# plus a short inference-cost bench of BOTH widths before training.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WHEEL__ __WATCHDOG__
#   __STEPS__ __SEEDS__ __THREADS__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/canary_n
PUB=$S3/evallab/nets_v8n

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
  for f in /opt/work/train.s*.log; do [ -f "\$f" ] && $AWS s3 cp "\$f" $DEST/$TAG/\$(basename \$f) --only-show-errors 2>/dev/null; done
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
test -f /opt/evallab/vt_n.py

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
cd /opt/evallab
status "deps ready, vcpu=$(nproc --all) torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"

status "STAGE fetch plc1"
$AWS s3 cp $S3/evallab/enc_plc1/ /opt/work/enc_plc1/ --recursive --only-show-errors
ls -la /opt/work/enc_plc1 | head -30
status "fetch done: $(du -sh /opt/work/enc_plc1 | cut -f1)"

# ---- inference cost of BOTH widths, on an otherwise idle box, 1 thread ------
status "STAGE bench"
OMP_NUM_THREADS=1 /opt/venv/bin/python vt_n.py bench /opt/work --threads 1 \
    > /opt/work/bench.log 2>&1
push
$AWS s3 cp /opt/work/bench.log $DEST/$TAG/bench.log --only-show-errors || true
status "bench done"

status "STAGE train 128/256 seeds=__SEEDS__ steps=__STEPS__ threads=__THREADS__"
PIDS=""
for S in $(echo __SEEDS__ | tr ',' ' '); do
  OMP_NUM_THREADS=__THREADS__ nohup /opt/venv/bin/python vt_n.py train /opt/work \
      --seed $S --steps __STEPS__ --threads __THREADS__ \
      --mon-hid 128 --trunk-hid 256 \
      > /opt/work/train.s$S.log 2>&1 &
  PIDS="$PIDS $!"
done
FAIL=0
for p in $PIDS; do wait $p || FAIL=1; done
push
[ "$FAIL" = "0" ] || { status "A SEED FAILED"; exit 9; }
status "train done"

status "STAGE pick"
/opt/venv/bin/python vt_n.py pick /opt/work --seeds __SEEDS__ > /opt/work/pick.log 2>&1
push
status "pick done"

$AWS s3 cp /opt/work/ $PUB/$TAG/ --recursive --exclude "*" \
    --include "REPORT.*.json" --include "ckpt_n_s*.pt" \
    --include "holdout_pred_*.npy" --include "*.log" --only-show-errors
status "PUBLISHED to $PUB/$TAG/"
DONE_OK=1
finish
