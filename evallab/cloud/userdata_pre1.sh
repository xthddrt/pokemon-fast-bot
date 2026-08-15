#!/bin/bash
# evallab STAGE 1 -- PRETRAIN on the full enc_pre1 cache. One job, self-terminate.
#
# Copied from userdata_canary.sh (the proven one). Differences:
#   * fetch pulls the WHOLE pre1 cache (21.7 GiB) + the whole plc1 cache;
#   * the page cache is warmed sequentially before training, so the first pass
#     is not 200k random EBS reads per step;
#   * TWO training processes (seeds) run concurrently at 32 threads each -- the
#     canary measured 32 threads to be as fast as 64, so this is free;
#   * every stage is RESUMABLE: a relaunch after a spot reclaim re-fetches,
#     recomputes the (deterministic) split, and each seed resumes from its
#     newest S3 checkpoint rather than restarting.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WHEEL__ __WATCHDOG__
#   __STEPS__ __THREADS__ __SEEDS__ __VAL_EVERY__ __CKPT_EVERY__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/nets_pre1

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
push(){ $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --only-show-errors 2>/dev/null; }

cat > /opt/uploader.sh <<UP
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2
while true; do
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors 2>/dev/null
  [ -f /opt/STATUS ] && $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null
  $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --only-show-errors 2>/dev/null
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
  $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED.$TAG $DEST/FINISHED.$TAG --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc --all) mem=$(free -g | awk '/Mem:/{print $2}')G tag=$TAG"
dnf install -y -q gcc python3.11 python3.11-devel tar gzip

cd /opt
$AWS s3 cp $S3/evallab/nets_pre1/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/vt_pre1.py
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
cd /opt/evallab
status "deps ready, vcpu=$(nproc --all) torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"

status "STAGE fetch (pre1 21.7 GiB + plc1 2.7 GiB)"
/opt/venv/bin/python vt_pre1.py fetch /opt/work
push; status "fetch done: $(du -sh /opt/work | cut -f1) free=$(df -h /opt | awk 'NR==2{print $4}')"

# Warm the page cache sequentially: 24 GiB off gp3 at ~1 GB/s takes ~30 s and
# turns every subsequent random gather into RAM access (box has 128 GiB).
status "STAGE warm"
cat /opt/work/enc_pre1/*.npy /opt/work/enc_plc1/*.npy > /dev/null
status "warm done, cached=$(free -g | awk '/Mem:/{print $6}')G"

status "STAGE split"
/opt/venv/bin/python vt_pre1.py split /opt/work
push; status "split done"

status "STAGE pretrain seeds=__SEEDS__ steps=__STEPS__ threads=__THREADS__"
PIDS=""
for SEED in __SEEDS__; do
  OMP_NUM_THREADS=__THREADS__ MKL_NUM_THREADS=__THREADS__ \
  nohup /opt/venv/bin/python vt_pre1.py pretrain /opt/work --seed $SEED \
    --steps __STEPS__ --threads __THREADS__ \
    --val-every __VAL_EVERY__ --ckpt-every __CKPT_EVERY__ \
    > /opt/work/pretrain_s$SEED.log 2>&1 &
  PIDS="$PIDS $!"
  status "launched seed $SEED pid $!"
done
RC=0
for P in $PIDS; do wait $P || RC=1; done
push
status "pretrain finished rc=$RC"
[ "$RC" = "0" ]

$AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --include "holdout_pred_*.npy" --only-show-errors
status "PRETRAIN DONE"
DONE_OK=1
finish
