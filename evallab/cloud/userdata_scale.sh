#!/bin/bash
# evallab DATA-SCALING-CURVE box. Ten trainings AT ONCE, then self-terminate.
#
# Structure copied verbatim from userdata_n.sh (the proven one). The only
# difference is the job: instead of 3 seeds at one training-set size, this runs
# 5 nested subset sizes x 2 seeds = 10 processes CONCURRENTLY, with the thread
# budget split so (sum of threads) == vCPU and the longest run (900k x 12k
# steps, the critical path) gets the largest share.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WHEEL__ __WATCHDOG__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/scale

mkdir -p /root/.aws /opt/work /opt/wheel
cat > /root/.aws/credentials <<'CRED'
[default]
aws_access_key_id = __AKID__
aws_secret_access_key = __ASEC__
CRED
chmod 600 /root/.aws/credentials
printf '[default]\nregion = us-east-2\n' > /root/.aws/config

AWS=$(command -v aws || echo /usr/bin/aws)
$AWS configure set default.s3.max_concurrent_requests 32
status(){ echo "$(date -u +%FT%TZ) $*" >> /opt/STATUS; $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null; }
push(){ $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --only-show-errors 2>/dev/null; }

cat > /opt/uploader.sh <<UP
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2
while true; do
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors 2>/dev/null
  [ -f /opt/STATUS ] && $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null
  $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "train.*.log" --only-show-errors 2>/dev/null
  sleep 45
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

status "boot on $(hostname) vcpu=$(nproc --all) tag=$TAG"
dnf install -y -q gcc python3.11 python3.11-devel tar gzip

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/vt_scale.py

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
$AWS s3 cp $S3/evallab/enc_plc1/ /opt/work/enc_plc1/ --recursive \
    --exclude "*" --include "*.npy" --include "*.npz" --include "*.json" \
    --only-show-errors
ls -la /opt/work/enc_plc1
status "fetch done: $(du -sh /opt/work/enc_plc1 | cut -f1)"

# warm the page cache once, so ten trainers do not each pay first-touch I/O
cat /opt/work/enc_plc1/*.npy > /dev/null || true
status "page cache warm"

# ---- THE GRID: N  steps  thread-weight --------------------------------------
# steps ~ sqrt(N): every schedule COMPLETES its cosine anneal, and smaller
# subsets still get MORE epochs than the 900k reference, so no subset is
# undertrained relative to it. Thread weights are shares of the vCPU budget
# (they sum to 96 per seed, 192 over both seeds) and are rescaled to whatever
# box we actually land, so the 10 runs always fit CONCURRENTLY.
VCPU=$(nproc --all)
GRID="50000:3000:6 100000:4000:10 200000:6000:16 500000:9000:26 900000:12000:38"
status "STAGE train 10 concurrent on $VCPU vCPU: $GRID x seeds 0,1"
PIDS=""
for G in $GRID; do
  NT=${G%%:*}; R=${G#*:}; ST=${R%%:*}; W=${R##*:}
  TH=$(( (VCPU * W + 96) / 192 )); [ "$TH" -lt 2 ] && TH=2
  for S in 0 1; do
    OMP_NUM_THREADS=$TH nohup /opt/venv/bin/python vt_scale.py train /opt/work \
        --n-train $NT --seed $S --steps $ST --threads $TH --eval-every 1000 \
        > /opt/work/train.n${NT}_s${S}.log 2>&1 &
    PIDS="$PIDS $!"
  done
done
sleep 20; status "launched: $(pgrep -fc vt_scale.py) processes"
FAIL=0
for p in $PIDS; do wait $p || FAIL=1; done
push
[ "$FAIL" = "0" ] || { status "A RUN FAILED"; grep -l Traceback /opt/work/train.*.log || true; exit 9; }
status "train done"

status "STAGE collect"
/opt/venv/bin/python vt_scale.py collect /opt/work > /opt/work/collect.log 2>&1
push

$AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" \
    --include "REPORT.*.json" --include "holdout_pred_*.npy" --include "*.log" \
    --only-show-errors
status "SCALE DONE -> $DEST/$TAG/"
DONE_OK=1
finish
