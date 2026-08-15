#!/bin/bash
# evallab STAGE 2 -- FINE-TUNE LEARNING CURVE + FROM-SCRATCH CONTROL.
#
# Copied from userdata_pre1.sh (the proven one). Differences:
#   * only the plc1 cache (2.7 GiB) + the two stage-1 checkpoints are fetched;
#     enc_pre1 is NOT needed and is never touched;
#   * 22 independent cells are run through a __LANES__-wide worker pool
#     (xargs -P) at __THREADS__ threads each, longest job first;
#   * every cell writes its own REPORT.cell.<job>.json and holdout prediction
#     vector, so a partial box still yields usable cells.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WHEEL__ __WATCHDOG__
#   __LANES__ __THREADS__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/ft1

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
  $AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --include "hpred_*.npy" --only-show-errors || true
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
$AWS s3 cp $S3/evallab/ft1/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/vt_ft1.py
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

status "STAGE fetch (plc1 2.7 GiB + stage-1 checkpoints)"
/opt/venv/bin/python vt_ft1.py fetch /opt/work
push; status "fetch done: $(du -sh /opt/work | cut -f1) free=$(df -h /opt | awk 'NR==2{print $4}')"

# Warm the page cache: 2.7 GiB off gp3 at ~1 GB/s, so every later random gather
# is a RAM access shared by all __LANES__ workers through one mmap.
status "STAGE warm"
cat /opt/work/enc_plc1/*.npy > /dev/null
status "warm done, cached=$(free -g | awk '/Mem:/{print $6}')G"

status "STAGE split (nested by team pair)"
/opt/venv/bin/python vt_ft1.py split /opt/work
push; status "split done"

# CANARY BEFORE THE SWEEP: one real cell for 6 steps. This walks the ENTIRE
# code path -- checkpoint load, mmap batching, the loop, a val check, best-ckpt
# save, the 100k-holdout scoring and the JSON dump -- in ~40 s, so a bug costs
# one minute of box time instead of the whole sweep. The report it writes is
# deleted so a later real failure cannot masquerade as a finished cell.
# seed 9 is NOT one of the real seeds, so the smoke can never overwrite a
# finished cell's report or prediction vector.
status "STAGE smoke (1 cell x 6 steps, full path incl. holdout eval)"
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /opt/venv/bin/python vt_ft1.py run /opt/work \
  --job ft:50000:9:s1 --threads 32 --steps 6 > /opt/work/smoke.log 2>&1
grep -q '"best_val"' /opt/work/REPORT.cell.ft_50000_9_s1.json
rm -f /opt/work/REPORT.cell.ft_50000_9_s1.json /opt/work/hpred_ft_50000_9_s1_*.npy \
      /opt/work/best_ft_50000_9_s1.pt
push; status "smoke ok: $(grep 'holdout best_val' /opt/work/smoke.log | head -c 200)"

/opt/venv/bin/python vt_ft1.py jobs /opt/work > /opt/work/jobs.txt
status "STAGE train: $(wc -l < /opt/work/jobs.txt) cells, lanes=__LANES__ threads=__THREADS__"
# PER-CELL SALVAGE. Cells are independent, so each one publishes its own report
# and prediction vectors the moment it finishes, and a cell whose report is
# already in S3 is skipped. A spot reclaim therefore costs only the <=4 cells
# in flight: relaunching this same box re-runs exactly the missing ones.
cat > /opt/one.sh <<ONE
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2 PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=__THREADS__ MKL_NUM_THREADS=__THREADS__
cd /opt/evallab
J=\$1
N=\${J//:/_}
R=REPORT.cell.\$N.json
if $AWS s3 ls $DEST/$TAG/\$R >/dev/null 2>&1; then
  $AWS s3 cp $DEST/$TAG/\$R /opt/work/\$R --only-show-errors
  $AWS s3 cp $DEST/$TAG/ /opt/work/ --recursive --exclude "*" \
    --include "hpred_\${N}_*.npy" --only-show-errors
  echo "SKIP \$J (already in S3)" > /opt/work/cell_\$N.log
  exit 0
fi
/opt/venv/bin/python vt_ft1.py run /opt/work --job "\$J" --threads __THREADS__ \
  > /opt/work/cell_\$N.log 2>&1 || exit 1
$AWS s3 cp /opt/work/\$R $DEST/$TAG/\$R --only-show-errors
$AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" \
  --include "hpred_\${N}_*.npy" --only-show-errors
ONE
chmod +x /opt/one.sh
RC=0
xargs -a /opt/work/jobs.txt -P __LANES__ -I{} /opt/one.sh {} || RC=1
push
NC=$(ls /opt/work/REPORT.cell.*.json 2>/dev/null | wc -l)
status "train finished rc=$RC cells=$NC/$(wc -l < /opt/work/jobs.txt)"
[ "$RC" = "0" ]
[ "$NC" = "$(wc -l < /opt/work/jobs.txt)" ]

$AWS s3 cp /opt/work/ $DEST/$TAG/ --recursive --exclude "*" --include "REPORT.*.json" --include "*.log" --include "hpred_*.npy" --include "jobs.txt" --only-show-errors
status "FT1 DONE"
DONE_OK=1
finish
