#!/bin/bash
# ENCODER VALUE TEST training grid box. No SSH, no generation, no engine build:
# the feature caches are already encoded and shipped in the bundle, so the box
# only installs torch+numpy and runs N independent vt_train.py jobs in parallel.
#
# Structure copied from evallab/cloud/userdata.sh (the proven one):
#   HOME=/root first; credentials via user-data; uploader started FIRST so a
#   boot failure is visible in S3; self-disarming trap; hard watchdog.
#
# Placeholders replaced by launch_vt.sh: __AKID__ __ASEC__ __S3__ __RUN__ __PAR__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
RUN=__RUN__
DEST=$S3/evallab/$RUN

mkdir -p /root/.aws /opt/out
cat > /root/.aws/credentials <<'CRED'
[default]
aws_access_key_id = __AKID__
aws_secret_access_key = __ASEC__
CRED
chmod 600 /root/.aws/credentials
printf '[default]\nregion = us-east-2\n' > /root/.aws/config

AWS=$(command -v aws || echo /usr/bin/aws)
status(){ echo "$(date -u +%FT%TZ) $*" >> /opt/STATUS; $AWS s3 cp /opt/STATUS $DEST/STATUS --only-show-errors 2>/dev/null; }

cat > /opt/uploader.sh <<UP
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2
while true; do
  $AWS s3 cp /root/boot.log $DEST/boot.log --only-show-errors 2>/dev/null
  [ -f /opt/STATUS ] && $AWS s3 cp /opt/STATUS $DEST/STATUS --only-show-errors 2>/dev/null
  $AWS s3 sync /opt/out $DEST/out --only-show-errors 2>/dev/null
  sleep 60
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

# hard watchdog: nothing survives past 2h
nohup bash -c "sleep 7200; $AWS s3 cp /root/boot.log $DEST/boot.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED; } || { echo "FAILED rc=$rc" > /opt/FINISHED; }
  $AWS s3 sync /opt/out $DEST/out --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED $DEST/FINISHED --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc) run=$RUN"
dnf install -y -q python3.11 python3.11-devel tar gzip
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
/opt/venv/bin/pip3 install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
/opt/venv/bin/pip3 install -q --no-cache-dir numpy
/opt/venv/bin/python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)"
status "python env ready"

cd /opt
$AWS s3 cp $DEST/code.tar.gz /opt/code.tar.gz --only-show-errors
$AWS s3 cp $DEST/jobs.txt /opt/jobs.txt --only-show-errors
tar xzf code.tar.gz
ls -la /opt/evallab/data/pl2/enc/ | head -20
status "bundle unpacked"

# ---- FAIL-LOUD CANARY: one short run must produce a sane Brier before the
# ---- grid is allowed to start. A broken cache or a wrong stub dies HERE.
cd /opt/evallab
/opt/venv/bin/python vt_train.py --arm old --cap 128,256 --seed 0 --epochs 2 \
    --out /opt/out/canary_old.json > /opt/out/canary_old.log 2>&1
/opt/venv/bin/python vt_train.py --arm enc2 --cap 32 --seed 0 --epochs 2 \
    --out /opt/out/canary_enc2.json > /opt/out/canary_enc2.log 2>&1
/opt/venv/bin/python vt_train.py --arm enc2 --arch shared --cap 64,256 --seed 0 \
    --epochs 2 --out /opt/out/canary_shared.json > /opt/out/canary_shared.log 2>&1
/opt/venv/bin/python - <<'PY'
import json, sys
for a in ("old", "enc2", "shared"):
    r = json.load(open("/opt/out/canary_%s.json" % a))
    b = r["test"]["all"]
    assert 0.005 < b < 0.09, "canary %s Brier %.5f out of range" % (a, b)
    print("canary", a, "OK brier=%.5f params=%d" % (b, r["params"]))
PY
status "canary passed"

# ---- the grid. Each line of jobs.txt is a full vt_train.py argument string.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cat /opt/jobs.txt | xargs -P __PAR__ -I{} sh -c \
  '/opt/venv/bin/python /opt/evallab/vt_train.py {} --threads 2 >> /opt/out/grid.log 2>&1 || echo "JOBFAIL {}" >> /opt/out/FAILURES'
status "grid finished; $(ls /opt/out/*.json | wc -l) result files"
[ -f /opt/out/FAILURES ] && { cat /opt/out/FAILURES; exit 9; }
DONE_OK=1
