#!/bin/bash
# ENCODER FINAL -- the recipe-fair search box. TWO PHASES on ONE box:
#   phase 1  every cell x every recipe, seed 0, ranked on VAL  (+ mechanism cells)
#   phase 2  every cell at its OWN best recipe AND at the shared published
#            recipe, 3 seeds each
# The selector between them is `vt_grid.py 2`, which reads /opt/out and writes
# jobs2.txt; it is tested locally on synthetic results before launch.
#
# Structure copied from userdata_vt3.sh (proven): HOME=/root first, credentials
# via user-data, uploader started FIRST, self-disarming trap, hard watchdog.
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
  $AWS s3 sync /opt/out $DEST/out --exclude '*_pred.npy' --only-show-errors 2>/dev/null
  sleep 60
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

# hard watchdog: nothing survives past 4h
nohup bash -c "sleep 14400; $AWS s3 cp /root/boot.log $DEST/boot.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED; } || { echo "FAILED rc=$rc" > /opt/FINISHED; }
  $AWS s3 sync /opt/out $DEST/out --exclude '*_pred.npy' --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED $DEST/FINISHED --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc) par=__PAR__ run=$RUN"
dnf install -y -q python3.11 python3.11-devel tar gzip
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
/opt/venv/bin/pip3 install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
/opt/venv/bin/pip3 install -q --no-cache-dir numpy
/opt/venv/bin/python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)"
status "python env ready"

cd /opt
$AWS s3 cp $DEST/code.tar.gz /opt/code.tar.gz --only-show-errors
tar xzf code.tar.gz
ls -la /opt/evallab/data/pl2/enc/ | head -20
status "bundle unpacked"

cd /opt/evallab
# ---- FAIL-LOUD CANARY. Three cells that exercise every new code path -- the
# ---- untouched control, an additive cell, and a pruned + bias-fixed cell --
# ---- must train and hit their EXACT published/derived parameter counts. A bad
# ---- stub, a mis-shipped cache or a broken --biasfix/--constok dies HERE.
canary(){ /opt/venv/bin/python vt_train.py --arm old --cap 256,512 --seed 0 --epochs 2 \
  --tag "$1" --out /opt/out/canary_$1.json "${@:2}" > /opt/out/canary_$1.log 2>&1; }
canary ctl
canary sc --add setup,caps
canary ns_bf --drop struct --biasfix 1
canary tok --drop last_item --constok 12
/opt/venv/bin/python - <<'CANARY'
import json
exp = {"ctl": 1091541, "sc": 1098453, "ns_bf": 1086165, "tok": 1091553}
for a, p in exp.items():
    r = json.load(open("/opt/out/canary_%s.json" % a))
    b = r["test"]["all"]
    assert 0.005 < b < 0.09, "canary %s Brier %.5f out of range" % (a, b)
    assert r["params"] == p, "canary %s params %d != %d" % (a, r["params"], p)
    print("canary", a, "OK brier=%.5f params=%d" % (b, r["params"]))
CANARY
status "canary passed"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run_jobs(){
  cat "$1" | xargs -P __PAR__ -I{} sh -c \
    'timeout 5400 /opt/venv/bin/python /opt/evallab/vt_train.py {} --threads 2 >> /opt/out/grid.log 2>&1 || echo "JOBFAIL {}" >> /opt/out/FAILURES'
}

# ---- phase 1: TUNE (every cell x every recipe, seed 0) + mechanism cells
/opt/venv/bin/python vt_grid.py 1 --out /opt/out --jobs /opt/jobs1.txt
status "phase 1: $(wc -l < /opt/jobs1.txt) jobs"
run_jobs /opt/jobs1.txt
status "phase 1 done; $(ls /opt/out/t.*.json 2>/dev/null | wc -l) tune results"
[ -f /opt/out/FAILURES ] && { cat /opt/out/FAILURES; exit 9; }

# ---- selector: each cell's best recipe on VAL -> phase 2 jobs
/opt/venv/bin/python vt_grid.py 2 --out /opt/out --results /opt/out --jobs /opt/jobs2.txt
$AWS s3 cp /opt/out/PICKED.json $DEST/out/PICKED.json --only-show-errors
status "phase 2: $(wc -l < /opt/jobs2.txt) jobs"

# ---- phase 2: CONFIRM (own-best recipe and shared recipe, 3 seeds each)
run_jobs /opt/jobs2.txt
status "phase 2 done; $(ls /opt/out/*.json | wc -l) result files"
[ -f /opt/out/FAILURES ] && { cat /opt/out/FAILURES; exit 9; }
DONE_OK=1
