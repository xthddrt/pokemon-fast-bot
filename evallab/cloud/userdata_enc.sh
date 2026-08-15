#!/bin/bash
# evallab ENCODE box. One job, then self-terminate.
#
#   JOB=plc1 -- encode the 1,000,000 fine-tune positions with the ADOPTED
#               encoder (arm A, DROP_TIMES_ATTACKED, + enc2's 14-col setup
#               block) -> s3://.../evallab/enc_plc1/
#   JOB=pre1 -- run the lambda-return selector over the r4 shards, then encode
#               the resulting ~7.9M positions the same way
#               -> s3://.../evallab/enc_pre1/  (selection -> evallab/pre1_sel/)
#
# Structure copied verbatim from userdata_corpus.sh (the proven one). The one
# structural difference: the poke_engine wheel is INSTALLED from S3, not built
# -- the corpus box already published it, so this box skips rustup entirely.
#
# CANARY FIRST, on this box, before the full pass: 5,000 rows encoded with the
# same code path and self-checked (NaNs, id range, fp16 round-trip, and the
# arm-A-vs-enc2 slot-alignment witness). A failure there trips the ERR trap and
# the box dies before spending anything on the full corpus.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __JOB__ __TAG__ __WHEEL__ __WATCHDOG__
#   __POS_PER_GAME__ __CHECK__ __SHARD_LIMIT__ __ROW_LIMIT__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
JOB=__JOB__
TAG=__TAG__
DEST=$S3/evallab/enc_$JOB

mkdir -p /root/.aws /opt/out /opt/data /opt/wheel
cat > /root/.aws/credentials <<'CRED'
[default]
aws_access_key_id = __AKID__
aws_secret_access_key = __ASEC__
CRED
chmod 600 /root/.aws/credentials
printf '[default]\nregion = us-east-2\n' > /root/.aws/config

AWS=$(command -v aws || echo /usr/bin/aws)
status(){ echo "$(date -u +%FT%TZ) $*" >> /opt/STATUS; $AWS s3 cp /opt/STATUS $DEST/STATUS.$TAG --only-show-errors 2>/dev/null; }

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
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED.$TAG; } || { echo "FAILED rc=$rc" > /opt/FINISHED.$TAG; cat /root/boot.log > /dev/console 2>/dev/null || true; }
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED.$TAG $DEST/FINISHED.$TAG --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc) job=$JOB tag=$TAG"
dnf install -y -q gcc python3.11 python3.11-devel tar gzip

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/enc_adopted.py
test -f /opt/valuenet/lossless_encoder.py

python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
# keep the SPEC filename: pip rejects a wheel renamed to anything else
$AWS s3 cp __WHEEL__ /opt/wheel/ --only-show-errors
# WHEEL PIN. Playout labels AND encoded states both come out of the engine, so
# a different wheel silently makes the corpus inhomogeneous. If a sha is given,
# it is a HARD gate: no match, no run, and no build fallback.
SHA="__WHEEL_SHA__"
echo "wheel sha256: $(sha256sum /opt/wheel/*.whl)"
if [ -n "$SHA" ]; then echo "$SHA  $(ls /opt/wheel/*.whl)" | sha256sum -c - ; fi
/opt/venv/bin/pip3 install -q /opt/wheel/*.whl
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy boto3
/opt/venv/bin/python -c "import poke_engine, numpy, boto3; print('deps ok', numpy.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
# `nproc` HONOURS $OMP_NUM_THREADS, so the worker count must be read BEFORE the
# BLAS pins are exported -- otherwise the box silently runs one worker on 64
# vCPU. Use --all so the ordering can never matter again.
W=$(nproc --all)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /opt/evallab
status "deps ready, workers=$W"

# ------------------------------------------------------------------ sources --
if [ "$JOB" = "plc1" ]; then
  status "downloading plc1 states + meta"
  $AWS s3 cp $S3/evallab/plc1/positions.jsonl.gz /opt/data/positions.jsonl.gz --only-show-errors
  $AWS s3 cp $DEST/meta.npz /opt/data/meta.npz --only-show-errors
  gunzip -f /opt/data/positions.jsonl.gz
  SRC=/opt/data/positions.jsonl
  META="--meta /opt/data/meta.npz"
  ROWS=""
  status "states: $(wc -l < $SRC) lines"
elif [ "$JOB" = "plc2" ]; then
  # STAGE 2. plc2's meta.npz does not exist yet (plc1's was built locally and
  # uploaded before its encode), and plc2 has a HOLE: i=409289 lost its label to
  # an engine panic. `prep` builds the meta in row order and writes the
  # gap-free state file, so row r of the cache is line r of the source is row r
  # of the meta -- densely, which is what makes --meta's row-by-row game-id
  # assertion meaningful.
  status "downloading plc2 states + labels + the stage-2 driver"
  $AWS s3 cp $DEST/plc2_stage2.py /opt/evallab/plc2_stage2.py --only-show-errors
  $AWS s3 cp $S3/evallab/plc2/positions.jsonl.gz /opt/data/positions.jsonl.gz --only-show-errors
  mkdir -p /opt/data/labels
  $AWS s3 cp $S3/evallab/plc2/out/ /opt/data/labels/ --recursive \
      --exclude "*" --include "labels_*.jsonl" --only-show-errors
  gunzip -f /opt/data/positions.jsonl.gz
  status "prep: meta.npz in row order + the gap-free state file"
  /opt/venv/bin/python plc2_stage2.py prep /opt/data/positions.jsonl /opt/data/labels \
      /opt/data/meta.npz /opt/data/src.jsonl
  $AWS s3 cp /opt/data/PREP_REPORT.json $DEST/PREP_REPORT.json --only-show-errors
  cat /opt/data/PREP_REPORT.json >> /opt/STATUS
  rm -f /opt/data/positions.jsonl
  SRC=/opt/data/src.jsonl
  META="--meta /opt/data/meta.npz"
  ROWS=""
  status "states: $(wc -l < $SRC) lines"
  # NEGATIVE CONTROL, before the full pass: corrupt one meta row and prove the
  # join assertion rejects it. An assertion that cannot fail proves nothing.
  status "TAMPER TEST on the row-by-row join check"
  /opt/venv/bin/python plc2_stage2.py tamper $SRC /opt/data/meta.npz /opt/out/tamper 2048
  $AWS s3 cp /opt/out/tamper/TAMPER_REPORT.json $DEST/TAMPER_REPORT.json --only-show-errors
  cat /opt/out/tamper/TAMPER_REPORT.json >> /opt/STATUS
  rm -rf /opt/out/tamper/clean /opt/out/tamper/dirty
  status "TAMPER TEST PASSED"
else
  status "resuming any finished selection shards from $S3/evallab/pre1_sel/"
  mkdir -p /opt/pt
  $AWS s3 sync $S3/evallab/pre1_sel/ /opt/pt/ --exclude "*" --include "*.pt.jsonl.gz*" --only-show-errors || true
  status "running the lambda-return selector over r4 (pos/game=__POS_PER_GAME__)"
  POS_PER_GAME=__POS_PER_GAME__ PT_WORKERS=$W OUT_S3=$S3/evallab/pre1_sel \
    /opt/venv/bin/python pretrain_select.py run /opt/pt 0 __SHARD_LIMIT__
  $AWS s3 cp /opt/pt/RUN_STATS.json $S3/evallab/pre1_sel/RUN_STATS.json --only-show-errors
  ROWS="--rows $(/opt/venv/bin/python -c "import json;print(json.load(open('/opt/pt/RUN_STATS.json'))['rows'])")"
  rm -f /opt/pt/plc1.positions.pairs.gz
  SRC=/opt/pt
  META=""
  status "selection done: $ROWS"
fi

# ------------------------------------------------------------------ canary ---
status "CANARY: 5,000 rows through the adopted encoder, fully self-checked"
/opt/venv/bin/python - "$SRC" <<'PY'
import glob, gzip, os, sys
src = sys.argv[1]
fs = (sorted(glob.glob(os.path.join(src, "*.jsonl.gz"))) if os.path.isdir(src) else [src])
n = 0
with open("/opt/data/canary.jsonl", "w") as o:
    for p in fs:
        f = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
        for line in f:
            o.write(line); n += 1
            if n >= 5000: break
        f.close()
        if n >= 5000: break
print("canary rows", n)
PY
KEEP_HPCHK=0 /opt/venv/bin/python enc_adopted.py encode /opt/data/canary.jsonl \
  /opt/out/canary --workers 8 --check 5000
$AWS s3 cp /opt/out/canary/ENC_STATS.json $DEST/CANARY_STATS.$TAG.json --only-show-errors
cat /opt/out/canary/ENC_STATS.json >> /opt/STATUS
status "CANARY PASSED"

# -------------------------------------------------------------------- full ---
status "FULL ENCODE starting"
/opt/venv/bin/python enc_adopted.py encode "$SRC" /opt/out/enc \
  --workers $W --check __CHECK__ $META $ROWS __ROW_LIMIT__
status "encode done: $(du -sh /opt/out/enc | cut -f1)"

# NOT `[ ... ] && cp`: under `set -e` a false test at top level exits the script.
case "$JOB" in plc1|plc2) cp /opt/data/meta.npz /opt/out/enc/meta.npz;; esac
$AWS s3 cp /opt/out/enc/ $DEST/ --recursive --only-show-errors
$AWS s3 ls $DEST/ --recursive --summarize | tail -5 >> /opt/STATUS
cat /opt/out/enc/ENC_STATS.json >> /opt/STATUS

# -------------------------------------------------------------------- merge --
# plc1 FIRST and in its own row order, so plc1's holdout_i.npy -- which the
# trainers use as raw row indices -- keeps pointing at exactly the same 100,000
# rows. Every plc2 row lands in the train pool.
if [ "$JOB" = "plc2" ]; then
  status "MERGE: fetching enc_plc1"
  mkdir -p /opt/plc1
  $AWS s3 cp $S3/evallab/enc_plc1/ /opt/plc1/ --recursive --exclude "*" \
      --include "*.npy" --include "*.npz" --include "addon_layout.json" \
      --include "split.json" --only-show-errors
  status "MERGE: building enc_plc12 (byte-for-byte read-back + leak check)"
  /opt/venv/bin/python plc2_stage2.py merge /opt/plc1 /opt/out/enc /opt/out/merged
  status "MERGE: uploading $(du -sh /opt/out/merged | cut -f1)"
  $AWS s3 cp /opt/out/merged/ $S3/evallab/enc_plc12/ --recursive --only-show-errors
  $AWS s3 cp /opt/out/merged/MERGE_REPORT.json $DEST/MERGE_REPORT.json --only-show-errors
  cat /opt/out/merged/MERGE_REPORT.json >> /opt/STATUS
  $AWS s3 ls $S3/evallab/enc_plc12/ --recursive --summarize | tail -3 >> /opt/STATUS
  status "MERGE DONE"
fi

status "ENCODE DONE"
DONE_OK=1
finish
