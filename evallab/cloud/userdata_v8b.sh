#!/bin/bash
# evallab v8b TRAINING box -- 128/256, from random init, on the MERGED
# plc1+plc2 cache (enc_plc12, 1,999,870 rows). One job, then self-terminate.
#
# Structure copied verbatim from userdata_n.sh (the box that trained the 900k
# baseline). Differences, and ONLY these:
#   * the cache fetched is enc_plc12 (merged) instead of enc_plc1 -- it is
#     dropped into the SAME local directory name so vt_n.py is untouched;
#   * a hard guard asserts 1,999,870 rows / 100,000 holdout / holdout max index
#     < 1,000,000 before a single step is taken;
#   * the engine wheel comes from the PRIVATE pinned copy (plc2/wheel), never
#     from evallab/wheel/, which a concurrent job has already added 0.0.60 to;
#   * the both-widths inference bench is dropped (already measured, and this
#     run must not train 256/512 in any form);
#   * publish prefix is nets_v8b.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __TAG__ __WATCHDOG__
#   __STEPS__ __SEEDS__ __THREADS__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
TAG=__TAG__
DEST=$S3/evallab/v8b
PUB=$S3/evallab/nets_v8b
WHEEL=$S3/evallab/plc2/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl
WHEEL_SHA=38e0c11ba765519dc55dbc7e26ab7dc34540b8b6241f7eb98547e51f8f655213

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
test -f /opt/valuenet/vocab.json

python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
# PINNED wheel, hard-gated. No build fallback: a wrong engine here would make
# this half of the corpus non-homogeneous with plc1's labels.
$AWS s3 cp $WHEEL /opt/wheel/ --only-show-errors
echo "$WHEEL_SHA  /opt/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl" > /opt/wheel/SHA
sha256sum -c /opt/wheel/SHA
/opt/venv/bin/pip3 install -q /opt/wheel/*.whl
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy boto3
/opt/venv/bin/pip3 install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
/opt/venv/bin/python -c "import poke_engine, numpy, boto3, torch; print('deps ok', numpy.__version__, torch.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
cd /opt/evallab
status "deps ready, vcpu=$(nproc --all) torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__)')"

# ---- the MERGED cache, into the directory name vt_n.py expects --------------
status "STAGE fetch enc_plc12"
$AWS s3 cp $S3/evallab/enc_plc12/ /opt/work/enc_plc1/ --recursive --only-show-errors
ls -la /opt/work/enc_plc1
status "fetch done: $(du -sh /opt/work/enc_plc1 | cut -f1)"

# ---- guard: the right cache, the right holdout, BEFORE any training ---------
status "STAGE guard"
/opt/venv/bin/python - <<'PY' > /opt/work/GUARD.json
import hashlib, json, numpy as np, os
d = "/opt/work/enc_plc1"
m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
h = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
n = len(m["label_p"])
sha = hashlib.sha256(open(os.path.join(d, "holdout_i.npy"), "rb").read()).hexdigest()
g = {"rows": int(n), "holdout": int(len(h)), "holdout_max": int(h.max()),
     "train_pool": int(n - len(h)), "holdout_i_sha256": sha,
     "corpus_counts": {k: int(v) for k, v in
                       zip(*np.unique(m["corpus"].astype(str), return_counts=True))}
     if "corpus" in m else None}
assert n == 1999870, g
assert len(h) == 100000 and len(np.unique(h)) == 100000, g
assert int(h.max()) < 1000000, g          # holdout is entirely inside plc1's rows
assert sha == "9c8bf63b216617de1b64bb39334fdd849c2d4a4d17c52d3a93050a075d3f61f3", g
assert n - len(h) == 1899870, g
for k in ("label_p", "band", "q_search", "pair", "g"):
    assert k in m, (k, sorted(m))
print(json.dumps(g, indent=1))
PY
cat /opt/work/GUARD.json
$AWS s3 cp /opt/work/GUARD.json $DEST/$TAG/GUARD.json --only-show-errors
status "guard PASS"

status "STAGE train 128/256 seeds=__SEEDS__ steps=__STEPS__ threads=__THREADS__"
PIDS=""
for S in $(echo __SEEDS__ | tr ',' ' '); do
  OMP_NUM_THREADS=__THREADS__ nohup /opt/venv/bin/python vt_n.py train /opt/work \
      --seed $S --steps __STEPS__ --threads __THREADS__ \
      --mon-hid 128 --trunk-hid 256 --eval-every 1000 \
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
    --include "holdout_pred_*.npy" --include "*.log" --include "GUARD.json" \
    --only-show-errors
status "PUBLISHED to $PUB/$TAG/"
DONE_OK=1
finish
