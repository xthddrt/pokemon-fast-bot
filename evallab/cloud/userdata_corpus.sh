#!/bin/bash
# evallab CORPUS-PREP box. Three jobs, then self-terminate:
#   1. build the poke_engine wheel from the SHIPPED source and publish it, so
#      every label box in the fleet installs the SAME binary in ~30 s instead
#      of re-running rustup+maturin for 7 minutes each.
#   2. select the TRAIN and EVAL position sets from the r-series shards
#      (one position per game) and publish them as positions.jsonl.gz.
#   3. run the CANARY: 500 train positions at N=__N_TRAIN__ and 50 eval
#      positions at N=__N_EVAL__, so cost/position is measured before the fleet.
#
# Structure copied verbatim from userdata_playout.sh (the proven one).
# Placeholders: __AKID__ __ASEC__ __S3__ __RUN__ __EVALRUN__ __SRC_TRAIN__
#   __SRC_EVAL__ __N_POS__ __N_EVAL_POS__ __N_TRAIN__ __N_EVAL__ __ITERS__
#   __W__ __WATCHDOG__ __CANARY__ __SHARD_LIMIT__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
RUN=__RUN__
EVALRUN=__EVALRUN__
TAG=prep
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

status "boot on $(hostname) vcpu=$(nproc) run=$RUN"
dnf install -y -q gcc gcc-c++ python3.11 python3.11-devel tar gzip
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source /root/.cargo/env

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/corpus_select.py
test -f /opt/evallab/playout.py

status "building poke_engine wheel from the SHIPPED source"
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip maturin
/opt/venv/bin/pip3 wheel --no-deps --wheel-dir /opt/wheelout /opt/poke-engine/poke-engine-py \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
WHL=$(ls /opt/wheelout/poke_engine-*.whl | head -1)
/opt/venv/bin/pip3 install -q "$WHL"
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy boto3
/opt/venv/bin/python -c "import poke_engine, numpy, boto3; print('wheel ok', numpy.__version__)"
# Publish the wheel FIRST: the fleet's only hard dependency on this box.
$AWS s3 cp "$WHL" $S3/evallab/wheel/poke_engine.whl --only-show-errors
status "wheel published -> $S3/evallab/wheel/poke_engine.whl ($(basename $WHL))"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /opt/evallab

status "selecting TRAIN: __N_POS__ positions, one per game, from __SRC_TRAIN__"
# 32, not $(nproc): each worker holds one decompressed+parsed shard (~1 GB), so
# 64 workers would be ~64 GB of peak RSS on a 128 GB box for no extra speed
# (the job is ~3 min either way).
SELECT_WORKERS=32 /opt/venv/bin/python corpus_select.py \
  __SRC_TRAIN__ __N_POS__ /opt/out/positions.jsonl 1 __SHARD_LIMIT__
status "train selected: $(wc -l < /opt/out/positions.jsonl) positions"
gzip -kf /opt/out/positions.jsonl
$AWS s3 cp /opt/out/positions.jsonl.gz $DEST/positions.jsonl.gz --only-show-errors
$AWS s3 cp /opt/out/positions.jsonl.stats.json $DEST/positions.stats.json --only-show-errors
gzip -f /opt/out/positions.jsonl.pairs
$AWS s3 cp /opt/out/positions.jsonl.pairs.gz $DEST/positions.pairs.gz --only-show-errors
gunzip -k /opt/out/positions.jsonl.pairs.gz

status "selecting EVAL: __N_EVAL_POS__ positions from __SRC_EVAL__, pair-disjoint from train"
EXCLUDE_PAIRS=/opt/out/positions.jsonl.pairs SELECT_WORKERS=32 \
  /opt/venv/bin/python corpus_select.py \
  __SRC_EVAL__ __N_EVAL_POS__ /opt/out/positions_eval.jsonl 2 0
status "eval selected: $(wc -l < /opt/out/positions_eval.jsonl) positions"
gzip -kf /opt/out/positions_eval.jsonl
$AWS s3 cp /opt/out/positions_eval.jsonl.gz $S3/evallab/$EVALRUN/positions.jsonl.gz --only-show-errors
$AWS s3 cp /opt/out/positions_eval.jsonl.stats.json $S3/evallab/$EVALRUN/positions.stats.json --only-show-errors

if [ "__CANARY__" = "1" ]; then
  status "CANARY A: 500 train positions N=__N_TRAIN__ iters=__ITERS__ workers=__W__"
  EVALLAB_WORKERS=__W__ /opt/venv/bin/python playout.py run \
    /opt/out/positions.jsonl /opt/out/canary_train.jsonl __N_TRAIN__ 0 500 __ITERS__
  $AWS s3 cp /opt/out/canary_train.jsonl $DEST/canary/canary_train.jsonl --only-show-errors

  status "CANARY B: 50 eval positions N=__N_EVAL__ iters=__ITERS__ workers=__W__"
  EVALLAB_WORKERS=__W__ /opt/venv/bin/python playout.py run \
    /opt/out/positions_eval.jsonl /opt/out/canary_eval.jsonl __N_EVAL__ 0 50 __ITERS__
  $AWS s3 cp /opt/out/canary_eval.jsonl $DEST/canary/canary_eval.jsonl --only-show-errors

  /opt/venv/bin/python - <<'PY' > /opt/out/CANARY_REPORT.json
import json, os
def summ(p, n_target):
    rs = [json.loads(l) for l in open(p)]
    rs = [r for r in rs if 'cs' in r]
    cs = sum(r['cs'] for r in rs) / len(rs)
    return {"file": os.path.basename(p), "n": len(rs), "n_playouts": n_target,
            "mean_cs_per_position": round(cs, 3),
            "core_s_per_playout": round(cs / n_target, 4),
            "mean_steps": round(sum(r['steps'] for r in rs) / len(rs), 2),
            "trunc_frac": round(sum(r['trunc'] for r in rs) / (len(rs) * n_target), 5),
            "errors": sum(1 for r in rs if 'error' in r),
            "mean_label_p": round(sum(r['label_p'] for r in rs) / len(rs), 4)}
a = summ('/opt/out/canary_train.jsonl', __N_TRAIN__)
b = summ('/opt/out/canary_eval.jsonl', __N_EVAL__)
n_tr = sum(1 for _ in open('/opt/out/positions.jsonl'))
n_ev = sum(1 for _ in open('/opt/out/positions_eval.jsonl'))
out = {"train_canary": a, "eval_canary": b, "n_train_positions": n_tr,
       "n_eval_positions": n_ev, "vcpu": os.cpu_count(),
       "projected_train_core_hours": round(a["mean_cs_per_position"] * n_tr / 3600, 1),
       "projected_eval_core_hours": round(b["mean_cs_per_position"] * n_ev / 3600, 1)}
print(json.dumps(out, indent=2))
PY
  $AWS s3 cp /opt/out/CANARY_REPORT.json $DEST/CANARY_REPORT.json --only-show-errors
  cat /opt/out/CANARY_REPORT.json >> /opt/STATUS
fi

status "PREP DONE"
DONE_OK=1
finish
