#!/bin/bash
# evallab plc2 CORPUS-PREP box. Derived from userdata_corpus.sh with three
# deliberate changes, then self-terminate:
#   1. NO WHEEL BUILD. plc2 must be labelled by the SAME engine binary as plc1
#      or the merged corpus is inhomogeneous, so the wheel is installed from a
#      PINNED private copy whose sha256 is verified. A concurrent job may
#      publish a newer wheel to evallab/wheel/; we never look there.
#   2. plc1's 1,000,000 team-pair hashes are passed as EXCLUDE_PAIRS, so plc2
#      draws only from the ~999,872 r4 games plc1 never consumed.
#   3. No eval selection. The plc1 100k holdout is reused verbatim downstream;
#      plc2 positions go into the TRAIN pool only.
# Everything else (band allocation 40/40/20, decided/opening caps, N, ITERS,
# MAX_STEPS, seed-as-a-function-of-i) is corpus_select.py / playout.py verbatim
# from plc1's own code.tar.gz.
#
# Placeholders: __AKID__ __ASEC__ __S3__ __RUN__ __SRC__ __N_POS__ __N_TRAIN__
#   __ITERS__ __W__ __WATCHDOG__ __CANARY_N__ __SEED__ __WHEEL_SHA__ __PLC1__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
RUN=__RUN__
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

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/corpus_select.py
test -f /opt/evallab/playout.py

# ---- PINNED ENGINE -------------------------------------------------------
status "installing PINNED poke_engine wheel from $DEST/wheel/"
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
mkdir -p /opt/wheel
$AWS s3 cp $DEST/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl /opt/wheel/ --only-show-errors
GOT=$(sha256sum /opt/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl | cut -d' ' -f1)
status "wheel sha256 $GOT (expect __WHEEL_SHA__)"
[ "$GOT" = "__WHEEL_SHA__" ] || { status "WHEEL SHA MISMATCH -- ABORT"; exit 9; }
/opt/venv/bin/pip3 install -q /opt/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy boto3
/opt/venv/bin/python -c "import poke_engine, numpy, boto3; print('wheel ok', numpy.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /opt/evallab

# ---- EXCLUDE plc1 --------------------------------------------------------
status "fetching plc1 pair hashes to exclude"
$AWS s3 cp $S3/evallab/__PLC1__/positions.pairs.gz /opt/plc1.pairs.gz --only-show-errors
gunzip -f /opt/plc1.pairs.gz
status "plc1 pairs: $(wc -l < /opt/plc1.pairs)"

status "selecting __N_POS__ positions (one per game) from __SRC__, plc1-excluded"
EXCLUDE_PAIRS=/opt/plc1.pairs SELECT_WORKERS=32 \
  /opt/venv/bin/python corpus_select.py __SRC__ __N_POS__ /opt/out/positions.jsonl __SEED__ 0
status "selected: $(wc -l < /opt/out/positions.jsonl) positions"

# ---- DISJOINTNESS PROOF --------------------------------------------------
status "verifying zero pair overlap and zero game overlap with plc1"
$AWS s3 cp $S3/evallab/__PLC1__/positions.jsonl.gz /opt/plc1.pos.jsonl.gz --only-show-errors
/opt/venv/bin/python - <<'PY' > /opt/out/DISJOINT.json
import gzip, json, re, os
G = re.compile(r'^\{"g": "([^"]+)"')
p1_pairs = {l.strip() for l in open('/opt/plc1.pairs') if l.strip()}
p1_games = set()
with gzip.open('/opt/plc1.pos.jsonl.gz', 'rt') as f:
    for line in f:
        m = G.match(line[:200])
        assert m, line[:80]
        p1_games.add(m.group(1))
p2_pairs, p2_games, n = set(), set(), 0
for line in open('/opt/out/positions.jsonl'):
    r = json.loads(line)
    p2_pairs.add(r['pair']); p2_games.add(r['g']); n += 1
out = {
    "plc1_pairs": len(p1_pairs), "plc1_games": len(p1_games),
    "plc2_rows": n, "plc2_distinct_pairs": len(p2_pairs),
    "plc2_distinct_games": len(p2_games),
    "pair_overlap_with_plc1": len(p1_pairs & p2_pairs),
    "game_overlap_with_plc1": len(p1_games & p2_games),
    "union_pairs": len(p1_pairs | p2_pairs), "union_games": len(p1_games | p2_games),
}
print(json.dumps(out, indent=2))
PY
cat /opt/out/DISJOINT.json >> /opt/STATUS
$AWS s3 cp /opt/out/DISJOINT.json $DEST/DISJOINT.json --only-show-errors
/opt/venv/bin/python -c "
import json,sys
d=json.load(open('/opt/out/DISJOINT.json'))
assert d['pair_overlap_with_plc1']==0, d
assert d['game_overlap_with_plc1']==0, d
assert d['plc2_distinct_games']==d['plc2_rows'], d
print('DISJOINT OK')"
rm -f /opt/plc1.pos.jsonl.gz

gzip -kf /opt/out/positions.jsonl
$AWS s3 cp /opt/out/positions.jsonl.gz $DEST/positions.jsonl.gz --only-show-errors
$AWS s3 cp /opt/out/positions.jsonl.stats.json $DEST/positions.stats.json --only-show-errors
gzip -f /opt/out/positions.jsonl.pairs
$AWS s3 cp /opt/out/positions.jsonl.pairs.gz $DEST/positions.pairs.gz --only-show-errors
status "positions published"

# ---- CANARY --------------------------------------------------------------
status "CANARY: __CANARY_N__ positions N=__N_TRAIN__ iters=__ITERS__ workers=__W__"
EVALLAB_WORKERS=__W__ /opt/venv/bin/python playout.py run \
  /opt/out/positions.jsonl /opt/out/canary_train.jsonl __N_TRAIN__ 0 __CANARY_N__ __ITERS__
$AWS s3 cp /opt/out/canary_train.jsonl $DEST/canary/canary_train.jsonl --only-show-errors

/opt/venv/bin/python - <<'PY' > /opt/out/CANARY_REPORT.json
import json, os
N = __N_TRAIN__
rs = [json.loads(l) for l in open('/opt/out/canary_train.jsonl')]
errs = sum(1 for r in rs if 'error' in r)
rs = [r for r in rs if 'cs' in r]
n = len(rs)
cs = sum(r['cs'] for r in rs) / n
# plc1's published full-corpus values, the homogeneity reference
REF = {"mean_se": 0.1019, "mean_label_p": 0.505, "trunc_frac": 0.00535,
       "mean_cs_per_position": 4.229, "mean_ply": 15.53}
got = {"n": n, "n_playouts": N, "errors": errs,
       "mean_cs_per_position": round(cs, 3),
       "core_s_per_playout": round(cs / N, 4),
       "mean_steps": round(sum(r['steps'] for r in rs) / n, 2),
       "trunc_frac": round(sum(r['trunc'] for r in rs) / (n * N), 5),
       "mean_label_p": round(sum(r['label_p'] for r in rs) / n, 4),
       "mean_se": round(sum(r['se'] for r in rs) / n, 4),
       "mean_ply": round(sum(r['ply'] for r in rs) / n, 2)}
# sampling-noise bands over n canary rows (sd/sqrt(n)); label_p sd~0.36, se sd~0.05
import math
band = {"mean_label_p": 3 * 0.36 / math.sqrt(n),
        "mean_se": 3 * 0.05 / math.sqrt(n),
        "trunc_frac": 3 * math.sqrt(max(REF['trunc_frac'], 1e-6) / (n * N))}
cmp = {k: {"plc1": REF[k], "plc2_canary": got[k],
           "delta": round(got[k] - REF[k], 5),
           "tol_3sigma": round(band[k], 5),
           "ok": abs(got[k] - REF[k]) <= band[k]} for k in band}
n_pos = sum(1 for _ in open('/opt/out/positions.jsonl'))
out = {"canary": got, "plc1_reference": REF, "comparison": cmp,
       "all_ok": all(v["ok"] for v in cmp.values()) and errs == 0,
       "n_positions": n_pos, "vcpu": os.cpu_count(),
       "projected_core_hours": round(got["mean_cs_per_position"] * n_pos / 3600, 1)}
print(json.dumps(out, indent=2))
PY
cat /opt/out/CANARY_REPORT.json >> /opt/STATUS
$AWS s3 cp /opt/out/CANARY_REPORT.json $DEST/CANARY_REPORT.json --only-show-errors

status "PREP DONE"
DONE_OK=1
finish
