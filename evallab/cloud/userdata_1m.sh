#!/bin/bash
# evallab CORPUS-SCALE box (1M-game pair-A corpus). Derived from cloud/userdata.sh
# -- the proven, already-flown bootstrap -- with exactly three changes:
#   1. MODE: `probe` (canary + diversity-knob comparison) or `bulk` (a slice of
#      the corpus). Nothing else differs between the two kinds of box.
#   2. generation is sliced into BLOCK-sized shards over [START, START+NGAMES),
#      so a spot interruption costs one block, not the box.
#   3. an analyze_corpus.py `scan` pass runs on the box and uploads ~17 bytes per
#      decision, so the 7.7 GB corpus never has to be downloaded to measure it.
# No oracle, no pairs B/C. Everything else (HOME=/root first, injected creds,
# self-disarming trap, watchdog, uploader-first) is unchanged and load-bearing.
#
# Placeholders replaced by launch_1m.sh: __AKID__ __ASEC__ __S3__ __RUN__
#   __MODE__ __ITERS__ __START__ __NGAMES__ __BLOCK__ __W__ __SUB__
#   __RAND__ __TEMP__ __EPS__ __PROBE_GAMES__ __CANARY_GAMES__ __WATCHDOG__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
RUN=__RUN__
DEST=$S3/evallab/$RUN
SUB=__SUB__

mkdir -p /root/.aws /opt/out
cat > /root/.aws/credentials <<'CRED'
[default]
aws_access_key_id = __AKID__
aws_secret_access_key = __ASEC__
CRED
chmod 600 /root/.aws/credentials
printf '[default]\nregion = us-east-2\n' > /root/.aws/config

AWS=$(command -v aws || echo /usr/bin/aws)
status(){ echo "$(date -u +%FT%TZ) $*" >> /opt/STATUS_$SUB; $AWS s3 cp /opt/STATUS_$SUB $DEST/STATUS_$SUB --only-show-errors 2>/dev/null; }

cat > /opt/uploader.sh <<UP
#!/bin/bash
export HOME=/root AWS_DEFAULT_REGION=us-east-2
while true; do
  $AWS s3 cp /root/boot.log $DEST/boot_$SUB.log --only-show-errors 2>/dev/null
  [ -f /opt/STATUS_$SUB ] && $AWS s3 cp /opt/STATUS_$SUB $DEST/STATUS_$SUB --only-show-errors 2>/dev/null
  $AWS s3 sync /opt/out $DEST/out --only-show-errors 2>/dev/null
  sleep 300
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

# hard watchdog: nothing survives past __WATCHDOG__ seconds, whatever happens
nohup bash -c "sleep __WATCHDOG__; $AWS s3 cp /root/boot.log $DEST/boot_$SUB.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/out/DONE_$SUB; } || { echo "FAILED rc=$rc" > /opt/out/FAILED_$SUB; cat /root/boot.log > /dev/console 2>/dev/null || true; }
  $AWS s3 sync /opt/out $DEST/out --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot_$SUB.log --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc) run=$RUN mode=__MODE__ sub=$SUB"
dnf install -y -q gcc gcc-c++ python3.11 python3.11-devel tar gzip
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source /root/.cargo/env

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/generate.py
test -f /opt/evallab/analyze_corpus.py

status "building poke_engine wheel from the SHIPPED source"
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip maturin
/opt/venv/bin/pip3 install --no-cache-dir /opt/poke-engine/poke-engine-py \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy
/opt/venv/bin/python -c "import poke_engine, numpy; print('wheel ok', numpy.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=__W__
cd /opt/evallab

# ---- relational-feature gate (same hand-checkable assertions as the local box).
# NOT piped: a pipeline's status is its LAST command, so `| tail` would mask a
# failing gate and the box would generate a corpus with broken semantics.
status "relfeat gate"
/opt/venv/bin/python test_relfeat.py > /opt/relfeat_gate.log 2>&1
tail -3 /opt/relfeat_gate.log
cp /opt/relfeat_gate.log /opt/out/relfeat_gate_$SUB.log

if [ "__MODE__" = "probe" ]; then
  # -------- CANARY: real config, uploaded IMMEDIATELY so the local box can
  # -------- verify parse + labels before a single bulk box is launched.
  status "canary: __CANARY_GAMES__ games at __ITERS__ iters, knobs __RAND__/__TEMP__/__EPS__"
  RAND_PLIES=__RAND__ TEMP_PLIES=__TEMP__ EPS=__EPS__ EVALLAB_WORKERS=$W \
    /opt/venv/bin/python generate.py A __CANARY_GAMES__ /opt/out/canary 9000000 __ITERS__
  EVALLAB_WORKERS=8 /opt/venv/bin/python analyze_corpus.py scan /opt/out/scan_canary.npz \
    '/opt/out/canary/shard_A_*.jsonl.gz'
  MERGE_BUCKET=100 /opt/venv/bin/python analyze_corpus.py merge /opt/out/probe_canary.json \
    /opt/out/scan_canary.npz
  $AWS s3 sync /opt/out $DEST/out --only-show-errors
  touch /opt/out/CANARY_DONE && $AWS s3 cp /opt/out/CANARY_DONE $DEST/out/CANARY_DONE
  status "CANARY UPLOADED"

  # -------- DIVERSITY-KNOB PROBE: identical game count and seeds per setting,
  # -------- so the only difference between arms is the knob triple.
  for cfg in __PROBE_CFGS__; do
    r=${cfg%%:*}; rest=${cfg#*:}; t=${rest%%:*}; ep=${rest##*:}
    name="p${r}_${t}_${ep}"
    status "probe $name: __PROBE_GAMES__ games"
    RAND_PLIES=$r TEMP_PLIES=$t EPS=$ep EVALLAB_WORKERS=$W \
      /opt/venv/bin/python generate.py A __PROBE_GAMES__ /opt/out/$name 1000000 __ITERS__
    EVALLAB_WORKERS=8 /opt/venv/bin/python analyze_corpus.py scan /opt/out/scan_$name.npz \
      "/opt/out/$name/shard_A_*.jsonl.gz"
    MERGE_BUCKET=1000 /opt/venv/bin/python analyze_corpus.py merge /opt/out/probe_$name.json \
      /opt/out/scan_$name.npz
    $AWS s3 sync /opt/out $DEST/out --only-show-errors
  done
  status "PROBE DONE"
else
  # -------- BULK: [START, START+NGAMES) in BLOCK-sized shards.
  status "bulk $SUB: games [__START__, __START__+__NGAMES__) block=__BLOCK__ iters=__ITERS__ knobs __RAND__/__TEMP__/__EPS__ workers=$W"
  export SUB START=__START__ NGAMES=__NGAMES__ BLOCK=__BLOCK__ ITERS=__ITERS__
  export RANDP=__RAND__ TEMPP=__TEMP__ EPSV=__EPS__
  mkdir -p /opt/out/$SUB
  s=__START__
  end=$(( __START__ + __NGAMES__ ))
  while [ $s -lt $end ]; do
    n=__BLOCK__
    [ $(( s + n )) -gt $end ] && n=$(( end - s ))
    RAND_PLIES=__RAND__ TEMP_PLIES=__TEMP__ EPS=__EPS__ EVALLAB_WORKERS=$W \
      /opt/venv/bin/python generate.py A $n /opt/out/$SUB $s __ITERS__
    status "block $s +$n done"
    s=$(( s + n ))
  done
  status "generation done, scanning"
  EVALLAB_WORKERS=$W /opt/venv/bin/python analyze_corpus.py scan /opt/out/scan_$SUB.npz \
    "/opt/out/$SUB/shard_A_*.jsonl.gz"
  MERGE_BUCKET=10000 /opt/venv/bin/python analyze_corpus.py merge /opt/out/report_$SUB.json \
    /opt/out/scan_$SUB.npz
  # manifest: what this box actually produced, with sizes and game counts
  /opt/venv/bin/python - <<'PYEOF' > /opt/out/MANIFEST_$SUB.json
import glob, gzip, json, os, sys
sub = os.environ["SUB"]
rows = []
for p in sorted(glob.glob("/opt/out/%s/shard_A_*.jsonl.gz" % sub)):
    ng = 0
    with gzip.open(p, "rt") as f:
        for line in f:
            ng += 1
    rows.append({"shard": os.path.basename(p), "bytes": os.path.getsize(p), "lines": ng})
json.dump({"sub": sub, "start": int(os.environ["START"]), "n_games": int(os.environ["NGAMES"]),
           "block": int(os.environ["BLOCK"]), "iters": int(os.environ["ITERS"]),
           "rand_plies": int(os.environ["RANDP"]), "temp_plies": int(os.environ["TEMPP"]),
           "eps": float(os.environ["EPSV"]), "shards": rows,
           "total_bytes": sum(r["bytes"] for r in rows),
           "total_games": sum(r["lines"] - 1 for r in rows)}, sys.stdout, indent=1)
PYEOF
  status "BULK DONE $(grep -o '\"total_games\": [0-9]*' /opt/out/MANIFEST_$SUB.json)"
fi

DONE_OK=1
status "ALL DONE"
finish
