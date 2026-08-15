#!/bin/bash
# evallab corpus box. Drives the ENTIRE cloud phase with no SSH:
#   build wheel -> CANARY (6 games, uploaded immediately) -> generate A/B/C
#   -> oracle -> upload -> self-terminate.
#
# Every structural choice here is copied from valuenet/cloud/gen_round1_bootstrap.sh
# and scratchpad/v15verify/userdata_template.sh, which are the two proven ones:
#   * HOME=/root before anything (cloud-init runs user-data with an empty HOME,
#     so "$HOME/.cargo/env" would resolve to /.cargo/env and die)
#   * credentials injected via user-data (the AMI has none and no instance
#     profile is attached, so even the FAILED marker upload needs them)
#   * a trap that disarms itself on its FIRST line (set -eE re-enters it)
#   * a hard watchdog so nothing outlives its budget whatever happens
#   * an uploader started FIRST so a boot failure is still visible in S3
#
# Placeholders replaced by launch.sh: __AKID__ __ASEC__ __S3__ __RUN__
#   __ITERS__ __ORACLE_ITERS__ __ORACLE_N__ __NA__ __NB__ __NC__ __W__
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
  sleep 120
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

# hard watchdog: nothing survives past 3h, whatever happens
nohup bash -c "sleep 10800; $AWS s3 cp /root/boot.log $DEST/boot.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED; } || { echo "FAILED rc=$rc" > /opt/FINISHED; cat /root/boot.log > /dev/console 2>/dev/null || true; }
  $AWS s3 sync /opt/out $DEST/out --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED $DEST/FINISHED --only-show-errors || true
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
test -f /opt/evallab/generate.py

status "building poke_engine wheel from the SHIPPED source (engine identity with the local box is what makes the corpus reusable locally)"
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip maturin
/opt/venv/bin/pip3 install --no-cache-dir /opt/poke-engine/poke-engine-py \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
# numpy is NOT in foul-play's requirements but valuenet/encoder.py needs it, and
# relfeat.py holds the type chart as an ndarray.
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy
/opt/venv/bin/python -c "import poke_engine, numpy; print('wheel ok', numpy.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=__W__
cd /opt/evallab

# ---- relational-feature gate: the same hand-checkable assertions the local box
# ---- runs. A wheel whose type/speed semantics drifted fails HERE, not silently
# ---- in the corpus.
status "relfeat gate"
# NOT `... | tail -3`: a pipeline's status is its LAST command, so tail's success
# would mask a failing gate and the box would carry on generating a corpus with
# broken relational features. Log to a file, then check the real exit code.
/opt/venv/bin/python test_relfeat.py > /opt/relfeat_gate.log 2>&1
tail -3 /opt/relfeat_gate.log
cp /opt/relfeat_gate.log /opt/out/relfeat_gate.log

# ---- CANARY: 6 games at the real config, uploaded IMMEDIATELY so the local box
# ---- can verify the remote engine's state strings parse before the bulk run.
status "canary: 6 games"
EVALLAB_WORKERS=6 /opt/venv/bin/python generate.py A 6 /opt/out/canary 9000000 __ITERS__
$AWS s3 sync /opt/out $DEST/out --only-show-errors
touch /opt/out/CANARY_DONE && $AWS s3 cp /opt/out/CANARY_DONE $DEST/out/CANARY_DONE
status "CANARY UPLOADED"

# ---- bulk generation. One generate.py per pair; each uses its own process pool.
# __SUB__ names pair A's output directory so two corpora at different search
# budgets can never be globbed together by accident.
status "generating A=__NA__ B=__NB__ C=__NC__ at __ITERS__ iterations, workers=$W, sub=__SUB__"
# `if`, NOT `[ n -gt 0 ] && { ... }`: under `set -eE` a false test makes the
# whole AND-list return 1, which fires the ERR trap and kills the box on the
# very branch that was supposed to be a no-op.
if [ __NA__ -gt 0 ]; then
  EVALLAB_WORKERS=$W /opt/venv/bin/python generate.py A __NA__ /opt/out/__SUB__ 0 __ITERS__
  status "pair A done"
fi
if [ __NB__ -gt 0 ]; then
  EVALLAB_WORKERS=$W /opt/venv/bin/python generate.py B __NB__ /opt/out/B 0 __ITERS__
  status "pair B done"
fi
if [ __NC__ -gt 0 ]; then
  EVALLAB_WORKERS=$W /opt/venv/bin/python generate.py C __NC__ /opt/out/C 0 __ITERS__
  status "pair C done"
fi
$AWS s3 sync /opt/out $DEST/out --only-show-errors

# ---- oracle. Fewer workers than vCPUs: a 5M-iteration tree is ~2 GB resident,
# ---- so the limit is RAM, not cores.
OW=$(( W / 2 )); [ $OW -lt 1 ] && OW=1
if [ __ORACLE_N__ -gt 0 ]; then
  status "oracle: __ORACLE_N__ positions at __ORACLE_ITERS__ iterations, workers=$OW"
  EVALLAB_WORKERS=$OW /opt/venv/bin/python oracle.py '/opt/out/__SUB__/shard_A_*.jsonl.gz' /opt/out/oracle_A.jsonl __ORACLE_N__ __ORACLE_ITERS__
  EVALLAB_WORKERS=$OW /opt/venv/bin/python oracle.py '/opt/out/B/shard_B_*.jsonl.gz' /opt/out/oracle_B.jsonl __ORACLE_NBC__ __ORACLE_ITERS__
  EVALLAB_WORKERS=$OW /opt/venv/bin/python oracle.py '/opt/out/C/shard_C_*.jsonl.gz' /opt/out/oracle_C.jsonl __ORACLE_NBC__ __ORACLE_ITERS__
  status "oracle done"
fi

DONE_OK=1
status "ALL DONE"
finish
