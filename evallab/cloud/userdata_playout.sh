#!/bin/bash
# evallab PLAYOUT-LABEL box. One job: N playouts from each of an assigned slice
# of positions, then self-terminate. No SSH, ever.
#
# Structure copied verbatim from cloud/userdata.sh, which is the proven one:
#   * HOME=/root before anything (cloud-init runs user-data with an empty HOME)
#   * credentials injected via user-data (no instance profile on this AMI)
#   * a trap that disarms itself on its FIRST line (set -eE re-enters it)
#   * a hard watchdog so nothing outlives its budget whatever happens
#   * an uploader started FIRST so a boot failure is still visible in S3
#   * the poke_engine wheel built from the SHIPPED source, so the playout policy
#     is the same engine that generated the corpus
#
# Placeholders replaced by launch_playout.sh:
#   __AKID__ __ASEC__ __S3__ __RUN__ __POSRUN__ __N__ __ITERS__ __START__
#   __COUNT__ __W__ __TAG__ __WATCHDOG__
export HOME=/root
exec > /root/boot.log 2>&1
set -x
export AWS_DEFAULT_REGION=us-east-2
S3=__S3__
RUN=__RUN__
TAG=__TAG__
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
  $AWS s3 sync /opt/out $DEST/out --only-show-errors 2>/dev/null
  sleep 120
done
UP
chmod +x /opt/uploader.sh
nohup /opt/uploader.sh > /opt/uploader.log 2>&1 &

# hard watchdog: nothing survives past its budget, whatever happens
nohup bash -c "sleep __WATCHDOG__; $AWS s3 sync /opt/out $DEST/out; $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log; shutdown -h now" >/dev/null 2>&1 &

DONE_OK=0
finish(){
  rc=$?
  trap - ERR EXIT                       # FIRST LINE, always
  [ "$DONE_OK" = "1" ] && { echo OK > /opt/FINISHED.$TAG; } || { echo "FAILED rc=$rc" > /opt/FINISHED.$TAG; cat /root/boot.log > /dev/console 2>/dev/null || true; }
  $AWS s3 sync /opt/out $DEST/out --only-show-errors || true
  $AWS s3 cp /root/boot.log $DEST/boot.$TAG.log --only-show-errors || true
  $AWS s3 cp /opt/FINISHED.$TAG $DEST/FINISHED.$TAG --only-show-errors || true
  sleep 15
  shutdown -h now
}
trap finish ERR EXIT
set -eE

status "boot on $(hostname) vcpu=$(nproc) run=$RUN tag=$TAG slice=__START__+__COUNT__"
dnf install -y -q gcc gcc-c++ python3.11 python3.11-devel tar gzip
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source /root/.cargo/env

cd /opt
$AWS s3 cp $DEST/code.tar.gz . && tar xzf code.tar.gz
test -f /opt/evallab/playout.py
# The positions file lives under the POSITION run prefix, so every box in every
# wave labels the SAME numbered positions and slices never disagree.
$AWS s3 cp $S3/evallab/__POSRUN__/positions.jsonl.gz /opt/positions.jsonl.gz
gunzip -f /opt/positions.jsonl.gz
wc -l /opt/positions.jsonl

# The wheel is built ONCE by the prep box from the SHIPPED source and published
# to $S3/evallab/wheel/. Reusing it saves ~7 min of rustup+maturin per box AND
# guarantees every box in the fleet runs the identical engine binary. If it is
# absent the box falls back to building its own, so this is an optimisation,
# never a new failure mode.
python3.11 -m venv /opt/venv
/opt/venv/bin/pip3 install -q --upgrade pip
# pip rejects a wheel whose FILENAME is not name-version-...-platform.whl, so
# the published copy must keep its real basename; sync the prefix and glob it.
mkdir -p /opt/wheel
$AWS s3 sync $S3/evallab/wheel/ /opt/wheel/ --exclude '*' --include 'poke_engine-*.whl' --only-show-errors || true
PUBWHL=$(ls /opt/wheel/poke_engine-*.whl 2>/dev/null | head -1)
if [ -n "$PUBWHL" ]; then
  status "installing published poke_engine wheel $(basename $PUBWHL)"
  /opt/venv/bin/pip3 install -q "$PUBWHL"
else
  status "building poke_engine wheel from the SHIPPED source"
  /opt/venv/bin/pip3 install -q maturin
  /opt/venv/bin/pip3 install --no-cache-dir /opt/poke-engine/poke-engine-py \
    --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
fi
grep -v poke-engine /opt/foul-play/requirements.txt > /tmp/req.txt
/opt/venv/bin/pip3 install -q -r /tmp/req.txt numpy
/opt/venv/bin/python -c "import poke_engine, numpy; print('wheel ok', numpy.__version__)"

export PYTHONPATH=/opt/evallab
export EVALLAB_NET=/opt/valuenet/m4_artifacts/valuenet_v6nopol.bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /opt/evallab

# Resume: if this slice already has partial output in S3, pull it back first.
# A spot reclaim then costs only the positions that were in flight.
$AWS s3 cp $DEST/out/labels_$TAG.jsonl /opt/out/labels_$TAG.jsonl --only-show-errors 2>/dev/null || true

status "playouts: N=__N__ iters=__ITERS__ slice __START__..+__COUNT__ workers=__W__"
EVALLAB_WORKERS=__W__ /opt/venv/bin/python playout.py run \
  /opt/positions.jsonl /opt/out/labels_$TAG.jsonl __N__ __START__ __COUNT__ __ITERS__
status "playouts done"
$AWS s3 sync /opt/out $DEST/out --only-show-errors

DONE_OK=1
status "ALL DONE"
finish
