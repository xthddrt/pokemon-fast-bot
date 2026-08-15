#!/bin/bash
# Launch the ONE end-to-end training canary box (spot, self-terminating, tagged).
#
#   BUNDLE=/tmp/code.tar.gz ./launch_canary.sh [tag] [instance-type]
#
# c7a.16xlarge by default -- deliberately the SAME instance type the full CPU
# training run would use, so the throughput this box measures is a measurement
# of the real target and not an extrapolation from a smaller box.
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
REGION=us-east-2
AMI=ami-028ba4d4ccb4b7b72
SG=sg-024075dbfb3236454
KEY=pokebot
WHEEL=$S3/evallab/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl

TAG="${1:-a}"
TYPE="${2:-c7a.16xlarge}"
WATCHDOG="${WATCHDOG:-5400}"
BLOCKS="${BLOCKS:-64}"
BLOCK_ROWS="${BLOCK_ROWS:-1250}"
PRE_EPOCHS="${PRE_EPOCHS:-60}"
FT_EPOCHS="${FT_EPOCHS:-80}"
DISK="${DISK:-60}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"
UD_SRC="${UD_SRC:-$ROOT/evallab/cloud/userdata_canary.sh}"
STEPS="${STEPS:-12000}"
SEEDS="${SEEDS:-0,1,2}"
THREADS="${THREADS:-21}"

DEST=$S3/evallab/canary
if [ "${SKIP_BUNDLE:-0}" != "1" ]; then
  echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__TAG__|$TAG|g" -e "s|__WHEEL__|$WHEEL|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__BLOCKS__|$BLOCKS|g" \
    -e "s|__BLOCK_ROWS__|$BLOCK_ROWS|g" -e "s|__PRE_EPOCHS__|$PRE_EPOCHS|g" \
    -e "s|__FT_EPOCHS__|$FT_EPOCHS|g" -e "s|__STEPS__|$STEPS|g" \
    -e "s|__SEEDS__|$SEEDS|g" -e "s|__THREADS__|$THREADS|g" \
    "$UD_SRC" > "$UD"
if grep -q '__[A-Z_]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (tag=$TAG disk=${DISK}G watchdog=${WATCHDOG}s)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"Throughput\":600,\"Iops\":6000,\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=canary-$TAG},{Key=proj,Value=evallab},{Key=job,Value=train_canary}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.$TAG -"
