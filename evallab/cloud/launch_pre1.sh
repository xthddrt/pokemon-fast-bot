#!/bin/bash
# Launch the STAGE 1 pretrain box (spot, self-terminating, tagged).
# Structure copied verbatim from launch_canary.sh; only the userdata and the
# knobs it substitutes differ.
#
#   BUNDLE=/tmp/pre1_code.tar.gz ./launch_pre1.sh [tag] [instance-type]
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
WATCHDOG="${WATCHDOG:-12600}"          # 3.5 h hard kill; run projects at ~1.6 h
STEPS="${STEPS:-6000}"
THREADS="${THREADS:-32}"
SEEDS="${SEEDS:-0 1}"
VAL_EVERY="${VAL_EVERY:-250}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
DISK="${DISK:-80}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST=$S3/evallab/nets_pre1
if [ "${SKIP_BUNDLE:-0}" != "1" ]; then
  echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__TAG__|$TAG|g" -e "s|__WHEEL__|$WHEEL|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__STEPS__|$STEPS|g" \
    -e "s|__THREADS__|$THREADS|g" -e "s|__SEEDS__|$SEEDS|g" \
    -e "s|__VAL_EVERY__|$VAL_EVERY|g" -e "s|__CKPT_EVERY__|$CKPT_EVERY|g" \
    "$ROOT/evallab/cloud/userdata_pre1.sh" > "$UD"
if grep -q '__[A-Z_]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (tag=$TAG disk=${DISK}G watchdog=${WATCHDOG}s steps=$STEPS seeds='$SEEDS')"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"Throughput\":1000,\"Iops\":16000,\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pre1-$TAG},{Key=proj,Value=evallab},{Key=job,Value=pretrain_pre1}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.$TAG -"
