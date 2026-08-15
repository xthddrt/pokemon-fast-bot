#!/bin/bash
# Launch the ONE data-scaling-curve box (spot, self-terminating, tagged).
#   BUNDLE=/tmp/code.tar.gz ./launch_scale.sh [tag] [instance-type]
# 192 vCPU by default: ten trainings run CONCURRENTLY, one thread block each.
set -euo pipefail
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
REGION=us-east-2
AMI=ami-028ba4d4ccb4b7b72
SG=sg-024075dbfb3236454
KEY=pokebot
WHEEL=$S3/evallab/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl

TAG="${1:-a}"
TYPE="${2:-c7a.48xlarge}"
WATCHDOG="${WATCHDOG:-3600}"
DISK="${DISK:-80}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"
UD_SRC="${UD_SRC:-$(dirname "$0")/userdata_scale.sh}"
DEST=$S3/evallab/scale

echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
"$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__TAG__|$TAG|g" -e "s|__WHEEL__|$WHEEL|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" "$UD_SRC" > "$UD"
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
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=scale-$TAG},{Key=proj,Value=evallab},{Key=job,Value=learning_curve}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.$TAG -"
