#!/bin/bash
# Launch ONE encode box (spot, self-terminating, tagged).
#
#   BUNDLE=/tmp/code.tar.gz ./launch_enc.sh <plc1|pre1> [tag] [instance-type]
#
# Each job is a single box because the whole thing is only a few core-hours:
# the adopted encoder measured 883 states/core-s locally, so 1M rows is ~0.3
# core-hours and 7.9M is ~2.5. Sharding across boxes would cost more in merge
# complexity than it saves in wall clock. Reclaim resilience comes from the
# selector's per-shard .done markers in S3 (pre1) and from the job being ~20
# minutes end to end.
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
REGION=us-east-2
AMI=ami-028ba4d4ccb4b7b72
SG=sg-024075dbfb3236454
KEY=pokebot
# PIN THE WHEEL. A concurrent engine rebuild publishes to evallab/wheel/, so
# `latest` there is NOT the engine a given corpus was labelled with. Override
# WHEEL with a private copy and WHEEL_SHA with its sha256; the box then refuses
# to run on anything else (no build fallback).
WHEEL="${WHEEL:-$S3/evallab/wheel/poke_engine-0.0.59-cp311-cp311-linux_x86_64.whl}"
WHEEL_SHA="${WHEEL_SHA:-}"

JOB="${1:?plc1, plc2 or pre1}"
TAG="${2:-a}"
TYPE="${3:-c7a.16xlarge}"
WATCHDOG="${WATCHDOG:-9000}"
POS_PER_GAME="${POS_PER_GAME:-8}"
CHECK="${CHECK:-20000}"
SHARD_LIMIT="${SHARD_LIMIT:-0}"
ROW_LIMIT="${ROW_LIMIT:-}"
DISK="${DISK:-$([ "$JOB" = pre1 ] && echo 200 || echo 60)}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST="$S3/evallab/enc_$JOB"
if [ "${SKIP_BUNDLE:-0}" != "1" ]; then
  echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__JOB__|$JOB|g" -e "s|__TAG__|$TAG|g" -e "s|__WHEEL__|$WHEEL|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__POS_PER_GAME__|$POS_PER_GAME|g" \
    -e "s|__CHECK__|$CHECK|g" -e "s|__SHARD_LIMIT__|$SHARD_LIMIT|g" \
    -e "s|__ROW_LIMIT__|$ROW_LIMIT|g" -e "s|__WHEEL_SHA__|$WHEEL_SHA|g" \
    "$ROOT/evallab/cloud/userdata_enc.sh" > "$UD"
if grep -q '__[A-Z_]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (job=$JOB tag=$TAG disk=${DISK}G)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"Throughput\":600,\"Iops\":6000,\"DeleteOnTermination\":true}}]" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=enc-$JOB-$TAG},{Key=proj,Value=evallab},{Key=job,Value=enc_$JOB}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.$TAG -"
