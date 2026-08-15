#!/bin/bash
# Launch ONE playout-label box (spot, self-terminating).
#
#   TAG=c0 START=0 COUNT=200 N=100 ./launch_playout.sh <run-name> [instance-type]
#
# Every box in a run labels a contiguous slice of the SAME positions.jsonl, which
# lives under $POSRUN (default: the run name). Slices are disjoint by START/COUNT,
# so the per-box outputs concatenate with no dedupe needed.
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
REGION=us-east-2
AMI=ami-028ba4d4ccb4b7b72
SG=sg-024075dbfb3236454
KEY=pokebot

RUN="${1:?run name}"
TYPE="${2:-c7a.16xlarge}"
POSRUN="${POSRUN:-$RUN}"
TAG="${TAG:-a}"
START="${START:-0}"; COUNT="${COUNT:-200}"
N="${N:-100}"; ITERS="${ITERS:-2000}"
W="${W:-64}"
WATCHDOG="${WATCHDOG:-10800}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST="$S3/evallab/$RUN"
if [ "${SKIP_BUNDLE:-0}" != "1" ]; then
  echo "uploading bundle -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__RUN__|$RUN|" -e "s|__POSRUN__|$POSRUN|g" -e "s|__TAG__|$TAG|g" \
    -e "s|__N__|$N|g" -e "s|__ITERS__|$ITERS|g" \
    -e "s|__START__|$START|g" -e "s|__COUNT__|$COUNT|g" -e "s|__W__|$W|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__WHEEL_SHA__|${WHEEL_SHA:-}|g" \
    "${UD_TEMPLATE:-$ROOT/evallab/cloud/userdata_playout.sh}" > "$UD"
if grep -q '__[A-Z_0-9]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_0-9]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (run=$RUN tag=$TAG slice=$START+$COUNT N=$N)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=plabel-$RUN-$TAG},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.$TAG -"
