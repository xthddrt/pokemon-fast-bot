#!/bin/bash
# Launch the single evallab corpus box (spot, self-terminating).
#
#   ./launch.sh <run-name> [instance-type]
#
# Everything the box needs is in code.tar.gz, uploaded here. Nothing is fetched
# from a repo on the box, and no SSH is required or expected.
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
NA="${NA:-10000}"; NB="${NB:-2000}"; NC="${NC:-2000}"
ITERS="${ITERS:-25000}"
ORACLE_ITERS="${ORACLE_ITERS:-5000000}"
ORACLE_N="${ORACLE_N:-200}"; ORACLE_NBC="${ORACLE_NBC:-60}"
W="${W:-64}"
SUB="${SUB:-A}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST="$S3/evallab/$RUN"
echo "uploading bundle -> $DEST/code.tar.gz"
"$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|" \
    -e "s|__RUN__|$RUN|" -e "s|__ITERS__|$ITERS|g" \
    -e "s|__ORACLE_ITERS__|$ORACLE_ITERS|g" -e "s|__ORACLE_NBC__|$ORACLE_NBC|g" \
    -e "s|__ORACLE_N__|$ORACLE_N|g" \
    -e "s|__NA__|$NA|g" -e "s|__NB__|$NB|g" -e "s|__NC__|$NC|g" -e "s|__W__|$W|" \
    -e "s|__SUB__|$SUB|g" \
    "$ROOT/evallab/cloud/userdata.sh" > "$UD"

echo "launching $TYPE spot in $REGION (run=$RUN)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=evallab-$RUN},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS - "
