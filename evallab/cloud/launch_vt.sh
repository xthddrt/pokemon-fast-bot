#!/bin/bash
# Launch the ENCODER VALUE TEST training-grid box (spot, self-terminating).
#   JOBS=/path/jobs.txt BUNDLE=/path/code.tar.gz ./launch_vt.sh <run-name> [type] [par]
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
# Region/AMI/SG are overridable: the us-east-2 spot vCPU quota is shared with
# whatever else is running, so a big box sometimes has to go to another region.
# The S3 bucket stays in us-east-2 and is reached cross-region.
REGION=${REGION:-us-east-2}
AMI=${AMI:-ami-028ba4d4ccb4b7b72}
SG=${SG:-sg-024075dbfb3236454}
KEY=${KEY:-pokebot}

RUN="${1:?run name}"
TYPE="${2:-c7a.16xlarge}"
PAR="${3:-24}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"
JOBS="${JOBS:?path to jobs.txt}"

DEST="$S3/evallab/$RUN"
echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
"$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
"$AWS" s3 cp "$JOBS" "$DEST/jobs.txt" --only-show-errors
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|" \
    -e "s|__RUN__|$RUN|g" -e "s|__PAR__|$PAR|g" \
    "${UD_SRC:-$ROOT/evallab/cloud/userdata_vt.sh}" > "$UD"
echo "user-data $(wc -c < "$UD") bytes, $(wc -l < "$JOBS") jobs"
[ "$(wc -c < "$UD")" -lt 16000 ] || { echo "user-data too large"; exit 1; }

# no SSH is used at all (results leave via S3), so a key pair is optional --
# KEY= lets the box run in a region where the pokebot key does not exist
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" ${KEY:+--key-name "$KEY"} \
  --security-group-ids "$SG" --count 1 \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vt-$RUN},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID  ($TYPE spot, $REGION)"
echo "watch:  $AWS s3 cp $DEST/STATUS -"
