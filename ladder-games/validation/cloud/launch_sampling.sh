#!/bin/bash
# Launch ONE sampling-validation box.
#
#   bash launch_sampling.sh                      # DRY RUN, prints only
#   CONFIRM=1 N_GAMES=20 TYPE=c7a.2xlarge bash launch_sampling.sh
#
# Nothing launches without CONFIRM=1: an accidental invocation must be a no-op
# rather than a second box quietly burning spend.
#
# Credentials are baked into user-data because the pokebot-cli IAM user has no
# IAM permissions and therefore cannot create an instance role (see the AWS
# setup note). User-data is readable from inside the instance only.
set -uo pipefail
# Region-parameterized (2026-08-21): eu-north-1 measured ~2.6x cheaper per
# vCPU than our us-east-2 c7a habit once its quota reached 300. Defaults keep
# the proven us-east-2 path; override all four together.
REGION="${REGION:-us-east-2}"
AMI="${AMI:-ami-028ba4d4ccb4b7b72}"
SG="${SG:-sg-024075dbfb3236454}"
KEY=pokebot
BUCKET=pokebot-valuenet-389825051723
AWSCLI=/Users/sallyliu/.awscli-venv/bin/aws
: "${TYPE:=c7a.16xlarge}"
: "${N_GAMES:=1000}"
: "${CONC:=12}"
: "${MS:=100}"
# SPOT BY DEFAULT. This job is restartable -- a reclaimed box loses at most one
# batch and we relaunch -- and spot is ~4x cheaper (c7a.16xlarge: $0.81/hr spot
# vs ~$3.0 on-demand, us-east-2 2026-08-20). SPOT=0 forces on-demand when spot
# capacity for the size is exhausted.
: "${SPOT:=1}"
: "${S3_PREFIX:=sampling/run-$(date +%Y%m%d-%H%M%S)}"

UD=$(mktemp)
{
  echo '#!/bin/bash'
  echo "export AWS_ACCESS_KEY_ID=$($AWSCLI configure get aws_access_key_id)"
  echo "export AWS_SECRET_ACCESS_KEY=$($AWSCLI configure get aws_secret_access_key)"
  echo "export AWS_DEFAULT_REGION=$REGION"
  echo "export S3_BUCKET=$BUCKET S3_CODE=sampling S3_PREFIX=$S3_PREFIX"
  echo "export N_GAMES=$N_GAMES CONC=$CONC MS=$MS"
  echo "aws s3 cp s3://$BUCKET/sampling/bootstrap_sampling.sh /root/b.sh"
  echo "bash /root/b.sh"
} > "$UD"

echo "=== type=$TYPE games=$N_GAMES conc=$CONC ms=$MS spot=$SPOT"
echo "=== results -> s3://$BUCKET/$S3_PREFIX/"
if [ "${CONFIRM:-0}" != "1" ]; then
  echo "DRY RUN -- re-run with CONFIRM=1 to launch"; rm -f "$UD"; exit 0
fi

# bash 3.2 (macOS) treats "${arr[@]}" on an EMPTY array as unbound under set -u,
# so pass the option as a plain string that is either empty or the flag pair.
MARKET=""
[ "$SPOT" = "1" ] && MARKET="--instance-market-options MarketType=spot"
ID=$($AWSCLI ec2 run-instances --region "$REGION" --image-id "$AMI" \
  --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=60,VolumeType=gp3}' \
  $MARKET \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sampling-validation}]" \
  --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "launched $ID"
echo "watch:  $AWSCLI s3 ls s3://$BUCKET/$S3_PREFIX/"
