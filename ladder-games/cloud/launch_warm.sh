#!/bin/bash
# Relaunch the v7warm2 warm ladder box (patched bootstrap). Prints instance id.
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
BUCKET=pokebot-valuenet-389825051723
PREFIX=ladder-fleet/v7warm2
SP="$(cd "$(dirname "$0")" && pwd)"
for m in DRAIN DONE READY RECLAIMED FAILED.log; do $AWS s3 rm "s3://$BUCKET/$PREFIX/$m" --quiet 2>/dev/null || true; done
$AWS s3 rm "s3://$BUCKET/$PREFIX/trigger/" --recursive --quiet 2>/dev/null || true
PS_PASSWORD=$(grep '^PS_PASSWORD=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')
UD=$(mktemp)
{
  echo '#!/bin/bash'
  echo "export AWS_ACCESS_KEY_ID=$($AWS configure get aws_access_key_id)"
  echo "export AWS_SECRET_ACCESS_KEY=$($AWS configure get aws_secret_access_key)"
  echo "export AWS_DEFAULT_REGION=us-east-2"
  echo "export S3_BUCKET=$BUCKET"
  echo "export S3_PREFIX=$PREFIX"
  echo "export PS_PASSWORD='$PS_PASSWORD'"
  echo "export ACCOUNT='fable foul play'"
  echo "export SEARCH_MS=4500"
  echo "export FIRST_MS=14500"
  echo "export WORLDS=16"
  echo "export POOL=16"
  echo "export THREADS=2"
  echo "export IDLE_MAX=7200"
  cat "$SP/v7warm2_bootstrap.sh"
} > "$UD"
$AWS ec2 run-instances --region us-east-2 \
  --image-id ami-028ba4d4ccb4b7b72 --instance-type c7a.8xlarge \
  --key-name pokebot --security-group-ids sg-024075dbfb3236454 \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fleet-v7game1}]' \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text
rm -f "$UD"
