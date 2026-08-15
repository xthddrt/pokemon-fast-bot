#!/bin/bash
# Build + upload the canary code tarball, compose user-data, launch ONE spot
# c7a.2xlarge running canary_bootstrap.sh. Prints the instance id.
set -euo pipefail

ROOT="/Users/sallyliu/pokemon-fast-bot"
AWS="/Users/sallyliu/.awscli-venv/bin/aws"
BUCKET="pokebot-valuenet-389825051723"
PREFIX="ladder-fleet/canary1"
REGION="us-east-2"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

echo "== tarball"
tar czf "$SCRATCH/code.tar.gz" -C "$ROOT" \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  --exclude='foul-play/logs' --exclude='poke-engine/target' \
  --exclude='*.jsonl' --exclude='ladder-games/games' \
  foul-play poke-engine \
  valuenet/m4_artifacts/valuenet_v6ref_nopuct.bin \
  valuenet/m4_artifacts/valuenet_v6ref_nopuct.constants.json \
  ladder-games/archive_game.py
ls -lh "$SCRATCH/code.tar.gz"
$AWS s3 cp "$SCRATCH/code.tar.gz" "s3://$BUCKET/$PREFIX/code.tar.gz"

echo "== user-data"
PS_PASSWORD=$(grep '^PS_PASSWORD=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')
{
  echo '#!/bin/bash'
  echo "export AWS_ACCESS_KEY_ID=$($AWS configure get aws_access_key_id)"
  echo "export AWS_SECRET_ACCESS_KEY=$($AWS configure get aws_secret_access_key)"
  echo "export AWS_DEFAULT_REGION=$REGION"
  echo "export S3_BUCKET=$BUCKET"
  echo "export S3_PREFIX=$PREFIX"
  echo "export PS_PASSWORD='$PS_PASSWORD'"
  echo "export ACCOUNT_1='endodontist'"
  echo "export ACCOUNT_2='1v6king'"
  echo "export GAMES_PER_ACCOUNT=8"
  echo "export PAR=2"
  echo "export SEARCH_MS=4000"
  cat "$ROOT/ladder-games/cloud/canary_bootstrap.sh"
} > "$SCRATCH/user-data.sh"

echo "== launch"
$AWS ec2 run-instances \
  --region "$REGION" \
  --image-id ami-028ba4d4ccb4b7b72 \
  --instance-type c7a.2xlarge \
  --key-name pokebot \
  --security-group-ids sg-024075dbfb3236454 \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ladder-canary1}]' \
  --user-data "file://$SCRATCH/user-data.sh" \
  --query 'Instances[0].InstanceId' --output text
