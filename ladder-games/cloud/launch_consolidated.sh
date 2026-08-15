#!/bin/bash
# Launch ONE consolidated fleet box: all 5 accounts, 1 slot each
# (max 5 concurrent games), 8 vCPU. Replaces the per-account fleet
# (Sally 2026-08-09). Same baked AMI + bootstrap; S3 prefix box-all.
set -euo pipefail
ROOT="/Users/sallyliu/pokemon-fast-bot"
AWS="/Users/sallyliu/.awscli-venv/bin/aws"
BUCKET="pokebot-valuenet-389825051723"
FLEET="ladder-fleet/fleet1"
REGION="us-east-2"
: "${AMI_ID:?set AMI_ID}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PS_PASSWORD=$(grep '^PS_PASSWORD=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')
AKID=$($AWS configure get aws_access_key_id)
ASEC=$($AWS configure get aws_secret_access_key)

TYPE="${FLEET_TYPE:-c7a.2xlarge}"
CHEAP_AZ=$($AWS ec2 describe-spot-price-history --region "$REGION" \
  --instance-types "$TYPE" --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice]' --output text \
  | sort -k2 -n | head -1 | cut -f1)
SUBNET=$($AWS ec2 describe-subnets --region "$REGION" \
  --filters "Name=availability-zone,Values=$CHEAP_AZ" "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' --output text)
echo "cheapest AZ: $CHEAP_AZ ($SUBNET)" >&2

{
  echo '#!/bin/bash'
  echo "export AWS_ACCESS_KEY_ID=$AKID"
  echo "export AWS_SECRET_ACCESS_KEY=$ASEC"
  echo "export AWS_DEFAULT_REGION=$REGION"
  echo "export S3_BUCKET=$BUCKET"
  echo "export S3_CODE=$FLEET"
  echo "export S3_PREFIX=$FLEET/box-all"
  echo "export PS_PASSWORD='$PS_PASSWORD'"
  echo "export ACCOUNTS='fable foul play,1v6king,beatmesilly,bobfamilyrules,endodontist'"
  echo "export GAMES_PER_ACCOUNT=0"
  echo "export PAR=1"
  echo "export FORMATS='gen9randombattle:1'"
  echo "export SEARCH_MS=4000"
  cat "$ROOT/ladder-games/cloud/fleet_bootstrap.sh"
} > "$SCRATCH/ud-all.sh"

ID=$($AWS ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$TYPE" \
  --subnet-id "$SUBNET" \
  --key-name pokebot \
  --security-group-ids sg-024075dbfb3236454 \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fleet-all}]" \
  --user-data "file://$SCRATCH/ud-all.sh" \
  --query 'Instances[0].InstanceId' --output text)
echo "consolidated $ID"
