#!/bin/bash
# Launch the standing ladder fleet: ONE BOX PER ACCOUNT (claim mutex is
# box-local, and per-account boxes cap a spot reclaim's blast radius at one
# account's games). Prints "account instance-id" per line.
# Usage: AMI_ID=ami-xxxx launch_fleet.sh
# Env: FLEET_ACCOUNTS (default all 5, comma-separated), FLEET_PAR (4),
# FLEET_GAMES (0 = continuous), FLEET_TYPE (c7a.2xlarge per box)
set -euo pipefail

ROOT="/Users/sallyliu/pokemon-fast-bot"
AWS="/Users/sallyliu/.awscli-venv/bin/aws"
BUCKET="pokebot-valuenet-389825051723"
FLEET="ladder-fleet/fleet1"
REGION="us-east-2"
: "${AMI_ID:?set AMI_ID}"
ACCOUNTS="${FLEET_ACCOUNTS:-fable foul play,1v6king,beatmesilly,bobfamilyrules,endodontist}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PS_PASSWORD=$(grep '^PS_PASSWORD=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')
AKID=$($AWS configure get aws_access_key_id)
ASEC=$($AWS configure get aws_secret_access_key)

# cheapest-AZ pin: query live spot prices for the type, launch every box in
# the cheapest AZ's default subnet (5-15% under an unpinned launch). Caveat:
# one AZ correlates reclaims — acceptable, AMI relaunch is ~2 min.
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

IFS=',' read -ra ACCTS <<< "$ACCOUNTS"
for acct in "${ACCTS[@]}"; do
  slug=$(echo "$acct" | tr -c 'A-Za-z0-9' '-' | sed 's/-*$//')
  # Slot split (Sally 2026-08-09, rev 2): target ~11.5 starts/3min against
  # the shared IP's 12/3min budget. Measured cycle 5.7 min/game -> 22 slots.
  # All accounts are now high enough that ~92% of opponents clear the >=1400
  # training bar, so the split is near-flat (concurrency cap is 6/account).
  case "$acct" in
    beatmesilly|1v6king) slots=5 ;;
    *)                   slots=4 ;;
  esac
  {
    echo '#!/bin/bash'
    echo "export AWS_ACCESS_KEY_ID=$AKID"
    echo "export AWS_SECRET_ACCESS_KEY=$ASEC"
    echo "export AWS_DEFAULT_REGION=$REGION"
    echo "export S3_BUCKET=$BUCKET"
    echo "export S3_CODE=$FLEET"
    echo "export S3_PREFIX=$FLEET/box-$slug"
    echo "export PS_PASSWORD='$PS_PASSWORD'"
    echo "export ACCOUNTS='$acct'"
    echo "export GAMES_PER_ACCOUNT=${FLEET_GAMES:-0}"
    echo "export PAR=${FLEET_PAR:-4}"
    echo "export FORMATS='gen9randombattle:$slots'"
    echo "export SEARCH_MS=4000"
    cat "$ROOT/ladder-games/cloud/fleet_bootstrap.sh"
  } > "$SCRATCH/ud-$slug.sh"
  ID=$($AWS ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$TYPE" \
    --subnet-id "$SUBNET" \
    --key-name pokebot \
    --security-group-ids sg-024075dbfb3236454 \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fleet-$slug}]" \
    --user-data "file://$SCRATCH/ud-$slug.sh" \
    --query 'Instances[0].InstanceId' --output text)
  echo "$acct $ID"
done