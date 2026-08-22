#!/bin/bash
# Launch the r64 remote-search box (eu-north c8g.16xlarge spot, 64 vCPU --
# cheapest 64-core arm box for this workload per the 2026-08-22 price scan)
# and write its URL to ladder-games/cloud/searchbox_ip.txt for run_game_r64.sh.
#
#   bash ladder-games/cloud/launch_search_box.sh
#
# Idempotent-ish: if a healthy box is already recorded, it is reused.
set -euo pipefail
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
HERE="$ROOT/ladder-games/cloud"
AWS="${AWS_CLI:-/Users/sallyliu/.awscli-venv/bin/aws}"
export AWS_MAX_ATTEMPTS=1
A="$AWS --cli-connect-timeout 8 --cli-read-timeout 45"
REGION=eu-north-1
TYPE="${SEARCHBOX_TYPE:-c8g.16xlarge}"
SG=sg-08aa12e53aa801f9d
AMI=ami-0b50f26215e9a0e77
BUCKET=pokebot-valuenet-389825051723

# reuse a live healthy box if one is recorded
if [ -f "$HERE/searchbox_ip.txt" ]; then
  URL=$(cat "$HERE/searchbox_ip.txt")
  if curl -sf -m 4 "$URL/health" >/dev/null 2>&1; then
    echo "reusing healthy search box at $URL"
    exit 0
  fi
fi

# keep the box's payload current (server file + net travel via S3)
$A s3 cp "$ROOT/foul-play/search_server.py" "s3://$BUCKET/nets/search_server.py" --only-show-errors
$A s3 cp "$ROOT/valuenet/nets_v10/v10c0.bin" "s3://$BUCKET/nets/v10c0.bin" --only-show-errors
$A s3 cp "$ROOT/valuenet/nets_v10/v10c0.constants.json" "s3://$BUCKET/nets/v10c0.constants.json" --only-show-errors
$A s3 cp "$HERE/searchbox_bootstrap.sh" "s3://$BUCKET/nets/searchbox_bootstrap.sh" --only-show-errors

AK=$($AWS configure get aws_access_key_id)
SK=$($AWS configure get aws_secret_access_key)
UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
export AWS_ACCESS_KEY_ID=$AK
export AWS_SECRET_ACCESS_KEY=$SK
export AWS_DEFAULT_REGION=$REGION
export S3_BUCKET=$BUCKET
aws s3 cp s3://$BUCKET/nets/searchbox_bootstrap.sh /root/b.sh
bash /root/b.sh
EOF
ID=$($A ec2 run-instances --region $REGION --image-id $AMI --instance-type "$TYPE" \
  --key-name pokebot --security-group-ids $SG \
  --instance-market-options 'MarketType=spot' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=r64-searchbox}]' \
  --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "launched $ID, waiting for health..."
echo "$ID" > "$HERE/searchbox_id.txt"
IP=""
for _ in $(seq 40); do
  sleep 5
  IP=$($A ec2 describe-instances --region $REGION --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
  [ -n "$IP" ] && [ "$IP" != "None" ] && break
done
[ -n "$IP" ] && [ "$IP" != "None" ] || { echo "no IP after 200s"; exit 1; }
URL="http://$IP:8000"
for _ in $(seq 60); do
  sleep 5
  if curl -sf -m 4 "$URL/health" 2>/dev/null | grep -q '"ok"'; then
    echo "$URL" > "$HERE/searchbox_ip.txt"
    echo "search box READY at $URL ($ID)"
    curl -sf -m 4 "$URL/health"; echo
    exit 0
  fi
done
echo "box never became healthy; check: $A s3 ... or ssh ec2-user@$IP"; exit 1
