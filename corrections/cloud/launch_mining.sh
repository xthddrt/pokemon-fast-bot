#!/bin/bash
# Run ONE value-mining round on ONE spot box, then bring the results home and
# kill the box. Blocking: it polls S3 until the tarball lands.
#
#   TAG=mine3 GAMES=10 bash corrections/cloud/launch_mining.sh
#   TAG=smoke GAMES=2 MS=1500 SEED_BASE=301 bash corrections/cloud/launch_mining.sh
#
# Env: TAG (required), GAMES (10), MS (4500), SEED_BASE (101), MINE_ARGS (""),
#      TIMEOUT_S (7200), TYPES, REGION, SKIP_PACK=1, DRY=1.
#
# The instance is terminated on EVERY exit path -- success, timeout, box-side
# failure, or Ctrl-C -- by an EXIT trap. It is also armed to self-terminate two
# ways independently of this script: instance-initiated-shutdown-behavior=terminate
# plus the bootstrap's own HARD_TIMEOUT_S watchdog. Three kill switches, because
# a forgotten 32-vCPU box costs ~$9/day.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AWS="/Users/sallyliu/.awscli-venv/bin/aws"
[ -x "$AWS" ] || AWS="$(command -v aws)"
BUCKET="${S3_BUCKET:-pokebot-valuenet-389825051723}"
S3_PREFIX="${S3_PREFIX:-mining}"
TAG="${TAG:?set TAG (round tag; names the S3 result key and the local work dir)}"
GAMES="${GAMES:-10}"
MS="${MS:-1000}"
SEED_BASE="${SEED_BASE:-101}"
MINE_ARGS="${MINE_ARGS:-}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
REGION="${REGION:-us-east-2}"
# c7a.8xlarge = 32 vCPU Zen4, the best $/search-second here; the fallbacks keep
# the vCPU count so MINE_CONCURRENT=nproc means the same thing on all of them.
TYPES="${TYPES:-c7a.8xlarge c7i.8xlarge m7a.8xlarge c6a.8xlarge}"
case "$REGION" in
  us-east-2) SG="${SG:-sg-024075dbfb3236454}"; AMI="${AMI:-ami-0a31e8ebaa0595a73}" ;;
  us-east-1) SG="${SG:-sg-095f0d0b0b2e3bf03}"; AMI="${AMI:-ami-09e41b4da286f446c}" ;;
  us-west-2) SG="${SG:-sg-06f5f185337f11a7f}"; AMI="${AMI:-ami-090bdb6d8c1222525}" ;;
  *) SG="${SG:?set SG for $REGION}"; AMI="${AMI:?set AMI for $REGION}" ;;
esac
RESULT_KEY="$S3_PREFIX/results/$TAG.tar.gz"
DEST="$(cd "$HERE/.." && pwd)/_mine_work"

IID=""; IREGION="$REGION"
cleanup() {
  if [ -n "$IID" ]; then
    echo "terminating $IID"
    $AWS ec2 terminate-instances --region "$IREGION" --instance-ids "$IID" \
      --query 'TerminatingInstances[0].CurrentState.Name' --output text || true
  fi
}
trap cleanup EXIT

if [ "${SKIP_PACK:-0}" != "1" ]; then
  S3_BUCKET="$BUCKET" S3_PREFIX="$S3_PREFIX" bash "$HERE/pack_mining_code.sh"
fi
$AWS s3 cp "$HERE/mining_bootstrap.sh" "s3://$BUCKET/$S3_PREFIX/mining_bootstrap.sh"
$AWS s3 rm "s3://$BUCKET/$RESULT_KEY" >/dev/null 2>&1 || true
$AWS s3 rm "s3://$BUCKET/$S3_PREFIX/results/$TAG.FAILED.log" >/dev/null 2>&1 || true

SPOT=$($AWS ec2 describe-spot-price-history --region "$REGION" \
  --instance-types "${TYPES%% *}" --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice]' --output text \
  | sort -k2 -n | head -1) || SPOT=""
CHEAP_AZ=$(echo "$SPOT" | cut -f1); PRICE=$(echo "$SPOT" | cut -f2)
SUBNET=""
if [ -n "$CHEAP_AZ" ] && [ "${ANY_AZ:-0}" != "1" ]; then
  SUBNET=$($AWS ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=$CHEAP_AZ" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null) || SUBNET=""
  if [ "$SUBNET" = "None" ]; then SUBNET=""; fi
fi

cat <<EOF

===== VALUE MINING ROUND =====
tag           : $TAG
round         : $GAMES games, ${MS} ms/decision, seed-base $SEED_BASE $MINE_ARGS
box           : SPOT ${TYPES%% *} (32 vCPU) in $REGION${CHEAP_AZ:+ / $CHEAP_AZ}, fallbacks: $TYPES
spot price    : \$${PRICE:-?}/hr  ->  ~\$$(awk -v p="${PRICE:-0.4}" 'BEGIN{printf "%.2f", p*0.5}') for the ~25 min build phase,
                ~\$$(awk -v p="${PRICE:-0.4}" 'BEGIN{printf "%.2f", p*2}') if it runs the full ${TIMEOUT_S}s cap
result        : s3://$BUCKET/$RESULT_KEY  ->  $DEST/$TAG/
live log      : $AWS s3 cp s3://$BUCKET/$S3_PREFIX/logs/$TAG.run.log - | tail -40
kill switches : EXIT trap here + shutdown-behavior=terminate + box watchdog ${TIMEOUT_S}s
EOF

if [ "${DRY:-0}" = "1" ]; then echo ""; echo "DRY=1 -- nothing launched."; IID=""; exit 0; fi

AKID=$($AWS configure get aws_access_key_id)
ASEC=$($AWS configure get aws_secret_access_key)
SCRATCH="$(mktemp -d)"
UD="$SCRATCH/ud.sh"
# Stub user-data: fetch the real bootstrap from S3. Keeps user-data far under
# the 16 KB limit and lets the bootstrap be edited without touching this file.
cat > "$UD" <<STUBEOF
#!/bin/bash
export AWS_ACCESS_KEY_ID=$AKID
export AWS_SECRET_ACCESS_KEY=$ASEC
export AWS_DEFAULT_REGION=us-east-2
export S3_BUCKET=$BUCKET
export S3_PREFIX=$S3_PREFIX
export TAG=$TAG
export GAMES=$GAMES
export MS=$MS
export SEED_BASE=$SEED_BASE
export MINE_ARGS="$MINE_ARGS"
export MODE=${MODE:-mine}
export CAND_KEY=${CAND_KEY:-}
export SHARD_START=${SHARD_START:-0}
export SHARD_COUNT=${SHARD_COUNT:-0}
export AUDIT_GAME="${AUDIT_GAME:-}"
export AUDIT_NET=${AUDIT_NET:-v8c_hz18}
export N=${N:-20}
export ARMS=${ARMS:-0}
export HARD_TIMEOUT_S=$TIMEOUT_S
export CODE_KEY=$S3_PREFIX/code.tar.gz
exec > /root/stub.log 2>&1
set -x
# Dumbest kill switch, independent of the bootstrap ever starting.
( sleep \$((HARD_TIMEOUT_S + 900)); shutdown -h now ) &
for i in 1 2 3 4 5; do
  aws s3 cp "s3://\$S3_BUCKET/\$S3_PREFIX/mining_bootstrap.sh" /root/mining_bootstrap.sh && break
  sleep 10
done
if [ ! -s /root/mining_bootstrap.sh ]; then
  aws s3 cp /root/stub.log "s3://\$S3_BUCKET/\$S3_PREFIX/results/\$TAG.FAILED.log" || true
  shutdown -h now
fi
bash /root/mining_bootstrap.sh
STUBEOF
[ "$(wc -c < "$UD")" -lt 15360 ] || { echo "user-data too big" >&2; exit 1; }

# Capacity walk: cheapest AZ first, then any AZ (the farm's ANY_AZ pattern),
# then the fallback types. Back-to-back run-instances calls draw
# RequestLimitExceeded, which looks exactly like a capacity miss, hence the sleep.
try_launch() {  # $1 = instance type, $2 = subnet ("" = let EC2 pick the AZ)
  local t="$1" sn="$2" id
  id=$($AWS ec2 run-instances --region "$REGION" --image-id "$AMI" \
    --instance-type "$t" ${sn:+--subnet-id "$sn"} \
    --count 1 --key-name pokebot --security-group-ids "$SG" \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=60,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mine-$TAG},{Key=Job,Value=mining},{Key=Owner,Value=sally}]" \
    --user-data "file://$UD" \
    --query 'Instances[0].InstanceId' --output text 2>/dev/null) || id=""
  if [ -n "$id" ] && [ "$id" != "None" ]; then echo "$id"; return 0; fi
  return 1
}
for t in $TYPES; do
  if [ -n "$SUBNET" ]; then
    IID=$(try_launch "$t" "$SUBNET") && break
    echo "  $t: no spot capacity in $CHEAP_AZ, retrying any AZ"
    sleep 5
  fi
  IID=$(try_launch "$t" "") && break
  echo "  $t: no spot capacity in any AZ"
  sleep 5
done
rm -rf "$SCRATCH"
[ -n "$IID" ] || { echo "NOTHING LAUNCHED (no spot capacity on: $TYPES)" >&2; exit 1; }
echo ""
echo "launched $IID in $REGION -- polling s3://$BUCKET/$RESULT_KEY"

DEADLINE=$(( $(date +%s) + TIMEOUT_S + 600 ))
while :; do
  if $AWS s3api head-object --bucket "$BUCKET" --key "$RESULT_KEY" >/dev/null 2>&1; then
    echo "result is up"; break
  fi
  if $AWS s3api head-object --bucket "$BUCKET" --key "$S3_PREFIX/results/$TAG.FAILED.log" >/dev/null 2>&1; then
    echo "BOX FAILED -- log:" >&2
    $AWS s3 cp "s3://$BUCKET/$S3_PREFIX/results/$TAG.FAILED.log" - | tail -40 >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "TIMEOUT after ${TIMEOUT_S}s -- terminating" >&2; exit 1
  fi
  sleep 30
  printf '.'
done

mkdir -p "$DEST"
$AWS s3 cp "s3://$BUCKET/$RESULT_KEY" "$DEST/$TAG.tar.gz"
tar xzf "$DEST/$TAG.tar.gz" -C "$DEST"
rm -f "$DEST/$TAG.tar.gz"
echo ""
ls -la "$DEST/$TAG"
echo ""
echo "=== tail of the round ==="
tail -25 "$DEST/$TAG/run.log" || true
echo ""
echo "results: $DEST/$TAG   (ledger_rows.json = staged rulings, NOT hammered)"
