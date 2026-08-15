#!/bin/bash
# Launch the ONE plc2 corpus-prep box (spot, self-terminating).
# Same shape as launch_corpus.sh, but it installs a PINNED wheel instead of
# building one, excludes plc1's team pairs, and runs a 5k homogeneity canary.
#
#   ./launch_plc2prep.sh [type]
set -euo pipefail
ROOT=/Users/sallyliu/pokemon-fast-bot
AWS=/Users/sallyliu/.awscli-venv/bin/aws
[ -x "$AWS" ] || AWS=$(command -v aws)
S3=s3://pokebot-valuenet-389825051723
REGION=us-east-2
AMI=ami-028ba4d4ccb4b7b72
SG=sg-024075dbfb3236454
KEY=pokebot

TYPE="${1:-c7a.16xlarge}"
RUN="${RUN:-plc2}"
PLC1="${PLC1:-plc1}"
SRC="${SRC:-r4}"
N_POS="${N_POS:-999872}"
N_TRAIN="${N_TRAIN:-10}"
ITERS="${ITERS:-2000}"
W="${W:-64}"
CANARY_N="${CANARY_N:-5000}"
SEED="${SEED:-3}"
WATCHDOG="${WATCHDOG:-7200}"
WHEEL_SHA="${WHEEL_SHA:?pinned wheel sha256}"
SUBNET="${SUBNET:-}"

DEST="$S3/evallab/$RUN"
AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__RUN__|$RUN|g" -e "s|__PLC1__|$PLC1|g" -e "s|__SRC__|$SRC|g" \
    -e "s|__N_POS__|$N_POS|g" -e "s|__N_TRAIN__|$N_TRAIN|g" \
    -e "s|__ITERS__|$ITERS|g" -e "s|__W__|$W|g" \
    -e "s|__CANARY_N__|$CANARY_N|g" -e "s|__SEED__|$SEED|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__WHEEL_SHA__|$WHEEL_SHA|g" \
    "$ROOT/evallab/cloud/userdata_plc2prep.sh" > "$UD"
if grep -q '__[A-Z_0-9]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_0-9]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (run=$RUN src=$SRC n=$N_POS canary=$CANARY_N)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prep-$RUN},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.prep -"
