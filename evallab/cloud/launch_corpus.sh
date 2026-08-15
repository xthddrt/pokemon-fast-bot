#!/bin/bash
# Launch the ONE corpus-prep box (spot, self-terminating):
# builds+publishes the poke_engine wheel, selects the train/eval position sets
# from the r-series shards, and runs the cost canary.
#
#   BUNDLE=/tmp/code.tar.gz ./launch_corpus.sh <run> <evalrun> [type]
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
EVALRUN="${2:?eval run name}"
TYPE="${3:-c7a.16xlarge}"
SRC_TRAIN="${SRC_TRAIN:-r4}"
SRC_EVAL="${SRC_EVAL:-r4x}"
N_POS="${N_POS:-1000000}"
N_EVAL_POS="${N_EVAL_POS:-10000}"
N_TRAIN="${N_TRAIN:-10}"
N_EVAL="${N_EVAL:-100}"
ITERS="${ITERS:-2000}"
W="${W:-64}"
WATCHDOG="${WATCHDOG:-10800}"
CANARY="${CANARY:-1}"
SHARD_LIMIT="${SHARD_LIMIT:-0}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST="$S3/evallab/$RUN"
if [ "${SKIP_BUNDLE:-0}" != "1" ]; then
  echo "uploading bundle ($(du -h "$BUNDLE" | cut -f1)) -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__RUN__|$RUN|g" -e "s|__EVALRUN__|$EVALRUN|g" \
    -e "s|__SRC_TRAIN__|$SRC_TRAIN|g" -e "s|__SRC_EVAL__|$SRC_EVAL|g" \
    -e "s|__N_POS__|$N_POS|g" -e "s|__N_EVAL_POS__|$N_EVAL_POS|g" \
    -e "s|__N_TRAIN__|$N_TRAIN|g" -e "s|__N_EVAL__|$N_EVAL|g" \
    -e "s|__ITERS__|$ITERS|g" -e "s|__W__|$W|g" \
    -e "s|__WATCHDOG__|$WATCHDOG|g" -e "s|__CANARY__|$CANARY|g" \
    -e "s|__SHARD_LIMIT__|$SHARD_LIMIT|g" \
    "$ROOT/evallab/cloud/userdata_corpus.sh" > "$UD"
if grep -q '__[A-Z_]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (run=$RUN src=$SRC_TRAIN/$SRC_EVAL n=$N_POS/$N_EVAL_POS canary=$CANARY)"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 ${SUBNET:+--subnet-id "$SUBNET"} \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prep-$RUN},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID"
echo "watch: $AWS s3 cp $DEST/STATUS.prep -"
