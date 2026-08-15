#!/bin/bash
# Launch one corpus-scale evallab box (spot, self-terminating).
# Same shape as cloud/launch.sh; parameterised for the 1M-game pair-A corpus.
#
#   MODE=probe                 ./launch_1m.sh <run> probe
#   MODE=bulk START=.. NGAMES=.. ./launch_1m.sh <run> <sub>
#
# Env: TYPE W ITERS RAND TEMP EPS BLOCK START NGAMES PROBE_GAMES CANARY_GAMES
#      PROBE_CFGS WATCHDOG BUNDLE SKIP_UPLOAD
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
SUB="${2:?sub name (probe|b0|b1|...)}"
MODE="${MODE:-bulk}"
TYPE="${TYPE:-c7a.16xlarge}"
W="${W:-64}"
ITERS="${ITERS:-2000}"
RAND="${RAND:-3}"; TEMP="${TEMP:-6}"; EPS="${EPS:-0.08}"
START="${START:-0}"; NGAMES="${NGAMES:-0}"; BLOCK="${BLOCK:-25000}"
PROBE_GAMES="${PROBE_GAMES:-6000}"; CANARY_GAMES="${CANARY_GAMES:-500}"
PROBE_CFGS="${PROBE_CFGS:-3:6:0.08 5:6:0.08 5:10:0.12}"
WATCHDOG="${WATCHDOG:-7200}"
BUNDLE="${BUNDLE:?path to code.tar.gz}"

DEST="$S3/evallab/$RUN"
if [ "${SKIP_UPLOAD:-0}" != "1" ]; then
  echo "uploading bundle -> $DEST/code.tar.gz"
  "$AWS" s3 cp "$BUNDLE" "$DEST/code.tar.gz" --only-show-errors
fi

AKID=$("$AWS" configure get aws_access_key_id)
ASEC=$("$AWS" configure get aws_secret_access_key)
UD=$(mktemp)
sed -e "s|__AKID__|$AKID|" -e "s|__ASEC__|$ASEC|" -e "s|__S3__|$S3|g" \
    -e "s|__RUN__|$RUN|g" -e "s|__MODE__|$MODE|g" -e "s|__SUB__|$SUB|g" \
    -e "s|__ITERS__|$ITERS|g" -e "s|__W__|$W|g" \
    -e "s|__RAND__|$RAND|g" -e "s|__TEMP__|$TEMP|g" -e "s|__EPS__|$EPS|g" \
    -e "s|__START__|$START|g" -e "s|__NGAMES__|$NGAMES|g" -e "s|__BLOCK__|$BLOCK|g" \
    -e "s|__PROBE_CFGS__|$PROBE_CFGS|g" -e "s|__PROBE_GAMES__|$PROBE_GAMES|g" \
    -e "s|__CANARY_GAMES__|$CANARY_GAMES|g" -e "s|__WATCHDOG__|$WATCHDOG|g" \
    "$ROOT/evallab/cloud/userdata_1m.sh" > "$UD"
# `if`, not `A && B`: under set -e a failing test in an AND-list kills the script
# on exactly the branch that was meant to be the no-op (same trap as userdata.sh).
if grep -q '__[A-Z_]*__' "$UD"; then
  echo "UNREPLACED PLACEHOLDER:"; grep -o '__[A-Z_]*__' "$UD" | sort -u; exit 3
fi

echo "launching $TYPE spot in $REGION (run=$RUN sub=$SUB mode=$MODE games=[$START,$((START+NGAMES)))"
ID=$("$AWS" ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --count 1 \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=evallab-$RUN-$SUB},{Key=proj,Value=evallab}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
echo "INSTANCE $ID $SUB"
