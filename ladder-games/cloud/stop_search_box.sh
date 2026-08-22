#!/bin/bash
# Terminate the r64 search box and clear its record.
set -euo pipefail
ROOT="${FP_ROOT:-/Users/sallyliu/pokemon-fast-bot}"
HERE="$ROOT/ladder-games/cloud"
AWS="${AWS_CLI:-/Users/sallyliu/.awscli-venv/bin/aws}"
[ -f "$HERE/searchbox_id.txt" ] || { echo "no recorded box"; exit 0; }
ID=$(cat "$HERE/searchbox_id.txt")
AWS_MAX_ATTEMPTS=1 $AWS --cli-connect-timeout 8 --cli-read-timeout 45 \
  ec2 terminate-instances --region eu-north-1 --instance-ids "$ID" \
  --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
rm -f "$HERE/searchbox_ip.txt" "$HERE/searchbox_id.txt"
echo "stopped"
