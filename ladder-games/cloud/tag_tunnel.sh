#!/bin/bash
# Tag-following reverse tunnel: keeps a relay tunnel to whatever instance
# currently carries Name=fleet-<slug>. Survives box relaunches (new IP) with
# no restart — it re-resolves the IP every reconnect. One per account.
# Usage: tag_tunnel.sh <name-tag>   e.g. tag_tunnel.sh fleet-1v6king
set -uo pipefail
AWS=/Users/sallyliu/.awscli-venv/bin/aws
TAG="$1"
while true; do
  IP=$($AWS ec2 describe-instances --region us-east-2 \
    --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running" \
    --query 'Reservations[-1].Instances[-1].PublicIpAddress' --output text 2>/dev/null)
  if [ -n "$IP" ] && [ "$IP" != "None" ]; then
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -i ~/.ssh/pokebot.pem -N -R 8765:127.0.0.1:8765 "ec2-user@$IP" 2>/dev/null || true
  fi
  sleep 8
done
