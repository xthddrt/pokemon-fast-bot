#!/bin/bash
# N parallel one-box mining rounds — "more hammers per iteration".
#
#   TAG=mine4 BOXES=4 GAMES=10 bash corrections/cloud/launch_mining_fleet.sh
#
# Box i runs TAG=<TAG>-b<i> with SEED_BASE offset i*1000 (disjoint games and
# fresh PS teams everywhere). Code is packed ONCE; each box launch reuses it.
# Results land per-box in corrections/_mine_work/<TAG>-b<i>/; assess the
# ledger_rows.json of every box together, then hammer once (the ledger is
# batch-native). Env: TAG (required), BOXES (4), GAMES (10), MS, SEED_BASE.
set -euo pipefail
BOXES="${BOXES:-4}"
TAG="${TAG:?set TAG}"
GAMES="${GAMES:-10}"
SEED_BASE="${SEED_BASE:-101}"
HERE="$(cd "$(dirname "$0")" && pwd)"

bash "$HERE/pack_mining_code.sh"
pids=()
for i in $(seq 1 "$BOXES"); do
  BT="$TAG-b$i"
  env TAG="$BT" SEED_BASE=$((SEED_BASE + i * 1000)) GAMES="$GAMES" \
      MS="${MS:-1000}" SKIP_PACK=1 \
      bash "$HERE/launch_mining.sh" > "/tmp/mining_$BT.log" 2>&1 &
  pids+=($!)
  sleep 5
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "fleet done: $(ls -d "$HERE/../_mine_work/$TAG"-b* 2>/dev/null | wc -l | tr -d ' ')/$BOXES boxes returned results"
exit $rc
