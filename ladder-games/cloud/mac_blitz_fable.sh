#!/bin/bash
# Continuous BLITZ on 'fable foul play' with v6nopol (rollback 2026-08-13).
# One game at a time: sanitize -> play -> log exact post-game blitz elo from
# the game's own rating line -> repeat. Stop: touch ladder-games/STOP_BLITZ.
ROOT=/Users/sallyliu/pokemon-fast-bot
STOP="$ROOT/ladder-games/STOP_BLITZ"
LOG="$ROOT/ladder-games/ELO_LOG.jsonl"
ACCT="fable foul play"
cd "$ROOT"

FAILS=0
while true; do
  [ -f "$STOP" ] && { echo "STOP flag — blitz run halted"; rm -f "$STOP"; exit 0; }
  "$ROOT/foul-play/.venv/bin/python" "$ROOT/ladder-games/scrape/sanitize_accounts.py" "$ACCT" 2>&1 | sed 's/^/SANITIZE: /' || true
  PRE_N=$(grep -c . "$ROOT/ladder-games/index.jsonl")
  echo "PLAYING blitz: $ACCT (v6nopol)"
  RG_USERNAME="$ACCT" RG_FORMAT=gen9randombattleblitz \
  RG_NN_WEIGHTS="../valuenet/m4_artifacts/valuenet_v6nopol.bin" \
    bash ladder-games/run_game.sh > /tmp/mac_blitz_last.log 2>&1 &
  GP=$!
  G0=$(date +%s)
  while kill -0 $GP 2>/dev/null; do
    if [ $(( $(date +%s) - G0 )) -gt 3900 ]; then
      echo "GAME TIMEOUT (65 min) — reaping"
      pkill -TERM -P $GP 2>/dev/null; kill -TERM $GP 2>/dev/null; sleep 5
      pkill -9 -f 'run.py.*search_ladder' 2>/dev/null
      break
    fi
    sleep 20
  done
  wait $GP 2>/dev/null
  sleep 3; pkill -f 'run.py.*search_ladder' 2>/dev/null || true
  POST_N=$(grep -c . "$ROOT/ladder-games/index.jsonl")
  if [ "$POST_N" -le "$PRE_N" ]; then
    FAILS=$((FAILS+1))
    echo "NO NEW GAME archived (#$FAILS) — retry in 90s"
    tail -3 /tmp/mac_blitz_last.log | head -3
    [ "$FAILS" -ge 3 ] && { echo "3 consecutive failures — halting"; exit 1; }
    sleep 90; continue
  fi
  FAILS=0
  python3 - <<'PY'
import json, os, re, time
ROOT = "/Users/sallyliu/pokemon-fast-bot"
idx = [json.loads(l) for l in open(ROOT + "/ladder-games/index.jsonl") if l.strip()]
row = idx[-1]
path = os.path.join(ROOT, "ladder-games", row["path"])
new = None
proto = os.path.join(path, "protocol.log")
if os.path.exists(proto):
    pat = re.compile(r"fable foul play's rating: (\d+) &rarr; <strong>(\d+)</strong>")
    for line in open(proto, errors="ignore"):
        m = pat.search(line)
        if m:
            new = float(m.group(2))
msg = "GAME %s vs %s (opp %s)" % (row.get("result"), row.get("opponent"), row.get("opp_elo"))
if new is not None:
    msg += " | BLITZ ELO NOW %.0f" % new
    with open(ROOT + "/ladder-games/ELO_LOG.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "account": "fable foul play",
                            "format": "gen9randombattleblitz", "elo": new,
                            "source": "game_log", "result": row.get("result"),
                            "opponent": row.get("opponent"), "net": "v6nopol"}) + "\n")
print(msg)
PY
  sleep 5
done
