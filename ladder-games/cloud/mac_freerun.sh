#!/bin/bash
# Mac free-run: continuously play the LOWEST-elo account on gen9randombattle
# with the champion config (8 worlds x 1 thread, PUCT active, tera gate).
# Post-game elo comes from the game's own |raw| rating line (instant, exact) --
# the ladder API is only used to seed accounts we have not played yet.
# Stop cleanly: touch ladder-games/STOP_MAC_FREERUN (finishes current game).
ROOT=/Users/sallyliu/pokemon-fast-bot
STATE="$ROOT/ladder-games/ELO_STATE.json"
STOP="$ROOT/ladder-games/STOP_MAC_FREERUN"
LOG="$ROOT/ladder-games/ELO_LOG.jsonl"
PICKER=/private/tmp/claude-501/-Users-sallyliu-pokemon-fast-bot/410e0c58-8931-45a0-8e25-e3a8ec37baef/scratchpad/elo_pick.py
cd "$ROOT"

# seed state from the ladder API once
if [ ! -s "$STATE" ]; then
  python3 "$PICKER" pick | python3 -c "
import json,sys
d=json.load(sys.stdin)
json.dump({r['account']: r['elo'] for r in d['table']}, open('$STATE','w'), indent=1)
print('seeded elo state from ladder API')
"
fi

FAILS=0
while true; do
  [ -f "$STOP" ] && { echo "STOP flag — mac free-run halted"; rm -f "$STOP"; exit 0; }
  DONE=$(python3 -c "
import json; s=json.load(open('$STATE')); print(1 if min(s.values())>=2300 else 0)")
  [ "$DONE" = "1" ] && { echo "GOAL REACHED: all 5 accounts >= 2300"; exit 0; }
  PRE_N=$(grep -c . "$ROOT/ladder-games/index.jsonl")
  ACCT=$(python3 -c "
import json; s=json.load(open('$STATE')); print(min(s, key=s.get))")
  ELO=$(python3 -c "
import json; s=json.load(open('$STATE')); print('%.1f' % min(s.values()))")
  "$ROOT/foul-play/.venv/bin/python" "$ROOT/ladder-games/scrape/sanitize_accounts.py" 2>&1 | sed 's/^/SANITIZE: /' || true
  echo "PLAYING: $ACCT ($ELO) — lowest of 5"
  RG_USERNAME="$ACCT" RG_FORMAT=gen9randombattle \
  RG_FIRST_TURN_MS=14500 RG_SEARCH_MS=4500 RG_WORLDS=8 RG_POOL=8 \
    bash ladder-games/run_game.sh > /tmp/mac_freerun_last.log 2>&1 &
  GP=$!
  G0=$(date +%s)
  while kill -0 $GP 2>/dev/null; do
    if [ $(( $(date +%s) - G0 )) -gt 3900 ]; then
      echo "GAME TIMEOUT (65 min) — reaping stuck processes"
      pkill -TERM -P $GP 2>/dev/null; kill -TERM $GP 2>/dev/null; sleep 5
      pkill -9 -f 'run.py.*search_ladder' 2>/dev/null
      break
    fi
    sleep 20
  done
  wait $GP 2>/dev/null
  # reap any run.py that finished its game but never exited (observed hang)
  sleep 3; pkill -f 'run.py.*search_ladder' 2>/dev/null || true
  POST_N=$(grep -c . "$ROOT/ladder-games/index.jsonl")
  if [ "$POST_N" -le "$PRE_N" ]; then
    FAILS=$((FAILS+1))
    echo "NO NEW GAME archived (crash/disconnect #$FAILS) — retry in 90s"
    tail -3 /tmp/mac_freerun_last.log | head -3
    [ "$FAILS" -ge 3 ] && { echo "3 consecutive failures — halting for inspection"; exit 1; }
    sleep 90; continue
  fi
  FAILS=0
  python3 - "$ACCT" <<'PY'
import json, os, re, sys, time
acct = sys.argv[1]
ROOT = "/Users/sallyliu/pokemon-fast-bot"
STATE = ROOT + "/ladder-games/ELO_STATE.json"
LOG = ROOT + "/ladder-games/ELO_LOG.jsonl"
idx = [json.loads(l) for l in open(ROOT + "/ladder-games/index.jsonl") if l.strip()]
row = idx[-1]
path = os.path.join(ROOT, "ladder-games", row["path"])
new = None
proto = os.path.join(path, "protocol.log")
if os.path.exists(proto):
    pat = re.compile(re.escape(acct) + r"'s rating: (\d+) &rarr; <strong>(\d+)</strong>")
    for line in open(proto, errors="ignore"):
        m = pat.search(line)
        if m:
            new = float(m.group(2))
if new is None:                       # forfeit/crash: keep old, flag it
    s = json.load(open(STATE)); new = s.get(acct, 1000.0)
    print("GAME %s vs %s (opp %s) acct=%s — NO rating line (unrated/aborted?)" % (
        row.get("result"), row.get("opponent"), row.get("opp_elo"), acct))
else:
    print("GAME %s vs %s (opp %s) acct=%s | ELO NOW %.0f" % (
        row.get("result"), row.get("opponent"), row.get("opp_elo"), acct, new))
    if row.get("result") == "L" and not row.get("infra"):
        with open(ROOT + "/ladder-games/LOSSES.md", "a") as f:
            f.write("| - | %s | %s | %s | (%s) |\n" % (
                row.get("ended_at", "")[:10], row.get("opponent"),
                row.get("replay_url"), acct))
s = json.load(open(STATE)); s[acct] = new
json.dump(s, open(STATE, "w"), indent=1)
with open(LOG, "a") as f:
    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "account": acct, "format": "gen9randombattle", "elo": new,
                        "source": "game_log", "result": row.get("result"),
                        "opponent": row.get("opponent")}) + "\n")
nxt = min(s, key=s.get)
print("NEXT UP: %s (%.0f)" % (nxt, s[nxt]))
PY
  sleep 5
done
