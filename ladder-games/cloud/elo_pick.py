#!/usr/bin/env python3
"""Min-elo account scheduler for the v7 ladder push.

pick            -> print elo table for all 5 accounts + lowest (JSON)
log <account>   -> re-query that account, append row to ladder-games/ELO_LOG.jsonl
Rating source: PS ladderget (gen9randombattle). Unrated => 1000 provisional.
"""
import json, re, sys, time, urllib.request

ACCOUNTS = ["fable foul play", "1v6king", "beatmesilly", "bobfamilyrules", "endodontist"]
FORMAT = "gen9randombattle"
LOG = "/Users/sallyliu/pokemon-fast-bot/ladder-games/ELO_LOG.jsonl"

def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def ladder_elo(account):
    url = ("https://play.pokemonshowdown.com/~~showdown/action.php"
           "?act=ladderget&user=" + toid(account))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode()
    data = json.loads(body[1:] if body.startswith("]") else body)
    for row in data:
        if row.get("formatid") == FORMAT:
            return float(row["elo"]), float(row.get("gxe", 0))
    return 1000.0, 0.0

def table():
    out = []
    for a in ACCOUNTS:
        try:
            elo, gxe = ladder_elo(a)
        except Exception as e:
            elo, gxe = 1000.0, 0.0
        out.append({"account": a, "elo": round(elo, 1), "gxe": gxe})
        time.sleep(0.3)
    out.sort(key=lambda r: r["elo"])
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if mode == "pick":
        t = table()
        print(json.dumps({"lowest": t[0]["account"], "table": t}, indent=1))
    elif mode == "log":
        acct = sys.argv[2]
        elo, gxe = ladder_elo(acct)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "account": acct, "format": FORMAT, "elo": elo, "gxe": gxe}
        with open(LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row))
