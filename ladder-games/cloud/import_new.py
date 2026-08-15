#!/usr/bin/env python3
"""Import new v7warm2 games from S3 into ladder-games/ (idempotent via counter).

Syncs game dirs, appends only-new index rows, appends non-infra losses to
LOSSES.md, prints one summary line per new game (consumed by freerun.sh).
"""
import json, os, subprocess, sys

AWS = "/Users/sallyliu/.awscli-venv/bin/aws"
B = "pokebot-valuenet-389825051723"
P = "ladder-fleet/v7warm2"
R = "/Users/sallyliu/pokemon-fast-bot/ladder-games"
CNT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".v7warm2_index_count")

subprocess.run([AWS, "s3", "sync", f"s3://{B}/{P}/archive/games/", f"{R}/games/",
                "--quiet"], check=False)
r = subprocess.run([AWS, "s3", "cp", f"s3://{B}/{P}/archive/index.jsonl", "-"],
                   capture_output=True, text=True)
lines = [l for l in r.stdout.splitlines() if l.strip()]
seen = int(open(CNT).read().strip()) if os.path.exists(CNT) else 0
new = lines[seen:]
for l in new:
    row = json.loads(l)
    with open(f"{R}/index.jsonl", "a") as f:
        f.write(l + "\n")
    if row.get("result") == "L" and not row.get("infra"):
        with open(f"{R}/LOSSES.md", "a") as f:
            f.write("| - | %s | %s | %s | (%s) |\n" % (
                row.get("ended_at", "")[:10], row.get("opponent"),
                row.get("replay_url"), row.get("account")))
    print("GAME %s: %s vs %s (opp %s, us %s) acct=%s" % (
        row.get("result"), row.get("path", "?").split("/")[-1],
        row.get("opponent"), row.get("opp_elo"), row.get("our_elo"),
        row.get("account")))
open(CNT, "w").write(str(len(lines)))
