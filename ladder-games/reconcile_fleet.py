"""Reconcile fleet S3 archives into the local ladder-games archive.

Idempotent; run any time (boxes may be live). Enforces the labeling rule:
losses caused by our machinery are tagged `infra` and excluded from the loss
register (training exclusion is enforced again in the dataset builder).

Steps:
 1. sync each box's archive/games -> local games/, append new index rows,
    append REAL (non-infra) losses to LOSSES.md
 2. re-run the infra retagger locally (idempotent)
 3. rawlogs-vs-archive diff: battle logs with no archived game and older than
    GRACE_MIN minutes are crashed/killed games — ladder losses with no ending
    in their log. Add index-only rows: result=L, infra="unarchived_crash".
 4. print summary incl. training-data yield under the >=1400 / x2 rule
"""
import datetime
import json
import os
import re
import subprocess
import sys

AWS = "/Users/sallyliu/.awscli-venv/bin/aws"
BUCKET = "pokebot-valuenet-389825051723"
FLEET = "ladder-fleet/fleet1"
ARCHIVE = "/Users/sallyliu/pokemon-fast-bot/ladder-games"
BOXES = {
    "box-fable-foul-play": "fable foul play",
    "box-1v6king": "1v6king",
    "box-beatmesilly": "beatmesilly",
    "box-bobfamilyrules": "bobfamilyrules",
    "box-endodontist": "endodontist",
    # consolidated single-box fleet (2026-08-09): all 5 accounts on one 8-core
    # box, 1 slot each. Archived metas carry the real account; this label is
    # only used for hidden-crash rows whose account is unknowable from the log.
    "box-all": "mixed",
}
GRACE_MIN = 30


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    os.chdir(ARCHIVE)
    have = {json.loads(l)["battle_tag"] for l in open("index.jsonl")}
    new_rows, new_losses = [], []

    # 1. merge
    for box, account in BOXES.items():
        local = os.path.join("fleet-sync", box)
        os.makedirs(local, exist_ok=True)
        run([AWS, "s3", "sync", f"s3://{BUCKET}/{FLEET}/{box}/archive/games/", local, "--quiet"])
        for d in sorted(os.listdir(local)):
            mp = os.path.join(local, d, "meta.json")
            if not os.path.exists(mp):
                continue
            meta = json.load(open(mp))
            if meta["battle_tag"] in have:
                continue
            dest = os.path.join("games", d)
            if not os.path.exists(dest):
                run(["cp", "-r", os.path.join(local, d), dest])
            row = {**meta, "path": "games/" + d, "source": box}
            new_rows.append(row)
            have.add(meta["battle_tag"])
            if meta.get("result") == "L" and not meta.get("infra"):
                new_losses.append(meta)

    with open("index.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in new_rows)
    if new_losses:
        n = sum(1 for l in open("index.jsonl") if '"result": "L"' in l and '"infra": null' in l or ('"result": "L"' in l and '"infra"' not in l))
        with open("LOSSES.md", "a") as f:
            for i, m in enumerate(new_losses):
                f.write("| - | {} | {} | {} | ({}) |\n".format(
                    m.get("ended_at", "")[:10], m["opponent"], m["replay_url"], m.get("account", "")))

    # 2. local retag
    sys.path.insert(0, ARCHIVE)
    import retag_infra
    retagged = retag_infra.retag_local(ARCHIVE)

    # 3. hidden crashed games
    now = datetime.datetime.now(datetime.timezone.utc)
    hidden = []
    for box, account in BOXES.items():
        listing = run([AWS, "s3", "ls", f"s3://{BUCKET}/{FLEET}/{box}/rawlogs/", "--recursive"]).stdout
        for line in listing.splitlines():
            m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*/(battle-gen9randombattle(?:blitz)?-\d+(?:-\w+)?)_(.+)\.log$", line)
            if not m:
                continue
            ts, tag, opponent = m.groups()
            if tag in have:
                continue
            age = (now - datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)).total_seconds() / 60
            if age < GRACE_MIN:
                continue  # likely a live game
            row = {
                "battle_tag": tag, "opponent": opponent, "account": account,
                "opp_elo": None, "our_elo": None, "result": "L",
                "winner": None, "infra": "unarchived_crash", "path": None,
                "source": box,
                "replay_url": "https://replay.pokemonshowdown.com/" + tag[len("battle-"):],
            }
            hidden.append(row)
            have.add(tag)
    with open("index.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in hidden)

    # 4. summary
    rows = [json.loads(l) for l in open("index.jsonl")]
    W = sum(1 for r in rows if r.get("result") == "W")
    L = sum(1 for r in rows if r.get("result") == "L" and not r.get("infra"))
    infra = sum(1 for r in rows if r.get("infra"))
    inc = sum(1 for r in rows if not r.get("infra") and r.get("result") in ("W", "L")
              and (r.get("opp_elo") or 0) >= 1400)
    x2 = sum(1 for r in rows if not r.get("infra") and (r.get("opp_elo") or 0) >= 1400
             and ((r.get("opp_elo") or 0) >= 1700 or r.get("result") == "L"))
    print(json.dumps({
        "merged_new_games": len(new_rows), "new_register_losses": len(new_losses),
        "retagged_infra": retagged, "hidden_crashes_recorded": len(hidden),
        "archive_totals": {"W": W, "real_L": L, "infra": infra, "all": len(rows)},
        "training_yield": {"included_games_opp1400+": inc, "x2_weighted": x2},
    }, indent=2))


if __name__ == "__main__":
    main()
