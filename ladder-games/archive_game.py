"""Archive the ladder game(s) played since --since into ladder-games/games/.

Called by run_game.sh after the bot exits. Best-effort on every optional step:
a failed replay fetch or parse never loses the battle log.
"""
import argparse
import datetime
import glob
import gzip
import json
import os
import re
import shutil
import sys
import urllib.request

# Env-overridable so the same archiver runs on fleet boxes (different roots)
ROOT = os.environ.get("FP_ARCHIVE_WORKSPACE", "/Users/sallyliu/pokemon-fast-bot")
LOGS = os.environ.get("FP_ARCHIVE_LOGS", os.path.join(ROOT, "foul-play", "logs"))
ARCHIVE = os.environ.get("FP_ARCHIVE_DIR", os.path.join(ROOT, "ladder-games"))
USERNAME = "fable foul play"  # fallback only; result/account are log-derived
# FP_ARCHIVE_NO_LOSSES=1 skips the LOSSES.md register append (fleet boxes:
# the register is curated locally, and spot-reclaimed losses are excluded)
NO_LOSSES = os.environ.get("FP_ARCHIVE_NO_LOSSES") == "1"


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "unknown"


SEARCH_MARKERS = (
    "Policy ",
    "WorldStats",
    "OppWorldStats",
    "PairTable",
    "ARGMAX-ONLY",
    "Tera gate",
    "TurnTiming",
    "Choice:",
    "Budget:",
    "World cap:",
)
LEVEL_PREFIX = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s")
STATESTRING = re.compile(r"INFO\s+StateString (\d+) chance=([\d.]+) (.*)$")


def segment_log(text, dest):
    """Split battle.log into typed files: worlds.jsonl (sampled world states),
    search.log (per-decision search internals), protocol.log (server messages).
    Lines in search.log are prefixed [d N]; worlds.jsonl rows carry the same
    decision number. Decision N ends at its "Choice:" line."""
    worlds, search, protocol = [], [], []
    decision = 1
    in_protocol = False
    for line in text.splitlines():
        if "Received message from websocket: " in line:
            in_protocol = True
            protocol.append(line.split("Received message from websocket: ", 1)[1])
            continue
        if in_protocol and not LEVEL_PREFIX.match(line):
            protocol.append(line)  # continuation of a multiline websocket chunk
            continue
        in_protocol = False
        m = STATESTRING.search(line)
        if m:
            worlds.append(
                json.dumps(
                    {
                        "decision": decision,
                        "world": int(m.group(1)),
                        "chance": float(m.group(2)),
                        "state": m.group(3),
                    }
                )
            )
            continue
        if any(k in line for k in SEARCH_MARKERS):
            search.append("[d {}] {}".format(decision, line))
            if "Choice:" in line:
                decision += 1
    for name, rows in (
        ("worlds.jsonl", worlds),
        ("search.log", search),
        ("protocol.log", protocol),
    ):
        with open(os.path.join(dest, name), "w") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))


def already_archived(tag):
    idx = os.path.join(ARCHIVE, "index.jsonl")
    if not os.path.exists(idx):
        return False
    return any(json.loads(l)["battle_tag"] == tag for l in open(idx))


def archive_one(log_path, flags):
    fname = os.path.basename(log_path)  # <battle_tag>_<opponent>.log
    stem = fname[: -len(".log")]
    tag, _, opponent = stem.partition("_")  # battle tags never contain "_"

    # Idempotency: mtime windows can re-catch an old log (rollover touches
    # files); a tag is archived exactly once, ever.
    if already_archived(tag):
        print("skip (already archived): {}".format(tag))
        return None

    text = open(log_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Winner: (.*)", text)
    winner = m.group(1).strip() if m else None
    # A log with no winner is a game still IN PROGRESS (a sibling slot's
    # archiver can catch it mid-game via the mtime window). Never archive it:
    # indexing a truncated copy makes the idempotency guard skip the real
    # archive when the game ends (7 truncated games, aws canary3 2026-08-08).
    if winner is None:
        print("skip (in progress, no winner yet): {}".format(tag))
        return None
    # account-agnostic: our side is whichever |player| is NOT the opponent
    # (from the filename), so multi-account fleets archive correctly
    result = (
        "L" if winner and winner == opponent else ("W" if winner else "unknown")
    )

    # INFRA losses are ours-by-machinery, not by play: they must never enter
    # training data (esp. not at the beat-us x2 weight) nor the loss register.
    # An inactivity timeout on a 4s/turn bot is always a clock/latency/bug
    # failure, never a strategic one.
    infra = None
    if result == "L" and re.search(
        r"lost due to inactivity|Invalid choice.*(?:nomove|No Move)", text
    ):
        infra = "inactivity_timeout"

    # pre-game Elos + our account name from the |player| lines (5th field is
    # the rating, present on rated games)
    our_elo = opp_elo = None
    account = None
    for pm in re.finditer(r"\|player\|p[12]\|([^|\n]+)\|[^|\n]*\|(\d+)", text):
        name, elo = pm.group(1).strip(), int(pm.group(2))
        if name == opponent:
            opp_elo = opp_elo if opp_elo is not None else elo
        else:
            our_elo = our_elo if our_elo is not None else elo
            account = account or name

    ended = datetime.datetime.fromtimestamp(os.path.getmtime(log_path))
    battle_id = tag.split("-", 2)[-1]  # numeric id (+ private suffix if any)
    dest = os.path.join(
        ARCHIVE,
        "games",
        "{}_{}_{}_{}".format(
            ended.strftime("%Y-%m-%d"), battle_id.split("-")[0], sanitize(opponent), result
        ),
    )
    os.makedirs(dest, exist_ok=True)
    segment_log(text, dest)
    with open(log_path, "rb") as src, gzip.open(
        os.path.join(dest, "battle.log.gz"), "wb"
    ) as dst:
        shutil.copyfileobj(src, dst)

    replay_id = tag[len("battle-"):] if tag.startswith("battle-") else tag
    replay_url = "https://replay.pokemonshowdown.com/{}.json".format(replay_id)
    replay_saved = False
    try:
        req = urllib.request.Request(
            replay_url, headers={"User-Agent": "Mozilla/5.0 (foul-play archive)"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            with open(os.path.join(dest, "replay.json"), "wb") as f:
                f.write(r.read())
        replay_saved = True
    except Exception as e:
        print("WARN: replay fetch failed for {}: {}".format(replay_id, e))

    meta = {
        "battle_tag": tag,
        "opponent": opponent,
        "account": account or USERNAME,
        "infra": infra,
        "opp_elo": opp_elo,
        "our_elo": our_elo,
        "result": result,
        "winner": winner,
        "ended_at": ended.isoformat(),
        "replay_url": "https://replay.pokemonshowdown.com/" + replay_id,
        "replay_saved": replay_saved,
        "flags": flags,
        "archived_at": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(dest, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(ARCHIVE, "index.jsonl"), "a") as f:
        f.write(json.dumps({**meta, "path": os.path.relpath(dest, ARCHIVE)}) + "\n")

    if result == "L" and infra is None and not NO_LOSSES:
        with open(os.path.join(ARCHIVE, "LOSSES.md"), "a") as f:
            n = sum(1 for line in open(os.path.join(ARCHIVE, "index.jsonl")) if '"result": "L"' in line)
            f.write(
                "| {} | {} | {} | {} |  |\n".format(
                    n, ended.strftime("%Y-%m-%d"), opponent, meta["replay_url"]
                )
            )

    print("archived {} ({} vs {}) -> {}".format(tag, result, opponent, dest))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, required=True, help="epoch seconds")
    ap.add_argument("--flags", default="", help="run flags for meta.json")
    args = ap.parse_args()

    logs = [
        p
        for p in glob.glob(os.path.join(LOGS, "battle-*.log"))
        + glob.glob(os.path.join(LOGS, "*", "battle-*.log"))
        if os.path.getmtime(p) > args.since
    ]
    if not logs:
        print("ERROR: no new battle log found since {}".format(args.since))
        sys.exit(1)
    for p in sorted(logs, key=os.path.getmtime):
        archive_one(p, args.flags)


if __name__ == "__main__":
    main()
