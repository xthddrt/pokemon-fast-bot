"""Build checker-ready logs from archived real ladder games.

The fleet runs several battles per account on one websocket, so a slot's raw
log interleaves OTHER battles' frames. Feeding that to the replay checker
produces OverlongPartyError builds and mass turn-skips (measured: 5 of 71
turns checked). Here we keep only the chunks addressed to the game's own
room, re-emitting them in the checker's expected
"Received message from websocket: >battle-TAG" form.
"""
import gzip
import json
import os
import sys

G = "/Users/sallyliu/pokemon-fast-bot/ladder-games/games"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/conf_logs2"
PREFIX = "Received message from websocket: "
# the checker's chunk regex requires a log-level prefix (fp/replay/protocol.py:17)
EMIT = "DEBUG    " + PREFIX


def chunks(fh):
    """Yield (room_tag_or_None, [lines]) for each websocket chunk."""
    cur, tag = None, None
    for line in fh:
        line = line.rstrip("\n")
        if PREFIX in line:
            if cur is not None:
                yield tag, cur
            body = line.split(PREFIX, 1)[1]
            cur = [body]
            tag = body.strip()[1:] if body.startswith(">battle-") else None
        elif cur is not None and (line.startswith("|") or line == ""):
            cur.append(line)
        elif cur is not None:
            yield tag, cur
            cur, tag = None, None
    if cur is not None:
        yield tag, cur


def main():
    os.makedirs(OUT, exist_ok=True)
    n = kept_tot = drop_tot = 0
    for d in sorted(os.listdir(G)):
        mp = os.path.join(G, d, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        if m.get("infra") or m.get("result") not in ("W", "L"):
            continue
        src = os.path.join(G, d, "battle.log.gz")
        if not os.path.exists(src):
            continue
        tag = m["battle_tag"]
        opp = "".join(c for c in (m.get("opponent") or "x") if c.isalnum())
        dst = os.path.join(OUT, "{}_{}.log".format(tag, opp))
        kept = dropped = 0
        with gzip.open(src, "rt", errors="replace") as fh, open(dst, "w") as o:
            for ctag, lines in chunks(fh):
                if ctag == tag:
                    o.write(EMIT + lines[0] + "\n")
                    for l in lines[1:]:
                        o.write(l + "\n")
                    kept += 1
                elif ctag is not None:
                    dropped += 1
        kept_tot += kept
        drop_tot += dropped
        n += 1
    print("prepared {} logs | own-room chunks {} | foreign chunks dropped {}".format(
        n, kept_tot, drop_tot))


if __name__ == "__main__":
    main()
