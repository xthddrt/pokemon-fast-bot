"""Minimal PS client that plays the OTHER side of a local validation game.

    .venv/bin/python synth_opponent.py --name synthopp --out truth.json

Connects to the local (--no-security) Showdown server, accepts the first
challenge it is offered, and plays uniformly-random legal choices. Its only
real job is to WRITE DOWN ITS OWN TEAM: the `|request|` it receives carries
every hidden attribute of the side our bot has to infer -- exact moves, item,
ability, tera type, level, stats -- so the file it dumps is the GROUND TRUTH
that ladder-games/validation/audit_sampling.py scores the sampler against.

Truth comes from the server's own `|request|`, not from our Python port of the
team generator (fp/search/ps_teams.py). That is deliberate: five of the open
sampling findings are bugs IN that port, and grading the sampler against the
same code it is meant to be checked against would hide exactly those.
"""
import argparse
import asyncio
import json
import random

import websockets


async def run(uri, name, out_path, seed):
    rng = random.Random(seed)
    truth = {"name": name, "team": None, "battle_tag": None, "requests": 0}
    async with websockets.connect(uri, max_size=None) as ws:
        battle = None
        sent_trn = False
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=300)
            except asyncio.TimeoutError:
                print("timeout waiting for server", flush=True)
                break
            room = ""
            for chunk in raw.split("\n"):
                if chunk.startswith(">"):
                    room = chunk[1:].strip()
                    continue
                if not chunk.startswith("|"):
                    continue
                parts = chunk.split("|")
                kind = parts[1] if len(parts) > 1 else ""

                if kind == "challstr" and not sent_trn:
                    # --no-security: any name, empty assertion. ONCE only --
                    # a second /trn is processed as a rename even when the name
                    # is unchanged, and the server cancels any pending challenge
                    # with "cancelled because they changed their username".
                    sent_trn = True
                    await ws.send("|/trn %s,0," % name)

                elif kind == "updateuser":
                    # Readiness must be the SERVER's confirmation of the rename,
                    # not the moment we sent /trn. A connection starts as
                    # "Guest N" and is renamed asynchronously; challenging the
                    # pre-rename identity makes the server drop the challenge
                    # with "cancelled because they changed their username".
                    who, named = parts[2].strip(), parts[3] if len(parts) > 3 else "0"
                    if named == "1" and who.lower().replace(" ", "") == \
                            name.lower().replace(" ", ""):
                        print("logged in as", who, flush=True)

                elif kind == "pm" and len(parts) > 4 and \
                        parts[4].startswith("/challenge") and parts[4].strip() != "/challenge":
                    # THIS is how a challenge actually arrives:
                    #   |pm| <from>| <to>|/challenge <format>|...
                    # `updatechallenges` is not delivered to the challenged
                    # side on this server, so keying the accept on it never
                    # fired at all. A bare "/challenge" is the WITHDRAWN form.
                    sender = parts[2].strip()
                    target = parts[3].strip()
                    if target.lower().replace(" ", "") == name.lower().replace(" ", "") \
                            and sender.lower().replace(" ", "") != name.lower().replace(" ", ""):
                        print("accepting challenge from", sender, flush=True)
                        await ws.send("|/accept %s" % sender)

                elif kind == "updatechallenges":
                    data = json.loads(parts[2])
                    for who in (data.get("challengesFrom") or {}):
                        print("accepting challenge from", who, flush=True)
                        await ws.send("|/accept %s" % who)

                elif kind == "request" and parts[2].strip():
                    req = json.loads(parts[2])
                    truth["requests"] += 1
                    if truth["team"] is None and req.get("side", {}).get("pokemon"):
                        truth["team"] = req["side"]["pokemon"]
                        truth["battle_tag"] = room
                        with open(out_path, "w") as f:
                            json.dump(truth, f, indent=1)
                        print("TRUTH written: %d mons -> %s"
                              % (len(truth["team"]), out_path), flush=True)
                    battle = room or battle
                    await respond(ws, battle, req, rng)

                elif kind == "win":
                    print("game over, winner:", parts[2], flush=True)
                    truth["winner"] = parts[2]
                    with open(out_path, "w") as f:
                        json.dump(truth, f, indent=1)
                    return


async def respond(ws, battle, req, rng):
    if req.get("wait"):
        return
    rqid = req.get("rqid", "")
    if req.get("forceSwitch"):
        alive = [
            i + 1 for i, p in enumerate(req["side"]["pokemon"])
            if not p.get("active") and not p.get("condition", "").endswith(" fnt")
            and p.get("condition") != "0 fnt"
        ]
        if alive:
            await ws.send("%s|/choose switch %d|%s"
                          % (battle, rng.choice(alive), rqid))
        return
    active = (req.get("active") or [{}])[0]
    moves = [
        i + 1 for i, m in enumerate(active.get("moves", []))
        if not m.get("disabled")
    ]
    if moves:
        # Terastallize sometimes. Without this the opponent NEVER teras, and the
        # "no tera action after a side has used its tera" check in
        # audit_sampling.py is only ever exercised on our own half.
        tera = " terastallize" if (active.get("canTerastallize") and rng.random() < 0.25) else ""
        await ws.send("%s|/choose move %d%s|%s"
                      % (battle, rng.choice(moves), tera, rqid))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://localhost:8000/showdown/websocket")
    ap.add_argument("--name", default="synthopp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    asyncio.run(run(a.uri, a.name, a.out, a.seed))
