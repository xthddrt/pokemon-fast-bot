#!/usr/bin/env python3
"""PS high-ladder spectator scraper.

Connects to the main Showdown server as a guest (no login, no account risk),
polls the public battle list per format, joins qualifying battles as a silent
spectator, and records the full protocol log of each (PS replays the entire
battle so far on join, so mid-game joins still capture turn 1 onward).

Formats & elo floors (Sally 2026-08-13): gen9randombattle >= 1900,
gen9randombattleblitz >= 1700.

Output (under --out, default ladder-scrape/):
  raw/<battleid>.log   full protocol stream for the battle room
  meta.jsonl           one row per completed battle (players, elo, turns, result)
  seen.txt             battle ids ever joined (dedupe across restarts)

Politeness: one websocket, join throttle 1/s, room cap, roomlist poll every
POLL_S. Spectating is passive; no messages are ever sent to a battle room.

Stats line printed every STATS_S seconds -> harness monitor events.
"""
import asyncio
import json
import os
import re
import signal
import sys
import time

import websockets

SERVER = "wss://sim3.psim.us/showdown/websocket"
FORMATS = {
    "gen9randombattle": 1900,
    "gen9randombattleblitz": 1700,
}
POLL_S = 15          # roomlist poll interval per format
STATS_S = 900        # stats event interval
ROOM_CAP = 80        # max simultaneously joined battles
JOIN_GAP_S = 1.0     # min seconds between joins
LINGER_S = 8         # wait after |win| for the rating |raw| lines

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
    "/Users/sallyliu/pokemon-fast-bot/ladder-scrape"
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
META = os.path.join(OUT, "meta.jsonl")
SEEN_F = os.path.join(OUT, "seen.txt")

seen = set()
if os.path.exists(SEEN_F):
    seen = set(l.strip() for l in open(SEEN_F) if l.strip())

open_rooms = {}   # roomid -> {"fh": file, "fmt": str, "minElo": int, "t0": ts,
                  #            "won": float|None, "turns": int, "ratings": {}}
stats = {"joined": 0, "completed": 0, "by_fmt": {f: 0 for f in FORMATS}}
_last_join = 0.0


def room_fmt(roomid):
    m = re.match(r"battle-([a-z0-9]+)-\d+", roomid)
    return m.group(1) if m else None


async def poll_roomlists(ws):
    while True:
        for fmt in FORMATS:
            try:
                # plain form only: ",<filter>,<page>" args make the server
                # return an empty list (probed 2026-08-13); filter client-side
                await ws.send(f"|/cmd roomlist {fmt}")
            except Exception:
                return
            await asyncio.sleep(2)
        await asyncio.sleep(POLL_S)


async def sweeper(ws):
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for rid, info in list(open_rooms.items()):
            idle = now - info["last_msg"]
            if (info["won_at"] and idle > 30) or idle > 2700:
                try:
                    await ws.send(f"|/leave {rid}")
                except Exception:
                    pass
                finish_room(rid, info)
                del open_rooms[rid]


async def stats_loop():
    while True:
        await asyncio.sleep(STATS_S)
        print(
            "SCRAPE: %d completed (%s) | %d open | %d joined total"
            % (
                stats["completed"],
                ", ".join(f"{k}={v}" for k, v in stats["by_fmt"].items()),
                len(open_rooms),
                stats["joined"],
            ),
            flush=True,
        )


def finish_room(roomid, info):
    info["fh"].close()
    row = {
        "id": roomid,
        "format": info["fmt"],
        "minElo": info["minElo"],
        "turns": info["turns"],
        "players": info.get("players", {}),
        "ratings_after": info.get("ratings", {}),
        "winner": info.get("winner"),
        "aborted": info["won_at"] is None,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META, "a") as f:
        f.write(json.dumps(row) + "\n")
    stats["completed"] += 1
    stats["by_fmt"][info["fmt"]] = stats["by_fmt"].get(info["fmt"], 0) + 1


async def handle(ws):
    global _last_join
    poller = asyncio.create_task(poll_roomlists(ws))
    stat_t = asyncio.create_task(stats_loop())
    sweep_t = asyncio.create_task(sweeper(ws))
    try:
        async for msg in ws:
            room = None
            lines = msg.split("\n")
            if lines[0].startswith(">"):
                room = lines[0][1:].strip()
                lines = lines[1:]

            if room and room in open_rooms:
                info = open_rooms[room]
                info["last_msg"] = time.time()
                info["fh"].write("\n".join(lines) + "\n")
                for ln in lines:
                    if ln.startswith("|turn|"):
                        info["turns"] = int(ln.split("|")[2])
                    elif ln.startswith("|player|"):
                        p = ln.split("|")
                        if len(p) >= 4 and p[3]:
                            info.setdefault("players", {})[p[2]] = {
                                "name": p[3],
                                "rating": p[5] if len(p) > 5 else None,
                            }
                    elif ln.startswith("|win|"):
                        info["won_at"] = time.time()
                        info["winner"] = ln.split("|")[2]
                    elif ln.startswith("|tie"):
                        info["won_at"] = time.time()
                        info["winner"] = None
                    elif ln.startswith("|raw|") and "rating:" in ln:
                        m = re.search(
                            r"(.+?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>", ln
                        )
                        if m:
                            info.setdefault("ratings", {})[m.group(1)] = {
                                "before": int(m.group(2)),
                                "after": int(m.group(3)),
                            }
                    elif ln.startswith("|deinit") or ln.startswith("|expire"):
                        info["dead"] = True
                # finalize: room died, or win seen + ratings in (or linger passed)
                if info["dead"] or (
                    info["won_at"] is not None
                    and (
                        len(info.get("ratings", {})) >= 2
                        or time.time() - info["won_at"] > LINGER_S
                    )
                ):
                    try:
                        await ws.send(f"|/leave {room}")
                    except Exception:
                        pass
                    finish_room(room, info)
                    del open_rooms[room]
                continue

            # roomlist responses arrive without a room prefix
            for ln in lines:
                if ln.startswith("|queryresponse|roomlist|"):
                    try:
                        data = json.loads(ln.split("|", 3)[3])
                    except (json.JSONDecodeError, IndexError):
                        continue
                    for rid, r in (data.get("rooms") or {}).items():
                        fmt = room_fmt(rid)
                        if fmt not in FORMATS or rid in seen:
                            continue
                        if len(open_rooms) >= ROOM_CAP:
                            break
                        try:
                            min_elo = int(r.get("minElo") or 0)
                        except (TypeError, ValueError):
                            min_elo = 0
                        if min_elo < FORMATS[fmt]:
                            continue
                        now = time.time()
                        wait = max(0.0, _last_join + JOIN_GAP_S - now)
                        if wait:
                            await asyncio.sleep(wait)
                        _last_join = time.time()
                        seen.add(rid)
                        with open(SEEN_F, "a") as f:
                            f.write(rid + "\n")
                        fh = open(os.path.join(RAW, rid + ".log"), "w")
                        open_rooms[rid] = {
                            "fh": fh,
                            "fmt": fmt,
                            "minElo": min_elo,
                            "t0": now,
                            "won_at": None,
                            "dead": False,
                            "last_msg": now,
                            "turns": 0,
                        }
                        stats["joined"] += 1
                        try:
                            await ws.send(f"|/join {rid}")
                        except Exception:
                            return
    finally:
        poller.cancel()
        stat_t.cancel()
        sweep_t.cancel()
        for rid, info in list(open_rooms.items()):
            finish_room(rid, info)
        open_rooms.clear()


async def main():
    backoff = 5
    while True:
        try:
            async with websockets.connect(SERVER, ping_interval=30) as ws:
                print("connected to showdown as guest spectator", flush=True)
                backoff = 5
                await handle(ws)
        except (websockets.WebSocketException, OSError) as e:
            print(f"connection lost ({e}); reconnect in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    asyncio.run(main())
