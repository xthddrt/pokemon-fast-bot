#!/usr/bin/env python3
"""Assert clean ladder state for our PS accounts.

For each account: log in on a throwaway connection, cancel any pending ladder
search, leave any battle rooms the account is still joined to, then disconnect.
Kills the 'stranded search / poisoned room -> ghost timer-loss' failure mode no
matter which client left the mess. Usage: sanitize_accounts.py [account ...]
(default: all five). Prints one line per account.
"""
import asyncio
import json
import os
import re
import sys

import websockets

SERVER = "wss://sim3.psim.us/showdown/websocket"
ACCOUNTS = ["fable foul play", "1v6king", "beatmesilly", "bobfamilyrules", "endodontist"]

def env_password():
    for line in open("/Users/sallyliu/pokemon-fast-bot/.env"):
        if line.startswith("PS_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("PS_PASSWORD not found")

def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

async def get_assertion(challstr, user, password):
    import urllib.parse, urllib.request
    data = urllib.parse.urlencode({
        "act": "login", "name": user, "pass": password, "challstr": challstr,
    }).encode()
    req = urllib.request.Request(
        "https://play.pokemonshowdown.com/~~showdown/action.php",
        data=data, headers={"User-Agent": "Mozilla/5.0"})
    body = await asyncio.get_event_loop().run_in_executor(
        None, lambda: urllib.request.urlopen(req, timeout=15).read().decode())
    j = json.loads(body[1:])
    return j["assertion"]

async def sanitize(account, password):
    async with websockets.connect(SERVER, ping_interval=None) as ws:
        challstr = None
        rooms = []
        async for msg in ws:
            for ln in msg.split("\n"):
                if ln.startswith("|challstr|"):
                    challstr = ln.split("|", 2)[2]
                    assertion = await get_assertion(challstr, account, password)
                    await ws.send(f"|/trn {account},0,{assertion}")
                elif ln.startswith("|updateuser|") and toid(account) in toid(ln.split("|")[2]):
                    await ws.send("|/cancelsearch")
                    await ws.send(f"|/cmd userdetails {account}")
                elif ln.startswith("|queryresponse|userdetails|"):
                    d = json.loads(ln.split("|", 3)[3])
                    rooms = [r.lstrip("☆* ") for r in (d.get("rooms") or {})]
                    for r in rooms:
                        if r.startswith("battle-"):
                            await ws.send(f"|/noreply /leave {r}")
                    await asyncio.sleep(1.5)
                    print(f"{account}: search cancelled, left {len(rooms)} room(s)")
                    return
            if challstr and rooms is None:
                break

async def main():
    password = env_password()
    targets = sys.argv[1:] or ACCOUNTS
    for a in targets:
        try:
            await asyncio.wait_for(sanitize(a, password), timeout=25)
        except Exception as e:
            print(f"{a}: sanitize FAILED ({type(e).__name__}: {e})")
        await asyncio.sleep(1)

asyncio.run(main())
