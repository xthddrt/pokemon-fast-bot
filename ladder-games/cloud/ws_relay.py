"""Websocket relay: bridges fleet bots (via SSH reverse tunnel) to Showdown.

Runs on the Mac. Each inbound connection on 127.0.0.1:8765 is bridged to
wss://sim3.psim.us/showdown/websocket, so authenticated logins originate from
this residential IP while the bots (state tracking, world sampling, search)
run on AWS. Pure byte relay — no parsing, no per-message CPU to speak of.

Run: foul-play/.venv/bin/python ladder-games/cloud/ws_relay.py
"""
import asyncio
import logging

try:
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 13
    from websockets import connect, serve

UPSTREAM = "wss://sim3.psim.us/showdown/websocket"
PORT = 8765

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("relay")
_n = 0


async def bridge(client):
    global _n
    _n += 1
    cid = _n
    log.info("[%d] client connected (%d active-ish)", cid, _n)
    try:
        async with connect(UPSTREAM) as upstream:
            async def pump(src, dst):
                async for msg in src:
                    await dst.send(msg)

            t1 = asyncio.create_task(pump(client, upstream))
            t2 = asyncio.create_task(pump(upstream, client))
            done, pending = await asyncio.wait(
                {t1, t2}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
    except Exception as e:
        log.info("[%d] bridge ended: %r", cid, e)
    finally:
        log.info("[%d] closed", cid)


async def main():
    async with serve(bridge, "127.0.0.1", PORT):
        log.info("relay listening on 127.0.0.1:%d -> %s", PORT, UPSTREAM)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
