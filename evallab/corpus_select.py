"""ONE-POSITION-PER-GAME selection from the v6-era full-information shards.

WHY THIS EXISTS
    The 200k playout-labelled set (evallab/data/pl2) draws ~10 positions from
    each of 20,000 games of ONE fixed team pair. Positions inside a game share
    a trajectory and a label, so 200k rows carry far less than 200k rows worth
    of independent information, and every row shares one roster. This selector
    replaces both limitations: EXACTLY ONE position per game, from games whose
    team pairs are drawn from a ~100k-team corpus, so N rows are N independent
    samples over N distinct team pairs.

WHY THE r-SERIES SHARDS AND NOT THE v7 CORPUS
    A playout needs the TRUE state of BOTH sides. The r-series shards
    (r4/r4x/r5/r5x/r7) are FULL-INFORMATION self-play: `valuenet/generate_selfplay.py`
    builds both teams explicitly and logs the exact engine state string per
    decision, so `s` already IS the true state -- no reconstruction, no
    determinizer. The v7 corpus_1m stores PS protocol logs plus per-side
    infostate rows; its true state exists only as a replay to be reconstructed.

SHARD SCHEMA (see generate_selfplay.py "SHARD SCHEMA v2")
    line 1: header {"schema":2,"kind":"header",...}
    then:   {"file": gid, "outcome": 1|0|0.5,
             "turns":[{"s":state,"v":root_value,"n":{arm:visits},"n2":{...},
                       "it":iters,"m":arm}, ...]}
    Some shards are stored as concatenated gzip members whose LAST member is
    truncated; `read_shard` recovers every complete line and drops the partial
    tail. r4/r4x parse whole; r5/r5x/r7 lose the tail of each shard.

SELECTION RULES (Sally's design)
    * one position per game, phase split 40/40/20 over early/mid/late
    * <= DECIDED_CAP of the set may be "already decided" (|q_search-0.5| > 0.4)
    * <= OPENING_CAP of the set may be a turn-1/2 opening (ply < 2)
    * a position must be a real decision (>= 2 arms with visits on side one)
    * the label distribution is NOT balanced and positions are NOT selected for
      model disagreement -- both would bias the calibration this set exists for

    Each shard is an independent uniform sample of the corpus, so the quotas
    are applied PER SHARD and the union satisfies them globally. That keeps the
    job embarrassingly parallel and single-pass over ~12 GB of shards.

OUTPUT (JSONL, one line per selected position -- the exact input contract of
`playout.py run`, plus provenance fields it passes through untouched)
    {"i": global index after shuffle, "g": "<prefix>/<shard>/<gid>", "ply": int,
     "band": "early|mid|late", "s": state_string, "y_single": game outcome,
     "q_search": root value, "src": prefix, "pair": 16-hex team-pair hash}

USAGE
    python corpus_select.py <prefix> <n_positions> <out.jsonl> [seed] [shard_limit]
      prefix        r4 | r4x | r5 | r5x | r7   (S3 prefix <p>/shards_<p>/)
      n_positions   how many positions to select in total
ENV
    SELECT_WORKERS  process pool size (default: cpu_count)
    BUCKET          default pokebot-valuenet-389825051723
    EXCLUDE_PAIRS   optional local file of pair hashes to skip (eval/train
                    disjointness); one hash per line
"""

import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
import zlib

import boto3
from botocore.config import Config

BUCKET = os.environ.get("BUCKET", "pokebot-valuenet-389825051723")
BANDS = (("early", 0, 10), ("mid", 10, 25), ("late", 25, 10 ** 9))
ALLOC = {"early": 0.40, "mid": 0.40, "late": 0.20}
DECIDED_CAP = 0.10          # fraction of the set with |q_search - 0.5| > 0.4
DECIDED_EDGE = 0.4
OPENING_CAP = 0.05          # fraction of the set with ply < 2
OPENING_PLY = 2

_CFG = Config(max_pool_connections=64, retries={"max_attempts": 10, "mode": "adaptive"})


def s3():
    return boto3.client("s3", config=_CFG)


def band_of(ply):
    for name, lo, hi in BANDS:
        if lo <= ply < hi:
            return name
    return "late"


def read_shard(body):
    """Concatenated-gzip-member reader that tolerates a truncated final member.

    zlib.decompressobj returns whatever decoded before the stream ran out
    instead of raising, which is exactly what we want: the complete lines of a
    truncated shard are perfectly good games. gzip.open would raise EOFError
    and lose the whole file.
    """
    parts = []
    data = body
    while data:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            parts.append(d.decompress(data))
        except zlib.error:
            break
        if d.unused_data == data:      # no progress: give up rather than spin
            break
        data = d.unused_data
    return b"".join(parts).decode("utf-8", "replace")


def pair_hash(state_str):
    """Stable id of the (team, team) matchup, order-insensitive.

    Species names are taken from the FIRST turn, where no form change
    (Palafin->PalafinHero, Terapagos->TerapagosTerastal, ...) has happened yet.
    """
    sides = state_str.split("/")
    t1 = sorted(m.split(",")[0] for m in sides[0].split("=")[:6])
    t2 = sorted(m.split(",")[0] for m in sides[1].split("=")[:6])
    key = "|".join(sorted(["+".join(t1), "+".join(t2)]))
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


def eligible(game):
    """-> {band: [(ply, v, is_decided, is_opening)]} for real decisions only."""
    per = {b[0]: [] for b in BANDS}
    for k, t in enumerate(game["turns"]):
        n = t.get("n") or {}
        if sum(1 for x in n.values() if x > 0) < 2:
            continue                      # forced: nothing to be uncertain about
        v = t.get("v")
        if v is None:
            continue
        per[band_of(k)].append((k, float(v), abs(float(v) - 0.5) > DECIDED_EDGE,
                                k < OPENING_PLY))
    return per


def select_shard(args):
    """Pick `quota` positions (one per game) out of ONE shard. Pure + parallel."""
    key, quota, seed, exclude = args
    if key.startswith("/"):               # local file: used by the offline test
        body = open(key, "rb").read()
    else:
        body = s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    rng = random.Random(seed)

    games = []
    bad = 0
    for line in read_shard(body).splitlines():
        try:
            g = json.loads(line)
        except Exception:
            bad += 1                       # truncated tail line
            continue
        if g.get("kind") == "header" or "error" in g or not g.get("turns"):
            continue
        games.append(g)

    band_q = {b: int(round(ALLOC[b] * quota)) for b in ALLOC}
    dec_q = int(round(DECIDED_CAP * quota))
    open_q = int(round(OPENING_CAP * quota))
    order = list(range(len(games)))
    rng.shuffle(order)

    picked, seen_pairs = [], set()
    n_dec = n_open = 0
    dup_pair = skipped = 0
    for gi in order:
        if sum(band_q.values()) <= 0:
            break
        g = games[gi]
        s0 = g["turns"][0]["s"]
        ph = pair_hash(s0)
        if ph in seen_pairs or ph in exclude:
            dup_pair += 1
            continue
        per = eligible(g)
        # Fill the band that is furthest from its quota first; that makes the
        # quotas exact without biasing WHICH games serve which band (games are
        # already in random order).
        cands = sorted((b for b in band_q if band_q[b] > 0 and per[b]),
                       key=lambda b: -band_q[b])
        chosen = None
        for b in cands:
            pool = per[b]
            ok = [p for p in pool
                  if (not p[2] or n_dec < dec_q) and (not p[3] or n_open < open_q)]
            if not ok:
                continue
            ply, v, is_dec, is_open = rng.choice(ok)
            chosen = (b, ply, v, is_dec, is_open)
            break
        if chosen is None:
            skipped += 1
            continue
        b, ply, v, is_dec, is_open = chosen
        band_q[b] -= 1
        n_dec += is_dec
        n_open += is_open
        seen_pairs.add(ph)
        picked.append({
            "g": "{}/{}/{}".format(key.split("/")[0],
                                   key.split("/")[-1].replace(".jsonl.gz", ""),
                                   g["file"]),
            "ply": ply, "band": b, "s": g["turns"][ply]["s"],
            "y_single": float(g["outcome"]), "q_search": round(v, 5),
            "src": key.split("/")[0], "pair": ph,
        })
    return picked, {"games": len(games), "picked": len(picked), "bad_lines": bad,
                    "dup_pair": dup_pair, "no_slot": skipped,
                    "short": {b: band_q[b] for b in band_q if band_q[b] > 0}}


def main():
    prefix = sys.argv[1]
    n_positions = int(sys.argv[2])
    out_path = sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shard_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    exclude = set()
    ex_path = os.environ.get("EXCLUDE_PAIRS")
    if ex_path and os.path.exists(ex_path):
        exclude = {l.strip() for l in open(ex_path) if l.strip()}
        print("excluding %d pair hashes" % len(exclude), flush=True)

    cli = s3()
    keys = []
    tok = None
    pfx = "{}/shards_{}/".format(prefix, prefix)
    while True:
        kw = {"Bucket": BUCKET, "Prefix": pfx}
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".jsonl.gz")]
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    keys.sort()
    if shard_limit:
        keys = keys[:shard_limit]
    assert keys, "no shards under s3://%s/%s" % (BUCKET, pfx)

    # Ask each shard for a little more than its exact share: some games are
    # skipped (duplicate pair, no band slot), and the merge trims to target.
    per_shard = int(n_positions / len(keys) * 1.06) + 2
    print("prefix %s | shards %d | per-shard quota %d | target %d"
          % (prefix, len(keys), per_shard, n_positions), flush=True)

    workers = int(os.environ.get("SELECT_WORKERS", os.cpu_count() or 8))
    tasks = [(k, per_shard, seed * 1_000_003 + i, exclude) for i, k in enumerate(keys)]
    rows, agg = [], {"games": 0, "picked": 0, "bad_lines": 0, "dup_pair": 0, "no_slot": 0}
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (picked, st) in enumerate(ex.map(select_shard, tasks, chunksize=1)):
            rows += picked
            for k in agg:
                agg[k] += st[k]
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print("  shards %d/%d rows %d elapsed %.0fs"
                      % (i + 1, len(tasks), len(rows), time.time() - t0), flush=True)

    rng = random.Random(seed ^ 0x5EED)
    rng.shuffle(rows)
    # Global de-dup: per-shard workers cannot see each other's pairs.
    seen, keep = set(), []
    for r in rows:
        if r["pair"] in seen:
            continue
        seen.add(r["pair"])
        keep.append(r)
        if len(keep) >= n_positions:
            break
    rows = keep

    from collections import Counter
    bands = Counter(r["band"] for r in rows)
    dec = sum(1 for r in rows if abs(r["q_search"] - 0.5) > DECIDED_EDGE)
    opn = sum(1 for r in rows if r["ply"] < OPENING_PLY)
    ysum = Counter(r["y_single"] for r in rows)
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            r["i"] = i
            f.write(json.dumps(r) + "\n")
    with open(out_path + ".pairs", "w") as f:
        for r in rows:
            f.write(r["pair"] + "\n")
    stats = {
        "prefix": prefix, "shards": len(keys), "target": n_positions,
        "selected": len(rows), "games_read": agg["games"],
        "distinct_games": len({r["g"] for r in rows}),
        "distinct_pairs": len(seen), "truncated_tail_lines": agg["bad_lines"],
        "skipped_dup_pair": agg["dup_pair"], "skipped_no_slot": agg["no_slot"],
        "bands": dict(bands),
        "band_frac": {k: round(v / max(1, len(rows)), 4) for k, v in bands.items()},
        "decided_frac": round(dec / max(1, len(rows)), 4),
        "opening_frac": round(opn / max(1, len(rows)), 4),
        "y_single": {str(k): v for k, v in ysum.items()},
        "mean_q_search": round(sum(r["q_search"] for r in rows) / max(1, len(rows)), 5),
        "mean_ply": round(sum(r["ply"] for r in rows) / max(1, len(rows)), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_path + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
