"""Corpus-scale diversity + recurrence pass (for corpora too large to hold in RAM).

`stats.py` answers the same questions on a 10k-game corpus by keeping every state
string in a python set. At 1,000,000 games / ~37M decisions that costs tens of
gigabytes, so this module keeps only a 64-bit hash per decision and does the set
algebra in numpy.

Two hashes per decision:
  h  blake2b-64 of the EXACT state string      -- the grouping the floor needs
  c  blake2b-64 of stats.coarse(state)         -- the grouping that actually
                                                  measures "did we explore?"
Both are stable across processes (blake2b, not python's randomised hash()).

USAGE
  python analyze_corpus.py scan  <out.npz> <shard-glob>...     # one box's shards
  python analyze_corpus.py merge <report.json> <scan.npz>...   # in GAME order
ENV
  EVALLAB_WORKERS  process pool size for `scan` (default 4)

The scan output is ~17 bytes per decision, i.e. ~600 MB for a 1M-game corpus --
small enough to pull off the boxes and merge locally, so the 7.7 GB corpus itself
never has to leave S3.
"""

import concurrent.futures as cf
import glob
import gzip
import hashlib
import json
import os
import sys

import numpy as np

import labenv  # noqa: F401  (pins encoder + engine env; must be first)
import stats as labstats  # noqa: E402  (reuse the SAME coarse key stats.py reports)
from poke_engine import State  # noqa: E402

TAGS = {"": 0, "rand": 1, "temp": 2, "eps": 3}


def _h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


def scan_shard(path):
    h, c, e, gn, go = [], [], [], [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "header" or "error" in row or not row.get("t"):
                continue
            turns = row["t"]
            gn.append(len(turns))
            go.append(row["outcome"])
            for t in turns:
                s = t["s"]
                h.append(_h64(s.encode()))
                c.append(_h64(repr(labstats.coarse(State.from_string(s))).encode()))
                e.append(TAGS.get(t["e"], 0))
    return (path,
            np.array(h, dtype=np.uint64), np.array(c, dtype=np.uint64),
            np.array(e, dtype=np.uint8), np.array(gn, dtype=np.int32),
            np.array(go, dtype=np.float32))


def scan(out, globs):
    paths = sorted(p for g in globs for p in glob.glob(g))
    assert paths, "no shards matched"
    workers = int(os.environ.get("EVALLAB_WORKERS", "4"))
    res = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(scan_shard, paths)):
            res[r[0]] = r[1:]
            print("scanned %d/%d %s (%d decisions)" % (i + 1, len(paths), r[0], len(r[1])), flush=True)
    # sorted filename order == global game order (shard names carry %07d start)
    parts = [res[p] for p in paths]
    np.savez(out,
             h=np.concatenate([p[0] for p in parts]),
             c=np.concatenate([p[1] for p in parts]),
             e=np.concatenate([p[2] for p in parts]),
             gn=np.concatenate([p[3] for p in parts]),
             go=np.concatenate([p[4] for p in parts]),
             files=np.array(paths))
    print("wrote %s" % out, flush=True)


def _groups(x, name, out, thresholds=(2, 5, 10, 30, 100, 300, 1000)):
    _, counts = np.unique(x, return_counts=True)
    n = int(x.size)
    out["distinct_" + name] = int(counts.size)
    out["frac_distinct_" + name] = counts.size / max(n, 1)
    out["mean_group_" + name] = float(n / max(counts.size, 1))
    out["max_group_" + name] = int(counts.max())
    out["positions_in_groups_ge_" + name] = {
        str(k): int(counts[counts >= k].sum()) for k in thresholds}
    out["frac_positions_in_groups_ge_" + name] = {
        str(k): float(counts[counts >= k].sum() / max(n, 1)) for k in thresholds}
    out["groups_ge_" + name] = {str(k): int((counts >= k).sum()) for k in thresholds}
    # size histogram, log-ish buckets
    edges = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 10 ** 9]
    hist = {}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (counts >= a) & (counts < b)
        hist["%d-%d" % (a, b - 1) if b < 10 ** 9 else "%d+" % a] = \
            {"groups": int(m.sum()), "positions": int(counts[m].sum())}
    out["group_size_hist_" + name] = hist
    return counts


def _curve(x, gidx, ngames, bucket, name, out):
    """New distinct keys discovered in each `bucket`-game window."""
    _, first = np.unique(x, return_index=True)
    g_first = gidx[first]
    nb = int(np.ceil(ngames / bucket))
    new = np.bincount(g_first // bucket, minlength=nb)[:nb]
    out["discovery_" + name] = {
        "bucket_games": bucket,
        "new_per_bucket": [int(v) for v in new],
        "cumulative": [int(v) for v in np.cumsum(new)],
    }


def merge(report, npzs, bucket=None):
    hs, cs, es, gns, gos = [], [], [], [], []
    for p in npzs:
        d = np.load(p, allow_pickle=False)
        hs.append(d["h"]); cs.append(d["c"]); es.append(d["e"])
        gns.append(d["gn"]); gos.append(d["go"])
        print("loaded %s: %d games %d decisions" % (p, d["gn"].size, d["h"].size), flush=True)
    h = np.concatenate(hs); del hs
    c = np.concatenate(cs); del cs
    e = np.concatenate(es); del es
    gn = np.concatenate(gns); go = np.concatenate(gos)
    ngames, n = int(gn.size), int(h.size)
    gidx = np.repeat(np.arange(ngames, dtype=np.int64), gn.astype(np.int64))
    assert gidx.size == n, (gidx.size, n)
    bucket = bucket or max(1, ngames // 100)

    out = {"games": ngames, "decisions": n, "mean_turns": n / max(ngames, 1),
           "win_rate": float(((go == 1.0).sum() + 0.5 * (go == 0.5).sum()) / max(ngames, 1)),
           "outcomes": {"s1_win": int((go == 1.0).sum()), "s1_loss": int((go == 0.0).sum()),
                        "draw": int((go == 0.5).sum())},
           "tags": {k: int((e == v).sum()) for k, v in TAGS.items()}}
    _curve(h, gidx, ngames, bucket, "exact", out)
    _groups(h, "exact", out)
    del h
    _curve(c, gidx, ngames, bucket, "coarse", out)
    _groups(c, "coarse", out)
    json.dump(out, open(report, "w"), indent=1)
    k = out["frac_positions_in_groups_ge_exact"]
    print("games=%d decisions=%d  distinct exact=%d (%.3f)  distinct coarse=%d"
          % (ngames, n, out["distinct_exact"], out["frac_distinct_exact"], out["distinct_coarse"]))
    print("exact groups: >=10 %.4f  >=30 %.4f  >=100 %.4f of positions"
          % (k["10"], k["30"], k["100"]))
    kc = out["frac_positions_in_groups_ge_coarse"]
    print("coarse groups: >=10 %.4f  >=30 %.4f  >=100 %.4f of positions"
          % (kc["10"], kc["30"], kc["100"]))
    print("wrote %s" % report)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        scan(sys.argv[2], sys.argv[3:])
    elif cmd == "merge":
        merge(sys.argv[2], sys.argv[3:],
              bucket=int(os.environ["MERGE_BUCKET"]) if os.environ.get("MERGE_BUCKET") else None)
    else:
        raise SystemExit("usage: analyze_corpus.py scan|merge ...")
