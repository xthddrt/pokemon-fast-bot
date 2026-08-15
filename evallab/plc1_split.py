"""plc1 MODEL-SELECTION SPLIT -- carve a ~100k holdout out of plc1 itself.

WHY: plc1 is 100% shard r4 and plc1e is 100% r4x (a different generator, 6.9
sigma apart in decidedness). Model selection must therefore NOT touch plc1e --
plc1e is read exactly once, at the very end. This carves the selection holdout
out of plc1, which is the same distribution as training.

UNIT OF THE SPLIT: the team pair (field `pair`). Every game lives under exactly
one pair (asserted below), so splitting by pair implies splitting by game and
guarantees zero pair-hash overlap -- the stronger of the two requirements.

OUTPUT (index only, never a copy of the data):
  data/plc1/holdout_i.npy   sorted int32 array of `i` values in the holdout
  data/plc1/split.json      seed, counts, and band/ply distributions of both

  python plc1_split.py [--target 100000] [--seed 20260814]
"""
import argparse
import glob
import json
import os

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
PLC1 = os.path.join(LAB, "data/plc1")
BANDS = ("early", "mid", "late")


def scan():
    """Read the 39 label files once. Returns i, pair-code, game-code, band-code, ply."""
    i, pair, game, band, ply = [], [], [], [], []
    pmap, gmap = {}, {}
    for f in sorted(glob.glob(os.path.join(PLC1, "labels_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                p, g = r["pair"], r["g"]
                pair.append(pmap.setdefault(p, len(pmap)))
                game.append(gmap.setdefault(g, len(gmap)))
                i.append(r["i"])
                band.append(BANDS.index(r["band"]))
                ply.append(r["ply"])
    return (np.array(i, np.int64), np.array(pair, np.int64), np.array(game, np.int64),
            np.array(band, np.int8), np.array(ply, np.int32))


def dist(band, ply, m):
    b, p = band[m], ply[m]
    return {"n": int(m.sum()),
            "band_frac": {k: round(float((b == j).mean()), 5) for j, k in enumerate(BANDS)},
            "ply_mean": round(float(p.mean()), 3),
            "ply_q": [int(x) for x in np.percentile(p, [5, 25, 50, 75, 95])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260814)
    a = ap.parse_args()

    i, pair, game, band, ply = scan()
    n = len(i)
    assert n == len(np.unique(i)) == 1000000, "expected 1,000,000 distinct i, got %d" % n

    # each game must sit under exactly one pair, else pair-grouping would not
    # imply game-grouping
    order = np.argsort(game, kind="stable")
    gs, ps = game[order], pair[order]
    cut = np.flatnonzero(np.diff(gs)) + 1
    assert all(len(np.unique(x)) == 1 for x in np.split(ps, cut)), \
        "a game spans more than one pair -- pair-grouping is not sufficient"

    # ---- choose whole pairs, in a fixed shuffled order, until target rows ----
    codes, counts = np.unique(pair, return_counts=True)
    perm = np.random.default_rng(a.seed).permutation(len(codes))
    k = int(np.searchsorted(np.cumsum(counts[perm]), a.target)) + 1
    hold_pairs = codes[perm[:k]]
    m_h = np.isin(pair, hold_pairs)
    m_t = ~m_h

    assert not (set(np.unique(pair[m_h])) & set(np.unique(pair[m_t]))), "pair overlap"
    assert not (set(np.unique(game[m_h])) & set(np.unique(game[m_t]))), "game overlap"

    hi = np.sort(i[m_h]).astype(np.int32)
    np.save(os.path.join(PLC1, "holdout_i.npy"), hi)
    rep = {"seed": a.seed, "target": a.target, "unit": "pair",
           "n_total": int(n), "n_pairs": int(len(codes)), "n_games": int(game.max() + 1),
           "n_holdout_pairs": int(len(hold_pairs)),
           "n_holdout_games": int(len(np.unique(game[m_h]))),
           "pair_overlap": 0, "game_overlap": 0,
           "holdout": dist(band, ply, m_h), "train": dist(band, ply, m_t),
           "holdout_i_file": "holdout_i.npy",
           "note": "train = every i NOT in holdout_i.npy; plc1e untouched"}
    json.dump(rep, open(os.path.join(PLC1, "split.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
