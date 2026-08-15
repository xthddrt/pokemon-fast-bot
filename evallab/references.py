"""Reference rows for the sibling metric -- the brackets every trained cell has
to be read between.

  RANDOM          uniform arm. The floor. A net below this is not steering.
  CORPUS SEARCH   the 25k-iteration search that generated the corpus, scored
                  against the 5M-iteration oracle on the SAME decisions. This
                  is the ceiling a value net could aspire to at this budget --
                  and the honest measure of how much better the oracle is than
                  the training signal.
  v6 INCUMBENT    run `evaluate.py <net_v6nopol.pt> ...` -- the ladder net,
                  5M games of real randbats, scored on the same positions.

USAGE  python references.py <oracle.jsonl> <shard_dir>
"""

import glob
import gzip
import json
import sys

import numpy as np

import labenv  # noqa: F401
from evaluate import spearman  # noqa: E402


def corpus_q_index(shard_dir):
    """(pair, game, ply) -> the generator's own 25k-iteration per-arm values."""
    ix = {}
    paths = sorted(glob.glob(shard_dir + "/*/shard_*.jsonl.gz")) or \
        sorted(glob.glob(shard_dir + "/shard_*.jsonl.gz"))
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                for pi, t in enumerate(row["t"]):
                    ix[(row["pair"], row["g"], pi)] = t["q"]
    return ix


def main():
    oracle_path, shard_dir = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(oracle_path)]
    ix = corpus_q_index(shard_dir)
    per = {}
    for r in rows:
        oq = r["q"]
        arms = list(oq)
        if len(arms) < 3:
            continue
        qs = np.array([oq[a] for a in arms])
        best = arms[int(np.argmax(qs))]
        top3 = {arms[int(i)] for i in np.argsort(-qs)[:3]}
        d = per.setdefault(r["pair"], {"n": 0, "rand": [], "s_top1": [], "s_top3": [],
                                       "s_rho": [], "s_reg": [], "gap": [],
                                       "inv_arms": [], "miss": 0})
        d["n"] += 1
        d["inv_arms"].append(1.0 / len(arms))
        d["rand"].append(float(qs.max() - qs.mean()))
        d["gap"].append(float(qs.max() - qs.min()))
        cq = ix.get((r["pair"], r["g"], r["ply"]))
        if not cq:
            d["miss"] += 1
            continue
        common = [a for a in arms if a in cq]
        if len(common) < 3:
            d["miss"] += 1
            continue
        cv = np.array([cq[a] for a in common])
        ov = np.array([oq[a] for a in common])
        pick = common[int(np.argmax(cv))]
        d["s_top1"].append(1.0 if pick == best else 0.0)
        d["s_top3"].append(1.0 if pick in top3 else 0.0)
        d["s_rho"].append(spearman(cv, ov))
        d["s_reg"].append(float(qs.max() - oq[pick]))

    print("| reference | pair | n | top1 | top3 | spearman | regret | oracle gap |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(per):
        d = per[k]
        print("| RANDOM arm | %s | %d | %.3f | - | 0.000 | %.4f | %.4f |"
              % (k, d["n"], float(np.mean(d["inv_arms"])), np.mean(d["rand"]), np.mean(d["gap"])))
        if d["s_top1"]:
            print("| CORPUS SEARCH (25k iters) | %s | %d | %.3f | %.3f | %.3f | %.4f | %.4f |"
                  % (k, len(d["s_top1"]), np.mean(d["s_top1"]), np.mean(d["s_top3"]),
                     np.nanmean(d["s_rho"]), np.mean(d["s_reg"]), np.mean(d["gap"])))
        if d["miss"]:
            print("|   (unjoined oracle rows: %d) | %s | | | | | | |" % (d["miss"], k))


if __name__ == "__main__":
    main()
