"""Paired per-decision comparison of two checkpoints on the sibling metric.

Marginal standard errors overstate the uncertainty of a cell-vs-cell comparison
here, because every cell is scored on the SAME oracle decisions. This does the
comparison the way it should be done: per decision, pooled over seeds, with an
exact-binomial (McNemar) test on the discordant top-1 pairs and a paired t on
regret.

USAGE  python paired.py <oracle.jsonl> <ckptA...> -- <ckptB...>
"""

import json
import sys

import numpy as np
import torch

import labenv  # noqa: F401
from dataset import encode_full, successor  # noqa: E402
from encoder import Vocab  # noqa: E402
from evaluate import batchify, load_net  # noqa: E402
from poke_engine import State  # noqa: E402


def per_decision(ckpts, rows, vocab):
    """[(top1, regret)] per (checkpoint, decision), aligned across checkpoints."""
    # encode every decision's siblings ONCE, then score with each net
    enc = []
    for r in rows:
        st = State.from_string(r["s"])
        bstar = r["best2"] or max(r["n2"], key=lambda k: r["n2"][k])
        arms, qs, feats = [], [], []
        for arm, q in r["q"].items():
            s2 = successor(st, arm, bstar)
            if s2 is None:
                continue
            arms.append(arm)
            qs.append(float(q))
            feats.append(encode_full(s2, vocab))
        enc.append((arms, np.asarray(qs), feats) if len(arms) >= 3 else None)
    out = {}
    for c in ckpts:
        net, _, _ = load_net(c)
        t1, reg = [], []
        for e in enc:
            if e is None:
                continue
            arms, qs, feats = e
            with torch.no_grad():
                v = torch.sigmoid(net(batchify(feats))).numpy()
            pick = arms[int(np.argmax(v))]
            t1.append(1.0 if pick == arms[int(np.argmax(qs))] else 0.0)
            reg.append(float(qs.max() - qs[arms.index(pick)]))
        out[c] = (np.asarray(t1), np.asarray(reg))
    return out


def main():
    oracle_path = sys.argv[1]
    rest = sys.argv[2:]
    i = rest.index("--")
    A, B = rest[:i], rest[i + 1:]
    rows = [json.loads(l) for l in open(oracle_path)]
    rows = [r for r in rows if r["pair"] == "A"]
    vocab = Vocab(frozen=True)
    res = per_decision(A + B, rows, vocab)
    ta = np.concatenate([res[c][0] for c in A])
    tb = np.concatenate([res[c][0] for c in B])
    ra = np.concatenate([res[c][1] for c in A])
    rb = np.concatenate([res[c][1] for c in B])
    n01 = int(((ta == 0) & (tb == 1)).sum())
    n10 = int(((ta == 1) & (tb == 0)).sum())
    # exact binomial on the discordant pairs (McNemar), two-sided
    from math import comb
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    d = rb - ra
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
    print("A = %s" % ", ".join(A))
    print("B = %s" % ", ".join(B))
    print("paired decisions per seed-set: %d (%d A-rows, %d B-rows)" % (len(rows), len(ta), len(tb)))
    print("top-1:  A %.4f   B %.4f   diff %+.4f" % (ta.mean(), tb.mean(), tb.mean() - ta.mean()))
    print("  discordant pairs: B-only-right %d, A-only-right %d -> McNemar exact p = %.4f"
          % (n01, n10, p))
    print("regret: A %.4f   B %.4f   diff %+.4f   paired t = %+.2f (n=%d)"
          % (ra.mean(), rb.mean(), rb.mean() - ra.mean(), tstat, len(d)))


if __name__ == "__main__":
    main()
