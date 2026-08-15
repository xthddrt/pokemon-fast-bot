"""How much move quality does a cheaper search budget actually cost?

The corpus is only worth what its trajectories and its labels are worth. Before
regenerating at a lower iteration count, measure the thing that matters: on
positions whose ground truth we already own (the 5M-iteration oracle), how often
does a search at budget B pick the oracle's best move, and how much win
probability does it give up when it does not?

Every budget is measured on the SAME positions with the SAME seeds, so the
comparison is paired and the differences are not sampling noise between sets.

USAGE  python budget_check.py <oracle.jsonl> [budgets...] [--seeds N] [--n N]
"""

import concurrent.futures as cf
import json
import os
import sys

import numpy as np

import labenv  # noqa: F401
from poke_engine import State, monte_carlo_tree_search  # noqa: E402


def one(task):
    s, iters, seed = task
    r = monte_carlo_tree_search(State.from_string(s), 0, iters, 1, seed)
    live = [m for m in r.side_one if m.visits > 0]
    if not live:
        return None
    # generate.py plays the visit argmax, so that is what is scored here
    return max(live, key=lambda m: m.visits).move_choice


def main():
    oracle_path = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    budgets = [int(a) for a in args] or [2000, 5000, 8000, 25000]
    seeds = int(os.environ.get("BC_SEEDS", "2"))
    nmax = int(os.environ.get("BC_N", "0"))
    rows = [json.loads(l) for l in open(oracle_path)]
    rows = [r for r in rows if len(r["q"]) >= 3 and r.get("best")]
    if nmax:
        rows = rows[:nmax]
    print("budget check on %d oracle positions x %d seeds (oracle = %d iterations)"
          % (len(rows), seeds, rows[0]["it"]), flush=True)

    tasks, meta = [], []
    for b in budgets:
        for i, r in enumerate(rows):
            for sd in range(seeds):
                tasks.append((r["s"], b, 777 + 13 * sd + i))
                meta.append((b, i))
    workers = int(os.environ.get("EVALLAB_WORKERS", "4"))
    out = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for (b, i), pick in zip(meta, ex.map(one, tasks, chunksize=8)):
            out.setdefault(b, []).append((i, pick))

    print("\n| budget (iters) | n | top-1 vs oracle | top-3 | regret (win prob) | rel. to 25k |")
    print("|---|---|---|---|---|---|")
    ref = None
    res = {}
    for b in budgets:
        t1, t3, reg = [], [], []
        for i, pick in out[b]:
            r = rows[i]
            q = r["q"]
            best = r["best"]
            top3 = set(sorted(q, key=lambda k: -q[k])[:3])
            if pick is None or pick not in q:
                # an arm the oracle never scored: treat as a miss with the worst regret
                t1.append(0.0)
                t3.append(0.0)
                reg.append(max(q.values()) - min(q.values()))
                continue
            t1.append(1.0 if pick == best else 0.0)
            t3.append(1.0 if pick in top3 else 0.0)
            reg.append(max(q.values()) - q[pick])
        m1, m3, mr = np.mean(t1), np.mean(t3), np.mean(reg)
        res[b] = {"top1": float(m1), "top3": float(m3), "regret": float(mr), "n": len(t1),
                  "top1_se": float(np.std(t1) / np.sqrt(len(t1)))}
        if b == 25000:
            ref = m1
    for b in budgets:
        d = res[b]
        rel = ("%.2f" % (d["top1"] / ref)) if ref else "-"
        print("| %d | %d | %.3f ± %.3f | %.3f | %.4f | %s |"
              % (b, d["n"], d["top1"], d["top1_se"], d["top3"], d["regret"], rel))
    print("\nJSON " + json.dumps(res))


if __name__ == "__main__":
    main()
