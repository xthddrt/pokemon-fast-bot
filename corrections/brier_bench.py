"""GENERALIZATION BENCH — Brier test of hammered nets on never-hammered states.

Sally's question (2026-08-15): beyond the exact ruled spots, is evaluation
getting better IN GENERAL as the ledger grows? Every mining round already
measures playout truth for every scanned state, and only flagged states get
hammered — so the non-ruled scanned states are a free, growing benchmark of
(state, truth) pairs no net was trained toward.

    foul-play/.venv/bin/python corrections/brier_bench.py mine2 mine3-b1 ... \
        [--nets s1,h1,h2,h3]

Per net: Brier = mean (eval - truth)^2 over the bench. Verdicts use the
PAIRED per-state difference (playout-label noise is common to all nets and
cancels in the pair), bootstrap 95% CI. Split: states from games that
produced a ruling (near-neighbor generalization) vs games with none
(strict generalization). Ruled states themselves are EXCLUDED, matched by
(round tag, game seed, decision).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEAF_PROF = os.path.join(ROOT, "poke-engine", "target", "release", "leaf_prof")
NETS_DIR = os.path.join(ROOT, "valuenet", "nets_v8b")
LEDGER = os.path.join(HERE, "value_ledger.jsonl")


def load_bench(tags):
    ruled = set()
    for line in open(LEDGER):
        e = json.loads(line)
        g = e.get("game", "")  # selfplay-<tag>-<seed>
        if g.startswith("selfplay-"):
            parts = g.split("-")
            ruled.add((parts[-1], e["decision"]))
    rows = []
    for tag in tags:
        work = os.path.join(HERE, "_mine_work", tag)
        i = 0
        while os.path.isfile(os.path.join(work, f"g{i}.json")):
            g = json.load(open(os.path.join(work, f"g{i}.json")))
            cand = json.load(open(os.path.join(work, f"cand{i}.json")))
            states = {r["t"]: r["s"] for r in g["states"]}
            has_ruling = any((str(g["seed"]), s["t"]) in ruled
                             for s in cand.get("scanned", []))
            for s in cand.get("scanned", []):
                if (str(g["seed"]), s["t"]) in ruled:
                    continue
                # truth: the scan's playout mean for this state (n=8/10 or 30)
                n = 30 if any(nm["t"] == s["t"] for nm in cand.get("near_misses", [])) else 10
                truth = s["p10"]
                for nm in cand.get("near_misses", []):
                    if nm["t"] == s["t"]:
                        truth = nm["p_confirm"]
                rows.append({"s": states[s["t"]], "truth": truth, "n": n,
                             "ruled_game": has_ruling,
                             "key": f"{tag}-g{g['seed']}-t{s['t']}"})
            i += 1
    return rows


def net_evals(bin_path, states_file, count):
    env = dict(os.environ, PE_NN_WEIGHTS=bin_path)
    out = subprocess.run([LEAF_PROF, "logits", states_file], env=env,
                        capture_output=True, text=True, check=True).stdout
    vals = [1 / (1 + math.exp(-float(l.split("\t")[1])))
            for l in out.splitlines() if "\t" in l]
    assert len(vals) == count
    return vals


def brier(evals, rows):
    return sum((e - r["truth"]) ** 2 for e, r in zip(evals, rows)) / len(rows)


def paired_ci(evals_a, evals_b, rows, n_boot=10000, seed=7):
    diffs = [(a - r["truth"]) ** 2 - (b - r["truth"]) ** 2
             for a, b, r in zip(evals_a, evals_b, rows)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return (sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--nets", default="s1,h3")
    a = ap.parse_args()
    rows = load_bench(a.tags)
    if not rows:
        raise SystemExit("no bench states found")
    sf = os.path.join(HERE, "_mine_work", "brier_states.txt")
    with open(sf, "w") as f:
        for r in rows:
            f.write(r["s"] + "\n")
    nets = a.nets.split(",")
    ev = {n: net_evals(os.path.join(NETS_DIR, f"v8b_{n}.bin"), sf, len(rows))
          for n in nets}
    print(f"bench: {len(rows)} never-hammered states from {a.tags} "
          f"({sum(r['ruled_game'] for r in rows)} in ruled games, "
          f"{sum(not r['ruled_game'] for r in rows)} in clean games)")
    for n in nets:
        print(f"  Brier {n}: {brier(ev[n], rows):.4f}")
    base = nets[0]
    for n in nets[1:]:
        for label, sub in (("ALL", rows),
                           ("ruled-game", [r for r in rows if r["ruled_game"]]),
                           ("clean-game", [r for r in rows if not r["ruled_game"]])):
            if len(sub) < 5:
                continue
            ia = [ev[base][i] for i, r in enumerate(rows) if r in sub]
            ib = [ev[n][i] for i, r in enumerate(rows) if r in sub]
            d, lo, hi = paired_ci(ia, ib, sub)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"  {base} vs {n} [{label}, n={len(sub)}]: ΔBrier "
                  f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"({'better' if d > 0 else 'worse'} for {n}, {sig})")


if __name__ == "__main__":
    main()
