"""DATA-SCALING CURVE -- vt_n.py's 128/256 job, run at nested training-set sizes.

The ONLY change from vt_n.train is that the training pool is restricted to a
NESTED, PAIR-GROUPED prefix of the 900,000 non-holdout plc1 rows:

  * whole team pairs are drawn in ONE fixed permutation (SUBSET_SEED, which is
    deliberately NOT the model seed, so both model seeds see identical data and
    the nesting is a property of the data, not of the run);
  * subset(N) = the shortest prefix of that permutation reaching N rows, so
    subset(50k) is a strict subset of subset(100k) is a strict subset of ...
    and a larger subset can never lack a pair a smaller one has.

Model, recipe, metrics and holdout are vt_n.py's, untouched.

  python vt_scale.py train <work> --n-train 200000 --seed 0 --steps 6000 --threads 16
  python vt_scale.py collect <work>
"""
import argparse
import json
import os
import sys

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402
import vt_n as N  # noqa: E402

SUBSET_SEED = 424242


def nested_pool(m, pool, n_target):
    """Pair-grouped nested prefix of `pool` with >= n_target rows."""
    _, inv = np.unique(m["pair"], return_inverse=True)
    pinv = inv[pool]
    upair = np.unique(pinv)
    perm = np.random.default_rng(SUBSET_SEED).permutation(len(upair))
    ordered = upair[perm]
    cnt = np.bincount(pinv, minlength=inv.max() + 1)[ordered]
    if n_target >= cnt.sum():
        return pool, len(ordered)
    k = int(np.searchsorted(np.cumsum(cnt), n_target)) + 1
    keep = np.zeros(inv.max() + 1, bool)
    keep[ordered[:k]] = True
    return pool[keep[pinv]], k


def train(work, n_train, seed, steps, threads, eval_every):
    """vt_n.train with the pool swapped for the nested subset. Patching
    np.flatnonzero is how we reuse that function byte-for-byte rather than
    forking a second copy of the training loop that could silently drift."""
    d = os.path.join(work, "enc_plc1")
    m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    n = len(m["label_p"])
    mask = np.ones(n, bool)
    mask[hold] = False
    full = np.flatnonzero(mask)
    sub, n_pairs = nested_pool(m, full, n_train)
    print("subset: target %d -> %d rows / %d pairs (of %d rows) seed %d steps %d"
          % (n_train, len(sub), n_pairs, len(full), seed, steps), flush=True)

    real = np.flatnonzero

    def patched(a):
        r = real(a)
        return sub if (r.shape == full.shape and np.array_equal(r, full)) else r

    np.flatnonzero = patched
    try:
        res = N.train(work, seed, steps, threads, eval_every, cap=N.CAP_N)
    finally:
        np.flatnonzero = real
    assert res["n_train_pool"] == len(sub), (res["n_train_pool"], len(sub))

    res["n_train_target"] = int(n_train)
    res["n_train_rows"] = int(len(sub))
    res["n_train_pairs"] = int(n_pairs)
    res["subset_seed"] = SUBSET_SEED
    res["epochs_equiv"] = round(steps * res["batch"] / len(sub), 2)
    res["curve_min_brier"] = min(c["holdout_brier"] for c in res["curve"])
    res["final_minus_min"] = res["brier"]["all"] - res["curve_min_brier"]
    tag = "n%d_s%d" % (n_train, seed)
    for src, dst in (("REPORT.n.s%d.json" % seed, "REPORT.scale.%s.json" % tag),
                     ("holdout_pred_s%d.npy" % seed, "holdout_pred_%s.npy" % tag),
                     ("ckpt_n_s%d.pt" % seed, "ckpt_%s.pt" % tag)):
        p = os.path.join(work, src)
        if os.path.exists(p) and not src.endswith(".json"):
            os.rename(p, os.path.join(work, dst))
        elif os.path.exists(p):
            os.remove(p)
    json.dump(res, open(os.path.join(work, "REPORT.scale.%s.json" % tag), "w"),
              indent=1)
    print("DONE %s brier_all %.6f (curve min %.6f)"
          % (tag, res["brier"]["all"], res["curve_min_brier"]), flush=True)
    return res


def collect(work):
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(work, "REPORT.scale.n*_s*.json"))):
        r = json.load(open(f))
        rows.append({k: r[k] for k in
                     ("n_train_target", "n_train_rows", "n_train_pairs", "seed",
                      "steps", "epochs_equiv", "brier", "auc", "curve_min_brier",
                      "final_minus_min", "wall_s", "rows_per_s", "n_params",
                      "paired_vs_q", "curve")})
    out = {"runs": rows, "noise_floor": N.NOISE_FLOOR, "cap": list(N.CAP_N),
           "subset_seed": SUBSET_SEED}
    json.dump(out, open(os.path.join(work, "REPORT.scale.collect.json"), "w"),
              indent=1)
    print(json.dumps([{k: r[k] for k in ("n_train_rows", "seed", "steps")} |
                      {"brier_all": r["brier"]["all"]} for r in rows], indent=1),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "collect"])
    ap.add_argument("work")
    ap.add_argument("--n-train", type=int, default=900000, dest="nt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000, dest="ee")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "train":
        train(a.work, a.nt, a.seed, a.steps, a.threads, a.ee)
    else:
        collect(a.work)


if __name__ == "__main__":
    main()
