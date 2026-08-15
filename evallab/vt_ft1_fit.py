"""STAGE 2 ANALYSIS -- aggregate the cells, fit the learning curve, price labels.

Reads the per-cell REPORT.cell.*.json + hpred_*.npy that vt_ft1.py wrote and
answers the two decision questions:

  1. reducible error E(N) = holdout Brier(N) - 0.013504 (the label-noise floor)
     fitted as E = A * N^-alpha on log-log by OLS over EVERY seed of EVERY
     learning-curve cell.  alpha's standard error is the regression's own,
     clustered by cell (the 3 seeds of one cell are not independent draws of
     the curve, so the cluster-robust se is the honest one).  Extrapolated to
     2M and 5M labels and priced at $19 per million.
  2. pretrained-then-fine-tuned vs from-scratch at 900k: mean gap, per-seed sd,
     the PAIRED z on the 100,000-row holdout, and -- the useful form -- the
     number of extra labels from-scratch would have to buy to catch up.

  python vt_ft1_fit.py <dir-with-REPORT.cell.*.json>
"""
import glob
import json
import math
import os
import sys

import numpy as np

FLOOR = {"all": 0.013504, "early": 0.0173, "mid": 0.0130, "late": 0.0069}
BANDS = ("early", "mid", "late")
PRICE_PER_M = 19.0
SIZES = (50000, 100000, 200000, 500000, 900000)


def load(d):
    cells = {}
    for f in sorted(glob.glob(os.path.join(d, "REPORT.cell.*.json"))):
        r = json.load(open(f))
        cells[r["job"]] = r
    return cells


def key(r):
    return (r["kind"], r["size"], r["src"])


def agg(cells, tag="best_val"):
    """(kind, size, src) -> per-seed vectors of holdout Brier, all + bands."""
    out = {}
    for r in cells.values():
        out.setdefault(key(r), {"seeds": [], "brier": {k: [] for k in ("all",) + BANDS},
                                "steps": [], "passes": [], "early_stop": [],
                                "best_step": [], "val": []})
        o = out[key(r)]
        o["seeds"].append(r["seed"])
        for k in ("all",) + BANDS:
            o["brier"][k].append(r["holdout_brier"][tag][k])
        o["steps"].append(r["steps_run"])
        o["passes"].append(round(r["passes"], 1))
        o["early_stop"].append(r["early_stop"]["step"] if r["early_stop"] else None)
        o["best_step"].append(r["best"]["step"])
        o["val"].append(r["best"]["val"])
    return out


def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def fit_powerlaw(pts):
    """pts = [(N, E, cell_id)]. OLS of log E on log N, cluster-robust se."""
    N = np.array([p[0] for p in pts], float)
    E = np.array([p[1] for p in pts], float)
    c = np.array([p[2] for p in pts])
    x = np.log(N)
    y = np.log(E)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    # cluster-robust (by cell): the 3 seeds inside a cell share the same subset
    meat = np.zeros((2, 2))
    for cid in np.unique(c):
        m = c == cid
        u = X[m].T @ res[m]
        meat += np.outer(u, u)
    G = len(np.unique(c))
    dof = max(1, G - 2)
    V = XtXi @ meat @ XtXi * (G / dof)
    se = np.sqrt(np.diag(V))
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"logA": float(beta[0]), "alpha": float(-beta[1]),
            "se_logA": float(se[0]), "se_alpha": float(se[1]),
            "A": float(math.exp(beta[0])), "n_points": len(pts), "n_clusters": int(G),
            "r2": 1 - ss_res / ss_tot, "resid_rms": float(np.sqrt((res ** 2).mean()))}


def predict_E(fit, N):
    return fit["A"] * N ** (-fit["alpha"])


def band(fit, N):
    """+-1 se on alpha, propagated through the extrapolation."""
    lo = math.exp(fit["logA"] - fit["se_logA"]) * N ** (-(fit["alpha"] + fit["se_alpha"]))
    hi = math.exp(fit["logA"] + fit["se_logA"]) * N ** (-(fit["alpha"] - fit["se_alpha"]))
    return lo, hi


def paired(pa, pb, y):
    d = (pa - y) ** 2 - (pb - y) ** 2
    return {"delta": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
            "z": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))}


def main(d):
    cells = load(d)
    meta = json.load(open(os.path.join(d, "REPORT.split.json")))
    out = {"n_cells": len(cells), "split": meta["cells"], "noise_floor": FLOOR,
           "price_per_million_labels_usd": PRICE_PER_M}

    for tag in ("best_val", "end_of_schedule"):
        a = agg(cells, tag)
        tab = {}
        for k, o in sorted(a.items()):
            name = "%s_%d_%s" % k
            tab[name] = {"n_seeds": len(o["seeds"]),
                         "steps": o["steps"], "passes": o["passes"],
                         "best_step": o["best_step"], "early_stop": o["early_stop"]}
            for b in ("all",) + BANDS:
                m, s = msd(o["brier"][b])
                tab[name][b] = {"mean": m, "sd": s, "per_seed": o["brier"][b]}
        out["cells_" + tag] = tab

    # ---------------- Q1: the learning curve -------------------------------
    a = agg(cells, "best_val")
    pts, curve = [], {}
    for N in SIZES:
        o = a[("ft", N, "s1")]
        curve[str(N)] = {"brier_mean": msd(o["brier"]["all"])[0],
                         "brier_sd": msd(o["brier"]["all"])[1],
                         "reducible_mean": msd(o["brier"]["all"])[0] - FLOOR["all"],
                         "per_seed": o["brier"]["all"]}
        for v in o["brier"]["all"]:
            pts.append((float(N), v - FLOOR["all"], str(N)))
    fit = fit_powerlaw(pts)
    out["Q1_curve"] = curve
    out["Q1_fit"] = fit
    proj = {}
    e900 = predict_E(fit, 900000.0)
    for N in (900000, 2000000, 5000000, 10000000):
        e = predict_E(fit, float(N))
        lo, hi = band(fit, float(N))
        proj[str(N)] = {"reducible": e, "brier": e + FLOOR["all"],
                        "brier_lo": lo + FLOOR["all"], "brier_hi": hi + FLOOR["all"],
                        "gain_vs_900k": e900 - e,
                        "extra_labels_M": (N - 900000) / 1e6,
                        "extra_cost_usd": (N - 900000) / 1e6 * PRICE_PER_M}
        if N > 900000:
            proj[str(N)]["usd_per_0.001_brier"] = (
                proj[str(N)]["extra_cost_usd"] / max(1e-12, (e900 - e) / 1e-3))
    out["Q1_projection"] = proj
    # measured check of the fit at the two ends
    out["Q1_fit_check"] = {str(N): {"measured": curve[str(N)]["reducible_mean"],
                                    "fitted": predict_E(fit, float(N))} for N in SIZES}

    # ---------------- Q2: what pretraining bought --------------------------
    ft9 = a[("ft", 900000, "s1")]["brier"]
    sc9 = a[("scratch", 900000, "none")]["brier"]
    q2 = {"finetuned_900k": {b: msd(ft9[b]) for b in ("all",) + BANDS},
          "scratch_900k": {b: msd(sc9[b]) for b in ("all",) + BANDS}}
    q2["gap_all"] = msd(sc9["all"])[0] - msd(ft9["all"])[0]
    q2["gap_rel_reducible"] = q2["gap_all"] / (msd(sc9["all"])[0] - FLOOR["all"])
    for b in BANDS:
        q2["gap_" + b] = msd(sc9[b])[0] - msd(ft9[b])[0]
    if ("scratchhi", 900000, "none") in a:
        q2["scratch_hi_lr_probe"] = {b: msd(a[("scratchhi", 900000, "none")]["brier"][b])
                                     for b in ("all",) + BANDS}
    if ("ft", 900000, "s0") in a:
        q2["finetuned_900k_from_s0"] = {b: msd(a[("ft", 900000, "s0")]["brier"][b])
                                        for b in ("all",) + BANDS}
    # data-equivalent value of pretraining: N on the fine-tune curve whose
    # reducible error equals from-scratch's at 900k
    e_sc = msd(sc9["all"])[0] - FLOOR["all"]
    if e_sc > 0 and fit["alpha"] > 0:
        n_eq = math.exp((fit["logA"] - math.log(e_sc)) / fit["alpha"])
        q2["scratch_equivalent_finetune_labels"] = n_eq
        q2["pretrain_worth_labels"] = 900000 - n_eq
        q2["pretrain_worth_usd"] = (900000 - n_eq) / 1e6 * PRICE_PER_M

    # paired test on the 100k holdout, seed 0 vs seed 0 (same holdout rows)
    def hp(job, tag="best_val"):
        p = os.path.join(d, "hpred_%s_%s.npy" % (job.replace(":", "_"), tag))
        return np.load(p).astype(np.float64) if os.path.exists(p) else None
    yh = None
    mp = os.path.join(d, "enc_plc1", "meta.npz")
    if os.path.exists(mp):
        z = np.load(mp, allow_pickle=False)
        h = np.load(os.path.join(d, "enc_plc1", "holdout_i.npy")).astype(np.int64)
        yh = z["label_p"][h]
    if yh is not None:
        pf = [hp("ft:900000:%d:s1" % s) for s in range(3)]
        ps = [hp("scratch:900000:%d:none" % s) for s in range(3)]
        if all(x is not None for x in pf + ps):
            q2["paired_scratch_minus_ft_seedmean"] = paired(
                np.mean(ps, 0), np.mean(pf, 0), yh)
    out["Q2_pretrain_value"] = q2

    json.dump(out, open(os.path.join(d, "REPORT.ft1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
