"""POWER-LAW FIT of the data-scaling curve produced by vt_scale.py.

Model:   Brier(N) = floor + A * N^-alpha     (floor = 0.013504, MEASURED)
Fit:     OLS of log(Brier - floor) on log(N), over ALL runs (not seed means),
         so the reported standard error on alpha includes seed-to-seed noise.

  python vt_scale_fit.py <dir-with-REPORT.scale.*.json>
"""
import glob
import json
import math
import os
import sys

import numpy as np

FLOOR = {"all": 0.013504, "early": 0.0173, "mid": 0.0130, "late": 0.0069}
REF_900K = 0.046959
COST_PER_M = 19.3                       # MEASURED $/million labelled positions
REGEN = {2_000_000: 130, 5_000_000: 320, 10_000_000: 650}   # REASONED


def ols(x, y):
    """slope, intercept, se(slope), R^2."""
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    res = y - (a + b * x)
    ss_res = (res ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 else float("nan")
    return b, a, se, 1 - ss_res / ss_tot


def load(d):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "REPORT.scale.n*_s*.json"))):
        r = json.load(open(f))
        runs.append(r)
    if not runs:
        raise SystemExit("no REPORT.scale.n*_s*.json in " + d)
    return runs


def fit_band(runs, band, key="brier"):
    N = np.array([r["n_train_rows"] for r in runs], float)
    if key == "brier":
        B = np.array([r["brier"][band] for r in runs], float)
    else:                                   # best point on the holdout curve
        B = np.array([r["curve_min_brier"] for r in runs], float)
    red = B - FLOOR[band]
    assert (red > 0).all(), "a run is at/below the noise floor: %s" % red
    b, a, se, r2 = ols(np.log(N), np.log(red))
    return {"alpha": -b, "se_alpha": se, "logA": a, "A": math.exp(a), "r2": r2,
            "n_points": len(N)}


def main():
    d = sys.argv[1]
    runs = load(d)
    sizes = sorted({r["n_train_rows"] for r in runs})

    # ---------------- per-size table ----------------------------------------
    tab = []
    for n in sizes:
        rs = [r for r in runs if r["n_train_rows"] == n]
        g = lambda k, sub: [x["brier"][k] if sub == "b" else x["auc"][k] for x in rs]  # noqa: E731
        row = {"n": n, "seeds": [r["seed"] for r in rs], "steps": rs[0]["steps"],
               "epochs": rs[0]["epochs_equiv"],
               "brier": {k: float(np.mean(g(k, "b"))) for k in FLOOR},
               "brier_per_seed": {r["seed"]: r["brier"]["all"] for r in rs},
               "spread": float(max(g("all", "b")) - min(g("all", "b"))),
               "auc": {k: float(np.mean(g(k, "a"))) for k in FLOOR},
               "curve_min": float(np.mean([r["curve_min_brier"] for r in rs])),
               "final_minus_min": float(np.mean([r["final_minus_min"] for r in rs])),
               "wall_s": float(np.mean([r["wall_s"] for r in rs])),
               "rows_per_s": float(np.mean([r["rows_per_s"] for r in rs]))}
        tab.append(row)

    fits = {b: fit_band(runs, b) for b in ("all", "early", "mid", "late")}
    # Same fit on the BEST point of each holdout curve. At small N the frozen
    # schedule overfits, so the end-of-schedule number is inflated there and the
    # end-of-schedule alpha is an UPPER bound; this one is the lower bound.
    fits["all_curvemin"] = fit_band(runs, "all", key="curve_min")
    # The decision-relevant fit: only the large-N end, where the frozen recipe
    # does NOT overfit, and which is the regime we extrapolate upward from.
    big = [r for r in runs if r["n_train_rows"] >= 200_000]
    fits["all_ge200k"] = fit_band(big, "all")
    f = fits["all_ge200k"]
    al, se, A = f["alpha"], f["se_alpha"], f["A"]

    # ---------------- local slope between consecutive sizes -----------------
    loc = []
    for i in range(1, len(tab)):
        n0, n1 = tab[i - 1]["n"], tab[i]["n"]
        r0 = tab[i - 1]["brier"]["all"] - FLOOR["all"]
        r1 = tab[i]["brier"]["all"] - FLOOR["all"]
        loc.append({"from": n0, "to": n1,
                    "alpha_local": -math.log(r1 / r0) / math.log(n1 / n0)})

    # ---------------- extrapolation + cost ----------------------------------
    # Anchored on OUR measured 900k point, so the projection starts exactly
    # where the net actually is rather than on the fitted intercept.
    b900 = [r for r in tab if r["n"] == max(sizes)][0]["brier"]["all"]
    red900 = b900 - FLOOR["all"]
    al_lo = fits["all_curvemin"]["alpha"]        # conservative (overfit removed)

    def proj(n, alpha):
        return FLOOR["all"] + red900 * (n / max(sizes)) ** (-alpha)

    ext = []
    for n in (2_000_000, 5_000_000, 10_000_000):
        p = proj(n, al)
        lo, hi = proj(n, al + se), proj(n, al - se)   # lo=faster decay
        add = n - 900_000
        ext.append({"n": n, "brier": p, "brier_lo": lo, "brier_hi": hi,
                    "brier_conservative": proj(n, al_lo),
                    "gain_vs_900k": b900 - p,
                    "gain_conservative": b900 - proj(n, al_lo),
                    "gain_lo": b900 - hi, "gain_hi": b900 - lo,
                    "extra_positions": add,
                    "cost_reuse": add / 1e6 * COST_PER_M,
                    "cost_regen": REGEN[n],
                    "pct_of_headroom": 100 * (b900 - p) / (b900 - FLOOR["all"])})

    out = {"table": tab, "fits": fits, "local_alpha": loc, "extrapolation": ext,
           "floor": FLOOR, "ref_900k": REF_900K,
           "model": "Brier(N) = %.6f + %.4f * N^-%.4f" % (FLOOR["all"], A, al),
           "n_to_halve_headroom": float(
               900_000 * (2.0 ** (1 / al))) if al > 0 else None}
    json.dump(out, open(os.path.join(d, "FIT.json"), "w"), indent=1)

    print("\n%-9s %-6s %-7s | %-9s %-9s %-9s %-9s | %-8s %-7s %-8s" %
          ("N", "steps", "epochs", "brier_all", "early", "mid", "late",
           "AUC_all", "spread", "fin-min"))
    for r in tab:
        print("%-9d %-6d %-7.1f | %-9.6f %-9.6f %-9.6f %-9.6f | %-8.5f %-7.5f %+.5f" %
              (r["n"], r["steps"], r["epochs"], r["brier"]["all"],
               r["brier"]["early"], r["brier"]["mid"], r["brier"]["late"],
               r["auc"]["all"], r["spread"], r["final_minus_min"]))
    print("\nfit (headline alpha = the N>=200k fit) " + out["model"])
    for b in fits:
        q = fits[b]
        print("  %-6s alpha = %.4f +/- %.4f   R2 = %.4f" %
              (b, q["alpha"], q["se_alpha"], q["r2"]))
    print("\nlocal alpha between consecutive sizes:")
    for l in loc:
        print("  %7d -> %7d : %.4f" % (l["from"], l["to"], l["alpha_local"]))
    print("\nextrapolation anchored on the MEASURED %d-row point (%.6f)"
          % (max(sizes), b900))
    print("%-11s %-9s %-19s %-10s %-11s %-10s %-10s %-7s" %
          ("N", "brier", "(alpha +/- 1se)", "gain", "gain_consrv", "cost_reuse",
           "cost_regen", "%headroom"))
    for e in ext:
        print("%-11d %-9.6f [%.6f,%.6f] %-10.6f %-11.6f $%-9.0f $%-9d %.1f%%" %
              (e["n"], e["brier"], e["brier_lo"], e["brier_hi"],
               e["gain_vs_900k"], e["gain_conservative"], e["cost_reuse"],
               e["cost_regen"], e["pct_of_headroom"]))
    print("\nwrote " + os.path.join(d, "FIT.json"))


if __name__ == "__main__":
    main()
