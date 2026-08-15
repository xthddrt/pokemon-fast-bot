"""ENCODER VALUE TEST -- aggregate every run into the report's tables."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

BANDS = ("early", "mid", "late")
BARS = {"current search": (0.1030, 0.1074, 0.0700),
        "recalibrated search": (0.0495, 0.0749, 0.0605),
        "constant": (0.0502, 0.0994, 0.1535),
        "floor (label noise)": (0.0151, 0.0098, 0.0054)}
ARM = {"old": "A BASELINE (shipped encoder + pooled net)",
       "enc2": "B NEW (enc2, 1413 cols + flat net)"}


def load(pats):
    out = []
    for p in pats:
        for f in sorted(glob.glob(p)):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            r["_f"] = os.path.basename(f)
            out.append(r)
    return out


def agg(rows, key=lambda r: (r["arm"], tuple(r["cap"]))):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r)
    return d


def ms(vals):
    v = np.asarray(vals, float)
    return v.mean(), (v.max() - v.min()), v.std(ddof=1) if len(v) > 1 else 0.0


def band_row(rs, band):
    return ms([r["test"][band] for r in rs])


def main():
    grid = load([sys.argv[1] + "/g_*.json"])
    dscl = load([sys.argv[1] + "/d_*.json"])
    wse = load([sys.argv[1] + "/w_*.json"])
    lmse = load([sys.argv[1] + "/l_*.json"])
    sweep = load(["/tmp/vt/c_*.json"])
    triv = json.load(open("/tmp/vt/trivial.json"))

    print("### capacity sweep (seed 0, wd=1e-2, early stop on val Brier)\n")
    print("| arm | capacity | params | best epoch | val Brier | test Brier all | early | mid | late |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(sweep, key=lambda r: (r["arm"], r["cap"][0])):
        t = r["test"]
        print("| %s | %s | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f |"
              % (r["arm"], "x".join(map(str, r["cap"])), r["params"], r["best_epoch"],
                 r["val_brier"], t["all"], t["early"], t["mid"], t["late"]))

    print("\n### multi-seed grid (3 seeds; mean +- spread=max-min)\n")
    print("| arm | capacity | val Brier | test all | early | mid | late |")
    print("|---|---|---:|---:|---:|---:|---:|")
    G = agg(grid)
    best = {}
    for k in sorted(G, key=lambda k: (k[0], k[1][0])):
        rs = G[k]
        vb = ms([r["val_brier"] for r in rs])
        cells = [band_row(rs, b) for b in ("all",) + BANDS]
        print("| %s | %s | %.5f | %s |" % (k[0], "x".join(map(str, k[1])), vb[0],
              " | ".join("%.5f +-%.5f" % (m, s) for m, s, _sd in cells)))
        if k[0] not in best or vb[0] < best[k[0]][0]:
            best[k[0]] = (vb[0], k, rs)
    json.dump({a: {"cap": list(v[1][1]), "val": v[0]} for a, v in best.items()},
              open("/tmp/vt/best.json", "w"))

    print("\n### HEADLINE per-band table\n")
    print("| band | current search | recal. search | constant | HP-diff linear | "
          "ARM A baseline | ARM B new | floor |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, b in enumerate(BANDS):
        a_m, a_s, _sd = band_row(best["old"][2], b)
        b_m, b_s, _sd = band_row(best["enc2"][2], b)
        print("| %s | %.4f | %.4f | %.4f | %.4f | **%.4f** +-%.4f | %.4f +-%.4f | %.4f |"
              % (b, BARS["current search"][i], BARS["recalibrated search"][i],
                 BARS["constant"][i], triv["hp_diff_logistic"][b],
                 a_m, a_s, b_m, b_s, BARS["floor (label noise)"][i]))

    print("\n### data-scaling (train games at 25/50/100%, 3 seeds)\n")
    print("| arm | frac | n train pos | test all | early | mid | late |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    D = agg(dscl, key=lambda r: (r["arm"], r["frac"]))
    for a in ("old", "enc2"):
        for f in (0.25, 0.5, 1.0):
            rs = D.get((a, f)) or (best[a][2] if f == 1.0 else None)
            if not rs:
                continue
            cells = [band_row(rs, x) for x in ("all",) + BANDS]
            print("| %s | %.2f | %d | %s |" % (a, f, rs[0]["n_train"],
                  " | ".join("%.5f +-%.5f" % (m, s) for m, s, _sd in cells)))

    print("\n### precision weighting 1/(se^2+s0^2) and MSE loss\n")
    print("| arm | variant | test all | early | mid | late |")
    print("|---|---|---:|---:|---:|---:|")
    for a in ("old", "enc2"):
        for nm, rs in (("BCE (default)", best[a][2]),
                       ("BCE + 1/se^2 weighting", [r for r in wse if r["arm"] == a]),
                       ("MSE", [r for r in lmse if r["arm"] == a])):
            if not rs:
                continue
            cells = [band_row(rs, x) for x in ("all",) + BANDS]
            print("| %s | %s (n=%d) | %s |" % (a, nm, len(rs),
                  " | ".join("%.5f" % m for m, _s, _d in cells)))

    print("\n### trivial baselines (same split)\n")
    print("| model | all | early | mid | late |")
    print("|---|---:|---:|---:|---:|")
    for k in ("constant", "hp_diff_logistic", "hp_diff_binned40", "search_raw",
              "search_recal40", "label_noise_floor_binomial"):
        v = triv[k]
        print("| %s | %.4f | %.4f | %.4f | %.4f |" % (k, v["all"], v["early"], v["mid"], v["late"]))


if __name__ == "__main__":
    main()
