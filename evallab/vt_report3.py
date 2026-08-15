"""ADDITIVE / SUBTRACTIVE ENCODER SEARCH -- grid report, under the anti-overfit
protocol.

Every cell is ARM A (shipped encoder, pooled net, 256x512) plus zero or more of
enc2's derived blocks and minus zero or more of arm A's own dead groups.

THE SPLIT IS THREE-WAY. Training and early stopping use train/val exactly as
published. The held-out TEST games are halved by game into SELECTION and
CONFIRMATION. Cells are ranked on SEL. CONF is printed only with --reveal, and
only for the cells named on the command line -- the mechanism that makes
"touched exactly once" auditable rather than a promise.

  python vt_report3.py <out-dir> [--reveal cell1,cell2]
"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np

BANDS = ("early", "mid", "late")
BARS = {"constant": (0.0889, 0.0502, 0.0994, 0.1535),
        "recalibrated search": (0.0618, 0.0495, 0.0749, 0.0605),
        "ARM A (published, full test)": (0.02146, 0.0215, 0.0206, 0.0230),
        "floor (label noise)": (0.0111, 0.0151, 0.0098, 0.0054)}
# F4 cells measure available parameter relief only: their columns are dead on
# THIS one-pair corpus and live on general data, so they cannot win.
INELIGIBLE = ("f4_cov", "f4_covonly")


def load(d):
    cells = {}
    for f in sorted(glob.glob(os.path.join(d, "r_*.json"))):
        name = os.path.basename(f)[2:].rsplit("_s", 1)[0]
        cells.setdefault(name, []).append(json.load(open(f)))
    return cells


def agg(runs, split, key):
    v = np.array([r[split][key] for r in runs])
    return float(v.mean()), float(v.max() - v.min()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ctl", default="b0_ctl")
    ap.add_argument("--reveal", default="")
    a = ap.parse_args()
    cells = load(a.out_dir)
    ctl = cells[a.ctl]
    n_cells = len(cells)
    order = sorted(cells, key=lambda k: agg(cells[k], "sel", "all")[0])

    print("\n%d CELLS x %d seeds = %d runs. Ranking is on SEL; the 3x-sd bar below "
          "is the multiple-comparison discipline for that many cells.\n"
          % (n_cells, len(ctl), sum(len(v) for v in cells.values())))

    print("== SELECTION half of the held-out test games ==")
    h = "%-22s %-34s %2s %8s %9s %9s %9s %9s %8s %7s"
    hdr = h % ("cell", "spec (add / -drop)", "n", "params", "sel all", "early", "mid",
               "late", "sd all", "d/sd")
    print(hdr); print("-" * len(hdr))
    cm, _, csd = agg(ctl, "sel", "all")
    for k in order:
        r = cells[k]
        spec = " ".join(filter(None, [r[0].get("add", ""),
                                      "-" + r[0]["drop"] if r[0].get("drop") else ""])) or "(none)"
        m, _, sd = agg(r, "sel", "all")
        pooled = ((sd ** 2 + csd ** 2) / 2) ** 0.5
        flag = "  INELIGIBLE(F4)" if k in INELIGIBLE else ""
        print(h % (k, spec[:34], len(r), r[0]["params"],
                   "%.5f" % m, "%.5f" % agg(r, "sel", "early")[0],
                   "%.5f" % agg(r, "sel", "mid")[0], "%.5f" % agg(r, "sel", "late")[0],
                   "%.5f" % sd, "%+.1f" % ((m - cm) / pooled) if pooled > 1e-9 else "-")
              + flag)

    print("\n== delta vs the ARM A control, on SEL (negative = the change WINS) ==")
    h2 = "%-22s %10s %10s %10s %10s %9s"
    print(h2 % ("cell", "d all", "d early", "d mid", "d late", "bar 3x sd"))
    for k in order:
        if k == a.ctl:
            continue
        r = cells[k]
        m, _, sd = agg(r, "sel", "all")
        pooled = ((sd ** 2 + csd ** 2) / 2) ** 0.5
        d = m - cm
        verdict = "PASS" if (d < -3 * pooled and k not in INELIGIBLE) else (
            "ineligible" if k in INELIGIBLE else "fail")
        print(h2 % (k, "%+.5f" % d,
                    *["%+.5f" % (agg(r, "sel", b)[0] - agg(ctl, "sel", b)[0]) for b in BANDS],
                    "%.5f %s" % (3 * pooled, verdict)))

    print("\n== full held-out test (union of sel+conf), for comparability with the "
          "two published reports ==")
    print(h % ("cell", "spec", "n", "params", "test all", "early", "mid", "late", "sd all", ""))
    for k, v in BARS.items():
        print(h % (k, "-", "-", "-", "%.5f" % v[0], "%.5f" % v[1], "%.5f" % v[2],
                   "%.5f" % v[3], "-", ""))
    for k in order:
        r = cells[k]
        print(h % (k, "", len(r), r[0]["params"],
                   *["%.5f" % agg(r, "test", b)[0] for b in ("all",) + BANDS],
                   "%.5f" % agg(r, "test", "all")[2], ""))

    if a.reveal:
        print("\n== CONFIRMATION half -- read ONCE, for these cells only ==")
        print(h2 % ("cell", "conf all", "d vs ctl", "sel d", "shrink", "sd all"))
        for k in [x for x in a.reveal.split(",") if x] + [a.ctl]:
            r = cells[k]
            m, _, sd = agg(r, "conf", "all")
            dc = m - agg(ctl, "conf", "all")[0]
            ds = agg(r, "sel", "all")[0] - cm
            print(h2 % (k, "%.5f" % m, "%+.5f" % dc, "%+.5f" % ds,
                        "%.0f%%" % (100 * (1 - dc / ds)) if abs(ds) > 1e-9 else "-",
                        "%.5f" % sd))

    print("\n== per-seed detail (sel all / test all) ==")
    for k in order:
        for r in sorted(cells[k], key=lambda r: r["seed"]):
            print("  %-22s seed=%d ep=%-3d val=%.5f sel=%.5f test=%.5f"
                  % (k, r["seed"], r["best_epoch"], r["val_brier"],
                     r["sel"]["all"], r["test"]["all"]))


if __name__ == "__main__":
    main()
