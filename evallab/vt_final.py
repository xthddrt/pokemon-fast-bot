"""ENCODER FINAL -- report the recipe-fair search.

Reads the two-phase output of `vt_grid.py`:
  t.<cell>.<recipe>.s0   phase 1, the tuning plane (VAL only is ever read)
  b.<cell>.s<seed>       phase 2, cell at its OWN best recipe
  p.<cell>.s<seed>       phase 2, cell at the SHARED published recipe
  m.<cell>.s<seed>       mechanism-diagnosis cells

Three comparisons, and the third is the point of the whole pass:
  (1) own-best vs own-best   -- the fair ranking
  (2) shared  vs shared      -- the ranking every previous pass produced
  (3) (1) - (2)              -- how much the fixed recipe biased those passes

Same pooled-seed-sd convention as vt_report3.py: pooled = sqrt((sd_a^2+sd_b^2)/2),
bar = 3x pooled. SEL ranks; CONF is printed only with --reveal.

  python vt_final.py <out-dir> [--reveal cell1,cell2]
"""
import argparse
import glob
import json
import os

import numpy as np

BANDS = ("early", "mid", "late")


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(f)
        if b.split(".")[0] not in ("t", "b", "p", "m"):
            continue
        r = json.load(open(f))
        kind, cell = r["tag"].split(".")[0], r["tag"].split(".")[1]
        runs.setdefault((kind, cell), []).append(r)
    return runs


def agg(rs, split, key="all"):
    v = np.array([r[split][key] for r in rs])
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def pooled(sa, sb):
    return ((sa ** 2 + sb ** 2) / 2) ** 0.5


def rec(rs):
    r = rs[0]
    return "lr %-6g wd %-5g do %.1f" % (r["lr"], r["wd"], r["dropout"])


def spec(rs):
    r = rs[0]
    s = " ".join(filter(None, [r.get("add", ""), "-" + r["drop"] if r.get("drop") else "",
                               "biasfix" if r.get("biasfix") else "",
                               "constok" if r.get("constok") else ""]))
    return s or "(arm A)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--reveal", default="")
    ap.add_argument("--ctl", default="ctl")
    a = ap.parse_args()
    R = load(a.out_dir)
    cells = sorted({c for k, c in R if k == "b"},
                   key=lambda c: agg(R[("b", c)], "sel")[0])
    n_runs = sum(len(v) for v in R.values())
    print("\n%d cells, %d runs total (%d tuning + %d confirm + %d mechanism).\n"
          % (len(cells), n_runs,
             sum(len(v) for k, v in R.items() if k[0] == "t"),
             sum(len(v) for k, v in R.items() if k[0] in "bp"),
             sum(len(v) for k, v in R.items() if k[0] == "m")))

    # ---------------- 1. the recipe each cell chose, and how far it moved -----
    print("== 1. CHOSEN RECIPE PER CELL (selected on VAL, seed 0, %d-point plane) =="
          % len(set(tuple(sorted(r.items())) for c in cells
                    for r in [{"lr": x["lr"], "wd": x["wd"], "do": x["dropout"]}
                              for x in R[("t", c)]])))
    h = "%-12s %-30s %9s  %-24s %10s %10s %10s"
    print(h % ("cell", "spec", "params", "own-best recipe", "val@own", "val@shared", "val gain"))
    for c in cells:
        vo, _, _ = agg(R[("b", c)], "val")
        vs, _, _ = agg(R[("p", c)], "val")
        print(h % (c, spec(R[("b", c)])[:30], R[("b", c)][0]["params"], rec(R[("b", c)]),
                   "%.5f" % vo, "%.5f" % vs, "%+.5f" % (vo - vs)))

    # ---------------- 2. fair comparison: own-best vs own-best ---------------
    for kind, title in (("b", "2. FAIR COMPARISON -- every cell at ITS OWN best recipe"),
                        ("p", "3. OLD COMPARISON -- every cell at the SHARED published recipe")):
        cm, csd, _ = agg(R[(kind, a.ctl)], "sel")
        print("\n== %s (SEL) ==" % title)
        h2 = "%-12s %-24s %9s %9s %9s %9s %8s %10s %9s %6s"
        print(h2 % ("cell", "recipe", "sel all", "early", "mid", "late", "sd", "d vs ctl",
                    "3x sd bar", "d/sd"))
        for c in sorted(cells, key=lambda c: agg(R[(kind, c)], "sel")[0]):
            m, sd, n = agg(R[(kind, c)], "sel")
            p = pooled(sd, csd)
            print(h2 % (c, rec(R[(kind, c)]), "%.5f" % m,
                        *["%.5f" % agg(R[(kind, c)], "sel", b)[0] for b in BANDS],
                        "%.5f" % sd, "%+.5f" % (m - cm), "%.5f" % (3 * p),
                        "%+.1f" % ((m - cm) / p) if p > 1e-9 else "-"))

    # ---------------- 4. the bias the fixed recipe introduced ----------------
    print("\n== 4. HOW MUCH THE FIXED RECIPE BIASED THE OLD COMPARISON ==")
    print("   d_fair  = (cell - ctl) with both at their own best recipe")
    print("   d_fixed = (cell - ctl) with both at the shared published recipe")
    cmb = agg(R[("b", a.ctl)], "sel")[0]
    cmp_ = agg(R[("p", a.ctl)], "sel")[0]
    h3 = "%-12s %11s %11s %11s   %s"
    print(h3 % ("cell", "d_fair", "d_fixed", "bias", "verdict change"))
    for c in cells:
        if c == a.ctl:
            continue
        df = agg(R[("b", c)], "sel")[0] - cmb
        dx = agg(R[("p", c)], "sel")[0] - cmp_
        pf = pooled(agg(R[("b", c)], "sel")[1], agg(R[("b", a.ctl)], "sel")[1])
        px = pooled(agg(R[("p", c)], "sel")[1], agg(R[("p", a.ctl)], "sel")[1])
        vf = "PASS" if df < -3 * pf else "fail"
        vx = "PASS" if dx < -3 * px else "fail"
        print(h3 % (c, "%+.5f" % df, "%+.5f" % dx, "%+.5f" % (df - dx),
                    "%s -> %s%s" % (vx, vf, "   <-- FLIPPED" if vf != vx else "")))

    # ---------------- 5. mechanism cells ------------------------------------
    mech = sorted({c for k, c in R if k == "m"})
    if mech:
        print("\n== 5. MECHANISM DIAGNOSIS (published recipe, 3 seeds) ==")
        cm, csd, _ = agg(R[("p", a.ctl)], "sel")
        h4 = "%-16s %-28s %9s %9s %8s %10s %6s"
        print(h4 % ("cell", "spec", "params", "sel all", "sd", "d vs ctl", "d/sd"))
        print(h4 % ("ctl", "(arm A)", R[("p", a.ctl)][0]["params"], "%.5f" % cm, "%.5f" % csd,
                    "-", "-"))
        for c in mech:
            m, sd, _ = agg(R[("m", c)], "sel")
            p = pooled(sd, csd)
            print(h4 % (c, spec(R[("m", c)])[:28], R[("m", c)][0]["params"], "%.5f" % m,
                        "%.5f" % sd, "%+.5f" % (m - cm),
                        "%+.1f" % ((m - cm) / p) if p > 1e-9 else "-"))

    # ---------------- 6. the tuning plane, per cell -------------------------
    print("\n== 6. THE TUNING PLANE (val Brier, seed 0; * = chosen, # = shared recipe) ==")
    for c in cells:
        ts = sorted(R[("t", c)], key=lambda r: r["val_brier"])
        best = (ts[0]["lr"], ts[0]["wd"], ts[0]["dropout"])
        print("  %-12s " % c + "  ".join(
            "%s%g/%g/%g:%.5f" % ("*" if (r["lr"], r["wd"], r["dropout"]) == best else
                                 ("#" if (r["lr"], r["wd"], r["dropout"]) == (1e-3, 1e-2, 0.0) else " "),
                                 r["lr"], r["wd"], r["dropout"], r["val_brier"])
            for r in ts[:6]))

    if a.reveal:
        print("\n== 7. CONFIRMATION half -- read ONCE, for these cells only ==")
        cm, csd, _ = agg(R[("b", a.ctl)], "conf")
        h5 = "%-12s %10s %10s %10s %9s %8s"
        print(h5 % ("cell", "conf all", "d vs ctl", "sel d", "shrink", "d/sd"))
        for c in [x for x in a.reveal.split(",") if x]:
            m, sd, _ = agg(R[("b", c)], "conf")
            dc, ds = m - cm, agg(R[("b", c)], "sel")[0] - agg(R[("b", a.ctl)], "sel")[0]
            p = pooled(sd, csd)
            print(h5 % (c, "%.5f" % m, "%+.5f" % dc, "%+.5f" % ds,
                        "%.0f%%" % (100 * (1 - dc / ds)) if abs(ds) > 1e-9 else "-",
                        "%+.1f" % (dc / p) if p > 1e-9 else "-"))

    print("\n== 8. per-seed detail (own-best recipe) ==")
    for c in cells:
        for r in sorted(R[("b", c)], key=lambda r: r["seed"]):
            print("  %-12s seed=%d ep=%-3d val=%.5f sel=%.5f test=%.5f"
                  % (c, r["seed"], r["best_epoch"], r["val_brier"], r["sel"]["all"],
                     r["test"]["all"]))


if __name__ == "__main__":
    main()
