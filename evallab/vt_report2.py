"""ENCODER V2 -- aggregate the architecture grid into the report's tables.

  python vt_report2.py <dir-of-result-jsons>

Same bars, same split, same metric as ENCODER_VALUE_TEST; the only thing that
moved is the enc2 columns (typing multi-hot restored, tera sweep repaired) and
the network architecture.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

BANDS = ("early", "mid", "late")
BARS = {"constant": (0.0502, 0.0994, 0.1535),
        "recal search": (0.0495, 0.0749, 0.0605),
        "floor": (0.0151, 0.0098, 0.0054)}
# ENCODER_VALUE_TEST published numbers, for the comparison rows
PUB_A = (0.0214, 0.0206, 0.0231)          # arm A, shipped encoder + pooled net
PUB_B = (0.0250, 0.0284, 0.0296)          # arm B, old enc2 columns + flat net


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(f)
        if b.startswith("canary"):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        r["_f"] = b
        out.append(r)
    return out


def cell(r):
    """(arch, mon_w, trunk_w, mon_depth, depth, wd, slot_emb) -- everything a
    run is identified by except the seed."""
    if r["arm"] == "old":
        return ("ARM A pooled/shipped", r["cap"][0], r["cap"][1], 2, 3, r["wd"], 1)
    c = r["cap"]
    return (r.get("arch", "flat"), c[0], c[1] if len(c) > 1 else 0,
            r.get("mon_depth", 2), r.get("depth", 3), r["wd"], r.get("slot_emb", 1))


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.max() - v.min()), (v.std(ddof=1) if len(v) > 1 else 0.0)


def label(k):
    a, mw, tw, md, d, wd, se = k
    if a.startswith("ARM A"):
        return "%s %dx%d" % (a, mw, tw)
    if a == "flat":
        return "flat (control) width %d" % mw
    s = "%s mon %dx%d, trunk %dx%d" % (a, mw, md, tw, d)
    if wd != 1e-2:
        s += ", wd %g" % wd
    if not se:
        s += ", no slot emb"
    return s


def main():
    rows = load(sys.argv[1])
    G = defaultdict(list)
    for r in rows:
        G[cell(r)].append(r)

    print("### every cell (mean over seeds; spread = max-min)\n")
    print("| cell | seeds | params | best epoch | val Brier | test all | early | mid | late |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = sorted(G, key=lambda k: ms([r["val_brier"] for r in G[k]])[0])
    for k in order:
        rs = G[k]
        vb = ms([r["val_brier"] for r in rs])
        c = {b: ms([r["test"][b] for r in rs]) for b in ("all",) + BANDS}
        print("| %s | %d | %d | %s | %.5f | %.5f%s | %.5f | %.5f | %.5f |"
              % (label(k), len(rs), rs[0]["params"],
                 "/".join(str(r["best_epoch"]) for r in rs), vb[0], c["all"][0],
                 (" ±%.5f" % c["all"][1]) if len(rs) > 1 else "",
                 c["early"][0], c["mid"][0], c["late"][0]))

    # best of each family, selected on VAL Brier (never on test)
    fam = {}
    for k in G:
        f = k[0]
        v = ms([r["val_brier"] for r in G[k]])[0]
        if f not in fam or v < fam[f][0]:
            fam[f] = (v, k)
    print("\n### best cell per family, selected on val Brier\n")
    print("| family | cell | seeds | test all | early | mid | late |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for f, (v, k) in sorted(fam.items(), key=lambda x: x[1][0]):
        rs = G[k]
        c = {b: ms([r["test"][b] for r in rs]) for b in ("all",) + BANDS}
        print("| %s | %s | %d | %.5f ±%.5f | %.5f ±%.5f | %.5f ±%.5f | %.5f ±%.5f |"
              % (f, label(k), len(rs), c["all"][0], c["all"][1], c["early"][0],
                 c["early"][1], c["mid"][0], c["mid"][1], c["late"][0], c["late"][1]))

    print("\n### HEADLINE: per-band Brier against the same bars\n")
    print("| band | constant | recal search | ARM A (this run) | ARM A (published) | "
          "ENC2-V2 shared | old arm B (published) | floor |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    A = G[fam["ARM A pooled/shipped"][1]] if "ARM A pooled/shipped" in fam else []
    B = G[fam["shared"][1]] if "shared" in fam else []
    gaps = {}
    for i, b in enumerate(BANDS):
        a = ms([r["test"][b] for r in A]) if A else (float("nan"),) * 3
        n = ms([r["test"][b] for r in B]) if B else (float("nan"),) * 3
        sd = np.sqrt((a[2] ** 2 + n[2] ** 2) / 2) if (A and B) else float("nan")
        gaps[b] = (n[0] - a[0], sd, a, n)
        print("| %s | %.4f | %.4f | %.4f ±%.4f | %.4f | **%.4f ±%.4f** | %.4f | %.4f |"
              % (b, BARS["constant"][i], BARS["recal search"][i], a[0], a[1],
                 PUB_A[i], n[0], n[1], PUB_B[i], BARS["floor"][i]))

    print("\n### gap, ENC2-V2 minus ARM A, in pooled seed standard deviations\n")
    print("| band | gap | pooled seed sd | gap / sd | verdict |")
    print("|---|---:|---:|---:|---|")
    for b in BANDS:
        g, sd, a, n = gaps[b]
        v = ("ENC2-V2 WINS" if g < -2 * sd else
             "ARM A WINS" if g > 2 * sd else "tie inside noise")
        print("| %s | %+.5f | %.5f | %.1fx | %s |"
              % (b, g, sd, abs(g) / sd if sd else float("nan"), v))
    if A and B:
        aa = ms([r["test"]["all"] for r in A])
        bb = ms([r["test"]["all"] for r in B])
        sd = np.sqrt((aa[2] ** 2 + bb[2] ** 2) / 2)
        print("| all | %+.5f | %.5f | %.1fx | %s |"
              % (bb[0] - aa[0], sd, abs(bb[0] - aa[0]) / sd if sd else float("nan"),
                 "ENC2-V2 WINS" if bb[0] - aa[0] < -2 * sd else
                 "ARM A WINS" if bb[0] - aa[0] > 2 * sd else "tie inside noise"))
    json.dump({f: {"cell": list(map(str, k)), "val": v} for f, (v, k) in fam.items()},
              open(os.path.join(sys.argv[1], "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
