"""ADDITIVE ENCODER SEARCH -- stratified read of a cell's gain.

The F2 prediction is falsifiable and this is what falsifies it: information arm A
structurally cannot compute (bench-vs-active and bench-vs-bench matchups, post-
setup sweep threats) should pay in MID and LATE and in positions where a setup
threat is actually live. A gain that is FLAT across phases and strata is
evidence of noise-fitting, not matchup capture.

Reads the per-row test predictions `vt_train.py` dumps, so no re-training.

  python vt_strat.py <out-dir> <ctl-cell> <cell> [<cell> ...]
"""
import glob
import json
import os
import sys

import numpy as np

import vt_lib as V


def preds(d, cell):
    f = sorted(glob.glob(os.path.join(d, "r_%s_s*_pred.npy" % cell)))
    assert f, "no predictions for %s" % cell
    return np.stack([np.load(x) for x in f])          # (seeds, n_test)


def main():
    d, ctl_cell, cells = sys.argv[1], sys.argv[2], sys.argv[3:]
    meta = V.load_meta()
    ix, _ = V.split_idx(meta)
    te = ix["test"]
    y, band = meta["label_p"][te], meta["band"][te]

    # strata built from the add-on cache, which is row-aligned to meta
    lay = V.load_addon_layout()
    am = np.load(os.path.join(V.ENC, "addon_mon.npy"), mmap_mode="r")
    ar = np.load(os.path.join(V.ENC, "addon_rest.npy"), mmap_mode="r")
    mn, rn = lay["mon_names"], lay["rest_names"]
    sweep = np.asarray(am[te][:, :, mn.index("cf_mon0.s1_sweep")], np.float32)
    tera_sweep = np.asarray(ar[te][:, [rn.index("cf_side0.tera_enabled_sweep"),
                                       rn.index("cf_side1.tera_enabled_sweep")]], np.float32)
    strata = {
        "ALL": np.ones(len(te), bool),
        "band=early": band == "early", "band=mid": band == "mid", "band=late": band == "late",
        "live sweep threat (any mon)": sweep.max(1) > 0.5,
        "no sweep threat": sweep.max(1) <= 0.5,
        "OUR sweep threat": sweep[:, :6].max(1) > 0.5,
        "THEIR sweep threat": sweep[:, 6:].max(1) > 0.5,
        # a setup MOVE exists in 99.8 % of this fixed-pair corpus, so that is not
        # a stratum; the strict sweep conjunction is
        "BOTH sides have a sweep threat": (sweep[:, :6].max(1) > 0.5) & (sweep[:, 6:].max(1) > 0.5),
        "late AND live sweep threat": (band == "late") & (sweep.max(1) > 0.5),
        "late AND no sweep threat": (band == "late") & (sweep.max(1) <= 0.5),
        "tera-enabled sweep live": tera_sweep.max(1) > 0.5,
    }
    P = {c: preds(d, c) for c in [ctl_cell] + cells}
    b = lambda p, m: float(np.mean((p[:, m] - y[m]) ** 2, axis=1).mean())

    w = max(len(s) for s in strata)
    hdr = "%-*s %7s %9s" % (w, "stratum", "n", ctl_cell)
    for c in cells:
        hdr += " %11s %10s" % (c[:11], "delta")
    print(hdr); print("-" * len(hdr))
    for name, m in strata.items():
        row = "%-*s %7d %9.5f" % (w, name, m.sum(), b(P[ctl_cell], m))
        for c in cells:
            row += " %11.5f %+10.5f" % (b(P[c], m), b(P[c], m) - b(P[ctl_cell], m))
        print(row)


if __name__ == "__main__":
    main()
