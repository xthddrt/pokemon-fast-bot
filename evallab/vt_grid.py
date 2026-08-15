"""ENCODER FINAL -- the recipe-fair search: cell list, recipe plane, job files.

WHY THIS FILE EXISTS
--------------------
Every previous pass compared configurations under ONE recipe (lr 1e-3, wd 1e-2,
no dropout) that was tuned for arm A. Changing the architecture changes the
optimal hyper-parameters, so that comparison measured "cell X at arm A's recipe",
not "cell X". This grid gives EVERY cell its OWN recipe sweep and compares
best-recipe against best-recipe.

TWO PHASES, one box:
  phase 1 TUNE     every cell x every recipe, seed 0, ranked on VAL Brier.
                   VAL -- not SEL -- so that SEL stays a clean comparison set and
                   CONF stays untouched. Early stopping already reads val, so no
                   new holdout is consumed by tuning.
  phase 2 CONFIRM  every cell at (a) its OWN best recipe and (b) the SHARED
                   published recipe, 3 seeds each. (a) is the fair comparison;
                   (b) is the old comparison, re-run in this code, so the
                   difference between them MEASURES the bias of every previous
                   pass.

`mech` cells are the mechanism diagnosis for the `-last_item` anomaly and run at
the published recipe only.
"""
import argparse
import glob
import json
import os

# ---------------------------------------------------------------- cells -----
# name -> extra vt_train.py flags. `add` blocks are enc2 derived features
# appended to arm A; `drop struct` prunes arm A's dead-by-construction columns.
CELLS = [
    ("ctl",        ""),
    ("setup",      "--add setup"),
    ("caps",       "--add caps"),
    ("tera",       "--add tera"),
    ("kopm",       "--add ko_pm"),
    ("kopms",      "--add ko_pm_s"),
    ("spdpm",      "--add spd_pm,mprio_pm"),
    ("sc",         "--add setup,caps"),
    ("sck",        "--add setup,caps,ko_pm_s"),
    ("sckt",       "--add setup,caps,ko_pm_s,tera"),
    ("sckfull",    "--add setup,caps,ko_pm"),
    ("ctl_ns",     "--drop struct"),
    ("sc_ns",      "--add setup,caps --drop struct"),
    ("ctl_ns_bf",  "--drop struct --biasfix 1"),
    ("sc_ns_bf",   "--add setup,caps --drop struct --biasfix 1"),
]

# ------------------------------------------------------------- recipes ------
# lr spans 30x (covers +/-3x around BOTH the published 1e-3 and the 3e-4 that
# ENCODER_V2_RESULT §4 found better for arm A). wd spans 10x.
# NOTE ON THE P/N PRIOR: optimal wd ~ P/N, and the widest cell here differs from
# the narrowest by 1.9 % of parameters, so that prior predicts a <2 % shift in
# optimal wd -- two orders of magnitude below this grid's 3x spacing. If an
# optimum moves here it is NOT the P/N mechanism; the live candidate is the
# first-layer fan_in change, which rescales PyTorch's 1/sqrt(fan_in) init.
LR = [1e-4, 3e-4, 1e-3, 3e-3]
WD = [1e-2, 3e-2, 1e-1]
RECIPES = [(lr, wd, 0.0) for lr in LR for wd in WD] + \
          [(3e-4, 3e-2, 0.1), (3e-4, 3e-2, 0.2)]
SHARED = (1e-3, 1e-2, 0.0)       # the recipe every previous pass used
BASE = "--arm old --cap 256,512 --epochs 150 --patience 12"

# mechanism-diagnosis cells for the -last_item anomaly (published recipe only)
MECH = [
    ("m_lastitem",     "--drop last_item"),
    ("m_lastitem_bf",  "--drop last_item --biasfix 1"),
    ("m_lastitem_tok", "--drop last_item --constok 12"),
    ("m_ctl_tok",      "--constok 12"),
    ("m_stellar",      "--drop tera_stellar"),
]


def rid(r):
    """Dot-free recipe id -- tags are split on '.', and 1e-4 is not."""
    return "L%d_W%d_D%d" % (LR.index(r[0]), WD.index(r[1]), int(round(r[2] * 10)))


def job(tag, flags, lr, wd, do, seed, out):
    return ("%s %s --lr %g --wd %g --dropout %g --seed %d --tag %s --out %s/%s.json"
            % (BASE, flags, lr, wd, do, seed, tag, out, tag)).replace("  ", " ")


def phase1(out):
    lines = []
    for name, flags in CELLS:
        for r in RECIPES:
            lines.append(job("t.%s.%s.s0" % (name, rid(r)), flags, *r, 0, out))
    for name, flags in MECH:
        for s in (0, 1, 2):
            lines.append(job("m.%s.s%d" % (name, s), flags, *SHARED, s, out))
    return lines


def phase2(out, results):
    """Pick each cell's best recipe on VAL, then emit 3-seed confirm jobs at
    that recipe AND at the shared published recipe."""
    by = {}
    for f in glob.glob(os.path.join(results, "t.*.json")):
        if f.endswith("_pred.npy"):
            continue
        r = json.load(open(f))
        cell = r["tag"].split(".")[1]
        by.setdefault(cell, {})[rid((r["lr"], r["wd"], r["dropout"]))] = r["val_brier"]
    lines, picked = [], {}
    for name, flags in CELLS:
        got = by.get(name, {})
        assert len(got) == len(RECIPES), "cell %s: %d/%d tune runs" % (
            name, len(got), len(RECIPES))
        best = min(got, key=got.get)
        r = next(x for x in RECIPES if rid(x) == best)
        picked[name] = {"recipe": list(r), "val": got[best],
                        "val_shared": got[rid(SHARED)],
                        "all": {k: got[k] for k in sorted(got, key=got.get)}}
        for s in (0, 1, 2):
            lines.append(job("b.%s.s%d" % (name, s), flags, *r, s, out))
            lines.append(job("p.%s.s%d" % (name, s), flags, *SHARED, s, out))
    json.dump(picked, open(os.path.join(results, "PICKED.json"), "w"), indent=1)
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["1", "2"])
    ap.add_argument("--out", default="/opt/out")
    ap.add_argument("--results", default="/opt/out")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--skip-existing", action="store_true", dest="skip_existing",
                    help="omit jobs whose output json is already present (salvage relaunch)")
    a = ap.parse_args()
    L = phase1(a.out) if a.phase == "1" else phase2(a.out, a.results)
    if a.skip_existing:
        # SALVAGE: a spot box can die mid-grid. Results sync to S3 every 60 s, so
        # a relaunch must run ONLY the missing cells -- re-running a completed
        # prefix wastes the whole point of having salvaged it.
        n0 = len(L)
        L = [j for j in L if not os.path.exists(j.rsplit(" ", 1)[1])]
        print("skip-existing: %d of %d jobs already done" % (n0 - len(L), n0))
    open(a.jobs, "w").write("\n".join(L) + "\n")
    print("phase %s: %d jobs -> %s" % (a.phase, len(L), a.jobs))
