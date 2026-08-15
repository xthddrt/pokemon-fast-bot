"""STAGE 2 -- FINE-TUNE LEARNING CURVE + FROM-SCRATCH CONTROL.

Same harness as vt_canary.py / vt_pre1.py: `vt_canary.arm` mmaps the published
plc1 cache, `vt_lib.build_net` builds the adopted "arm A + setup" net, the recipe
knobs come from `vt_canary.REC["ft"]` untouched, the LR schedule is
`vt_canary.sched`, and the holdout metric is `vt_canary.brier_bands`.

WHAT THIS STAGE MEASURES
  A. learning curve: fine-tune the STAGE-1 pretrained net on NESTED subsets of
     the 900,000 non-holdout plc1 rows (50k/100k/200k/500k/900k), 3 seeds each.
  B. from-scratch control at 900k: identical data + recipe, RANDOM init, no
     pretraining, a 2x step budget so it is not budget-starved.
  C. every cell is scored on the SAME published 100,000-row plc1 holdout.

SUBSET NESTING is by TEAM PAIR: pairs are shuffled once (fixed seed) and each
subset is a PREFIX of that order, so subset(50k) is a strict subset of
subset(100k) and a bigger subset can never introduce a pair the smaller one
lacked.  10 % of pairs are flagged `val` at draw time, so the val split is
nested too and no pair is ever on both sides.  Model selection (best checkpoint)
uses that internal val ONLY -- the 100k holdout is read for reporting, and
plc1e is never touched by any code path in this file.

BUDGET.  The cosine schedule is tied to the total-step argument, so `total` is
always the number of steps actually run: every cell anneals to its own floor.
Fine-tune cells get constant PASSES (the derivation's 6,000 steps at 900k =
27.3 passes) with a 1,500-step floor so small cells are not optimizer-limited.

  python vt_ft1.py fetch  <work>
  python vt_ft1.py split  <work>
  python vt_ft1.py jobs   <work>            # job list, longest first (LPT)
  python vt_ft1.py run    <work> --job ft:900000:0:s1
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402
import vt_canary as C  # noqa: E402

BUCKET = os.environ.get("CANARY_BUCKET", "pokebot-valuenet-389825051723")
P_PLC1 = "evallab/enc_plc1/"
P_PRE = os.environ.get("VT_PRE_PREFIX", "evallab/nets_pre1/")
P_OUT = os.environ.get("VT_OUT_PREFIX", "evallab/ft1/")

SIZES = (50000, 100000, 200000, 500000, 900000)
SEEDS = (0, 1, 2)
VAL_PAIR_FRAC = 0.10
VAL_ROW_CAP = 40000        # rows actually scored per val check
SPLIT_SEED = 31
FT_STEPS_AT_900K = 6000    # RECIPE_DERIVATION.md's fine-tune budget
FT_STEPS_MIN = 1500
SCRATCH_STEPS = 12000      # 2x, so the control is not budget-starved
VAL_CHECKS = 24            # val points per run, as in stage 1
PATIENCE = 8               # val checks; a SAFETY NET, not the stopping rule
SRC_CKPT = {"s1": "pre1_s1_best.pt", "s0": "pre1_s0_best.pt"}


def _cp(key, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    subprocess.check_call(["aws", "s3", "cp", "s3://%s/%s" % (BUCKET, key), dst,
                           "--only-show-errors"])


def _put(path, key):
    subprocess.check_call(["aws", "s3", "cp", path, "s3://%s/%s%s" % (BUCKET, P_OUT, key),
                           "--only-show-errors"])


# =============================================================== S3 fetching ==
def fetch(work):
    t0 = time.time()
    d = os.path.join(work, "enc_plc1")
    os.makedirs(d, exist_ok=True)
    for k in C.ARRAYS:
        _cp(P_PLC1 + k + ".npy", os.path.join(d, k + ".npy"))
    for k in ("addon_layout.json", "meta.npz", "holdout_i.npy", "split.json"):
        _cp(P_PLC1 + k, os.path.join(d, k))
    n = os.path.join(work, "nets")
    os.makedirs(n, exist_ok=True)
    for f in SRC_CKPT.values():
        _cp(P_PRE + f, os.path.join(n, f))
    out = {"plc1_bytes": sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)),
           "ckpts": sorted(os.listdir(n)), "s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(work, "REPORT.fetch.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


# ====================================================== nested by-pair split ==
def split(work):
    """Shuffle team pairs once; every subset is a PREFIX of that order."""
    d = os.path.join(work, "enc_plc1")
    p = os.path.join(work, "ft1_split.npz")
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=False)
    pair = m["pair"]
    n = len(pair)
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    keep = np.ones(n, bool)
    keep[hold] = False
    pool = np.flatnonzero(keep)

    codes, inv = np.unique(pair, return_inverse=True)
    hold_codes = np.unique(inv[hold])
    pool_codes = np.unique(inv[pool])
    assert not (set(hold_codes.tolist()) & set(pool_codes.tolist())), \
        "a team pair straddles the published holdout -- the holdout is not pair-clean"

    rng = np.random.default_rng(SPLIT_SEED)
    order = pool_codes[rng.permutation(len(pool_codes))]
    is_val = rng.random(len(order)) < VAL_PAIR_FRAC
    cnt = np.bincount(inv, minlength=len(codes))
    cum = np.cumsum(cnt[order])

    rank = np.full(len(codes), np.iinfo(np.int64).max, np.int64)
    rank[order] = np.arange(len(order))
    valf = np.zeros(len(codes), bool)
    valf[order] = is_val
    row_rank = rank[inv]
    row_val = valf[inv]

    sub = {}
    prev = None
    rep = {"n_rows": int(n), "n_pool": int(len(pool)), "n_holdout": int(len(hold)),
           "n_pairs_total": int(len(codes)), "n_pairs_pool": int(len(pool_codes)),
           "rows_per_pair": float(len(pool) / len(pool_codes)),
           "val_pair_frac": VAL_PAIR_FRAC, "split_seed": SPLIT_SEED, "cells": {}}
    for N in SIZES:
        k = min(len(cum), int(np.searchsorted(cum, N)) + 1)
        sel = keep & (row_rank < k)
        tr = np.flatnonzero(sel & ~row_val).astype(np.int64)
        va = np.flatnonzero(sel & row_val).astype(np.int64)
        vs = va if len(va) <= VAL_ROW_CAP else np.sort(np.random.default_rng(
            SPLIT_SEED + 7).choice(va, VAL_ROW_CAP, replace=False))
        cur = np.flatnonzero(sel)
        if prev is not None:                    # NESTING, asserted not assumed
            assert np.all(np.isin(prev, cur)), "subset %d is not nested in %d" % (0, N)
        prev = cur
        assert not (set(np.unique(inv[tr]).tolist()) & set(np.unique(inv[va]).tolist())), \
            "a pair is on both sides of the internal val split at N=%d" % N
        sub["tr_%d" % N] = tr
        sub["va_%d" % N] = va
        sub["vs_%d" % N] = vs
        rep["cells"][str(N)] = {"n_pairs": int(k), "n_subset": int(len(cur)),
                                "n_train": int(len(tr)), "n_val": int(len(va)),
                                "n_val_scored": int(len(vs))}
    np.savez(p, **sub)
    json.dump(rep, open(os.path.join(work, "REPORT.split.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ======================================================================= jobs ==
def steps_for(kind, size):
    if kind.startswith("scratch"):
        return SCRATCH_STEPS
    return max(FT_STEPS_MIN, int(round(FT_STEPS_AT_900K * size / 900000)))


def job_list():
    js = []
    for N in SIZES:                                    # A. learning curve
        for s in SEEDS:
            js.append(("ft", N, s, "s1"))
    for s in SEEDS:                                    # pretrain-ckpt sensitivity
        js.append(("ft", 900000, s, "s0"))
    for s in SEEDS:                                    # B. from-scratch control
        js.append(("scratch", 900000, s, "none"))
    js.append(("scratchhi", 900000, 0, "none"))        # LR-adequacy probe
    js.sort(key=lambda j: -steps_for(j[0], j[1]))      # LPT: longest first
    return ["%s:%d:%d:%s" % j for j in js]


# ==================================================================== one cell ==
def run(work, job, threads=16, steps=0):
    import torch
    import torch.nn as nn
    import vt_lib as V
    kind, size, seed, src = job.split(":")
    size, seed = int(size), int(seed)
    if threads:
        torch.set_num_threads(threads)
    log = lambda s: print("[%s] %s" % (job, s), flush=True)  # noqa: E731

    d = os.path.join(work, "enc_plc1")
    cfg = dict(C.REC["ft"])
    if kind in ("scratch", "scratchhi"):
        # No pretrained embeddings to protect: the recipe's 3x-reduced embedding
        # LR exists only to keep a pretrained table from being washed out, so
        # holding it here would handicap the control for no reason.
        cfg["lr_emb"] = cfg["lr"]
    if kind == "scratchhi":
        # LR-ADEQUACY PROBE: the pretrain LR mapped to B=4,096 by the
        # derivation's own saturating law. If this does not beat `scratch`, the
        # control was not LR-limited.
        f = C.lr_scale(C.REC["pre"]["batch"], cfg["batch"])
        cfg["lr"] = C.REC["pre"]["lr"] * f
        cfg["lr_emb"] = cfg["lr"]
    batch = cfg["batch"]
    total = steps or steps_for(kind, size)   # `steps` is the SMOKE-TEST override
    val_every = max(50, total // VAL_CHECKS)

    V.ENC = d
    ft = C.arm(d)
    mft = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
    net = V.build_net("old", C.CAP, 1000 + seed, add="setup", dropout=cfg["dropout"])
    nparam = sum(p.numel() for p in net.parameters())
    assert nparam == 1094869, "not the adopted 1,094,869-param model: %d" % nparam
    if kind == "ft":
        ck = os.path.join(work, "nets", SRC_CKPT[src])
        net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["sd"])
        log("loaded pretrained %s" % SRC_CKPT[src])

    z = np.load(os.path.join(work, "ft1_split.npz"))
    tr, va_s = z["tr_%d" % size], z["vs_%d" % size]
    y = torch.from_numpy(mft["label_p"].astype(np.float32))
    y_np = mft["label_p"]
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    band_h = mft["band"][hold].astype(str)
    y_h = y_np[hold]

    emb, dense = C.groups(net)
    sc = C.lr_scale(C.REC["ft"]["batch"], batch)       # == 1.0: at recipe batch
    lr_d, lr_e = cfg["lr"] * sc, cfg["lr_emb"] * sc
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": lr_d, "weight_decay": cfg["wd"]},
         {"params": emb, "lr": lr_e, "weight_decay": cfg["wd_emb"]}])
    warm = max(1, int(round(cfg["warm_frac"] * total)))
    spe = max(1, int(math.ceil(len(tr) / batch)))

    def val():
        p = C.predict(net, ft, va_s, batch=8192)
        net.train()
        return float(np.mean((p - y_np[va_s]) ** 2))

    hist, best = [], {"val": 1e18, "step": -1}
    bpath = os.path.join(work, "best_%s.pt" % job.replace(":", "_"))
    log("N=%d train=%d val_scored=%d steps=%d B=%d lr=%.3e lr_emb=%.3e warm=%d spe=%d"
        % (size, len(tr), len(va_s), total, batch, lr_d, lr_e, warm, spe))

    rng = np.random.default_rng(seed + 202)
    step, gmax, gmed, nonfinite = 0, 0.0, [], 0
    stopped = None
    t0 = time.time()
    net.train()
    while step < total and not stopped:
        order = rng.permutation(len(tr))
        for i in range(spe):
            b = np.sort(tr[order[i * batch:(i + 1) * batch]])
            if not len(b):
                continue
            m = C.sched(step, total, warm, cfg["cos_floor"])
            opt.param_groups[0]["lr"] = lr_d * m
            opt.param_groups[1]["lr"] = lr_e * m
            ls = nn.functional.binary_cross_entropy_with_logits(
                net(ft.batch(b)), y[b], reduction="none")
            loss = ls.mean()
            opt.zero_grad()
            loss.backward()
            g = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
            if not (math.isfinite(g) and torch.isfinite(loss)):
                nonfinite += 1
            gmax = max(gmax, g if math.isfinite(g) else 0.0)
            if step % 25 == 0:
                gmed.append(g)
            opt.step()
            step += 1
            if step % val_every == 0 or step == total:
                vv = val()
                hist.append({"step": step, "train_loss": float(loss.detach()), "val": vv,
                             "cos_mult": m, "lr": lr_d * m, "s": round(time.time() - t0, 1)})
                log("step %-5d train=%.6f val=%.6f cos=%.4f lr=%.3e %.0fs"
                    % (step, hist[-1]["train_loss"], vv, m, lr_d * m, time.time() - t0))
                if vv < best["val"] - 1e-9:
                    best = {"val": vv, "step": step}
                    torch.save({"sd": net.state_dict(), "step": step, "val": vv}, bpath)
                elif len([h for h in hist if h["step"] > best["step"]]) >= PATIENCE:
                    stopped = {"step": step, "cos_mult": m, "lr": lr_d * m,
                               "reason": "val not improved for %d checks" % PATIENCE}
                    log("EARLY STOP %s" % json.dumps(stopped))
                    break
            if step >= total:
                break
    wall = time.time() - t0

    res = {"job": job, "kind": kind, "size": size, "seed": seed, "src": src,
           "n_params": nparam, "recipe": cfg, "batch": batch,
           "total_steps": total, "steps_run": step, "warmup_steps": warm,
           "steps_per_epoch": spe, "passes": step * batch / len(tr),
           "sample_visits": step * batch, "val_every": val_every,
           "n_train": int(len(tr)), "n_val_scored": int(len(va_s)),
           "early_stop": stopped, "wall_s": round(wall, 1),
           "rows_per_s": round(step * batch / max(1e-9, wall), 1),
           "grad_norm_max": gmax,
           "grad_norm_median": float(np.median(gmed)) if gmed else None,
           "nonfinite_events": nonfinite, "torch_threads": torch.get_num_threads(),
           "val_curve": hist, "best": best, "noise_floor": C.NOISE_FLOOR,
           "holdout_brier": {}}
    cur = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for tag, sd in (("end_of_schedule", cur),
                    ("best_val", torch.load(bpath, map_location="cpu",
                                            weights_only=False)["sd"])):
        net.load_state_dict(sd)
        pr = C.predict(net, ft, hold)
        res["holdout_brier"][tag] = C.brier_bands(pr, y_h, band_h)
        np.save(os.path.join(work, "hpred_%s_%s.npy" % (job.replace(":", "_"), tag)),
                pr.astype(np.float32))
        log("holdout %-16s %s" % (tag, json.dumps(res["holdout_brier"][tag])))
    json.dump(res, open(os.path.join(work, "REPORT.cell.%s.json"
                                     % job.replace(":", "_")), "w"), indent=1)
    log("DONE wall=%.0fs rows/s=%.0f best=%s" % (wall, res["rows_per_s"],
                                                 json.dumps(best)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "split", "jobs", "run"])
    ap.add_argument("work")
    ap.add_argument("--job", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0, help="SMOKE TEST ONLY: override")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "fetch":
        fetch(a.work)
    elif a.cmd == "split":
        split(a.work)
    elif a.cmd == "jobs":
        print("\n".join(job_list()))
    else:
        run(a.work, a.job, a.threads, a.steps)


if __name__ == "__main__":
    main()
