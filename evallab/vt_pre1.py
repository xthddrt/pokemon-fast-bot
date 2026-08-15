"""STAGE 1 -- PRETRAIN the adopted net on the FULL enc_pre1 cache (7,998,924 rows).

Same harness as vt_canary.py: `vt_canary.arm` mmaps the published cache,
`vt_lib.build_net` builds the adopted "arm A + setup" net, the recipe knobs come
from `vt_canary.REC["pre"]` untouched, the LR schedule is `vt_canary.sched`, and
the holdout metric is `vt_canary.brier_bands`.  What changes versus the canary:

  * row subsetting: the WHOLE cache, not a 1 % block sample;
  * the stopping rule: a STEP budget (6,000 steps at B=16,384) instead of an
    epoch budget, so the cosine schedule -- which is tied to the total-step
    argument -- actually COMPLETES.  Early stopping is a safety net only.
  * checkpoints every 2,000 steps to S3, and resume from the newest one, so a
    spot reclaim costs at most one checkpoint interval.

  python vt_pre1.py fetch    <work>                  # full pre1 + plc1 caches
  python vt_pre1.py split    <work>                  # by-GAME train/val split
  python vt_pre1.py pretrain <work> --seed 0         # the run itself
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
P_PRE1 = "evallab/enc_pre1/"
P_PLC1 = "evallab/enc_plc1/"
P_NETS = os.environ.get("VT_NETS_PREFIX", "evallab/nets_pre1/")

VAL_GAME_FRAC = 0.05      # 5 % of GAMES held out -- never split inside a game
VAL_ROW_CAP = 300000      # rows actually scored per val check (subset of val)
SPLIT_SEED = 11           # FIXED: both model seeds share one split, so the two
                          # val curves and their best-val picks are comparable


# =============================================================== S3 fetching ==
def _cp(key, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    subprocess.check_call(["aws", "s3", "cp", "s3://%s/%s" % (BUCKET, key), dst,
                           "--only-show-errors"])


def fetch(work):
    """Both caches, whole. pre1 = 21.7 GiB, plc1 = 2.7 GiB."""
    t0 = time.time()
    out = {}
    for name, pref, extra in (("enc_pre1", P_PRE1, ["addon_layout.json", "meta.npz"]),
                              ("enc_plc1", P_PLC1, ["addon_layout.json", "meta.npz",
                                                    "holdout_i.npy", "split.json"])):
        d = os.path.join(work, name)
        os.makedirs(d, exist_ok=True)
        t1 = time.time()
        for k in C.ARRAYS:
            _cp(pref + k + ".npy", os.path.join(d, k + ".npy"))
        for k in extra:
            _cp(pref + k, os.path.join(d, k))
        out[name + "_dl_s"] = round(time.time() - t1, 1)
        out[name + "_bytes"] = sum(os.path.getsize(os.path.join(d, f))
                                   for f in os.listdir(d))
    out["total_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(work, "REPORT.fetch.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


# ============================================================ by-game split ==
def split(work):
    """Hold out 5 % of GAMES. Positions inside one game share the outcome that
    the lambda-return is built from, so anything finer than a game leaks."""
    d = os.path.join(work, "enc_pre1")
    p = os.path.join(work, "pre1_split.npz")
    if os.path.exists(p):
        z = np.load(p)
        print("split cached: train %d val %d" % (len(z["tr"]), len(z["va"])), flush=True)
        return
    t0 = time.time()
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=False)
    g = m["g"]
    ug, ginv = np.unique(g, return_inverse=True)
    rg = np.random.default_rng(SPLIT_SEED)
    perm = rg.permutation(len(ug))
    sel = np.zeros(len(ug), bool)
    sel[perm[:max(1, int(round(VAL_GAME_FRAC * len(ug))))]] = True
    is_v = sel[ginv]
    tr = np.flatnonzero(~is_v).astype(np.int64)
    va = np.flatnonzero(is_v).astype(np.int64)
    # the rows actually scored at each val check: a random subset of the val
    # rows (whole games are already on the val side, so no leak is possible)
    vs = va if len(va) <= VAL_ROW_CAP else np.sort(
        np.random.default_rng(SPLIT_SEED + 1).choice(va, VAL_ROW_CAP, replace=False))
    np.savez(p, tr=tr, va=va, vs=vs)
    rep = {"n_rows": int(len(g)), "n_games": int(len(ug)), "n_train": int(len(tr)),
           "n_val": int(len(va)), "n_val_scored": int(len(vs)),
           "val_game_frac": VAL_GAME_FRAC, "split_seed": SPLIT_SEED,
           "rows_per_game": float(len(g) / len(ug)), "s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(work, "REPORT.split.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ============================================================== checkpoints ==
def ck_name(seed, step):
    return "pre1_s%d_step%05d.pt" % (seed, step)


def s3_latest(seed):
    """Newest checkpoint step for this seed already in S3, or -1."""
    import boto3
    cli = boto3.client("s3")
    pre = P_NETS + "pre1_s%d_step" % seed
    best = -1
    tok = {}
    while True:
        r = cli.list_objects_v2(Bucket=BUCKET, Prefix=pre, **tok)
        for o in r.get("Contents", []):
            try:
                best = max(best, int(os.path.basename(o["Key"])[-8:-3]))
            except ValueError:
                pass
        if not r.get("IsTruncated"):
            return best
        tok = {"ContinuationToken": r["NextContinuationToken"]}


def ck_put(path, key):
    subprocess.check_call(["aws", "s3", "cp", path, "s3://%s/%s%s" % (BUCKET, P_NETS, key),
                           "--only-show-errors"])


# ================================================================== pretrain ==
def pretrain(work, seed=0, total=6000, threads=32, val_every=250, ckpt_every=2000,
             patience=8, no_s3=False):
    import torch
    import torch.nn as nn
    import vt_lib as V
    if threads:
        torch.set_num_threads(threads)
    log = lambda s: print("[s%d] %s" % (seed, s), flush=True)  # noqa: E731

    d_pre = os.path.join(work, "enc_pre1")
    d_ft = os.path.join(work, "enc_plc1")
    cfg = C.REC["pre"]
    batch = cfg["batch"]

    assert (json.load(open(os.path.join(d_pre, "addon_layout.json")))
            == json.load(open(os.path.join(d_ft, "addon_layout.json")))), \
        "pre1 and plc1 addon layouts differ -- the same net cannot read both"
    pre = C.arm(d_pre)
    mpre = np.load(os.path.join(d_pre, "meta.npz"), allow_pickle=False)
    ft = C.arm(d_ft)
    mft = dict(np.load(os.path.join(d_ft, "meta.npz"), allow_pickle=False))
    V.ENC = d_pre
    net = V.build_net("old", C.CAP, seed, add="setup", dropout=cfg["dropout"])
    nparam = sum(p.numel() for p in net.parameters())
    assert nparam == 1094869, "net is not the adopted 1,094,869-param model: %d" % nparam

    z = np.load(os.path.join(work, "pre1_split.npz"))
    tr, va_s = z["tr"], z["vs"]
    y = torch.from_numpy(mpre["y"].astype(np.float32))
    w_np = mpre["w"].astype(np.float64)
    w_np = w_np / w_np[tr].mean()
    w = torch.from_numpy(w_np.astype(np.float32))

    hold = np.load(os.path.join(d_ft, "holdout_i.npy")).astype(np.int64)
    band_h = mft["band"][hold].astype(str)
    y_h = mft["label_p"][hold]

    emb, dense = C.groups(net)
    sc = C.lr_scale(cfg["batch"], batch)       # == 1.0: we train AT the recipe batch
    lr_d, lr_e = cfg["lr"] * sc, cfg["lr_emb"] * sc
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": lr_d, "weight_decay": cfg["wd"]},
         {"params": emb, "lr": lr_e, "weight_decay": cfg["wd_emb"]}])
    warm = max(1, int(round(cfg["warm_frac"] * total)))
    spe = max(1, int(math.ceil(len(tr) / batch)))

    def val():
        net.eval()
        tot, seen = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_s), 8192):
                b = va_s[i:i + 8192]
                ls = nn.functional.binary_cross_entropy_with_logits(
                    net(pre.batch(b)), y[b], reduction="none")
                tot += float((ls * w[b]).sum())
                seen += len(b)
        net.train()
        return tot / seen

    # ---------------- resume ------------------------------------------------
    step, hist, best = 0, [], {"val": 1e18, "step": -1}
    resumed_from = None
    if not no_s3:
        last = s3_latest(seed)
        if last >= 0:
            p = os.path.join(work, ck_name(seed, last))
            _cp(P_NETS + ck_name(seed, last), p)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            net.load_state_dict(ck["sd"])
            opt.load_state_dict(ck["opt"])
            step, hist, best = ck["step"], ck["hist"], ck["best"]
            resumed_from = ck_name(seed, last)
            log("RESUMED from %s at step %d" % (resumed_from, step))

    log("rows %d (train %d, val scored %d)  steps %d  B %d  lr %.2e  warm %d  spe %d"
        % (len(y), len(tr), len(va_s), total, batch, lr_d, warm, spe))

    rng = np.random.default_rng(seed + 101)
    ep0 = step // spe
    for _ in range(ep0):                       # replay to the resumed epoch
        rng.permutation(len(tr))
    ep = ep0
    t0 = time.time()
    stopped = None
    net.train()
    gmax, gmed, nonfinite = 0.0, [], 0

    def save(tag_step):
        """Full state (weights + AdamW moments + step + history) so a reclaim
        costs at most one interval, never a restart."""
        p = os.path.join(work, ck_name(seed, tag_step))
        torch.save({"sd": net.state_dict(), "opt": opt.state_dict(), "step": tag_step,
                    "hist": hist, "best": best, "seed": seed, "total": total}, p)
        if not no_s3:
            ck_put(p, ck_name(seed, tag_step))
        return ck_name(seed, tag_step)

    while step < total:
        order = rng.permutation(len(tr))
        i0 = (step % spe) if ep == ep0 else 0
        for i in range(i0, spe):
            b = np.sort(tr[order[i * batch:(i + 1) * batch]])
            if not len(b):
                continue
            m = C.sched(step, total, warm, cfg["cos_floor"])
            opt.param_groups[0]["lr"] = lr_d * m
            opt.param_groups[1]["lr"] = lr_e * m
            ls = nn.functional.binary_cross_entropy_with_logits(
                net(pre.batch(b)), y[b], reduction="none")
            loss = (ls * w[b]).mean()
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
                tl = float(loss.detach())
                hist.append({"step": step, "train_loss": tl, "val": vv,
                             "cos_mult": m, "lr": lr_d * m,
                             "s": round(time.time() - t0, 1)})
                log("step %-5d train=%.6f val=%.6f cos=%.4f lr=%.3e %.0fs"
                    % (step, tl, vv, m, lr_d * m, time.time() - t0))
                if vv < best["val"] - 1e-7:
                    best = {"val": vv, "step": step}
                    torch.save({"sd": net.state_dict(), "step": step, "val": vv,
                                "seed": seed}, os.path.join(work, "best_s%d.pt" % seed))
                    if not no_s3:
                        ck_put(os.path.join(work, "best_s%d.pt" % seed),
                               "pre1_s%d_best.pt" % seed)
                elif len([h for h in hist if h["step"] > best["step"]]) >= patience:
                    stopped = {"step": step, "cos_mult": m, "lr": lr_d * m,
                               "reason": "val not improved for %d checks" % patience}
                    log("EARLY STOP %s" % json.dumps(stopped))
                    break
            if step % ckpt_every == 0 or step == total:
                save(step)
            if step >= total:
                break
        if stopped:
            break
        ep += 1
    wall = time.time() - t0
    if stopped:
        save(step)

    # ---------------- holdout Brier on the 100k plc1 holdout, BOTH ckpts -----
    cur = {k: v.detach().clone() for k, v in net.state_dict().items()}
    res = {"seed": seed, "n_params": nparam, "recipe": cfg, "batch": batch,
           "total_steps": total, "steps_run": step, "warmup_steps": warm,
           "steps_per_epoch": spe, "passes": step * batch / len(tr),
           "n_rows": int(len(y)), "n_train": int(len(tr)), "n_val_scored": int(len(va_s)),
           "resumed_from": resumed_from, "early_stop": stopped,
           "wall_s": round(wall, 1), "rows_per_s": round(step * batch / max(1e-9, wall), 1),
           "grad_norm_max": gmax, "grad_norm_median": float(np.median(gmed)) if gmed else None,
           "nonfinite_events": nonfinite, "torch_threads": torch.get_num_threads(),
           "val_curve": hist, "best": best,
           "noise_floor": C.NOISE_FLOOR, "holdout_brier": {}}
    p_const = float(y_h.mean())
    res["holdout_brier"]["constant"] = C.brier_bands(np.full(len(hold), p_const), y_h, band_h)
    res["holdout_brier"]["q_search"] = C.brier_bands(mft["q_search"][hold], y_h, band_h)
    for tag, sd in (("end_of_schedule", cur),
                    ("best_val", torch.load(os.path.join(work, "best_s%d.pt" % seed),
                                            map_location="cpu",
                                            weights_only=False)["sd"])):
        net.load_state_dict(sd)
        pr = C.predict(net, ft, hold)
        res["holdout_brier"][tag] = C.brier_bands(pr, y_h, band_h)
        np.save(os.path.join(work, "holdout_pred_%s_s%d.npy" % (tag, seed)),
                pr.astype(np.float32))
        log("holdout %-16s %s" % (tag, json.dumps(res["holdout_brier"][tag])))
    res["ckpt_keys"] = {"best_val": P_NETS + "pre1_s%d_best.pt" % seed,
                        "end_of_schedule": P_NETS + ck_name(seed, step)}
    json.dump(res, open(os.path.join(work, "REPORT.pre1.s%d.json" % seed), "w"), indent=1)
    log("DONE " + json.dumps({k: res[k] for k in ("steps_run", "passes", "wall_s",
                                                  "rows_per_s", "best", "early_stop",
                                                  "holdout_brier")})[:1200])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "split", "pretrain"])
    ap.add_argument("work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=250, dest="ve")
    ap.add_argument("--ckpt-every", type=int, default=2000, dest="ce")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-s3", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "fetch":
        fetch(a.work)
    elif a.cmd == "split":
        split(a.work)
    else:
        pretrain(a.work, a.seed, a.steps, a.threads, a.ve, a.ce, a.patience, a.no_s3)


if __name__ == "__main__":
    main()
