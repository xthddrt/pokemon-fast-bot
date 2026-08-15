"""FAST PATH -- train the adopted net FROM RANDOM INIT on the 900k non-holdout
plc1 rows at the RECIPE_DERIVATION fine-tune knobs, one seed per process.

No pretrain, no sweep, no early stopping: the cosine schedule ALWAYS completes
and the PUBLISHED checkpoint is the final, fully-annealed one. Holdout Brier is
logged every `--eval-every` steps purely as a progress curve.

  python vt_fast.py train <work> --seed 0 --steps 12000 --threads 21
  python vt_fast.py pick  <work>                  # merge seeds, pick, publish
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402
from vt_canary import (  # noqa: E402
    BANDS, CAP, NOISE_FLOOR, REC, brier_bands, groups, paired, predict, sched,
)


def soft_auc(pred, y):
    """AUC with SOFT labels: P(a sampled win outranks a sampled loss), where
    row i contributes weight y_i to the positive pool and (1-y_i) to the
    negative pool. Ties get 0.5. Well-defined for label_p in [0,1] and applied
    identically to every model compared."""
    o = np.argsort(pred, kind="mergesort")
    p, w = pred[o], y[o]
    n = 1.0 - w
    tot_p, tot_n = w.sum(), n.sum()
    if tot_p <= 0 or tot_n <= 0:
        return float("nan")
    cn = np.concatenate([[0.0], np.cumsum(n)])          # negatives strictly before
    # group boundaries of equal predictions
    starts = np.flatnonzero(np.concatenate([[True], p[1:] != p[:-1]]))
    ends = np.concatenate([starts[1:], [len(p)]])
    acc = 0.0
    for s, e in zip(starts, ends):
        neg_before = cn[s]
        neg_in = cn[e] - cn[s]
        acc += w[s:e].sum() * (neg_before + 0.5 * neg_in)
    return float(acc / (tot_p * tot_n))


def auc_bands(pred, y, band):
    r = {"all": soft_auc(pred, y)}
    for b in BANDS:
        m = band == b
        r[b] = soft_auc(pred[m], y[m])
    return r


def train(work, seed, steps, threads, eval_every=1000, batch=None):
    import torch
    import torch.nn as nn
    import vt_lib as V
    if threads:
        torch.set_num_threads(threads)
    d = os.path.join(work, "enc_plc1")
    V.ENC = d
    ft = V.Arm("old", add="setup")
    m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    n = len(m["label_p"])
    mask = np.ones(n, bool)
    mask[hold] = False
    pool = np.flatnonzero(mask)
    y_np = m["label_p"].astype(np.float64)
    y = torch.from_numpy(m["label_p"].astype(np.float32))
    y_h, band_h = y_np[hold], m["band"][hold].astype(str)
    qs = m["q_search"][hold].astype(np.float64)

    cfg = dict(REC["ft"])
    B = batch or cfg["batch"]
    net = V.build_net("old", CAP, seed, add="setup", dropout=cfg["dropout"])
    nparam = sum(p.numel() for p in net.parameters())
    assert nparam == 1094869, "net is not the adopted arm A + setup: %d" % nparam
    emb, dense = groups(net)
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
         {"params": emb, "lr": cfg["lr_emb"], "weight_decay": cfg["wd_emb"]}])
    warm = max(1, int(round(cfg["warm_frac"] * steps)))
    rng = np.random.default_rng(seed + 909)
    log = lambda s: print(s, flush=True)  # noqa: E731
    log("seed %d: params %d train_pool %d holdout %d B %d steps %d warm %d threads %d"
        % (seed, nparam, len(pool), len(hold), B, steps, warm, torch.get_num_threads()))

    curve = []
    order = rng.permutation(len(pool))
    cur = 0
    t0 = time.time()
    gn_max, nonfinite = 0.0, 0
    for step in range(steps):
        if cur + B > len(order):
            order = rng.permutation(len(pool))
            cur = 0
        b = np.sort(pool[order[cur:cur + B]])
        cur += B
        mlt = sched(step, steps, warm, cfg["cos_floor"])
        opt.param_groups[0]["lr"] = cfg["lr"] * mlt
        opt.param_groups[1]["lr"] = cfg["lr_emb"] * mlt
        net.train()
        z = net(ft.batch(b))
        loss = nn.functional.binary_cross_entropy_with_logits(z, y[b])
        opt.zero_grad()
        loss.backward()
        g = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
        gn_max = max(gn_max, g)
        if not (math.isfinite(g) and torch.isfinite(loss)):
            nonfinite += 1
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            pr = predict(net, ft, hold)
            bs = float(np.mean((pr - y_h) ** 2))
            curve.append({"step": step + 1, "train_loss": float(loss),
                          "lr": cfg["lr"] * mlt, "holdout_brier": bs,
                          "s": round(time.time() - t0, 1)})
            log("  seed %d step %-6d loss %.5f lr %.2e holdout %.6f  %.0fs"
                % (seed, step + 1, float(loss), cfg["lr"] * mlt, bs, time.time() - t0))

    pr = predict(net, ft, hold)
    res = {"seed": seed, "n_params": nparam, "steps": steps, "batch": B,
           "warm_steps": warm, "recipe": cfg, "cap": list(CAP), "add": "setup",
           "from_random_init": True, "pretrain": None,
           "n_train_pool": int(len(pool)), "n_holdout": int(len(hold)),
           "wall_s": round(time.time() - t0, 1),
           "rows_per_s": round(steps * B / max(1e-9, time.time() - t0), 1),
           "grad_norm_max": gn_max, "nonfinite_events": nonfinite,
           "torch_threads": torch.get_num_threads(),
           "curve": curve,
           "brier": brier_bands(pr, y_h, band_h),
           "auc": auc_bands(pr, y_h, band_h),
           "brier_q_search": brier_bands(qs, y_h, band_h),
           "auc_q_search": auc_bands(qs, y_h, band_h),
           "noise_floor": NOISE_FLOOR,
           "pred_stats": {"min": float(pr.min()), "max": float(pr.max()),
                          "mean": float(pr.mean()), "std": float(pr.std()),
                          "nonfinite": int((~np.isfinite(pr)).sum())}}
    res["paired_vs_q"] = {"all": paired(pr, qs, y_h)}
    for b in BANDS:
        k = band_h == b
        res["paired_vs_q"][b] = paired(pr[k], qs[k], y_h[k])
    torch.save({"sd": net.state_dict(), "cfg": {k: v for k, v in res.items()
                                                if k != "curve"}},
               os.path.join(work, "ckpt_fast_s%d.pt" % seed))
    np.save(os.path.join(work, "holdout_pred_s%d.npy" % seed), pr.astype(np.float32))
    json.dump(res, open(os.path.join(work, "REPORT.fast.s%d.json" % seed), "w"), indent=1)
    log(json.dumps({k: res[k] for k in ("brier", "auc", "paired_vs_q",
                                        "pred_stats")}, indent=1))
    return res


def pick(work, seeds):
    """Merge the per-seed reports, pick the best holdout Brier, add the
    seed-mean ensemble as a free reference."""
    rs = [json.load(open(os.path.join(work, "REPORT.fast.s%d.json" % s))) for s in seeds]
    d = os.path.join(work, "enc_plc1")
    m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    y_h, band_h = m["label_p"][hold].astype(np.float64), m["band"][hold].astype(str)
    qs = m["q_search"][hold].astype(np.float64)
    P = np.stack([np.load(os.path.join(work, "holdout_pred_s%d.npy" % s)).astype(np.float64)
                  for s in seeds])
    ens = P.mean(0)
    best = min(rs, key=lambda r: r["brier"]["all"])
    out = {"seeds": seeds, "best_seed": best["seed"],
           "per_seed_brier_all": {r["seed"]: r["brier"]["all"] for r in rs},
           "per_seed": {r["seed"]: {"brier": r["brier"], "auc": r["auc"],
                                    "paired_vs_q": r["paired_vs_q"],
                                    "wall_s": r["wall_s"],
                                    "rows_per_s": r["rows_per_s"]} for r in rs},
           "spread_brier_all": {
               "min": min(r["brier"]["all"] for r in rs),
               "max": max(r["brier"]["all"] for r in rs),
               "mean": float(np.mean([r["brier"]["all"] for r in rs])),
               "std": float(np.std([r["brier"]["all"] for r in rs], ddof=1))},
           "ensemble": {"brier": brier_bands(ens, y_h, band_h),
                        "auc": auc_bands(ens, y_h, band_h),
                        "paired_vs_q": {"all": paired(ens, qs, y_h)}},
           "q_search": {"brier": brier_bands(qs, y_h, band_h),
                        "auc": auc_bands(qs, y_h, band_h)},
           "noise_floor": NOISE_FLOOR}
    for b in BANDS:
        k = band_h == b
        out["ensemble"]["paired_vs_q"][b] = paired(ens[k], qs[k], y_h[k])
    np.save(os.path.join(work, "holdout_pred_ens.npy"), ens.astype(np.float32))
    json.dump(out, open(os.path.join(work, "REPORT.fast.pick.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "pick"])
    ap.add_argument("work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000, dest="ee")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "train":
        train(a.work, a.seed, a.steps, a.threads, a.ee)
    else:
        pick(a.work, [int(x) for x in a.seeds.split(",")])


if __name__ == "__main__":
    main()
