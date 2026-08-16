"""ENGINE-NATIVE WIDTH -- vt_fast.py with the capacity as a parameter.

Identical recipe, identical data, identical metrics as the v8fast-256 job; the
ONLY change is (MON_HID, TRUNK_HID) = (128, 256), which is what
poke-engine/src/genx/evaluate_nn.rs declares as compile-time consts, so the
resulting checkpoint can actually be loaded by the shipping engine.

  python vt_n.py train <work> --seed 0 --steps 12000 --threads 21
  python vt_n.py pick  <work> --seeds 0,1,2
  python vt_n.py bench <work>          # params / MACs / rows-per-s, both widths
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
    BANDS, NOISE_FLOOR, REC, brier_bands, groups, paired, predict, sched,
)
from vt_fast import auc_bands  # noqa: E402

CAP_N = (128, 256)      # THE ENGINE'S consts (evaluate_nn.rs:28-29)
CAP_WIDE = (256, 512)   # what v8fast-256 was trained at


def _load(work):
    import vt_lib as V
    d = os.path.join(work, "enc_plc1")
    V.ENC = d
    return d, V


def train(work, seed, steps, threads, eval_every=1000, cap=CAP_N, batch=None,
          wd=None, swap_aug=False):
    import torch
    import torch.nn as nn
    if threads:
        torch.set_num_threads(threads)
    d, V = _load(work)
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
    if wd is not None:
        cfg["wd"] = wd
    B = batch or cfg["batch"]
    net = V.build_net("old", cap, seed, add="setup", dropout=cfg["dropout"])
    nparam = sum(p.numel() for p in net.parameters())
    emb, dense = groups(net)
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
         {"params": emb, "lr": cfg["lr_emb"], "weight_decay": cfg["wd_emb"]}])
    warm = max(1, int(round(cfg["warm_frac"] * steps)))
    rng = np.random.default_rng(seed + 909)
    log = lambda s: print(s, flush=True)  # noqa: E731
    log("seed %d cap %s: params %d train_pool %d holdout %d B %d steps %d "
        "warm %d threads %d wd %g"
        % (seed, cap, nparam, len(pool), len(hold), B, steps, warm,
           torch.get_num_threads(), cfg["wd"]))

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
        bt = ft.batch(b)
        yb = y[b]
        if swap_aug:
            # per-row 50% side swap with flipped label: exact-antisymmetry
            # training (v8c). Holdout eval stays unswapped.
            import torch as _t
            sm = _t.from_numpy(rng.random(len(b)) < 0.5)
            if bool(sm.any()):
                V.swap_rows_(bt, sm)
                yb = yb.clone()
                yb[sm] = 1.0 - yb[sm]
        z = net(bt)
        loss = nn.functional.binary_cross_entropy_with_logits(z, yb)
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
                % (seed, step + 1, float(loss), cfg["lr"] * mlt, bs,
                   time.time() - t0))

    pr = predict(net, ft, hold)
    res = {"seed": seed, "n_params": nparam, "steps": steps, "batch": B,
           "warm_steps": warm, "recipe": cfg, "cap": list(cap), "add": "setup",
           "from_random_init": True, "pretrain": None,
           "engine_native_width": list(cap) == list(CAP_N),
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
               os.path.join(work, "ckpt_n_s%d.pt" % seed))
    np.save(os.path.join(work, "holdout_pred_s%d.npy" % seed),
            pr.astype(np.float32))
    json.dump(res, open(os.path.join(work, "REPORT.n.s%d.json" % seed), "w"),
              indent=1)
    log(json.dumps({k: res[k] for k in ("brier", "auc", "paired_vs_q",
                                        "pred_stats")}, indent=1))
    return res


def pick(work, seeds):
    rs = [json.load(open(os.path.join(work, "REPORT.n.s%d.json" % s)))
          for s in seeds]
    d, V = _load(work)
    m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=False))
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    y_h = m["label_p"][hold].astype(np.float64)
    band_h = m["band"][hold].astype(str)
    qs = m["q_search"][hold].astype(np.float64)
    P = np.stack([np.load(os.path.join(work, "holdout_pred_s%d.npy" % s))
                  .astype(np.float64) for s in seeds])
    ens = P.mean(0)
    best = min(rs, key=lambda r: r["brier"]["all"])
    out = {"seeds": seeds, "best_seed": best["seed"], "cap": rs[0]["cap"],
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
    json.dump(out, open(os.path.join(work, "REPORT.n.pick.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


# ------------------------------------------------------------- inference ----
def _macs_and_rate(cap, ft, idx, threads, reps=3):
    """Multiply-accumulates per SAMPLE (Linear layers -- the only FLOP-bearing
    ops; embeddings are gathers) plus measured torch rows/s at batch 4096."""
    import torch
    import torch.nn as nn
    import vt_lib as V
    if threads:
        torch.set_num_threads(threads)
    net = V.build_net("old", cap, 0, add="setup", dropout=0.0)
    net.eval()
    macs = [0]
    hs = []
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            def h(m, i, o, _m=macs):
                _m[0] += int(m.in_features) * int(m.out_features) * \
                    int(o.numel() // o.shape[-1])
            hs.append(mod.register_forward_hook(h))
    B = 4096
    with torch.no_grad():
        net(ft.batch(idx[:B]))          # one pass counts the MACs
    for h in hs:
        h.remove()
    per_sample = macs[0] / B
    # timing: pre-gather the batch so we time the NET, not the mmap
    batches = [ft.batch(idx[i * B:(i + 1) * B]) for i in range(4)]
    with torch.no_grad():
        for b in batches:
            net(b)                      # warm
        t0 = time.time()
        for _ in range(reps):
            for b in batches:
                net(b)
        dt = time.time() - t0
    rows = reps * len(batches) * B
    return {"cap": list(cap),
            "params": int(sum(p.numel() for p in net.parameters())),
            "macs_per_sample": int(per_sample),
            "rows_per_s": round(rows / dt, 1),
            "torch_threads": torch.get_num_threads(),
            "batch": B, "rows_timed": rows, "wall_s": round(dt, 3)}


def bench(work, threads):
    d, V = _load(work)
    ft = V.Arm("old", add="setup")
    hold = np.load(os.path.join(d, "holdout_i.npy")).astype(np.int64)
    out = {"note": "MACs = Linear multiply-accumulates per sample (embeddings "
                   "are gathers, 0 MACs). rows/s is torch CPU at batch 4096.",
           "nets": {}}
    for cap in (CAP_N, CAP_WIDE):
        r = _macs_and_rate(cap, ft, hold, threads)
        out["nets"]["%dx%d" % cap] = r
        print(json.dumps(r), flush=True)
    a = out["nets"]["128x256"]
    b = out["nets"]["256x512"]
    out["ratio_wide_over_narrow"] = {
        "params": round(b["params"] / a["params"], 3),
        "macs": round(b["macs_per_sample"] / a["macs_per_sample"], 3),
        "narrow_rows_per_s_over_wide": round(a["rows_per_s"] / b["rows_per_s"], 3)}
    json.dump(out, open(os.path.join(work, "REPORT.n.bench.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "pick", "bench"])
    ap.add_argument("work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mon-hid", type=int, default=CAP_N[0], dest="mh")
    ap.add_argument("--trunk-hid", type=int, default=CAP_N[1], dest="th")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=1000, dest="ee")
    ap.add_argument("--swap-aug", action="store_true", help="v8c: per-row 50%% "
                    "side-swap with flipped label (exact-antisymmetry training)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "train":
        train(a.work, a.seed, a.steps, a.threads, a.ee, cap=(a.mh, a.th),
              wd=a.wd, swap_aug=a.swap_aug)
    elif a.cmd == "bench":
        bench(a.work, a.threads)
    else:
        pick(a.work, [int(x) for x in a.seeds.split(",")])


if __name__ == "__main__":
    main()
