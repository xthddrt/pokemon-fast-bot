"""END-TO-END TRAINING CANARY -- 1 % pretrain + 1 % fine-tune, the REAL pipeline.

The gate before any paid GPU training. Everything here is the code the full run
will use: `vt_lib.Arm` mmaps the published caches, `vt_lib.build_net` builds the
adopted "arm A + setup" net, and the two stages use RECIPE_DERIVATION.md's knobs
(only the batch size is scaled down for the 1 % subsets, and the learning rate
is moved with it by the derivation's OWN saturating law
eps(B) = eps_max / (1 + B_noise/B), B_noise = 648 -- not by any folklore rule).

  python vt_canary.py fetch   <workdir>          # build the 1 % local caches
  python vt_canary.py train   <workdir>          # pretrain -> fine-tune -> eval
  python vt_canary.py verify  <workdir>          # 20+20 re-encode join checks
  python vt_canary.py bench   <workdir>          # throughput for the projection
  python vt_canary.py selftest <workdir>         # synthetic cache, no S3, no wheel

Every stage writes <workdir>/REPORT.<stage>.json.
"""
import argparse
import gzip
import io
import json
import math
import os
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402

BUCKET = os.environ.get("CANARY_BUCKET", "pokebot-valuenet-389825051723")
P_PLC1 = "evallab/enc_plc1/"
P_PRE1 = "evallab/enc_pre1/"
P_SEL = "evallab/pre1_sel/"
P_STATES = "evallab/plc1/positions.jsonl.gz"

ARRAYS = ["old_a1_f", "old_a1_ids", "old_a2_f", "old_a2_ids",
          "old_b1_f", "old_b1_ids", "old_b2_f", "old_b2_ids",
          "old_sf1", "old_sf2", "old_g", "addon_mon"]

# ---- RECIPE (RECIPE_DERIVATION.md) -----------------------------------------
B_NOISE = 648.0
REC = {
    "pre": dict(batch=16384, lr=2e-4, lr_emb=2e-4, wd=5e-3, wd_emb=0.0,
                dropout=0.0, warm_frac=0.037, cos_floor=0.1),   # 1000/27000
    "ft":  dict(batch=4096, lr=1e-4, lr_emb=1e-4 / 3, wd=2e-3, wd_emb=0.0,
                dropout=0.0, warm_frac=0.167, cos_floor=0.0),   # 1000/6000
}
CAP = (256, 512)
BANDS = ("early", "mid", "late")
NOISE_FLOOR = {"all": 0.013504, "early": 0.0173, "mid": 0.0130, "late": 0.0069}


def lr_scale(b_ref, b_new):
    """The derivation's saturating law, used in BOTH directions."""
    return (1.0 + B_NOISE / b_ref) / (1.0 + B_NOISE / b_new)


# =============================================================== S3 fetching ==
def s3c():
    import boto3
    return boto3.client("s3")


def npy_header(cli, key):
    raw = cli.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-511")["Body"].read()
    f = io.BytesIO(raw)
    ver = np.lib.format.read_magic(f)
    if ver == (1, 0):
        shape, _, dt = np.lib.format.read_array_header_1_0(f)
    else:
        shape, _, dt = np.lib.format.read_array_header_2_0(f)
    return shape, dt, f.tell()


def fetch_blocks(cli, key, out_path, blocks):
    """Range-GET the given (start, count) row blocks of an .npy into a local .npy."""
    shape, dt, hdr = npy_header(cli, key)
    row_b = int(np.prod(shape[1:])) * dt.itemsize
    n = sum(c for _, c in blocks)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=dt,
                                    shape=(n,) + tuple(shape[1:]))
    off = 0
    for st, c in blocks:
        lo = hdr + st * row_b
        buf = cli.get_object(Bucket=BUCKET, Key=key,
                             Range="bytes=%d-%d" % (lo, lo + c * row_b - 1))["Body"].read()
        assert len(buf) == c * row_b, (key, st, c, len(buf))
        arr[off:off + c] = np.frombuffer(buf, dt).reshape((c,) + tuple(shape[1:]))
        off += c
    arr.flush()
    del arr
    return shape[0], n


def fetch(work, n_blocks=64, block_rows=1250):
    """plc1: full cache (2.7 GB, needed whole because the holdout is scattered).
    pre1: `n_blocks` evenly spaced contiguous row blocks = 1 % of the corpus."""
    cli = s3c()
    t0 = time.time()
    d_ft = os.path.join(work, "enc_plc1")
    d_pre = os.path.join(work, "enc_pre1_sub")
    os.makedirs(d_ft, exist_ok=True)
    os.makedirs(d_pre, exist_ok=True)

    # ---- plc1, whole ----
    import subprocess
    for k in ARRAYS + ["addon_layout.json", "meta.npz", "holdout_i.npy", "split.json"]:
        key = P_PLC1 + (k if k.endswith((".json", ".npz", ".npy")) else k + ".npy")
        dst = os.path.join(d_ft, os.path.basename(key))
        if not os.path.exists(dst):
            subprocess.check_call(["aws", "s3", "cp", "s3://%s/%s" % (BUCKET, key), dst,
                                   "--only-show-errors"])
    t_ft = time.time() - t0

    # ---- pre1, 1 % in evenly spaced blocks ----
    shape, _, _ = npy_header(cli, P_PRE1 + "old_a1_f.npy")
    N = int(shape[0])
    starts = [int(round(k * (N - block_rows) / (n_blocks - 1))) for k in range(n_blocks)]
    blocks = [(s, block_rows) for s in starts]
    gidx = np.concatenate([np.arange(s, s + c) for s, c in blocks]).astype(np.int64)
    assert len(gidx) == len(np.unique(gidx)), "blocks overlap"
    np.save(os.path.join(d_pre, "gidx.npy"), gidx)

    from concurrent.futures import ThreadPoolExecutor
    t1 = time.time()
    with ThreadPoolExecutor(12) as ex:
        futs = {k: ex.submit(fetch_blocks, s3c(), P_PRE1 + k + ".npy",
                             os.path.join(d_pre, k + ".npy"), blocks) for k in ARRAYS}
        got = {k: f.result() for k, f in futs.items()}
    assert len({v[0] for v in got.values()}) == 1, got
    assert len({v[1] for v in got.values()}) == 1, got

    for k in ["addon_layout.json"]:
        cli.download_file(BUCKET, P_PRE1 + k, os.path.join(d_pre, k))
    mp = os.path.join(work, "pre1_meta_full.npz")
    if not os.path.exists(mp):
        cli.download_file(BUCKET, P_PRE1 + "meta.npz", mp)
    z = np.load(mp, allow_pickle=False)
    np.savez_compressed(os.path.join(d_pre, "meta.npz"),
                        **{k: z[k][gidx] for k in z.files})
    rep = {"pre1_rows_total": N, "pre1_rows_sub": len(gidx), "pre1_frac": len(gidx) / N,
           "pre1_blocks": n_blocks, "pre1_block_rows": block_rows,
           "plc1_dl_s": round(t_ft, 1), "pre1_dl_s": round(time.time() - t1, 1),
           "plc1_dir": d_ft, "pre1_dir": d_pre}
    json.dump(rep, open(os.path.join(work, "REPORT.fetch.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ================================================================== training ==
def arm(d):
    import vt_lib as V
    V.ENC = d
    return V.Arm("old", add="setup")


def groups(net):
    emb = [p for n, p in net.named_parameters() if "emb." in n]
    ids = {id(p) for p in emb}
    dense = [p for p in net.parameters() if id(p) not in ids]
    return emb, dense


def sched(step, total, warm, floor):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def predict(net, data, idx, batch=4096):
    import torch
    net.eval()
    out = np.empty(len(idx), np.float64)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            out[i:i + len(b)] = torch.sigmoid(net(data.batch(b))).numpy()
    return out


def brier_bands(pred, y, band):
    r = {"all": float(np.mean((pred - y) ** 2))}
    for b in BANDS:
        m = band == b
        r[b] = float(np.mean((pred[m] - y[m]) ** 2))
    return r


def paired(pred_a, pred_b, y):
    """mean Brier(a) - Brier(b) with the PAIRED standard error."""
    d = (pred_a - y) ** 2 - (pred_b - y) ** 2
    return {"delta": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
            "z": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))}


def run_stage(net, data, tr, va, y, w, cfg, batch, epochs, patience, tag,
              val_fn, rng, log):
    """One training stage. Returns history + the gradient census."""
    import torch
    import torch.nn as nn
    emb, dense = groups(net)
    sc = lr_scale(cfg["batch"], batch)
    lr_d, lr_e = cfg["lr"] * sc, cfg["lr_emb"] * sc
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": lr_d, "weight_decay": cfg["wd"]},
         {"params": emb, "lr": lr_e, "weight_decay": cfg["wd_emb"]}])
    spe = max(1, int(math.ceil(len(tr) / batch)))
    total = spe * epochs
    warm = max(1, int(round(cfg["warm_frac"] * total)))
    sp_emb = net.base.emb["species"].weight
    gacc = np.zeros(sp_emb.shape[0], np.float64)
    gn = []
    nonfinite = 0
    hist = []
    best = (1e18, None, -1)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        tot, seen = 0.0, 0
        for i in range(0, len(order), batch):
            b = np.sort(tr[order[i:i + batch]])
            m = sched(step, total, warm, cfg["cos_floor"])
            opt.param_groups[0]["lr"] = lr_d * m
            opt.param_groups[1]["lr"] = lr_e * m
            z = net(data.batch(b))
            ls = nn.functional.binary_cross_entropy_with_logits(
                z, y[b], reduction="none")
            loss = (ls * w[b]).mean() if w is not None else ls.mean()
            if not torch.isfinite(loss):
                nonfinite += 1
            opt.zero_grad()
            loss.backward()
            g = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
            gn.append(g)
            if not math.isfinite(g):
                nonfinite += 1
            if sp_emb.grad is not None:
                gacc += sp_emb.grad.abs().sum(1).detach().numpy()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
            step += 1
        vv = val_fn(net)
        hist.append({"ep": ep, "train_loss": tot / seen, "val": vv,
                     "lr": lr_d * m, "s": round(time.time() - t0, 1)})
        log("  %s ep%-3d train=%.5f val=%.5f  %.0fs" % (tag, ep, tot / seen, vv,
                                                        time.time() - t0))
        if vv < best[0] - 1e-7:
            best = (vv, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] >= patience:
            break
    net.load_state_dict(best[1])
    gn = np.array(gn)
    cen = {"steps": int(step), "epochs_run": len(hist), "best_epoch": best[2],
           "best_val": best[0], "wall_s": round(time.time() - t0, 1),
           "batch": batch, "lr_dense": lr_d, "lr_emb": lr_e,
           "lr_scale_vs_recipe": sc, "warmup_steps": warm,
           "grad_norm_median": float(np.median(gn)), "grad_norm_max": float(gn.max()),
           "grad_norm_p99": float(np.percentile(gn, 99)),
           "grad_norm_max_over_median": float(gn.max() / np.median(gn)),
           "nonfinite_events": int(nonfinite),
           "rows_per_s": round(step * batch / max(1e-9, time.time() - t0), 1),
           "hist": hist}
    return cen, gacc


def species_counts(data, n):
    c = np.zeros(2048, np.int64)
    for k, s in (("a1_ids", None), ("a2_ids", None), ("b1_ids", 1), ("b2_ids", 1)):
        a = np.asarray(data.a[k][:n])
        v = (a[:, 0] if s is None else a[:, :, 0]).reshape(-1).astype(np.int64)
        np.add.at(c, v, 1)
    return c


def train(work, pre_epochs=60, ft_epochs=80, seed=0, threads=0):
    import torch
    import vt_lib as V
    if threads:
        torch.set_num_threads(threads)
    log = lambda s: print(s, flush=True)  # noqa: E731
    d_ft = os.path.join(work, "enc_plc1")
    d_pre = os.path.join(work, "enc_pre1_sub")

    pre = arm(d_pre)
    mpre = dict(np.load(os.path.join(d_pre, "meta.npz"), allow_pickle=False))
    ft = arm(d_ft)
    mft = dict(np.load(os.path.join(d_ft, "meta.npz"), allow_pickle=False))
    V.ENC = d_ft
    net = V.build_net("old", CAP, seed, add="setup", dropout=REC["ft"]["dropout"])
    nparam = sum(p.numel() for p in net.parameters())
    log("net params = %d" % nparam)

    # ---------------- pretrain split: BY GAME, 95/5 -------------------------
    n_pre = len(mpre["y"])
    gm = mpre["g"].astype(str)
    ug = np.unique(gm)
    rg = np.random.default_rng(seed + 11)
    perm = rg.permutation(len(ug))
    hold_g = set(ug[perm[:max(1, len(ug) // 20)]].tolist())
    is_v = np.array([g in hold_g for g in gm])
    tr_p, va_p = np.flatnonzero(~is_v), np.flatnonzero(is_v)
    yp = torch.from_numpy(mpre["y"].astype(np.float32))
    wp_np = mpre["w"].astype(np.float64)
    wp_np = wp_np / wp_np[tr_p].mean()
    wp = torch.from_numpy(wp_np.astype(np.float32))

    # ---------------- fine-tune split ---------------------------------------
    hold = np.load(os.path.join(d_ft, "holdout_i.npy")).astype(np.int64)
    n_ft = len(mft["label_p"])
    mask = np.ones(n_ft, bool)
    mask[hold] = False
    pool = np.flatnonzero(mask)
    rf = np.random.default_rng(seed + 23)
    sub = np.sort(rf.choice(pool, size=int(round(0.01 * len(pool))), replace=False))
    k = int(round(0.9 * len(sub)))
    pm = rf.permutation(len(sub))
    tr_f, va_f = np.sort(sub[pm[:k]]), np.sort(sub[pm[k:]])
    yf = torch.from_numpy(mft["label_p"].astype(np.float32))
    band_h = mft["band"][hold].astype(str)
    y_h = mft["label_p"][hold]

    log("pretrain rows %d (train %d / val %d, %d games)  fine-tune rows %d "
        "(train %d / val %d)  holdout %d"
        % (n_pre, len(tr_p), len(va_p), len(ug), len(sub), len(tr_f), len(va_f), len(hold)))

    # ---------------- baselines + init --------------------------------------
    p_const = float(mft["label_p"][tr_f].mean())
    res = {"n_params": nparam, "n_pre": int(n_pre), "n_pre_train": int(len(tr_p)),
           "n_pre_games": int(len(ug)), "n_ft_sub": int(len(sub)),
           "n_ft_train": int(len(tr_f)), "n_holdout": int(len(hold)),
           "const_p": p_const, "recipe": REC, "cap": list(CAP), "seed": seed,
           "torch_threads": torch.get_num_threads()}
    pr_init = predict(net, ft, hold)
    res["holdout_brier"] = {"init": brier_bands(pr_init, y_h, band_h),
                            "constant": brier_bands(np.full(len(hold), p_const), y_h, band_h),
                            "q_search": brier_bands(mft["q_search"][hold], y_h, band_h)}
    res["noise_floor"] = NOISE_FLOOR
    log("baselines: const %.5f  q_search %.5f  init %.5f"
        % (res["holdout_brier"]["constant"]["all"], res["holdout_brier"]["q_search"]["all"],
           res["holdout_brier"]["init"]["all"]))

    # ---------------- STAGE 1: pretrain -------------------------------------
    import torch.nn as nn

    def val_pre(n):
        n.eval()
        tot, seen = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_p), 4096):
                b = va_p[i:i + 4096]
                z = n(pre.batch(b))
                ls = nn.functional.binary_cross_entropy_with_logits(
                    z, yp[b], reduction="none")
                tot += float((ls * wp[b]).sum())
                seen += len(b)
        return tot / seen

    b_pre = max(256, int(REC["pre"]["batch"] * 0.0625))       # 16,384 -> 1,024
    cen_p, gacc_p = run_stage(net, pre, tr_p, va_p, yp, wp, REC["pre"], b_pre,
                              pre_epochs, 5, "PRE", val_pre,
                              np.random.default_rng(seed + 101), log)
    res["pretrain"] = cen_p
    pr_pre = predict(net, ft, hold)
    res["holdout_brier"]["after_pretrain"] = brier_bands(pr_pre, y_h, band_h)
    torch.save({"sd": net.state_dict()}, os.path.join(work, "ckpt_pretrain.pt"))
    log("after pretrain: holdout Brier %.5f" % res["holdout_brier"]["after_pretrain"]["all"])

    # rare-species gradient census, measured on the PRETRAIN stage
    sc = species_counts(pre, n_pre)
    present = np.flatnonzero(sc > 0)
    order = present[np.argsort(sc[present])]
    rare = order[:25]
    res["rare_species_grad"] = {
        "n_species_present": int(len(present)),
        "rarest_25_ids": [int(x) for x in rare],
        "rarest_25_counts": [int(sc[x]) for x in rare],
        "rarest_25_absgrad": [float(gacc_p[x]) for x in rare],
        "n_rarest_25_with_zero_grad": int((gacc_p[rare] == 0).sum()),
        "n_present_with_zero_grad": int((gacc_p[present] == 0).sum()),
        "min_absgrad_over_present": float(gacc_p[present].min()),
        "median_absgrad_over_present": float(np.median(gacc_p[present])),
        "absent_rows_grad_sum": float(gacc_p[np.setdiff1d(
            np.arange(len(gacc_p)), present)].sum())}

    # ---------------- STAGE 2: fine-tune ------------------------------------
    def val_ft(n):
        p = predict(n, ft, va_f)
        return float(np.mean((p - mft["label_p"][va_f]) ** 2))

    b_ft = max(128, int(REC["ft"]["batch"] * 0.125))          # 4,096 -> 512
    cen_f, gacc_f = run_stage(net, ft, tr_f, va_f, yf, None, REC["ft"], b_ft,
                              ft_epochs, 10, "FT", val_ft,
                              np.random.default_rng(seed + 202), log)
    res["finetune"] = cen_f
    pr_ft = predict(net, ft, hold)
    res["holdout_brier"]["after_finetune"] = brier_bands(pr_ft, y_h, band_h)
    torch.save({"sd": net.state_dict()}, os.path.join(work, "ckpt_finetune.pt"))

    # ---------------- comparisons -------------------------------------------
    cp = np.full(len(hold), p_const)
    qs = mft["q_search"][hold]
    res["paired"] = {
        "ft_vs_const": paired(pr_ft, cp, y_h),
        "ft_vs_q": paired(pr_ft, qs, y_h),
        "ft_vs_pretrain": paired(pr_ft, pr_pre, y_h),
        "pretrain_vs_const": paired(pr_pre, cp, y_h)}
    for b in BANDS:
        m = band_h == b
        res["paired"]["ft_vs_const_" + b] = paired(pr_ft[m], cp[m], y_h[m])
        res["paired"]["ft_vs_q_" + b] = paired(pr_ft[m], qs[m], y_h[m])

    # ---------------- finite / sanity census --------------------------------
    bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
    res["nan_census"] = {
        "nonfinite_params": bad,
        "pred_nonfinite": int((~np.isfinite(pr_ft)).sum()),
        "pred_min": float(pr_ft.min()), "pred_max": float(pr_ft.max()),
        "pred_mean": float(pr_ft.mean()), "pred_std": float(pr_ft.std())}
    res["loss_went_down"] = {
        "pretrain_train_first": cen_p["hist"][0]["train_loss"],
        "pretrain_train_last": cen_p["hist"][-1]["train_loss"],
        "pretrain_val_first": cen_p["hist"][0]["val"],
        "pretrain_val_best": cen_p["best_val"],
        "finetune_train_first": cen_f["hist"][0]["train_loss"],
        "finetune_train_last": cen_f["hist"][-1]["train_loss"],
        "finetune_val_first": cen_f["hist"][0]["val"],
        "finetune_val_best": cen_f["best_val"]}
    np.save(os.path.join(work, "holdout_pred_ft.npy"), pr_ft.astype(np.float32))
    np.save(os.path.join(work, "holdout_pred_pre.npy"), pr_pre.astype(np.float32))
    json.dump(res, open(os.path.join(work, "REPORT.train.json"), "w"), indent=1)
    log(json.dumps({k: res[k] for k in ("holdout_brier", "paired", "loss_went_down",
                                        "nan_census", "rare_species_grad")}, indent=1))
    return res


# ================================================================ throughput ==
def bench(work, iters=10):
    import torch
    d_pre = os.path.join(work, "enc_pre1_sub")
    d_ft = os.path.join(work, "enc_plc1")
    import vt_lib as V
    pre = arm(d_pre)
    V.ENC = d_ft
    net = V.build_net("old", CAP, 0, add="setup")
    n = pre.a["a1_f"].shape[0]
    emb, dense = groups(net)
    opt = torch.optim.AdamW([{"params": dense, "lr": 1e-4, "weight_decay": 5e-3},
                             {"params": emb, "lr": 1e-4, "weight_decay": 0.0}])
    import torch.nn as nn
    rng = np.random.default_rng(7)
    out = []
    hw = os.cpu_count()
    for th in sorted({8, 16, 32, hw}):
        if th > hw:
            continue
        torch.set_num_threads(th)
        for B in (4096, 16384):
            if B > n:
                continue
            # warm-up
            for _ in range(2):
                b = np.sort(rng.choice(n, B, replace=False))
                d = pre.batch(b)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    net(d), torch.zeros(B))
                opt.zero_grad(); loss.backward(); opt.step()
            t_gather = t_step = 0.0
            t0 = time.time()
            for _ in range(iters):
                b = np.sort(rng.choice(n, B, replace=False))
                ta = time.time()
                d = pre.batch(b)
                tb = time.time()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    net(d), torch.zeros(B))
                opt.zero_grad(); loss.backward(); opt.step()
                t_gather += tb - ta
                t_step += time.time() - tb
            el = time.time() - t0
            out.append({"threads": th, "batch": B, "iters": iters,
                        "s_per_step": el / iters, "rows_per_s": iters * B / el,
                        "gather_frac": t_gather / el, "compute_frac": t_step / el})
            print("  threads=%-3d B=%-6d %.4f s/step  %.0f rows/s (gather %.0f%%)"
                  % (th, B, el / iters, iters * B / el, 100 * t_gather / el), flush=True)
    # inference-only rate (what an MCTS leaf costs, and what eval passes cost)
    torch.set_num_threads(hw)
    net.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            b = np.sort(rng.choice(n, 4096, replace=False))
            torch.sigmoid(net(pre.batch(b)))
    inf = iters * 4096 / (time.time() - t0)
    rep = {"host_vcpu": hw, "torch": torch.__version__, "train": out,
           "infer_rows_per_s_4096": inf,
           "params": sum(p.numel() for p in net.parameters())}
    json.dump(rep, open(os.path.join(work, "REPORT.bench.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


# ============================================================ join verification =
def _cmp_rows(sub_dir, sub_rows, tmpdir):
    """Bit-for-bit compare freshly encoded rows against the stored cache rows."""
    bad, checked = [], 0
    for k in ARRAYS:
        a = np.load(os.path.join(tmpdir, k + ".npy"), mmap_mode="r")
        b = np.load(os.path.join(sub_dir, k + ".npy"), mmap_mode="r")
        got = np.asarray(a)
        want = np.asarray(b[sub_rows])
        checked += got.size
        if not np.array_equal(got.view(np.uint16 if got.dtype == np.float16 else got.dtype),
                              want.view(np.uint16 if want.dtype == np.float16 else want.dtype)):
            d = np.flatnonzero((got != want).reshape(len(sub_rows), -1).any(1))
            bad.append({"array": k, "n_bad_rows": int(len(d)), "first": int(d[0])})
    return bad, checked


def verify(work, k=20, seed=5):
    import subprocess
    import enc_adopted as EA
    d_ft = os.path.join(work, "enc_plc1")
    d_pre = os.path.join(work, "enc_pre1_sub")
    rep = {}

    # ---------------- plc1: independent source of truth = positions.jsonl ----
    mft = dict(np.load(os.path.join(d_ft, "meta.npz"), allow_pickle=False))
    n = len(mft["label_p"])
    rows = np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    src = os.path.join(work, "positions.jsonl.gz")
    if not os.path.exists(src):
        subprocess.check_call(["aws", "s3", "cp", "s3://%s/%s" % (BUCKET, P_STATES),
                               src, "--only-show-errors"])
    want = set(int(x) for x in rows)
    hi = int(rows[-1])
    got = {}
    t0 = time.time()
    with gzip.open(src, "rb") as f:
        for i, line in enumerate(f):
            if i in want:
                got[i] = json.loads(line)
                if i == hi:
                    break
    assert len(got) == k, (len(got), k)
    sel = os.path.join(work, "plc1_sel.jsonl")
    with open(sel, "w") as o:
        for r in rows:
            o.write(json.dumps(got[int(r)]) + "\n")
    gmis = [int(r) for r in rows if got[int(r)]["g"] != str(mft["g"][int(r)])]
    tmp = os.path.join(work, "reenc_plc1")
    EA.encode(sel, tmp, workers=1, check=0)
    bad, checked = _cmp_rows(d_ft, rows, tmp)
    # label side: the label file's own `i` is the line number, checked at build
    # time for all 1M rows; here we re-assert it for the sampled rows.
    rep["plc1"] = {"k": k, "rows": [int(x) for x in rows], "scan_s": round(time.time() - t0, 1),
                   "game_id_mismatches": gmis, "arrays_mismatched": bad,
                   "values_compared": int(checked),
                   "label_i_matches": bool(np.array_equal(mft["i"][rows], rows)),
                   "label_p": [float(mft["label_p"][int(r)]) for r in rows[:5]]}
    print("plc1 verify:", json.dumps(rep["plc1"])[:400], flush=True)

    # ---------------- pre1: source of truth = the selector's own shard files --
    mp = dict(np.load(os.path.join(d_pre, "meta.npz"), allow_pickle=False))
    m = len(mp["y"])
    srows = np.sort(np.random.default_rng(seed + 1).choice(m, k, replace=False))
    gs = mp["g"].astype(str)
    need = {}
    for r in srows:
        shard = gs[int(r)].split("/")[1]
        need.setdefault(shard, []).append(int(r))
    recs = {}
    for shard, rs in need.items():
        p = os.path.join(work, shard + ".pt.jsonl.gz")
        if not os.path.exists(p):
            subprocess.check_call(["aws", "s3", "cp",
                                   "s3://%s/%s%s.pt.jsonl.gz" % (BUCKET, P_SEL, shard),
                                   p, "--only-show-errors"])
        keys = {(gs[r], int(mp["t"][r])): r for r in rs}
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                kk = (d["g"], int(d["t"]))
                if kk in keys:
                    recs[keys[kk]] = d
    miss = [int(r) for r in srows if r not in recs]
    sel2 = os.path.join(work, "pre1_sel.jsonl")
    with open(sel2, "w") as o:
        for r in srows:
            o.write(json.dumps(recs[int(r)]) + "\n")
    tmp2 = os.path.join(work, "reenc_pre1")
    EA.encode(sel2, tmp2, workers=1, check=0)
    bad2, checked2 = _cmp_rows(d_pre, srows, tmp2)
    ymis = [int(r) for r in srows
            if abs(float(recs[int(r)]["y"]) - float(mp["y"][int(r)])) > 1e-9
            or abs(float(recs[int(r)]["w"]) - float(mp["w"][int(r)])) > 1e-9]
    rep["pre1"] = {"k": k, "rows": [int(x) for x in srows], "missing_records": miss,
                   "arrays_mismatched": bad2, "values_compared": int(checked2),
                   "target_mismatches": ymis,
                   "shards_touched": sorted(need)}
    print("pre1 verify:", json.dumps(rep["pre1"])[:400], flush=True)
    rep["PASS"] = bool(not gmis and not bad and not bad2 and not miss and not ymis
                       and rep["plc1"]["label_i_matches"])
    json.dump(rep, open(os.path.join(work, "REPORT.verify.json"), "w"), indent=1)
    assert rep["PASS"], "JOIN VERIFICATION FAILED"
    return rep


# =================================================================== selftest ==
def selftest(work):
    """Synthetic caches with the real shapes -- exercises every code path in
    train() without S3, the wheel, or the real corpora."""
    from encoder import NUM_MON_FEATS, N_IDS
    rng = np.random.default_rng(0)
    for name, n, meta in (("enc_pre1_sub", 4000, "pre"), ("enc_plc1", 6000, "ft")):
        d = os.path.join(work, name)
        os.makedirs(d, exist_ok=True)
        sh = {"old_a1_f": (n, NUM_MON_FEATS), "old_a2_f": (n, NUM_MON_FEATS),
              "old_b1_f": (n, 5, NUM_MON_FEATS), "old_b2_f": (n, 5, NUM_MON_FEATS),
              "old_sf1": (n, 99), "old_sf2": (n, 99), "old_g": (n, 18),
              "old_a1_ids": (n, N_IDS), "old_a2_ids": (n, N_IDS),
              "old_b1_ids": (n, 5, N_IDS), "old_b2_ids": (n, 5, N_IDS),
              "addon_mon": (n, 12, 14)}
        sig = rng.normal(size=n)
        for k, s in sh.items():
            if "ids" in k:
                a = rng.integers(1, 64, size=s).astype(np.int16)
            else:
                a = rng.normal(size=s).astype(np.float16)
                a[..., 0] = sig.reshape([-1] + [1] * (len(s) - 2)).astype(np.float16)
            np.save(os.path.join(d, k + ".npy"), a)
        json.dump({"D_mon": 14, "D_rest": 0, "groups": {"mon.setup": [0, 14]},
                   "mon_names": ["x%d" % i for i in range(14)], "rest_names": []},
                  open(os.path.join(d, "addon_layout.json"), "w"))
        p = 1 / (1 + np.exp(-sig))
        if meta == "pre":
            np.savez(os.path.join(d, "meta.npz"), y=p, w=np.ones(n),
                     g=np.array(["g%04d" % (i // 8) for i in range(n)]),
                     t=np.arange(n) % 8, i=np.arange(n))
        else:
            np.savez(os.path.join(d, "meta.npz"), label_p=p, q_search=0.5 + 0.1 * sig,
                     se=np.full(n, 0.1), band=np.array(list(BANDS) * (n // 3)),
                     g=np.array(["h%05d" % i for i in range(n)]), i=np.arange(n))
            np.save(os.path.join(d, "holdout_i.npy"),
                    np.arange(n - 1000, n).astype(np.int32))
    r = train(work, pre_epochs=3, ft_epochs=3, threads=2)
    print("SELFTEST OK", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "train", "bench", "verify", "selftest", "all"])
    ap.add_argument("work")
    ap.add_argument("--pre-epochs", type=int, default=60, dest="pe")
    ap.add_argument("--ft-epochs", type=int, default=80, dest="fe")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--block-rows", type=int, default=1250, dest="br")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.cmd in ("fetch", "all"):
        fetch(a.work, a.blocks, a.br)
    if a.cmd in ("verify", "all"):
        verify(a.work)
    if a.cmd in ("train", "all"):
        train(a.work, a.pe, a.fe, threads=a.threads)
    if a.cmd in ("bench", "all"):
        bench(a.work)
    if a.cmd == "selftest":
        selftest(a.work)


if __name__ == "__main__":
    main()
