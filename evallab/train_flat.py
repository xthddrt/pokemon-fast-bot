"""SMOKE TRAINER for the flat net on lossless features -- and the switchable
old-encoder + pooled baseline, in the same harness.

THIS IS NOT THE EXPERIMENT. The labels here are the existing single-outcome
game results (0/1), which is the loss the lab already showed is the wrong
objective; the real labels come from the playout-averaging agent. What this
proves is that the pipeline RUNS and LEARNS: encoder -> cache -> flat net ->
falling CE against a stated base rate.

TWO ARMS, one code path, one split, one metric:
  ARCH=flat   llencoder features (VARIANT=full|lean) + flatnet.FlatValueNet
  ARCH=old    encoder.py features + labmodel.LabValueNet (bit-identical to the
              shipped train.ValueNet, per labmodel.assert_baseline_parity)
The arm is inferred from the .npz: a "feats" key means the flat cache, an
"a1_ids" key means the incumbent cache. So `train_flat.py A2k_pos.npz` and
`train_flat.py A2k_llfull_pos.npz` are the two arms and nothing else differs.

SPLIT
  By GAME (labenv/train_lab.py convention). Positions inside one game share an
  outcome label, so anything finer leaks it.

BASE RATE
  Every CE is printed next to the CE of the constant predictor fitted on the
  train split. Pair A is unbalanced (README), so an absolute CE means nothing
  on its own.

USAGE
  python train_flat.py <pos.npz> <out.pt> [<hold_pos.npz>]
ENV
  WIDTH DEPTH DROPOUT EPOCHS LR BATCH SEED VAL_FRAC EVALLAB_THREADS
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import labenv  # noqa: F401
import flatnet
from encoder import Vocab  # noqa: E402

WIDTH = int(os.environ.get("WIDTH", "512"))
DEPTH = int(os.environ.get("DEPTH", "3"))
DROPOUT = float(os.environ.get("DROPOUT", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "1024"))
SEED = int(os.environ.get("SEED", "7"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.10"))
THREADS = int(os.environ.get("EVALLAB_THREADS", "4"))
# Keep only plies with ply % PLY_STRIDE == 0. Filtering on the PLY column (not
# on the row index) means the flat cache built with POS_STRIDE=k and the
# incumbent cache built with POS_STRIDE=1 select the IDENTICAL positions, so the
# two arms are apples-to-apples without rebuilding either cache.
PLY_STRIDE = int(os.environ.get("PLY_STRIDE", "1"))

OLD_KEYS = ("a1_ids", "a1_f", "b1_ids", "b1_f", "sf1",
            "a2_ids", "a2_f", "b2_ids", "b2_f", "sf2", "g", "gx", "sx1", "sx2")


def load(path):
    z = np.load(path)
    flat = "feats" in z.files
    t = {}
    for k in z.files:
        a = z[k]
        if k == "feats":
            t[k] = torch.from_numpy(a)
        elif k == "ids" or k in OLD_KEYS:
            t[k] = torch.from_numpy(a.astype(np.int16) if "ids" in k or k == "ids"
                                    else a.astype(np.float16))
        else:
            t[k] = torch.tensor(a, dtype=torch.float32 if a.dtype.kind == "f" else torch.long)
    return t, flat


def batch_of(t, idx, flat):
    if flat:
        return {"ids": t["ids"][idx].long(), "feats": t["feats"][idx].float()}
    return {k: (t[k][idx].long() if "ids" in k else t[k][idx].float()) for k in OLD_KEYS}


def base_rate_ce(p, y):
    """CE of the constant predictor p, evaluated on labels y."""
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(net, t, idx, flat, batch):
    net.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            z = net(batch_of(t, b, flat))
            s += float(nn.functional.binary_cross_entropy_with_logits(
                z, t["y"][b], reduction="sum"))
            n += len(b)
    return s / max(n, 1)


def main():
    torch.set_num_threads(THREADS)
    pos_path, out_path = sys.argv[1], sys.argv[2]
    hold_path = sys.argv[3] if len(sys.argv) > 3 else None
    variant = os.environ.get("VARIANT", "full")

    t, flat = load(pos_path)
    gid = t["gid"].numpy()
    games = np.unique(gid)
    rs = np.random.default_rng(SEED)
    perm = rs.permutation(games)
    n_val = max(1, int(len(games) * VAL_FRAC))
    val_games = set(perm[:n_val].tolist())
    is_val = np.isin(gid, list(val_games))
    keep = (t["ply"].numpy() % PLY_STRIDE) == 0
    tr_idx = torch.tensor(np.flatnonzero(~is_val & keep))
    va_idx = torch.tensor(np.flatnonzero(is_val & keep))

    vocab = Vocab(frozen=True)
    torch.manual_seed(SEED)
    if flat:
        net = flatnet.build("flat", vocab.sizes(), variant, WIDTH, DEPTH, DROPOUT)
        arch = "flat[%s] width=%d depth=%d in=%d" % (variant, WIDTH, DEPTH, net.in_dim)
    else:
        net = flatnet.build("old", vocab.sizes())
        arch = "old(pooled) trunk_in=%d" % net.trunk_in
    nparam = sum(p.numel() for p in net.parameters())

    y_np = t["y"].numpy()
    p_tr = float(y_np[tr_idx.numpy()].mean())
    br_tr = base_rate_ce(p_tr, y_np[tr_idx.numpy()])
    br_va = base_rate_ce(p_tr, y_np[va_idx.numpy()])
    print("ARCH %s  params=%d" % (arch, nparam))
    print("pos=%d (train %d / val %d over %d games)  train base rate p=%.4f"
          % (len(gid), len(tr_idx), len(va_idx), len(games), p_tr))
    print("BASE-RATE CONSTANT PREDICTOR CE: train=%.4f  val=%.4f" % (br_tr, br_va), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + 1)
    hist, best = [], (1e9, None)
    t0 = time.time()
    for ep in range(EPOCHS):
        te = time.time()
        net.train()
        order = torch.tensor(rng.permutation(len(tr_idx)))
        tot, seen = 0.0, 0
        for i in range(0, len(order), BATCH):
            b = tr_idx[order[i:i + BATCH]]
            z = net(batch_of(t, b, flat))
            loss = nn.functional.binary_cross_entropy_with_logits(z, t["y"][b])
            tot += loss.item() * len(b)
            seen += len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ep_s = time.time() - te
        vce = evaluate(net, t, va_idx, flat, BATCH)
        hist.append({"epoch": ep, "train_ce": tot / max(seen, 1), "val_ce": vce,
                     "epoch_s": round(ep_s, 1)})
        print("  ep%-2d train_ce=%.4f val_ce=%.4f  (base %.4f)  %.0fs/epoch"
              % (ep, tot / max(seen, 1), vce, br_va, ep_s), flush=True)
        if vce < best[0]:
            best = (vce, {k: v.detach().clone() for k, v in net.state_dict().items()})

    net.load_state_dict(best[1])
    res = {"out": out_path, "arch": arch, "params": nparam, "flat": flat,
           "variant": variant if flat else None, "base_rate_p": p_tr,
           "base_ce_train": br_tr, "base_ce_val": br_va,
           "val_ce": best[0], "hist": hist,
           "wall_s": round(time.time() - t0, 1)}
    if hold_path and os.path.exists(hold_path):
        th, fh = load(hold_path)
        idx = torch.tensor(np.flatnonzero((th["ply"].numpy() % PLY_STRIDE) == 0))
        hce = evaluate(net, th, idx, fh, BATCH)
        res["hold_ce"] = hce
        res["base_ce_hold"] = base_rate_ce(p_tr, th["y"].numpy())
        res["hold_rows"] = int(len(idx))
        print("HELD-OUT GAMES: ce=%.4f  (base %.4f)  rows=%d"
              % (hce, res["base_ce_hold"], len(idx)), flush=True)
    torch.save({"sd": net.state_dict(), "res": res}, out_path)
    print("saved %s  best val_ce=%.4f  (%.0fs)" % (out_path, best[0], res["wall_s"]))
    print("JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
