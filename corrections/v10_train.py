"""V10 TRAINING — single-corpus, single-trunk, mirror-augmented (Sally 2026-08-19).

Deliberately NOT v9_train.py. That script is built around TWO corpora
(enc_plc12 + fresh v9) and the 3-trunk phase net, and its control arm won, so
production `v9c0.bin` IS the single-trunk control. Rather than strip a
two-corpus phase-blend script down, this is the minimal thing that trains the
shipped architecture on one corpus.

    python v10_train.py --enc /root/v10/enc --seed 0 --steps 60000 --tag v10s0

WHAT THIS DOES AND WHY

Data: ONE encoded corpus (enc_adopted.py output). Labels come straight from
`meta.npz["y"]` -- the v10 playout posterior. No blending, no relabeling.

MIRROR AUGMENTATION (the point of this run). One epoch = every row exactly
once in its original seating AND exactly once side-swapped, shuffled together.
Not a random half-batch: a virtual index over [0, 2n) with the high half
meaning "mirrored". `vt_lib.swap_rows_` flips every side-scoped feature and the
label becomes 1 - y. This is what killed the measured 53.4% side-one bias.
Honest caveat, from mine_value.py:84 -- the label player is measurably
side-asymmetric (mean 0.027 antisymmetry deviation), so 1 - y is a very good
approximation, not ground truth.

HOLDOUT IS SPLIT BY GAME, NOT BY ROW. Up to 3 rows are mined per game and rows
from one game share a team pair and a trajectory, so a row-level split leaks
across the boundary and flatters the holdout. `meta.npz["g"]` carries the game
seed; whole games go to one side or the other.

Selection = best fresh-holdout Brier, checkpointed, same as v9.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")


def load_enc(vt_lib, np, torch, dev, enc_dir):
    """Load one encoded corpus onto the device. Mirrors v9_train.load_enc:
    ids are int64 (embedding indices), everything else float32."""
    os.environ["VT_ENC"] = enc_dir
    import importlib
    importlib.reload(vt_lib)
    arm = vt_lib.Arm("old", add="setup")
    n = np.load(os.path.join(enc_dir, "old_a1_f.npy"), mmap_mode="r").shape[0]
    g = {}
    for k in vt_lib.OLD_KEYS:
        # np.array(copy) not asarray: enc output is mmap-backed and read-only,
        # and torch.from_numpy on a non-writable buffer warns + is UB to write.
        g[k] = torch.from_numpy(np.array(arm.a[k])).to(
            dev, torch.int64 if "ids" in k else torch.float32)
    g["am"] = torch.from_numpy(
        np.array(arm.am[:, :, arm.mcol], np.float32)).to(dev)
    return g, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, help="encoded corpus dir (enc_adopted.py output)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--holdout-frac", type=float, default=0.02,
                    help="fraction of GAMES (not rows) held out")
    ap.add_argument("--tag", default="v10s0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label-key", default="y",
                    help="meta.npz column to train on (e.g. y11 for max-CV labels)")
    ap.add_argument("--weight-key", default="",
                    help="optional meta.npz column of per-row loss weights")
    a = ap.parse_args()
    # vt_lib resolves VT_ENC from its own module dir on reload, so a relative
    # --enc silently becomes evallab/<rel>. Pin it absolute at the boundary.
    a.enc = os.path.abspath(a.enc)

    sys.path.insert(0, LAB)
    import numpy as np
    import torch
    import vt_lib

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={dev} enc={a.enc} seed={a.seed}", flush=True)

    g, n = load_enc(vt_lib, np, torch, dev, a.enc)
    meta = np.load(os.path.join(a.enc, "meta.npz"))
    y_np = meta[a.label_key].astype(np.float32)
    # enc_adopted turns a key missing on later rows into nan silently; a nan
    # label destroys the run with no error, so refuse it here.
    assert np.isfinite(y_np).all(), f"non-finite values in label column {a.label_key!r}"
    assert (y_np >= 0.0).all() and (y_np <= 1.0).all(), \
        f"labels outside [0,1] in {a.label_key!r}"
    y_all = torch.from_numpy(y_np).to(dev)
    assert len(y_all) == n, (len(y_all), n)
    w_all = None
    if a.weight_key:
        w_np = meta[a.weight_key].astype(np.float32)
        assert np.isfinite(w_np).all() and (w_np > 0).all(), \
            f"bad weights in {a.weight_key!r}"
        w_all = torch.from_numpy(w_np).to(dev)
    games = meta["g"].astype(np.int64)
    print(f"corpus: {n:,} rows, {len(np.unique(games)):,} distinct games", flush=True)

    # GAME-LEVEL holdout: rows from one game never straddle the split.
    rng_np = np.random.default_rng(20260819)
    uq = np.unique(games)
    rng_np.shuffle(uq)
    n_hold_g = max(1, int(len(uq) * a.holdout_frac))
    hold_games = set(uq[:n_hold_g].tolist())
    is_hold = np.array([gg in hold_games for gg in games])
    tr_idx = torch.from_numpy(np.flatnonzero(~is_hold)).to(dev)
    ho_idx = torch.from_numpy(np.flatnonzero(is_hold)).to(dev)
    ntr, nho = len(tr_idx), len(ho_idx)
    print(f"split: train {ntr:,} rows / holdout {nho:,} rows "
          f"({n_hold_g:,} games held out)", flush=True)

    hg = {k: g[k][ho_idx] for k in g}
    hy = y_all[ho_idx]

    net = vt_lib.build_net("old", (128, 256), a.seed + 1, add="setup").to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"net: {nparam:,} params", flush=True)

    def brier(model, gg, nn_, ys, slices=False):
        with torch.no_grad():
            ps = []
            for i in range(0, nn_, 8192):
                idx = torch.arange(i, min(i + 8192, nn_), device=dev)
                ps.append(torch.sigmoid(model({k: gg[k][idx] for k in gg})))
            e2 = (torch.cat(ps) - ys) ** 2
        if not slices:
            return float(e2.mean())
        ph = torch.from_numpy(meta["ph"].astype(np.float32)).to(dev)[ho_idx]
        out = {"all": round(float(e2.mean()), 6)}
        for nm, lo, hi in (("early", 0, 1/3), ("mid", 1/3, 2/3), ("late", 2/3, 1.01)):
            m = (ph >= lo) & (ph < hi)
            out[nm] = (round(float(e2[m].mean()), 6), int(m.sum())) if int(m.sum()) else None
        return out

    opt = torch.optim.AdamW([
        {"params": [p for nm, p in net.named_parameters() if "emb" in nm],
         "lr": a.lr, "weight_decay": 0.0},
        {"params": [p for nm, p in net.named_parameters() if "emb" not in nm],
         "lr": a.lr, "weight_decay": a.wd},
    ])
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    # FULL-COVERAGE MIRROR EPOCHS: every train row once seated, once mirrored.
    steps_per_epoch = max(1, (2 * ntr) // a.batch)
    print(f"epoch = {2*ntr:,} samples (corpus+mirror) = {steps_per_epoch:,} steps; "
          f"{a.steps} steps ~= {a.steps/steps_per_epoch:.1f} epochs", flush=True)

    rng = np.random.default_rng(a.seed * 7919 + 13)
    best = (1e9, None, -1)
    perm = None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        mlt = (step / a.warmup if step <= a.warmup else
               0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup))))
        for grp in opt.param_groups:
            grp["lr"] = a.lr * mlt
        pos = ((step - 1) % steps_per_epoch) * a.batch
        if pos == 0:
            perm = torch.from_numpy(rng.permutation(2 * ntr)).to(dev)
        v = perm[pos:pos + a.batch]
        mirror = v >= ntr
        rows = tr_idx[torch.where(mirror, v - ntr, v)]
        b = {k: g[k][rows] for k in g}
        yb = y_all[rows]
        vt_lib.swap_rows_(b, mirror)          # flip every side-scoped feature
        yb = torch.where(mirror, 1.0 - yb, yb)
        net.train()
        per = lossf(net(b), yb)
        if w_all is not None:
            wb = w_all[rows]        # mirrored rows keep their weight
            loss = (per * wb).sum() / wb.sum()
        else:
            loss = per.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        lval = float(loss.detach())
        if step % a.eval_every == 0 or step == a.steps:
            net.eval()
            hb = brier(net, hg, nho, hy)
            print(f"  step {step}: loss {lval:.4f} holdout-brier {hb:.6f} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if hb < best[0]:
                best = (hb, {k: v_.detach().cpu().clone()
                             for k, v_ in net.state_dict().items()}, step)
    net.load_state_dict(best[1])
    net.eval()
    print(f"BEST holdout-brier {best[0]:.6f} @ step {best[2]}", flush=True)
    print(f"SLICES {a.tag}: {json.dumps(brier(net, hg, nho, hy, slices=True))}", flush=True)

    out = a.out or os.path.join(os.path.dirname(a.enc.rstrip("/")), f"v10_{a.tag}.pt")
    torch.save({"sd": net.state_dict(),
                "cfg": {"arch": "single-trunk", "seed": a.seed, "enc": a.enc,
                        "steps": a.steps, "batch": a.batch, "lr": a.lr,
                        "label_key": a.label_key, "weight_key": a.weight_key,
                        "holdout_brier": best[0], "best_step": best[2],
                        "mirror": "full-coverage corpus+mirror per epoch"}}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
