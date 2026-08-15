"""ENCODER VALUE TEST -- one training run.

  python vt_train.py --arm enc2|old --cap W[,W2] --seed S [--frac F] [--wse]
                     [--loss bce|mse] --out run.json

TARGET  `label_p`, the measured win rate from 10 clean playouts.
LOSS    BCE with logits against the continuous target. Justification: the label
        is the mean of n = 10 Bernoulli playouts, so n * BCE(sigma(z), label_p)
        IS the negative log-likelihood of those 10 playouts up to a constant --
        the exact likelihood of the data-generating process. It is a proper
        scoring rule, so its population minimiser is the true win rate, which
        is what Brier also scores. `--loss mse` is available as a check.
SELECT  Early stopping and capacity selection on VAL BRIER (the reported
        metric), identically for both arms.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import vt_lib as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["enc2", "old"])
    ap.add_argument("--arch", default="flat", choices=["flat", "shared", "pool"],
                    help="enc2 only: flat = the ENCODER_VALUE_TEST arm-B net; "
                         "shared = shared per-mon encoder + concatenation; "
                         "pool = shared per-mon encoder + sum-pooled bench")
    ap.add_argument("--mon-depth", type=int, default=2, dest="mon_depth")
    ap.add_argument("--depth", type=int, default=3, help="trunk depth")
    ap.add_argument("--no-slot-emb", action="store_true")
    ap.add_argument("--add", default="", help="ADDITIVE arm (--arm old): comma list of "
                    "enc2 derived blocks appended to arm A -- setup,caps,ko,speed,tera. "
                    "Empty = the untouched arm-A control.")
    ap.add_argument("--drop", default="", help="SUBTRACTIVE arm (--arm old): prune arm A's "
                    "own dead columns. 'struct' = dead by construction on any corpus; "
                    "'cov' = struct + groups dead only on THIS one-pair corpus (measurement "
                    "only, NOT a prune recommendation); or a comma list of groups.")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--cap", required=True)          # "128" or "128,256"
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--wse", action="store_true", help="precision weighting 1/(se^2+s0^2)")
    ap.add_argument("--loss", default="bce", choices=["bce", "mse"])
    ap.add_argument("--biasfix", type=int, default=0, help="MECHANISM PROBE: when "
                    "last_item is pruned, restore the per-unit bias spread its 12 "
                    "constant columns supplied at init, and nothing else")
    ap.add_argument("--constok", type=int, default=0, help="MECHANISM PROBE: a free "
                    "learnable K-vector broadcast to every mon (last_item's actual "
                    "function class on this corpus)")
    ap.add_argument("--tag", default="", help="cell name, carried into the result json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    meta = V.load_meta()
    ix, ginfo = V.split_idx(meta, a.frac, sub_seed=0)
    data = V.Arm(a.arm, add=a.add, drop=a.drop)
    cap = tuple(int(x) for x in a.cap.split(","))
    net = V.build_net(a.arm, cap, a.seed, arch=a.arch, mon_depth=a.mon_depth,
                      depth=a.depth, slot_emb=not a.no_slot_emb,
                      dropout=a.dropout, add=a.add, drop=a.drop,
                      biasfix=a.biasfix, constok=a.constok)
    nparam = sum(p.numel() for p in net.parameters())

    y = torch.from_numpy(meta["label_p"].astype(np.float32))
    se = meta["se"].astype(np.float64)
    if a.wse:
        s0 = float(np.median(se[se > 0]))
        w_all = 1.0 / (se ** 2 + s0 ** 2)
        w_all /= w_all[ix["train"]].mean()
        w = torch.from_numpy(w_all.astype(np.float32))
    else:
        w = None

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    rng = np.random.default_rng(a.seed + 1)
    tr = ix["train"]
    best = (1e9, None, -1)
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(len(tr))
        tot, seen = 0.0, 0
        for i in range(0, len(order), a.batch):
            b = np.sort(tr[order[i:i + a.batch]])
            z = net(data.batch(b))
            tb = y[b]
            if a.loss == "bce":
                ls = nn.functional.binary_cross_entropy_with_logits(z, tb, reduction="none")
            else:
                ls = (torch.sigmoid(z) - tb) ** 2
            loss = (ls * w[b]).mean() if w is not None else ls.mean()
            tot += float(loss) * len(b)
            seen += len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = V.predict(net, data, ix["val"])
        vb = V.brier_by_band(pv, meta, ix["val"])
        hist.append({"ep": ep, "train_loss": tot / seen, "val_brier": vb["all"],
                     "s": round(time.time() - t0, 1)})
        print("  ep%-2d loss=%.5f val_brier=%.5f  %.0fs" % (ep, tot / seen, vb["all"],
                                                            time.time() - t0), flush=True)
        if vb["all"] < best[0] - 1e-6:
            best = (vb["all"], {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] >= a.patience:
            break
    net.load_state_dict(best[1])

    pt = V.predict(net, data, ix["test"])
    # SEL / CONF are sub-slices of the same test predictions, so this costs
    # nothing and cannot leak: no decision anywhere reads `conf`.
    pos = {int(r): i for i, r in enumerate(ix["test"])}
    sub = {k: np.array([pos[int(r)] for r in ix[k]]) for k in ("sel", "conf")}
    res = {"arm": a.arm, "cap": cap, "seed": a.seed, "frac": a.frac, "wd": a.wd,
           "add": a.add, "drop": a.drop, "arch": a.arch, "mon_depth": a.mon_depth, "depth": a.depth,
           "slot_emb": int(not a.no_slot_emb), "dropout": a.dropout, "lr": a.lr,
           "batch": a.batch, "biasfix": a.biasfix, "constok": a.constok, "tag": a.tag,
           "loss": a.loss, "wse": a.wse, "params": nparam, "in_dim": getattr(net, "in_dim", None),
           "best_epoch": best[2], "val_brier": best[0],
           "test": V.brier_by_band(pt, meta, ix["test"]),
           "sel": V.brier_by_band(pt[sub["sel"]], meta, ix["sel"]),
           "conf": V.brier_by_band(pt[sub["conf"]], meta, ix["conf"]),
           "val": V.brier_by_band(V.predict(net, data, ix["val"]), meta, ix["val"]),
           "n_train": int(len(tr)), "wall_s": round(time.time() - t0, 1), **ginfo,
           "hist": hist}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    # per-row test predictions, so every stratified analysis (phase, live setup
    # threat, ...) is a local numpy op and needs no re-training
    np.save(a.out.replace(".json", "_pred.npy"), pt.astype(np.float32))
    if a.save:
        torch.save({"sd": net.state_dict(), "res": res}, a.save)
    print("RESULT " + json.dumps({k: res[k] for k in
                                  ("arm", "add", "drop", "arch", "cap", "mon_depth", "depth", "seed",
                                   "frac", "params", "best_epoch",
                                   "val_brier", "test", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
