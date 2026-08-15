"""Fast local trainer for the lab corpus. Minutes per run on 4 CPU cores.

FOUR CELLS (the first experiment)
    REL=0/1   x   RANK_W=0/>0
  REL=0 is `valuenet/train.py`'s net, bit-exact (labmodel.assert_baseline_parity).
  REL=1 adds the 48 relational columns at the trunk input.
  RANK_W=0 is today's loss: outcome-only BCE.
  RANK_W>0 adds a SIBLING RANKING term.

THE RANKING TERM
  For each sampled decision, the net scores every sibling successor state; for
  every ordered pair (i, j) with q_i > q_j the loss pays
      softplus(-(z_i - z_j)) * (q_i - q_j)
  z = the value head's LOGIT, so a unit gap is a natural scale and no
  temperature has to be invented. Weighting by the oracle gap means the term
  spends its gradient on the pairs whose ordering actually costs win
  probability, and stops caring about ties -- which is exactly the failure the
  metric measures.

SPLIT
  By GAME, never by position: positions inside one game share an outcome label,
  so a position-level split leaks the label and every CE below would be a lie.

NO SIDE-SWAP AUGMENTATION
  train.py swaps sides 50% of the time. The relational blocks CAN be swapped
  correctly (relfeat.gx_swap), but turning it on in one cell and not another
  would confound the comparison, and at this corpus size it buys little. It is
  off in ALL four cells. Stated, not hidden.

USAGE
  python train_lab.py <pos.npz> <sib.npz> <out.pt> [epochs]
ENV
  REL=0|1  RANK_W=float  EPOCHS  LR  BATCH  SEED  VAL_FRAC  SIB_GROUPS
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import labenv  # noqa: F401
import labmodel
from encoder import Vocab  # noqa: E402

KEYS = ("a1_ids", "a1_f", "b1_ids", "b1_f", "sf1",
        "a2_ids", "a2_f", "b2_ids", "b2_f", "sf2", "g", "gx", "sx1", "sx2")

REL = os.environ.get("REL", "0") == "1"
RANK_W = float(os.environ.get("RANK_W", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "4096"))
SEED = int(os.environ.get("SEED", "7"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.10"))
# Fraction of TRAIN GAMES to keep. The data-scaling series ("how much data does
# a correct memory need") is a sweep of this, and it subsamples GAMES rather
# than positions so the learning curve is not contaminated by within-game
# correlation.
GAME_FRAC = float(os.environ.get("GAME_FRAC", "1.0"))
SIB_GROUPS = int(os.environ.get("SIB_GROUPS", "96"))   # decisions per rank step
THREADS = int(os.environ.get("EVALLAB_THREADS", "4"))


def load(path):
    """Features stay float16/int16 in memory (8.6 GB box); batches are upcast."""
    z = np.load(path)
    t = {}
    for k in z.files:
        a = z[k]
        if k in KEYS:
            t[k] = torch.tensor(a, dtype=torch.int16 if "ids" in k else torch.float16)
        else:
            t[k] = torch.tensor(a, dtype=torch.float32 if a.dtype.kind == "f" else torch.long)
    return t


def batch_of(t, idx):
    return {k: (t[k][idx].long() if "ids" in k else t[k][idx].float()) for k in KEYS}


def group_ranges(grp):
    """[(start, end)] for each contiguous decision group in a sib tensor."""
    g = grp.numpy()
    edges = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(g)]])
    return list(zip(starts.tolist(), ends.tolist()))


def rank_loss(z, q):
    """Gap-weighted pairwise logistic ranking loss inside one decision."""
    dz = z[:, None] - z[None, :]
    dq = q[:, None] - q[None, :]
    m = dq > 0
    if not bool(m.any()):
        return z.sum() * 0.0
    return (nn.functional.softplus(-dz[m]) * dq[m]).sum() / m.sum()


def main():
    torch.set_num_threads(THREADS)
    pos_path, sib_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS

    vocab = Vocab(frozen=True)
    pos = load(pos_path)
    sib = load(sib_path) if RANK_W > 0 else None

    # ---- split by game
    gid = pos["gid"].numpy()
    games = np.unique(gid)
    rs = np.random.default_rng(SEED)
    perm = rs.permutation(games)
    n_val = max(1, int(len(games) * VAL_FRAC))
    val_games = set(perm[:n_val].tolist())
    is_val = np.isin(gid, list(val_games))
    keep = ~is_val
    if GAME_FRAC < 1.0:
        tr_games = perm[n_val:]
        keep_games = set(tr_games[:max(1, int(len(tr_games) * GAME_FRAC))].tolist())
        keep = keep & np.isin(gid, list(keep_games))
    tr_idx = torch.tensor(np.flatnonzero(keep))
    va_idx = torch.tensor(np.flatnonzero(is_val))

    torch.manual_seed(SEED)
    net = labmodel.LabValueNet(vocab.sizes(), use_rel=REL)
    nparam = sum(p.numel() for p in net.parameters())
    print("REL=%d RANK_W=%g game_frac=%g epochs=%d params=%d pos=%d (train %d / val %d over %d games) sib=%s"
          % (REL, RANK_W, GAME_FRAC, epochs, nparam, len(gid), len(tr_idx), len(va_idx), len(games),
             (len(sib["q"]) if sib is not None else 0)), flush=True)

    # No leak by construction: dataset.py builds `sib` from TRAIN games only
    # (labenv.is_holdout), and every reported metric is computed on oracle
    # positions drawn from the HELD-OUT games. The internal VAL_FRAC split below
    # is a monitoring/early-stopping device inside the train games, nothing more.
    groups = group_ranges(sib["grp"]) if sib is not None else []

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    hist = []
    rng = np.random.default_rng(SEED + 1)
    t0 = time.time()
    best = (1e9, None)
    for ep in range(epochs):
        net.train()
        order = torch.tensor(rng.permutation(len(tr_idx)))
        tot, seen, rtot, rn = 0.0, 0, 0.0, 0
        for i in range(0, len(order), BATCH):
            bidx = tr_idx[order[i:i + BATCH]]
            b = batch_of(pos, bidx)
            z = net(b)
            loss = nn.functional.binary_cross_entropy_with_logits(z, pos["y"][bidx])
            tot += loss.item() * len(bidx)
            seen += len(bidx)
            if RANK_W > 0 and groups:
                picks = rng.choice(len(groups), size=min(SIB_GROUPS, len(groups)), replace=False)
                rows = np.concatenate([np.arange(*groups[p]) for p in picks])
                rt = torch.tensor(rows)
                sb = batch_of(sib, rt)
                zs = net(sb)
                rl = 0.0
                off = 0
                for p in picks:
                    a, bnd = groups[p]
                    n = bnd - a
                    rl = rl + rank_loss(zs[off:off + n], sib["q"][rt[off:off + n]])
                    off += n
                rl = rl / len(picks)
                rtot += float(rl.detach()) * len(bidx)
                rn += len(bidx)
                loss = loss + RANK_W * rl
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for i in range(0, len(va_idx), BATCH):
                bidx = va_idx[i:i + BATCH]
                b = batch_of(pos, bidx)
                z = net(b)
                vs += float(nn.functional.binary_cross_entropy_with_logits(
                    z, pos["y"][bidx], reduction="sum"))
                vn += len(bidx)
        vce = vs / max(vn, 1)
        hist.append({"epoch": ep, "train_ce": tot / max(seen, 1),
                     "rank": rtot / max(rn, 1), "val_ce": vce, "s": round(time.time() - t0, 1)})
        print("  ep%-2d train_ce=%.4f rank=%.4f val_ce=%.4f  %.0fs"
              % (ep, tot / max(seen, 1), rtot / max(rn, 1), vce, time.time() - t0), flush=True)
        if vce < best[0]:
            best = (vce, {k: v.detach().clone() for k, v in net.state_dict().items()})

    net.load_state_dict(best[1])
    torch.save({"sd": net.state_dict(), "rel": REL, "rank_w": RANK_W, "game_frac": GAME_FRAC,
                "vocab": vocab.d, "hist": hist, "val_ce": best[0],
                "val_games": sorted(val_games)}, out_path)
    print("saved %s  best val_ce=%.4f  (%.0fs)" % (out_path, best[0], time.time() - t0), flush=True)
    print("JSON " + json.dumps({"out": out_path, "rel": int(REL), "rank_w": RANK_W,
                                "val_ce": best[0], "hist": hist}))


if __name__ == "__main__":
    main()
