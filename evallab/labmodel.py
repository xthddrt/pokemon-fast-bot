"""The lab's value net: `valuenet/train.py:ValueNet` with an optional
relational-feature port.

REL=0 must be the incumbent architecture EXACTLY -- same embedding tables, same
shared mon MLP, same sum-pooled bench, same 3-layer trunk, same parameter count.
`assert_baseline_parity()` proves it against the real `train.ValueNet` rather
than asserting it in a comment, because "the baseline arm was subtly a different
net" is the failure mode that would make every number in the experiment
meaningless.

REL=1 concatenates the relational blocks onto the trunk input:
    [a1, pool(b1), a2, pool(b2), sf1, sf2, g]  +  [gx, sx1, sx2]
i.e. 48 extra columns arriving at the TRUNK, never at the shared per-mon MLP
and never through the bench sum-pool. That placement is the hypothesis.
"""

import os

import torch
import torch.nn as nn

import labenv  # noqa: F401
import relfeat
from encoder import NUM_GLOBAL_FEATS, NUM_MON_FEATS, NUM_SIDE_FEATS  # noqa: E402

EMB = {"species": 32, "item": 12, "ability": 12, "move": 16, "teratype": 8}
N_MOVES = 4
MON_HID = int(os.environ.get("MON_HID", "128"))
TRUNK = int(os.environ.get("TRUNK", "256"))
BENCH_POOL = os.environ.get("BENCH_POOL", "sum")

N_EXTRA = relfeat.N_GX + 2 * relfeat.N_SX  # 16 + 2*16 = 48


class LabValueNet(nn.Module):
    def __init__(self, vocab_sizes, use_rel=False):
        super().__init__()
        self.use_rel = use_rel
        self.emb = nn.ModuleDict(
            {k: nn.Embedding(max(vocab_sizes[k], 2) + 64, d, padding_idx=0) for k, d in EMB.items()}
        )
        mon_in = (EMB["species"] + EMB["item"] + EMB["ability"]
                  + N_MOVES * EMB["move"] + EMB["teratype"] + EMB["item"] + NUM_MON_FEATS)
        self.mon = nn.Sequential(nn.Linear(mon_in, MON_HID), nn.ReLU(),
                                 nn.Linear(MON_HID, MON_HID), nn.ReLU())
        bench_mult = 5 if BENCH_POOL == "concat" else 2 if BENCH_POOL == "summax" else 1
        trunk_in = (2 + 2 * bench_mult) * MON_HID + 2 * NUM_SIDE_FEATS + NUM_GLOBAL_FEATS
        if use_rel:
            trunk_in += N_EXTRA
        self.trunk_in = trunk_in
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, TRUNK), nn.ReLU(),
            nn.Linear(TRUNK, TRUNK), nn.ReLU(),
            nn.Linear(TRUNK, 1),
        )

    def pool(self, bench):
        if BENCH_POOL == "concat":
            return bench.reshape(bench.shape[0], -1)
        if BENCH_POOL == "max":
            return bench.max(dim=1).values
        if BENCH_POOL == "summax":
            return torch.cat([bench.sum(dim=1), bench.max(dim=1).values], dim=-1)
        return bench.sum(dim=1)

    def embed_mon(self, ids, feats):
        e = torch.cat([
            self.emb["species"](ids[..., 0]),
            self.emb["item"](ids[..., 1]),
            self.emb["ability"](ids[..., 2]),
            self.emb["move"](ids[..., 3:3 + N_MOVES]).flatten(-2),
            self.emb["teratype"](ids[..., 3 + N_MOVES]),
            self.emb["item"](ids[..., 4 + N_MOVES]),
            feats,
        ], dim=-1)
        return self.mon(e)

    def forward(self, b):
        a1 = self.embed_mon(b["a1_ids"], b["a1_f"])
        a2 = self.embed_mon(b["a2_ids"], b["a2_f"])
        b1 = self.embed_mon(b["b1_ids"], b["b1_f"])
        b2 = self.embed_mon(b["b2_ids"], b["b2_f"])
        parts = [a1, self.pool(b1), a2, self.pool(b2), b["sf1"], b["sf2"], b["g"]]
        if self.use_rel:
            parts += [b["gx"], b["sx1"], b["sx2"]]
        return self.trunk(torch.cat(parts, dim=-1)).squeeze(-1)


def assert_baseline_parity(vocab_sizes, verbose=True):
    """REL=0 must be bit-identical to valuenet/train.py's ValueNet."""
    import numpy as np
    import train as vtrain

    torch.manual_seed(7)
    ref = vtrain.ValueNet(vocab_sizes)
    torch.manual_seed(7)
    lab = LabValueNet(vocab_sizes, use_rel=False)
    nref = sum(p.numel() for p in ref.parameters())
    nlab = sum(p.numel() for p in lab.parameters())
    assert nref == nlab, "param count %d != %d" % (nref, nlab)
    rng = np.random.default_rng(0)
    b = {
        "a1_ids": torch.tensor(rng.integers(0, 50, (7, 9))), "a1_f": torch.rand(7, NUM_MON_FEATS),
        "b1_ids": torch.tensor(rng.integers(0, 50, (7, 5, 9))), "b1_f": torch.rand(7, 5, NUM_MON_FEATS),
        "sf1": torch.rand(7, NUM_SIDE_FEATS),
        "a2_ids": torch.tensor(rng.integers(0, 50, (7, 9))), "a2_f": torch.rand(7, NUM_MON_FEATS),
        "b2_ids": torch.tensor(rng.integers(0, 50, (7, 5, 9))), "b2_f": torch.rand(7, 5, NUM_MON_FEATS),
        "sf2": torch.rand(7, NUM_SIDE_FEATS), "g": torch.rand(7, NUM_GLOBAL_FEATS),
    }
    with torch.no_grad():
        d = float((ref(b) - lab(b)).abs().max())
    assert d == 0.0, "forward mismatch %g" % d
    if verbose:
        print("baseline parity OK: params=%d, max|dv|=%g vs train.ValueNet" % (nlab, d))
    return nlab
