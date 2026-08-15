"""FLAT value net -- no pooling anywhere -- plus the switchable OLD baseline.

THE HYPOTHESIS THIS TESTS
  The incumbent (`valuenet/train.py:ValueNet`, mirrored by `labmodel.py`) pushes
  every Pokemon through ONE shared MLP and then SUM-POOLS the five bench slots:

      trunk_in = [ mlp(a1), sum_i mlp(b1_i), mlp(a2), sum_i mlp(b2_i), sf1, sf2, g ]

  Two things die there. (1) The sum is permutation-invariant, so which bench slot
  a feature belongs to is unrecoverable AFTER the pool -- and BENCH_SORT already
  destroyed party identity BEFORE it. (2) The shared MLP forces one function to
  serve "my active", "my bench", "their active" and "their bench", so the net
  cannot learn a slot-specific reading without spending capacity re-deriving the
  slot from the features.

THE FLAT LAYOUT (what this file builds)
  Every per-slot feature block goes into a FIXED position in one flat vector, in
  this canonical order, and the trunk sees all of it at once:

      [ our active | our bench 1..5 | their active | their bench 1..5 |
        side block 1 | side block 2 | global block ]

  which is exactly `llencoder`'s own column order (12 mon blocks, 2 side blocks,
  1 global block), so the tensor coming out of the encoder is ALREADY the input
  vector -- no reshaping, no pooling, no per-slot module. Slot identity is
  positional and therefore free.

  Embedding ids are expanded the same way: each of the 12 slots has its own 15
  ids, they are looked up in SHARED tables (a species is a species wherever it
  stands -- sharing the TABLE is not the same as sharing the MLP) and the 12
  resulting vectors are CONCATENATED into their own fixed positions.

CAPACITY KNOB
  `width` and `depth` size the trunk. Everything else is fixed, so a capacity
  sweep is a sweep of two integers and nothing else changes.

BASELINE
  `build("old")` returns `labmodel.LabValueNet`, which `assert_baseline_parity`
  proves is bit-identical to the shipped `train.ValueNet`. Every future
  comparison therefore runs old-encoder+pooled and new-encoder+flat in ONE
  harness, on one split, with one metric.

ASSUMPTIONS, STATED
  * Flat means the input layer is dense over 1,863 (or 1,206) columns; at
    width 512 that is a ~1M-parameter first layer. This buys slot identity at
    the cost of parameters, and the capacity knob exists precisely so the
    tradeoff is measurable rather than argued.
  * No side-swap augmentation, matching the rest of the lab (README).
  * The flat net has no weight sharing across slots, so it needs to see each
    slot in each role. Pair A is a FIXED roster, which is the regime where that
    is cheapest -- generalisation across rosters is explicitly out of scope here
    and must be re-measured before anything ships.
"""

import os

import torch
import torch.nn as nn

import labenv  # noqa: F401
import llencoder as ll

# shared embedding tables, sized as the incumbent sizes them
EMB = {"species": 32, "item": 12, "ability": 12, "move": 16, "ptype": 8, "nature": 8}
# llencoder mon-id column order (see Layout._build in lossless_encoder.py)
ID_TABLE = ["species", "item", "ability", "ability", "item", "nature", "ptype",
            "ptype", "ptype", "ptype", "ptype", "move", "move", "move", "move"]
SIDE_ID_TABLE = ["move", "move"]
N_TYPE, N_NATURE = len(ll.TYPE_ORDER), len(ll.NATURE_ORDER)


class FlatValueNet(nn.Module):
    """One flat vector in, one logit out. No pooling, no shared per-mon MLP."""

    def __init__(self, vocab_sizes, variant="full", width=512, depth=3, dropout=0.0):
        super().__init__()
        self.variant = variant
        V = ll.VARIANTS[variant]
        self.n_feats, self.n_ids = V.N_FEATS, V.N_IDS
        sizes = {"species": vocab_sizes["species"] + 64, "item": vocab_sizes["item"] + 64,
                 "ability": vocab_sizes["ability"] + 64, "move": vocab_sizes["move"] + 64,
                 "ptype": N_TYPE, "nature": N_NATURE}
        self.emb = nn.ModuleDict({k: nn.Embedding(max(sizes[k], 2), d, padding_idx=0)
                                  for k, d in EMB.items()})
        self.mon_emb_dim = sum(EMB[t] for t in ID_TABLE)          # 192
        self.side_emb_dim = sum(EMB[t] for t in SIDE_ID_TABLE)    # 32
        in_dim = V.N_FEATS + 12 * self.mon_emb_dim + 2 * self.side_emb_dim
        self.in_dim = in_dim
        layers, d_prev = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d_prev, width), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d_prev = width
        layers.append(nn.Linear(d_prev, 1))
        self.trunk = nn.Sequential(*layers)

    def embed(self, ids):
        """ids int64[B, 12*15 + 2*2] -> [B, 12*192 + 2*32], slot-positional."""
        b = ids.shape[0]
        mon = ids[:, :12 * ll.N_MON_IDS].reshape(b, 12, ll.N_MON_IDS)
        side = ids[:, 12 * ll.N_MON_IDS:].reshape(b, 2, ll.N_SIDE_IDS)
        # one table lookup per id COLUMN (15 of them), each over all 12 slots at
        # once; the results keep their slot position in the concatenation.
        mparts = [self.emb[t](mon[:, :, i]) for i, t in enumerate(ID_TABLE)]
        m = torch.cat(mparts, dim=-1).reshape(b, -1)
        sparts = [self.emb[t](side[:, :, i]) for i, t in enumerate(SIDE_ID_TABLE)]
        s = torch.cat(sparts, dim=-1).reshape(b, -1)
        return m, s

    def forward(self, batch):
        m, s = self.embed(batch["ids"])
        return self.trunk(torch.cat([batch["feats"], m, s], dim=-1)).squeeze(-1)


def build(arch, vocab_sizes, variant="full", width=512, depth=3, dropout=0.0):
    """arch="flat" -> FlatValueNet on the lossless features.
       arch="old"  -> the incumbent encoder + pooled architecture, unchanged."""
    if arch == "flat":
        return FlatValueNet(vocab_sizes, variant, width, depth, dropout)
    if arch == "old":
        import labmodel
        return labmodel.LabValueNet(vocab_sizes, use_rel=os.environ.get("REL", "0") == "1")
    raise ValueError("unknown arch %r" % arch)


def layout_doc(variant="full"):
    """The flat layout, printed. This is the documentation the spec asks for and
    it is generated from the code, so it cannot go stale."""
    V = ll.VARIANTS[variant]
    lines = ["FLAT INPUT LAYOUT -- variant %r" % variant,
             "  numeric block: %d columns" % V.N_FEATS]
    off = 0
    names = (["our active"] + ["our bench %d" % i for i in range(1, 6)]
             + ["their active"] + ["their bench %d" % i for i in range(1, 6)])
    for i, nm in enumerate(names):
        lines.append("    [%5d:%5d) %-14s %d per-slot columns" % (off, off + V.NM, nm, V.NM))
        off += V.NM
    for i, nm in enumerate(("our side block", "their side block")):
        lines.append("    [%5d:%5d) %-14s %d columns" % (off, off + V.NS, nm, V.NS))
        off += V.NS
    lines.append("    [%5d:%5d) %-14s %d columns" % (off, off + V.NG, "global", V.NG))
    off += V.NG
    assert off == V.N_FEATS
    md = sum(EMB[t] for t in ID_TABLE)
    sd = sum(EMB[t] for t in SIDE_ID_TABLE)
    lines.append("  embedding block: 12 x %d (per-slot, shared TABLES) + 2 x %d = %d columns"
                 % (md, sd, 12 * md + 2 * sd))
    lines.append("  total trunk input: %d" % (V.N_FEATS + 12 * md + 2 * sd))
    lines.append("  pooling operations: 0   shared per-mon MLPs: 0")
    return "\n".join(lines)


if __name__ == "__main__":
    for v in ("full", "lean"):
        print(layout_doc(v))
        print()
