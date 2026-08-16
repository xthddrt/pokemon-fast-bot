"""ENCODER VALUE TEST -- shared split / data / model / metric code.

ONE split, fixed for every run in the experiment, so seed spread measures
TRAINING noise only and the two arms are compared on identical rows.
Split is BY GAME (20,000 games x 10 positions); 70 % train / 15 % val (early
stopping and capacity selection) / 15 % test (every number reported).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402

# The BASELINE arm's net lives in labmodel.py, which imports relfeat, which
# imports poke_engine purely for `calculate_damage`. Nothing on the training
# path calls it: with use_rel=False (the shipped architecture) labmodel reads
# exactly two integers from relfeat, N_GX and N_SX, and never a feature. On a
# box with no Rust wheel we therefore stub the ONE symbol and then PROVE the
# stub changed nothing -- the two integers are checked, and every net's
# parameter count is checked against the count measured locally with the real
# wheel present. A stub that mattered would fail here, loudly.
PARAM_COUNTS = {(16, 32): 86181, (32, 64): 106613, (64, 128): 167445,
                (128, 256): 368981, (256, 512): 1091541}
try:
    import poke_engine  # noqa: F401
    STUBBED_PE = False
except ImportError:
    import types
    _m = types.ModuleType("poke_engine")

    def _nope(*a, **k):
        raise RuntimeError("poke_engine is stubbed on this box and was CALLED")
    _m.calculate_damage = _nope
    _m.State = _nope
    _m.generate_instructions = _nope
    sys.modules["poke_engine"] = _m
    STUBBED_PE = True

# ENCODING DIRECTORY. Was hardcoded to the 200k pl2 pilot encoding; now a
# parameter. Override with $VT_ENC (absolute, or relative to evallab/); the
# default is the plc1 corpus. Set VT_ENC=data/pl2/enc to reproduce old runs.
ENC = os.environ.get("VT_ENC") or "data/plc1/enc"
if not os.path.isabs(ENC):
    ENC = os.path.join(LAB, ENC)
OLD_KEYS = ("a1_ids", "a1_f", "b1_ids", "b1_f", "sf1",
            "a2_ids", "a2_f", "b2_ids", "b2_f", "sf2", "g")
SPLIT_SEED = 0
BANDS = ("early", "mid", "late")


# ---------------------------------------------------------------- data ------
def load_meta():
    z = np.load(os.path.join(ENC, "meta.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def split_idx(meta, train_frac_of_games=1.0, sub_seed=0):
    """Game-level 70/15/15. `train_frac_of_games` subsamples the TRAIN games
    only (data-scaling curve); val and test never move."""
    g = meta["g"]
    games = np.unique(g)
    perm = np.random.default_rng(SPLIT_SEED).permutation(games)
    n = len(games)
    n_te, n_va = int(0.15 * n), int(0.15 * n)
    te_g, va_g, tr_g = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    if train_frac_of_games < 1.0:
        k = max(1, int(round(len(tr_g) * train_frac_of_games)))
        tr_g = np.random.default_rng(1000 + sub_seed).permutation(tr_g)[:k]
    # ANTI-OVERFIT: the held-out TEST games are split once, by game, into a
    # SELECTION half and a CONFIRMATION half. Every cell of the grid is ranked
    # on SEL; CONF is read exactly once, for the single winner. `test` (the
    # union) is still reported unchanged so the control is directly comparable
    # to the published 0.0215 / 0.0206 / 0.0230. Training and early stopping do
    # not move at all: train and val are untouched.
    hp = np.random.default_rng(SPLIT_SEED + 777).permutation(te_g)
    sel_g, conf_g = hp[:len(hp) // 2], hp[len(hp) // 2:]
    ix = {k: np.flatnonzero(np.isin(g, v)) for k, v in
          (("train", tr_g), ("val", va_g), ("test", te_g),
           ("sel", sel_g), ("conf", conf_g))}
    return ix, {"train_games": len(tr_g), "val_games": len(va_g), "test_games": len(te_g),
                "sel_games": len(sel_g), "conf_games": len(conf_g)}


# ------------------------------------------------------- additive add-ons ----
# ADDITIVE SEARCH: arm A, unchanged, PLUS selected enc2 derived blocks.
# `vt_addon.py` wrote two caches, both row-aligned to meta.npz:
#   addon_mon.npy  (n, 12, D)  per-mon blocks REMAPPED into ARM A's slot order
#                              (arm A sorts the bench, enc2 does not) so each
#                              block is appended to the RIGHT mon's vector.
#   addon_rest.npy (n, R)      pair/side blocks, which belong to no slot.
def load_addon_layout():
    return json.load(open(os.path.join(ENC, "addon_layout.json")))


def addon_cols(spec):
    """"setup,ko" -> (mon column indices, rest column indices) into the caches.

    Blocks are ATOMS of the layout, each of which is either per-mon (goes
    through arm A's shared per-mon MLP) or trunk-level. `ko`/`speed` name the
    ORIGINAL trunk-level forms; `ko_pm`/`spd_pm`/`mprio_pm` name the SAME
    information RESHAPED per-mon, and `ko_pm_s` its 12-column scalar form."""
    if not spec:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    lay = load_addon_layout()
    g, al = lay["groups"], lay.get("aliases", {})
    want = []
    for s in spec.split(","):
        if s:
            want += al.get(s, [s])
    known = {k.split(".", 1)[1] for k in g}
    assert set(want) <= known, "unknown add-on block(s): %s" % (set(want) - known)
    assert len(want) == len(set(want)), "block listed twice in %r" % spec
    def pick(pre):
        v = [np.arange(*g["%s.%s" % (pre, w)]) for w in want if "%s.%s" % (pre, w) in g]
        return np.concatenate(v).astype(np.int64) if v else np.zeros(0, np.int64)
    return pick("mon"), pick("rest")


# ------------------------------------------- subtractive: ARM A's dead weight -
# The other direction: arm A's own permutation ablation (ENCODER_VALUE_TEST
# §7.1) put a long list of its groups at EXACTLY +0.0000. Column indices below
# are the documented layout of `valuenet/encoder.py` (per-mon 73, per-side 99,
# global 18) and match `vt_ablate.old_groups`.
# Indices past `times_attacked` shift by one under DROP_TIMES_ATTACKED, so they
# are read from the encoder rather than restated here.
from encoder import (  # noqa: E402
    F_TERA_STELLAR, F_WEIGHT, NUM_MON_FEATS,
)

ARMA_MON_G = {"level": [1], "stats": list(range(2, 7)), "maxhp": [7],
              "weight": [F_WEIGHT], "tera_stellar": [F_TERA_STELLAR]}
ARMA_SIDE_G = {"screens": [4, 5, 6, 7, 8], "extras": list(range(13, 39)),
               "paradox": list(range(73, 78)), "durations": list(range(78, 91))}
ARMA_GLOB_G = {"field": list(range(18))}
ARMA_EMB_G = ("last_item",)          # ids[..., 8], the 6th embedding term

# TWO SETS, AND THE DISTINCTION IS THE POINT.
#   struct  -- dead BY CONSTRUCTION, on any corpus: level and raw stats are
#              deterministic functions of species/EVs that the computed stats
#              and the KO matrix already carry, maxhp and weight matter to two
#              moves, tera_stellar and the last-consumed item are unreachable
#              signal for this net.
#   cov     -- struct PLUS the groups that read zero only because THIS corpus is
#              one fixed pair with no screen setter, no weather/terrain setter
#              and no paradox ability. These would be LIVE on general data. The
#              cell exists to measure available parameter relief; it is NOT a
#              recommendation to prune.
PRUNE_SETS = {
    "struct": ["level", "stats", "maxhp", "weight", "tera_stellar", "last_item"],
    "cov": ["level", "stats", "maxhp", "weight", "tera_stellar", "last_item",
            "screens", "extras", "paradox", "durations", "field"],
}


def prune_keep(spec):
    """-> (keep_mon, keep_side, keep_glob, drop_last_item) as index arrays."""
    want = PRUNE_SETS.get(spec, [s for s in spec.split(",") if s]) if spec else []
    known = set(ARMA_MON_G) | set(ARMA_SIDE_G) | set(ARMA_GLOB_G) | set(ARMA_EMB_G)
    assert set(want) <= known, "unknown prune group(s): %s" % (set(want) - known)
    def keep(groups, n):
        drop = sorted({c for g in want if g in groups for c in groups[g]})
        return np.setdiff1d(np.arange(n), drop).astype(np.int64)
    return (keep(ARMA_MON_G, NUM_MON_FEATS), keep(ARMA_SIDE_G, 99),
            keep(ARMA_GLOB_G, 18), "last_item" in want)


# ---- side-swap augmentation (v8c, Sally 2026-08-15) -------------------------
# Mirroring a state = swapping the two sides everywhere they appear in the
# batch dict, with the label flipped by the caller (y -> 1-y). The game is
# side-symmetric, so this teaches exact antisymmetry (measured v8b deviation:
# 0.027). Slot layout per ArmAPlusNet.forward: 0=a1, 1-5=b1, 6=a2, 7-11=b2.
SWAP_KEYS = (("a1_ids", "a2_ids"), ("a1_f", "a2_f"),
             ("b1_ids", "b2_ids"), ("b1_f", "b2_f"), ("sf1", "sf2"))
AM_SLOT_SWAP = np.concatenate([np.arange(6, 12), np.arange(0, 6)])


def swap_rows_(b, mask):
    """In-place side swap of the masked rows of a batch dict; g is global and
    stays. Caller flips the labels."""
    for k1, k2 in SWAP_KEYS:
        t = b[k1][mask].clone()
        b[k1][mask] = b[k2][mask]
        b[k2][mask] = t
    if "am" in b:
        b["am"][mask] = b["am"][mask][:, AM_SLOT_SWAP]
    if "ar" in b and b["ar"].shape[1]:
        raise SystemExit("swap_rows_: addon_rest present but has no defined "
                         "side swap -- refuse to train silently wrong")
    return b


class Arm:
    """Holds mmap'd feature arrays and yields batches. mmap means the four
    concurrent training processes share ONE page-cache copy of the 565 MB
    feature file instead of four resident copies (8.6 GB box)."""

    def __init__(self, arm, add="", drop=""):
        self.arm = arm
        self.add = add
        if arm == "enc2":
            self.feats = np.load(os.path.join(ENC, "enc2_feats.npy"), mmap_mode="r")
            self.ids = np.load(os.path.join(ENC, "enc2_ids.npy"), mmap_mode="r")
            self.n_feats = self.feats.shape[1]
        else:
            self.a = {k: np.load(os.path.join(ENC, "old_%s.npy" % k), mmap_mode="r")
                      for k in OLD_KEYS}
            # DROP_TIMES_ATTACKED changes arm A's per-mon width, so a cache built
            # under the other setting would train a 72-wide net on 73-wide rows
            # (or silently mis-align every column past index 32). Fail loudly.
            assert self.a["a1_f"].shape[1] == NUM_MON_FEATS, (
                "cache %s has %d per-mon numerics but the encoder produces %d "
                "-- DROP_TIMES_ATTACKED disagrees with how this cache was built"
                % (ENC, self.a["a1_f"].shape[1], NUM_MON_FEATS))
            self.mcol, self.rcol = addon_cols(add)
            self.km, self.ks, self.kg, _ = prune_keep(drop)
            self.drop = drop
            if len(self.mcol):
                self.am = np.load(os.path.join(ENC, "addon_mon.npy"), mmap_mode="r")
            if len(self.rcol):
                self.ar = np.load(os.path.join(ENC, "addon_rest.npy"), mmap_mode="r")

    def batch(self, idx):
        if self.arm == "enc2":
            return {"feats": torch.from_numpy(np.asarray(self.feats[idx], np.float32)),
                    "ids": torch.from_numpy(np.asarray(self.ids[idx], np.int64))}
        b = {k: torch.from_numpy(np.asarray(
            self.a[k][idx], np.int64 if "ids" in k else np.float32)) for k in OLD_KEYS}
        if self.drop:
            for k in ("a1_f", "a2_f"):
                b[k] = b[k][:, self.km]
            for k in ("b1_f", "b2_f"):
                b[k] = b[k][:, :, self.km]
            for k in ("sf1", "sf2"):
                b[k] = b[k][:, self.ks]
            b["g"] = b["g"][:, self.kg]
        if len(self.mcol):
            b["am"] = torch.from_numpy(np.asarray(self.am[idx][:, :, self.mcol], np.float32))
        if len(self.rcol):
            b["ar"] = torch.from_numpy(np.asarray(self.ar[idx][:, self.rcol], np.float32))
        return b

    def id_max(self):
        return np.asarray(self.ids).max(axis=0) if self.arm == "enc2" else None


# --------------------------------------------------------------- models -----
EMB2 = {"item": 12, "ability": 12, "ptype": 8, "move": 16}
# enc2 MON_IDS = item, ability, type1, type2, tera_type, move0..3 ; SIDE_IDS = 2 moves
ID_TABLE2 = ["item", "ability", "ptype", "ptype", "ptype", "move", "move", "move", "move"]
SIDE_TABLE2 = ["move", "move"]


def load_layout():
    """enc2's column layout, written by `vt_encode2.py join` next to the cache.

    Read from JSON rather than by importing enc2, so a training box needs
    neither the Rust wheel nor the randbats data files."""
    return json.load(open(os.path.join(ENC, "enc2_layout.json")))


def mon_and_rest_cols(lay):
    """THE SPLIT, stated explicitly (this is the whole architecture change).

    PER-MON, 12 blocks in canonical slot order (our active, our bench 1-5,
    their active, their bench 1-5) -- each block is that slot's §1 per-mon
    columns CONCATENATED with its §4 per-mon counterfactual columns, plus its
    nine embedding ids:
        feats[O_MON + k*NM : O_MON + (k+1)*NM]      §1 per-mon
        feats[O_CFM + k*NCM : O_CFM + (k+1)*NCM]    §4 setup counterfactual
        ids[k*9 : (k+1)*9]                          item/ability/t1/t2/tera/4 moves

    NOT PER-MON, straight to the trunk -- everything else, i.e. the two §2 side
    blocks, the §3 relational block (it is about PAIRS of mons, so it belongs to
    no single slot), the two §4 per-side tera/aggregate blocks, the §5 global
    block, and the four side embedding ids.

    Returns (mon_cols int64[12, NM+NCM], rest_cols int64[...])."""
    NM, NCM, O_MON, O_CFM = lay["NM"], lay["NCM"], lay["O_MON"], lay["O_CFM"]
    mc = np.stack([np.concatenate([
        np.arange(O_MON + k * NM, O_MON + (k + 1) * NM),
        np.arange(O_CFM + k * NCM, O_CFM + (k + 1) * NCM)]) for k in range(12)])
    rest = np.setdiff1d(np.arange(lay["N"]), mc.reshape(-1))
    assert len(rest) == 2 * lay["NS"] + lay["NR"] + 2 * lay["NCS"] + lay["NG"]
    return mc.astype(np.int64), rest.astype(np.int64)


class Enc2SharedNet(nn.Module):
    """SHARED per-mon encoder + CONCATENATION -- the architecture this pass is
    testing.

    ONE MLP is applied to each of the twelve per-mon blocks with SHARED weights,
    and its twelve outputs are CONCATENATED in canonical slot order (never
    summed, never pooled), a learned per-slot embedding is added so the trunk
    can tell the slots apart, and the not-per-mon blocks are concatenated on
    before the trunk.

    WHY: the incumbent pooled net does two separable things -- it shares weights
    across the twelve mons AND it sum-pools the bench. The value test discarded
    both and lost. Sharing is what regularises: without it the flat net has to
    learn "low HP is bad" twelve separate times, from a twelfth of the effective
    data each, which is exactly the generalisation failure that was measured
    (flat peaks at width 32 and degrades above; pooled improves to 3.8M params).
    Concatenation is what keeps per-mon identity, which is what pooling loses.

    `mode="pool"` is the pure-pooled CONTROL: same shared encoder, bench slots
    sum-pooled exactly as `labmodel.LabValueNet` does, so the two hypotheses can
    be separated inside one harness."""

    def __init__(self, sizes, lay, mon_w=64, mon_depth=2, width=256, depth=3,
                 mode="shared", slot_emb=True, dropout=0.0):
        super().__init__()
        self.mode = mode
        self.emb = nn.ModuleDict({k: nn.Embedding(max(sizes[k], 2), d, padding_idx=0)
                                  for k, d in EMB2.items()})
        mc, rest = mon_and_rest_cols(lay)
        self.register_buffer("mon_cols", torch.from_numpy(mc))
        self.register_buffer("rest_cols", torch.from_numpy(rest))
        self.mon_emb_dim = sum(EMB2[t] for t in ID_TABLE2)
        self.side_dim = sum(EMB2[t] for t in SIDE_TABLE2)
        mon_in = lay["NM"] + lay["NCM"] + self.mon_emb_dim
        layers, d = [], mon_in
        for _ in range(mon_depth):
            layers += [nn.Linear(d, mon_w), nn.ReLU()]
            d = mon_w
        self.mon = nn.Sequential(*layers)
        # a pooled net must stay permutation-invariant over the bench, so the
        # slot embedding exists only in the concatenating variant
        self.slot = nn.Parameter(torch.zeros(12, mon_w)) if (
            slot_emb and mode == "shared") else None
        self.in_dim = (12 if mode == "shared" else 4) * mon_w + len(rest) + 2 * self.side_dim
        layers, d = [], self.in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, b):
        f, ids = b["feats"], b["ids"]
        n = ids.shape[0]
        mon_ids = ids[:, :108].reshape(n, 12, 9)
        e = torch.cat([self.emb[t](mon_ids[:, :, i]) for i, t in enumerate(ID_TABLE2)],
                      dim=-1)                                   # (n, 12, mon_emb_dim)
        h = self.mon(torch.cat([f[:, self.mon_cols], e], dim=-1))   # (n, 12, mon_w)
        if self.slot is not None:
            h = h + self.slot
        if self.mode == "shared":
            g = h.reshape(n, -1)
        else:
            g = torch.cat([h[:, 0], h[:, 1:6].sum(1), h[:, 6], h[:, 7:12].sum(1)], dim=-1)
        side = ids[:, 108:].reshape(n, 2, 2)
        s = torch.cat([self.emb[t](side[:, :, i]) for i, t in enumerate(SIDE_TABLE2)],
                      dim=-1).reshape(n, -1)
        return self.trunk(torch.cat([g, f[:, self.rest_cols], s], dim=-1)).squeeze(-1)


class Enc2FlatNet(nn.Module):
    """The flat, no-pooling net of flatnet.py, over enc2's columns. Same
    construction (shared embedding TABLES, slot-positional concatenation, one
    dense trunk); only the numeric block and the id table differ. Kept as the
    pure-flat CONTROL -- this is the arm-B net of ENCODER_VALUE_TEST."""

    def __init__(self, sizes, n_feats=1413, width=128, depth=3, dropout=0.0):
        super().__init__()
        self.emb = nn.ModuleDict({k: nn.Embedding(max(sizes[k], 2), d, padding_idx=0)
                                  for k, d in EMB2.items()})
        self.mon_dim = sum(EMB2[t] for t in ID_TABLE2)
        self.side_dim = sum(EMB2[t] for t in SIDE_TABLE2)
        self.in_dim = n_feats + 12 * self.mon_dim + 2 * self.side_dim
        layers, d_prev = [], self.in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d_prev, width), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d_prev = width
        layers.append(nn.Linear(d_prev, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, b):
        ids = b["ids"]
        n = ids.shape[0]
        mon, side = ids[:, :108].reshape(n, 12, 9), ids[:, 108:].reshape(n, 2, 2)
        m = torch.cat([self.emb[t](mon[:, :, i]) for i, t in enumerate(ID_TABLE2)],
                      dim=-1).reshape(n, -1)
        s = torch.cat([self.emb[t](side[:, :, i]) for i, t in enumerate(SIDE_TABLE2)],
                      dim=-1).reshape(n, -1)
        return self.trunk(torch.cat([b["feats"], m, s], dim=-1)).squeeze(-1)


class ArmAPlusNet(nn.Module):
    """ARM A, UNCHANGED, PLUS enc2's derived blocks -- the additive arm.

    Every part of arm A is reused as-is: the same embedding tables, the same
    SHARED per-mon MLP, the same SUM-POOLED bench, the same 3-layer trunk. Only
    two matrices get wider:

      * the per-mon MLP's first layer, by `d_mon` -- the per-mon add-ons (setup
        counterfactual, capabilities) are appended to arm A's 73 per-mon
        numerics, so they pass through arm A's own aggregation. `vt_addon.py`
        has already remapped them into arm A's (BENCH_SORT) slot order, which
        is NOT enc2's; without that remap each mon would be handed a different
        mon's setup flags.
      * the trunk's first layer, by `d_rest` -- the KO matrix, the speed /
        priority block and the per-side tera block are about PAIRS or about a
        whole side, so they belong to no slot. This is exactly where `labmodel`
        already puts relational features when use_rel=1.

    With d_mon = d_rest = 0 and no drop this class is not used at all:
    `build_net` returns the untouched `labmodel.LabValueNet`, so the CONTROL
    cell is the published code path bit for bit.

    BIDIRECTIONAL: `drop` prunes arm A's own dead weight at the same time --
    the pruned columns are removed from the DATA (`Arm.batch`) and the same two
    matrices get correspondingly narrower, so this measures a genuinely smaller
    encoder, not a masked one. `drop` may include `last_item`, which removes the
    sixth embedding term from the per-mon MLP entirely."""

    def __init__(self, vocab_sizes, d_mon, d_rest, drop="", biasfix=0, constok=0):
        super().__init__()
        import labmodel as LM
        self.d_mon, self.d_rest = d_mon, d_rest
        km, ks, kg, self.drop_last_item = prune_keep(drop)
        self.n_mon, self.n_side, self.n_glob = len(km), len(ks), len(kg)
        self.base = LM.LabValueNet(vocab_sizes, use_rel=False)
        E, H = LM.EMB, self.base.mon[0].out_features
        emb_in = (E["species"] + E["item"] + E["ability"] + 4 * E["move"]
                  + E["teratype"] + (0 if self.drop_last_item else E["item"]))
        # MECHANISM PROBE `constok`: a free learnable K-vector broadcast to every
        # mon. `last_item` is the CONSTANT id 158 in every row and every slot of
        # this corpus, so arm A's sixth embedding term IS exactly this object --
        # an over-parameterised extra bias into the shared per-mon MLP, not an
        # information channel. constok reproduces that function class without the
        # embedding lookup, so `-last_item +constok 12` and arm A are the same net.
        self.tok = nn.Parameter(torch.randn(constok)) if constok else None
        mon_in = emb_in + self.n_mon + d_mon + constok
        if mon_in != self.base.mon[0].in_features:
            self.base.mon[0] = nn.Linear(mon_in, H)
        # MECHANISM PROBE `biasfix`: removing the 12 constant embedding columns
        # removes a per-unit random bias of std sqrt(12*Var(W)) = sqrt(12/(3*fan_in))
        # -- ~3.5x the Linear's own bias init. biasfix restores exactly that much
        # bias spread and changes nothing else, separating "init-time bias scale"
        # from "a learnable, Adam-preconditioned bias path".
        if biasfix and self.drop_last_item:
            import math
            nn.init.normal_(self.base.mon[0].bias, 0.0,
                            math.sqrt(E["item"] / (3.0 * mon_in)))
        trunk_in = 4 * H + 2 * self.n_side + self.n_glob + d_rest
        if trunk_in != self.base.trunk[0].in_features:
            self.base.trunk[0] = nn.Linear(trunk_in, self.base.trunk[0].out_features)
        self.in_dim = trunk_in

    def embed_mon(self, ids, f):
        B, e = self.base, self.base.emb
        parts = [e["species"](ids[..., 0]), e["item"](ids[..., 1]),
                 e["ability"](ids[..., 2]), e["move"](ids[..., 3:7]).flatten(-2),
                 e["teratype"](ids[..., 7])]
        if not self.drop_last_item:
            parts.append(e["item"](ids[..., 8]))
        parts.append(f)
        if self.tok is not None:
            parts.append(self.tok.expand(*f.shape[:-1], self.tok.shape[0]))
        return B.mon(torch.cat(parts, dim=-1))

    def forward(self, b):
        B = self.base
        am = b.get("am")

        def mon(ids, f, sl):
            return self.embed_mon(ids, f if am is None else torch.cat([f, am[:, sl]], dim=-1))
        parts = [mon(b["a1_ids"], b["a1_f"], 0),
                 B.pool(mon(b["b1_ids"], b["b1_f"], slice(1, 6))),
                 mon(b["a2_ids"], b["a2_f"], 6),
                 B.pool(mon(b["b2_ids"], b["b2_f"], slice(7, 12))),
                 b["sf1"], b["sf2"], b["g"]]
        if "ar" in b:
            parts.append(b["ar"])
        return B.trunk(torch.cat(parts, dim=-1)).squeeze(-1)


def add_trunk_dropout(net, p):
    """Insert Dropout after every trunk ReLU of an ARM-A-family net.

    `LabValueNet` has no dropout at all, so `--dropout` was a SILENT NO-OP for
    `--arm old` in every previous pass. Applied here, after construction, so at
    p=0 the control cell is still the published code path bit for bit and no RNG
    is consumed differently."""
    if not p:
        return net
    base = getattr(net, "base", net)
    layers = []
    for m in base.trunk:
        layers.append(m)
        if isinstance(m, nn.ReLU):
            layers.append(nn.Dropout(p))
    base.trunk = nn.Sequential(*layers)
    return net


def build_net(arm, cap, seed, arch="flat", mon_depth=2, depth=3, slot_emb=1,
              dropout=0.0, add="", drop="", biasfix=0, constok=0):
    torch.manual_seed(seed)
    if arm == "enc2":
        sizes = {"item": 1024, "ability": 1024, "ptype": 32, "move": 2048}
        lay = load_layout()
        if arch == "flat":
            return Enc2FlatNet(sizes, n_feats=lay["N"], width=cap[0], depth=depth,
                               dropout=dropout)
        return Enc2SharedNet(sizes, lay, mon_w=cap[0], mon_depth=mon_depth,
                             width=cap[1], depth=depth, mode=arch,
                             slot_emb=bool(slot_emb), dropout=dropout)
    os.environ["MON_HID"], os.environ["TRUNK"] = str(cap[0]), str(cap[1])
    import importlib
    import labmodel
    import relfeat
    importlib.reload(labmodel)
    assert (relfeat.N_GX, relfeat.N_SX) == (16, 16), "relfeat constants moved"
    from encoder import Vocab
    torch.manual_seed(seed)
    mc, rc = addon_cols(add)
    if len(mc) or len(rc) or drop or constok:
        return add_trunk_dropout(ArmAPlusNet(
            Vocab(frozen=True).sizes(), len(mc), len(rc), drop=drop,
            biasfix=biasfix, constok=constok), dropout)
    net = labmodel.LabValueNet(Vocab(frozen=True).sizes(), use_rel=False)
    n = sum(p.numel() for p in net.parameters())
    if cap in PARAM_COUNTS:
        # PARAM_COUNTS was measured at NUM_MON_FEATS = 73; each per-mon numeric
        # column costs exactly cap[0] weights in the shared per-mon MLP's first
        # layer, so DROP_TIMES_ATTACKED shifts every entry by -cap[0].
        want = PARAM_COUNTS[cap] + (NUM_MON_FEATS - 73) * cap[0]
        assert n == want, "cap %s: %d params, expected %d (stubbed=%s)" % (
            cap, n, want, STUBBED_PE)
    return add_trunk_dropout(net, dropout)


# -------------------------------------------------------------- metrics -----
def predict(net, data, idx, batch=4096):
    net.eval()
    out = np.empty(len(idx), np.float64)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            out[i:i + len(b)] = torch.sigmoid(net(data.batch(b))).numpy()
    return out


def brier_by_band(pred, meta, idx):
    y, band = meta["label_p"][idx], meta["band"][idx]
    r = {"all": float(np.mean((pred - y) ** 2))}
    for b in BANDS:
        m = band == b
        r[b] = float(np.mean((pred[m] - y[m]) ** 2))
    return r
