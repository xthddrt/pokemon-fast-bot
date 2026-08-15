"""ADDITIVE ENCODER SEARCH -- build the enc2 "add-on" caches, aligned to ARM A.

The premise: arm A's per-mon block is demonstrably good (typing +0.0084, species
+0.0062, moves +0.0089, boosts +0.0131 on permutation). enc2 REPLACED it. This
pass instead APPENDS to arm A only the derived blocks arm A structurally lacks,
and makes each block earn its place.

THE ONE HARD PART -- SLOT ALIGNMENT. The two caches order the twelve mons
DIFFERENTLY:

  * arm A  (`valuenet/encoder.py`, BENCH_SORT=1): slot 0 = active, slots 1..5 =
    the bench sorted by (species vocab id, party index).
  * enc2   (`llencoder._slot_order`):             slot 0 = active, slots 1..5 =
    the rest in ASCENDING PARTY ORDER.

Both are deterministic but they are not the same permutation, and the corpus
uses 720 distinct party orders, so it is not a constant. Gluing enc2's per-mon
setup block onto arm A's per-mon vector without the remap would pair each mon's
arm-A features with a DIFFERENT mon's setup features -- silently, and the net
would still train. So this builds, per row, the map

    perm[row, j] = enc2 slot holding the mon that arm A puts in slot j

by re-parsing the state and calling `encoder.bench_order` -- the ONE definition
of arm A's bench permutation -- rather than reimplementing it.

The NOT-per-mon blocks (KO matrix, speed/priority, per-side tera) go straight to
the trunk, exactly where `labmodel` already puts relational features when
use_rel=1. They are internally self-consistent in enc2 slot space and belong to
no arm-A slot, so they need no remap.

  python vt_addon.py perm <labels_shard.jsonl> <out_dir> <tag>
  python vt_addon.py join <out_dir>
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labenv  # noqa: F401,E402

TAGS = ("a", "b", "c", "d")

# ------------------------------------------------------------------ groups ---
# PER-MON add-ons: appended to arm A's 73 per-mon numerics, so they pass through
# the SHARED per-mon MLP and the bench sum-pool -- arm A's own aggregation,
# which ENCODER_V2_RESULT proved beats concatenation at every matched capacity.
MON_GROUPS = {
    "setup": r"^cf_mon%d\.",          # §4 per-mon setup counterfactual, 14 cols
    "caps": r"^mon%d\.cap_",          # §2 per-mon capabilities, 13 bools
}
# TRUNK add-ons: about pairs or about a whole side, so they belong to no slot.
REST_GROUPS = {
    "ko": r"^rel\.ko_now_",                                              # 360
    "speed": r"^rel\.(pairspeed|active_speed|order_|prio_|revenge_|mprio)",  # 106
    "tera": r"^cf_side\d\.",                                             # 10
}


def group_cols(lay):
    """-> (mon_cols {name: int64[12, d]}, rest_cols {name: int64[d]})."""
    nm = lay["names"]
    mon, rest = {}, {}
    for g, pat in MON_GROUPS.items():
        blocks = [np.array([i for i, n in enumerate(nm) if re.match(pat % k, n)], np.int64)
                  for k in range(12)]
        w = len(blocks[0])
        assert all(len(b) == w for b in blocks), g
        mon[g] = np.stack(blocks)
    for g, pat in REST_GROUPS.items():
        rest[g] = np.array([i for i, n in enumerate(nm) if re.match(pat, n)], np.int64)
    return mon, rest


# -------------------------------------------------------------------- perm ---
def perm_shard(src, out_dir, tag):
    from poke_engine import State
    from encoder import Vocab, bench_order
    v = Vocab(frozen=True)
    os.makedirs(out_dir, exist_ok=True)
    states = []
    for line in open(src):
        r = json.loads(line)
        if "label_p" in r:
            states.append(r["s"])
    n = len(states)
    perm = np.zeros((n, 12), np.int8)
    t0 = time.time()
    for i, s in enumerate(states):
        st = State.from_string(s)
        for si, side in enumerate((st.side_one, st.side_two)):
            a = int(side.active_index)
            mons = list(side.pokemon)
            # arm A slot order, as PARTY indices: active, then bench_order
            party = [a] + [t[0] for t in bench_order(mons, a, v)]
            # party index -> enc2 slot (llencoder._party_to_slot), + 6 for side two
            for j, p in enumerate(party):
                perm[i, si * 6 + j] = (0 if p == a else (p + 1 if p < a else p)) + 6 * si
    np.save(os.path.join(out_dir, "perm_%s.npy" % tag), perm)
    print("[%s] %d rows %.0fs (%.3f ms/state)" % (tag, n, time.time() - t0,
                                                  1000 * (time.time() - t0) / n), flush=True)


def join(out_dir):
    lay = json.load(open(os.path.join(out_dir, "enc2_layout.json")))
    mon_g, rest_g = group_cols(lay)
    perm = np.concatenate([np.load(os.path.join(out_dir, "perm_%s.npy" % t)) for t in TAGS])
    feats = np.load(os.path.join(out_dir, "enc2_feats.npy"), mmap_mode="r")
    n = len(perm)
    assert n == feats.shape[0], (n, feats.shape)
    # a permutation of 0..11 in every row, and side-blocks never cross
    assert (np.sort(perm, axis=1) == np.arange(12)).all(), "perm is not a permutation"
    assert (perm[:, :6] < 6).all() and (perm[:, 6:] >= 6).all(), "perm crosses sides"

    mon_names = sum(([lay["names"][c] for c in mon_g[g][0]] for g in sorted(mon_g)), [])
    mon_cols = np.concatenate([mon_g[g] for g in sorted(mon_g)], axis=1)   # (12, D)
    rest_names = sum(([lay["names"][c] for c in rest_g[g]] for g in sorted(rest_g)), [])
    rest_cols = np.concatenate([rest_g[g] for g in sorted(rest_g)])
    D, R = mon_cols.shape[1], len(rest_cols)

    am = np.lib.format.open_memmap(os.path.join(out_dir, "addon_mon.npy"), mode="w+",
                                   dtype=np.float16, shape=(n, 12, D))
    ar = np.lib.format.open_memmap(os.path.join(out_dir, "addon_rest.npy"), mode="w+",
                                   dtype=np.float16, shape=(n, R))
    CH = 20000
    for i in range(0, n, CH):
        f = np.asarray(feats[i:i + CH])
        k = len(f)
        # gather in ENC2 slot order, then permute rows into ARM A slot order
        tmp = f[:, mon_cols.reshape(-1)].reshape(k, 12, D)
        am[i:i + k] = np.take_along_axis(
            tmp, perm[i:i + k].astype(np.int64)[:, :, None], axis=1)
        ar[i:i + k] = f[:, rest_cols]
    am.flush(), ar.flush()

    off, groups = 0, {}
    for g in sorted(mon_g):
        w = mon_g[g].shape[1]
        groups["mon." + g] = [off, off + w]
        off += w
    off = 0
    for g in sorted(rest_g):
        w = len(rest_g[g])
        groups["rest." + g] = [off, off + w]
        off += w
    json.dump({"D_mon": int(D), "D_rest": int(R), "groups": groups,
               "mon_names": mon_names, "rest_names": rest_names},
              open(os.path.join(out_dir, "addon_layout.json"), "w"))
    print("addon_mon %s  addon_rest %s" % ((n, 12, D), (n, R)))
    print(json.dumps(groups, indent=1))

    # constant-column census: which added columns never move on THIS corpus
    for nmlist, arr, lbl in ((mon_names, am, "mon"), (rest_names, ar, "rest")):
        x = np.asarray(arr[::20], np.float32)
        x = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        const = [nmlist[i] for i in np.flatnonzero(x.std(axis=0) == 0)]
        print("%s: %d/%d columns CONSTANT on this corpus" % (lbl, len(const), len(nmlist)))
        json.dump(const, open(os.path.join(out_dir, "addon_const_%s.json" % lbl), "w"))


# ------------------------------------------------- reshape: derived PER-MON --
# WHY THIS EXISTS. enc2's derived blocks are TRUNK-level: one flat 360-column KO
# matrix, one 36-column pairwise speed matrix. Trunk parameters are learned once
# each from n training rows. The shared per-mon MLP's parameters are learned
# from 12n, because the same weights see all twelve slots -- that is the same
# mechanism that made sum-pooling beat concatenation. So the KO matrix is
# REFACTORED, not re-computed: every mon gets ITS OWN ROW of it,
#     "how I fare against each of the six opponents"  and
#     "how each of the six fares against me",
# which is the identical information, factored so ONE "am I winning my
# matchups" function is learned and applied twelve times.
#
# ORIENTATION, which is the part that is easy to get wrong: enc2 stores BOTH
# KO directions with OUR mon on the row axis (`enc2.py` transposes block 1
# explicitly), and `pairspeed[i][j]` is "our i moves first vs their j". So for a
# mon on side TWO the row/column roles swap, and the speed bit must be
# complemented -- exact, including ties, because `_first` returns 0.5 on a tie.
KB = ("ohko", "2hko", "3hko", "4plus", "never")
KO_SCALAR = (1.0, 0.5, 1.0 / 3.0, 0.25, 0.0)     # "fraction of the job per turn"


def permon_derived(lay):
    """-> (cols int64[12, 70], flip int64[...]) column indices into enc2's
    feature vector, in ENC2 slot order; `flip` indexes the columns of that block
    that must be replaced by 1-x for the side-two slots."""
    c = {n: i for i, n in enumerate(lay["names"])}
    ko = lambda d, i, j: [c["rel.ko_now_%s_%d%d_%s" % (d, i, j, b)] for b in KB]
    rows = []
    for k in range(12):
        if k < 6:                       # side one: row index is this mon
            out = sum((ko("us_to_them", k, o) for o in range(6)), [])
            inc = sum((ko("them_to_us", k, o) for o in range(6)), [])
            spd = [c["rel.pairspeed_%d%d" % (k, o)] for o in range(6)]
        else:                           # side two: this mon is the COLUMN
            j = k - 6
            out = sum((ko("them_to_us", o, j) for o in range(6)), [])
            inc = sum((ko("us_to_them", o, j) for o in range(6)), [])
            spd = [c["rel.pairspeed_%d%d" % (o, j)] for o in range(6)]
        mp = [c["rel.mprio_s%d_m%d" % (k, m)] for m in range(4)]
        rows.append(out + inc + spd + mp)             # 30 + 30 + 6 + 4 = 70
    return np.array(rows, np.int64)


def join2(out_dir):
    """Rewrite addon_mon.npy as a SUPERSET: the original per-mon add-ons at
    unchanged offsets, plus the reshaped per-mon KO / speed / priority blocks.
    addon_rest.npy is not touched -- the trunk atoms are sub-ranges of what is
    already there."""
    lay = json.load(open(os.path.join(out_dir, "enc2_layout.json")))
    old = json.load(open(os.path.join(out_dir, "addon_layout.json")))
    mon_g, _ = group_cols(lay)
    perm = np.concatenate([np.load(os.path.join(out_dir, "perm_%s.npy" % t)) for t in TAGS])
    feats = np.load(os.path.join(out_dir, "enc2_feats.npy"), mmap_mode="r")
    n = len(perm)
    base = np.concatenate([mon_g[g] for g in sorted(mon_g)], axis=1)     # (12, 27)
    pmd = permon_derived(lay)                                            # (12, 70)
    B, P = base.shape[1], pmd.shape[1]
    D = B + P + 12                       # + the 12-column scalar KO variant
    am = np.lib.format.open_memmap(os.path.join(out_dir, "addon_mon.npy"), mode="w+",
                                   dtype=np.float16, shape=(n, 12, D))
    kw = np.array(KO_SCALAR, np.float32)
    CH = 20000
    for i in range(0, n, CH):
        f = np.asarray(feats[i:i + CH], np.float32)
        k = len(f)
        blk = np.concatenate([f[:, base.reshape(-1)].reshape(k, 12, B),
                              f[:, pmd.reshape(-1)].reshape(k, 12, P)], axis=2)
        # side two: "do I move first" is the complement of the stored bit
        blk[:, 6:, B + 60:B + 66] = 1.0 - blk[:, 6:, B + 60:B + 66]
        # scalar KO variant: collapse each 5-way bucket to 1 / (hits to KO)
        sc = (blk[:, :, B:B + 60].reshape(k, 12, 12, 5) * kw).sum(-1)     # (k,12,12)
        blk = np.concatenate([blk, sc], axis=2)
        am[i:i + k] = np.take_along_axis(blk, perm[i:i + k].astype(np.int64)[:, :, None],
                                         axis=1)
    am.flush()

    nm = lay["names"]
    names = old["mon_names"] + [nm[c] + "|pm" for c in pmd[0]] + \
        ["ko_pm_scalar_%d" % i for i in range(12)]
    g = dict(old["groups"])
    g["mon.ko_pm"] = [B, B + 60]
    g["mon.spd_pm"] = [B + 60, B + 66]
    g["mon.mprio_pm"] = [B + 66, B + 70]
    g["mon.ko_pm_s"] = [B + 70, B + 82]
    # trunk ATOMS: sub-ranges of the existing rest cache (ko | pairspeed |
    # active-pair speed & priority | per-move priority | tera)
    g["rest.pairspeed"] = [360, 396]
    g["rest.spdrest"] = [396, 418]
    g["rest.mprio"] = [418, 466]
    json.dump({"D_mon": int(D), "D_rest": old["D_rest"], "groups": g,
               "aliases": {"speed": ["pairspeed", "spdrest", "mprio"]},
               "mon_names": names, "rest_names": old["rest_names"]},
              open(os.path.join(out_dir, "addon_layout.json"), "w"))
    print("addon_mon rewritten %s; groups:" % ((n, 12, D),))
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "join2":
        join2(sys.argv[2])
    elif sys.argv[1] == "perm":
        perm_shard(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        join(sys.argv[2])
