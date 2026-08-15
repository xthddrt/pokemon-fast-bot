"""The incremental ceiling, measured on the real dataflow graph.

Not a guess and not a redesign: run the encoder on a PARENT and on each of its
real MCTS CHILDREN, recording the output of every array operation. The op
sequence is data-independent, so the two traces align call for call. Then:

  * an operation whose output is bit-identical between parent and child is work
    a perfect incremental encoder skips ENTIRELY.
  * within an operation that did change, the fraction of elements that changed
    bounds how much a finer-grained scheme could save.

The first number is the realistic ceiling (you recompute whole arrays); the
second is the theoretical floor of remaining work. Both are measured.

Also reports the same split for the CHEAP-WIN candidates so they can be priced
individually.
"""
import collections
import json
import os
import random
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import numpy as np  # noqa: E402
import labenv  # noqa: E402,F401
import enc2  # noqa: E402
import enc2_count as EC  # noqa: E402
import npcount  # noqa: E402
import enc2_gate as G  # noqa: E402
import llencoder as LL  # noqa: E402
from perf_delta import children_of  # noqa: E402

BLOCKS = ["1_static_reorder", "2_gather_parse", "3_spread_boosts", "4_speed",
          "5_dmg_gather+mults", "6_dmg_combine_now", "7_ko_matrix_calc",
          "8_ids+per_mon", "9_per_side", "10_relational", "11_cf_prep",
          "12_cf_setup_lv1+lv2", "13_cf_side+tera", "14_global"]


def seed(d):
    return {k: (x.view(npcount.CA) if isinstance(x, np.ndarray) else x)
            for k, x in d.items()}


def trace_one(state_str, v, T, marks):
    """-> (list of (block, opname, output array), block boundaries)."""
    C = LL.parse_batch([state_str], v)
    order1 = LL._slot_order(C["si"][:, :, enc2._SI["active_index"]])
    S = EC.StaticCtx(seed(C), EC.Tables(v), order1)
    marks.clear()
    npcount.reset(record=True)

    def tk(label, _orig=EC._TK_ORIG):
        marks.append((label, len(npcount.TRACE)))
        _orig(label)

    EC._TK = tk
    EC.encode_columnar(seed(C), v, S=S)
    npcount.RECORD[0] = False
    return list(npcount.TRACE), list(marks)


def block_for(i, marks):
    """A checkpoint labels the block that just FINISHED, so ops with index in
    [marks[k], marks[k+1]) belong to the block named by marks[k+1]."""
    for lab, idx in marks:
        if i < idx:
            return lab
    return "14_global"


def tiled_changed(a, b):
    """Elements in CHANGED TILES, under the active/bench tiling an incremental
    encoder would actually use: every axis of length 6 splits into {active} and
    {5 bench}, every axis of length 12 into {2 actives} and {10 bench}. Any
    other axis is left whole. -> elements that must be recomputed."""
    d = a != b
    if not d.any():
        return 0
    sel = []
    for ax, s in enumerate(a.shape):
        if s == 6:
            sel.append([slice(0, 1), slice(1, 6)])
        elif s == 12:
            sel.append([[0, 6], [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]])
        else:
            sel.append([slice(None)])
    tot = 0
    import itertools
    for combo in itertools.product(*sel):
        sub = d
        for ax, ix in enumerate(combo):
            sub = np.take(sub, ix, axis=ax) if isinstance(ix, list) else \
                sub[(slice(None),) * ax + (ix,)]
        if sub.any():
            tot += sub.size
    return tot


def slot_changed(a, b):
    """Per-SLOT granularity: recompute row i / column j of a pair tensor, not
    the whole active/bench half. Every axis of length 6 or 12 is indexed
    individually; all other axes are recomputed whole (they are the channel /
    move / one-hot axes, which are not separable)."""
    d = a != b
    if not d.any():
        return 0
    slot_ax = [i for i, s in enumerate(a.shape) if s in (6, 12)]
    if not slot_ax:
        return a.size
    other = tuple(i for i in range(a.ndim) if i not in slot_ax)
    m = d.any(axis=other) if other else d
    osz = 1
    for i in other:
        osz *= a.shape[i]
    return int(m.sum()) * osz


def main():
    n_parents = int(os.environ.get("NPAR", "40"))
    rng = random.Random(4242)
    rows = []
    for line in open(os.path.join(LAB, "data/pl2/out/labels_a.jsonl")):
        rows.append(json.loads(line))
        if len(rows) >= n_parents * 4:
            break
    rng.shuffle(rows)
    rows = rows[:n_parents]

    v = G.vocab()
    T = enc2.Tables(v)
    EC._TK_ORIG = EC._TK

    tot = collections.Counter()      # block -> total elems
    chg_op = collections.Counter()   # block -> elems in ops whose output changed
    chg_el = collections.Counter()   # block -> individual elements changed
    chg_tile = collections.Counter()  # block -> elems in changed active/bench tiles
    chg_slot = collections.Counter()  # block -> elems in changed per-slot rows/cols
    ncalls = collections.Counter()
    nchg_calls = collections.Counter()
    op_tot = collections.Counter()
    op_chg = collections.Counter()
    n_pairs = 0
    marks = []

    for r in rows:
        kids = children_of(r["s"], rng, max_children=4)
        if not kids:
            continue
        try:
            tp, mp = trace_one(r["s"], v, T, marks)
        except Exception:
            continue
        for k in kids:
            try:
                tc, mc = trace_one(k, v, T, marks)
            except Exception:
                continue
            if len(tc) != len(tp):
                continue          # control flow diverged; skip (never observed)
            n_pairs += 1
            for i, ((na, a), (nb, b)) in enumerate(zip(tp, tc)):
                if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                    continue
                if a.shape != b.shape:
                    continue
                blk = block_for(i, mp)
                tot[blk] += a.size
                ncalls[blk] += 1
                op_tot[na] += a.size
                d = (a != b)
                nd = int(d.sum())
                if nd:
                    chg_op[blk] += a.size
                    chg_el[blk] += nd
                    chg_tile[blk] += tiled_changed(a, b)
                    chg_slot[blk] += slot_changed(a, b)
                    nchg_calls[blk] += 1
                    op_chg[na] += a.size

    print("parent->child pairs traced: %d\n" % n_pairs)
    print("=== incremental ceiling, per block (mean elems per transition) ===")
    print("%-24s %8s %8s %8s %8s %8s" %
          ("block", "elems", "whole", "act/bnch", "per-slot", "per-elem"))
    print("-" * 80)
    TT = CO = CE = CT = CS = 0
    NC = NCC = 0
    for b in BLOCKS:
        if not tot[b]:
            continue
        t = tot[b] / n_pairs
        co, ce = chg_op[b] / n_pairs, chg_el[b] / n_pairs
        ct, cs = chg_tile[b] / n_pairs, chg_slot[b] / n_pairs
        TT += t; CO += co; CE += ce; CT += ct; CS += cs
        NC += ncalls[b] / n_pairs
        NCC += nchg_calls[b] / n_pairs
        print("%-24s %8.0f %8.0f %8.0f %8.0f %8.0f"
              % (b, t, co, ct, cs, ce))
    print("-" * 80)
    print("%-24s %8.0f %8.0f %8.0f %8.0f %8.0f"
          % ("TOTAL", TT, CO, CT, CS, CE))
    print("%-24s %8s %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
          % ("  as % of full", "", 100*CO/TT, 100*CT/TT, 100*CS/TT, 100*CE/TT))
    print("\narray ops per leaf: %.0f, of which output changed: %.0f (%.1f%%)"
          % (NC, NCC, 100 * NCC / NC))
    print("\nINCREMENTAL SPEEDUP CEILING (arithmetic, per leaf)")
    print("  whole-array granularity  (cache an op or redo it): %.2fx  [%.2fx calls]"
          % (TT / max(CO, 1), NC / max(NCC, 1)))
    print("  active/bench tiling      (coarse design):          %.2fx" % (TT / max(CT, 1)))
    print("  per-slot rows/columns    (the design to build):    %.2fx" % (TT / max(CS, 1)))
    print("  per-element granularity  (unreachable floor):      %.2fx" % (TT / max(CE, 1)))

    print("\n=== ops that NEVER changed (pure cacheable work), by elems ===")
    print("%-22s %10s %10s %8s" % ("op", "elems", "changed", "%"))
    for k, e in sorted(op_tot.items(), key=lambda x: -x[1])[:14]:
        c = op_chg.get(k, 0)
        print("%-22s %10.0f %10.0f %7.1f%%"
              % (k, e / n_pairs, c / n_pairs, 100 * c / max(e, 1)))


if __name__ == "__main__":
    main()
