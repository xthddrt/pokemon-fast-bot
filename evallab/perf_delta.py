"""How much of the 1,511-column vector actually CHANGES from an MCTS parent to
its child?

A child differs from its parent by ONE joint move. This measures, on REAL
transitions produced by the engine (`generate_instructions` + `apply_instructions`,
the same calls the search itself makes), what fraction of the columns move --
overall and per feature block. That fraction is the ceiling on what an
incremental ("efficiently updatable") encoder can save.

Also reports how often the shared StaticCtx is invalidated by the transition
(a terastallization, a PP exhaustion or a new disable -- enc2 §6's precondition).
"""
import json
import os
import random
import sys
import collections

os.environ.setdefault("OMP_NUM_THREADS", "1")
LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import numpy as np  # noqa: E402
import labenv  # noqa: E402,F401
import enc2  # noqa: E402
import enc2_gate as G  # noqa: E402
import llencoder as LL  # noqa: E402
from poke_engine import State, generate_instructions, monte_carlo_tree_search  # noqa: E402
from playout import arm_to_move  # noqa: E402

L = enc2.DEFAULT_LAYOUT


def legal_arms(state, seed):
    res = monte_carlo_tree_search(state, 0, 1, 1, seed)
    return [m.move_choice for m in res.side_one], [m.move_choice for m in res.side_two]


def children_of(state_str, rng, max_children=8):
    """Real children: every (my arm, their arm) pair the search would expand,
    taking the MODAL branch of each (the search expands all branches; the modal
    one is representative and keeps the corpus size sane)."""
    st = State.from_string(state_str)
    try:
        a1, a2 = legal_arms(st, rng.randrange(1 << 30))
    except Exception:
        return []
    if not a1 or not a2:
        return []
    pairs = [(x, y) for x in a1 for y in a2]
    rng.shuffle(pairs)
    out = []
    for p1, p2 in pairs[:max_children]:
        try:
            br = [b for b in generate_instructions(st, arm_to_move(p1), arm_to_move(p2))
                  if b.percentage > 0]
            if not br:
                continue
            b = max(br, key=lambda x: x.percentage)
            out.append(State.from_string(state_str).apply_instructions(b).to_string())
        except Exception:
            continue
    return out


def block_of(name):
    if name.startswith("mon"):
        return "1_per_mon"
    if name.startswith("side"):
        return "2_per_side"
    if name.startswith("cf_mon"):
        return "4_cf_per_mon"
    if name.startswith("cf_side"):
        return "4_cf_per_side"
    if name.startswith("g."):
        return "5_global"
    return "3_relational"


def rel_sub(name, i):
    """finer split of the relational block by column index"""
    if i < 360:
        return "3a_ko_matrix(360)"
    if i < 396:
        return "3b_pair_speed(36)"
    if i < 398:
        return "3c_speed_margin(2)"
    if i < 414:
        return "3d_move_order16(16)"
    if i < 418:
        return "3e_prio_flags(4)"
    return "3f_prio_brackets(48)"


def main():
    n_parents = int(os.environ.get("NPAR", "150"))
    rng = random.Random(20260814)
    rows = []
    for line in open(os.path.join(LAB, "data/pl2/out/labels_a.jsonl")):
        rows.append(json.loads(line))
        if len(rows) >= n_parents * 4:
            break
    rng.shuffle(rows)
    rows = rows[:n_parents]

    v = G.vocab()
    T = enc2.Tables(v)
    names = L.names
    O_REL = L.O_REL

    tot_changed = np.zeros(L.N, np.int64)
    n_pairs = 0
    n_static_invalid = 0
    per_child_counts = []
    parent_used = 0
    cause = collections.Counter()

    for r in rows:
        kids = children_of(r["s"], rng)
        if not kids:
            continue
        parent_used += 1
        allst = [r["s"]] + kids
        C = LL.parse_batch(allst, v)
        C1 = LL.parse_batch(allst[:1], v)
        order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])
        S = enc2.StaticCtx(C1, T, order1)
        _, f = enc2.encode_columnar(C, v, S=None)   # cold path: fully correct
        # static-validity check: the twelve mons' pp / disabled / tera must hold
        # Only what StaticCtx actually keys on: has_pp is `pp > 0`, so a
        # decrement 16->15 is NOT an invalidation -- only exhaustion is.
        m = C["mi"].reshape(C["n"], 12, -1)
        keys = {
            "pp_exhausted": (m[:, :, enc2._MI["pp0"]:enc2._MI["pp0"] + 4] > 0),
            "disabled": (m[:, :, enc2._MI["disabled0"]:enc2._MI["disabled0"] + 4] > 0),
            "terastallized": (m[:, :, enc2._MI["terastallized"]] > 0),
        }
        key = np.concatenate([v.reshape(C["n"], -1) for v in keys.values()], 1)
        for j in range(1, len(allst)):
            d = f[j] != f[0]
            tot_changed += d
            per_child_counts.append(int(d.sum()))
            if not np.array_equal(key[j], key[0]):
                n_static_invalid += 1
            for kn, kv in keys.items():
                if not np.array_equal(kv[j], kv[0]):
                    cause[kn] += 1
            n_pairs += 1

    print("parents used            %d" % parent_used)
    print("parent->child pairs     %d" % n_pairs)
    c = np.array(per_child_counts)
    print("columns changed / child: mean %.1f  median %d  p90 %d  max %d  "
          "(of %d = %.1f%% mean)"
          % (c.mean(), np.median(c), np.percentile(c, 90), c.max(), L.N,
             100 * c.mean() / L.N))
    print("static ctx invalidated  %d / %d = %.1f%% of transitions"
          % (n_static_invalid, n_pairs, 100 * n_static_invalid / max(n_pairs, 1)))
    for kn, kc in cause.most_common():
        print("    by %-16s %5d = %.1f%%" % (kn, kc, 100 * kc / max(n_pairs, 1)))

    # ---- per block ------------------------------------------------------
    bl_tot = collections.Counter()
    bl_chg = collections.Counter()
    for i, nm in enumerate(names):
        b = block_of(nm)
        if b == "3_relational":
            b = rel_sub(nm, i - O_REL)
        bl_tot[b] += 1
        bl_chg[b] += tot_changed[i]
    print("\n%-22s %8s %14s %12s" % ("block", "cols", "chg/child", "% of block"))
    print("-" * 60)
    for b in sorted(bl_tot):
        per = bl_chg[b] / max(n_pairs, 1)
        print("%-22s %8d %14.1f %11.1f%%"
              % (b, bl_tot[b], per, 100 * per / bl_tot[b]))
    print("-" * 60)
    print("%-22s %8d %14.1f %11.1f%%"
          % ("TOTAL", L.N, tot_changed.sum() / max(n_pairs, 1),
             100 * tot_changed.sum() / max(n_pairs, 1) / L.N))

    # never-changing columns
    dead = int((tot_changed == 0).sum())
    print("\ncolumns that NEVER changed in any transition: %d (%.1f%%)"
          % (dead, 100 * dead / L.N))
    np.save(os.path.join(LAB, "data/enc2/delta_counts.npy"), tot_changed)


if __name__ == "__main__":
    main()
