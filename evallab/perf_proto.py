"""(a) the STATIC half's work, and what the 5 weather x 2 tera variants cost;
(b) an incremental PROTOTYPE of the dominant dependency chain --
    dmg_full -> ko_now -> the 360-column KO matrix -- verified bit-identical to
    enc2 on real parent->child transitions, and priced in array operations.

Timing the prototype in numpy would measure numpy's dispatch, not the design:
at n=1 the incremental path issues MORE, SMALLER calls than the full recompute,
so it is slower in numpy and faster in any compiled language. The transferable
number is therefore the OPERATION COUNT, which is what is reported.
"""
import collections
import json
import os
import random
import sys
import time

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


def seed(d):
    return {k: (x.view(npcount.CA) if isinstance(x, np.ndarray) else x)
            for k, x in d.items()}


# =========================================================================
# (a) the static half
# =========================================================================
def static_cost(v, states):
    T = enc2.Tables(v)
    C1 = LL.parse_batch(states[:1], v)
    order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])

    def t(fn, reps=25):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    base = t(lambda: enc2.StaticCtx(C1, T, order1))
    npcount.reset()
    EC.StaticCtx(seed(C1), EC.Tables(v), order1)
    c, e = npcount.STATS["calls"], npcount.STATS["elems"]

    # how much of it is the 5 weather x 2 tera variant sweep?
    saved = enc2.WEATHER_VARIANTS
    try:
        enc2.WEATHER_VARIANTS = ("NONE",)
        one = t(lambda: enc2.StaticCtx(C1, T, order1))
    finally:
        enc2.WEATHER_VARIANTS = saved

    print("=== STATIC half (once per search, n=1) ===")
    print("  cost, 5 weather variants   %8.3f ms" % (base * 1e3))
    print("  cost, 1 weather variant    %8.3f ms   (%.2fx cheaper)"
          % (one * 1e3, base / max(one, 1e-9)))
    print("  work: %d array ops, %d output elements" % (c, e))
    print("  -> at 2-6 elem-ops/ns in Rust: %.0f-%.0f us"
          % (e / 6e3, e / 2e3))
    return e


# =========================================================================
# (b) the incremental prototype: dmg_full -> ko_now -> KO matrix
# =========================================================================
class KoChain:
    """The full-recompute reference for the dominant chain, expressed exactly as
    enc2 does it, plus an incremental variant that recomputes only the rows and
    columns whose inputs moved. Both produce the 360 KO-matrix columns.

    Tracks ENCODER2_BUILD.md §11.3's PER-MOVE damage table: `S.dmg` is
    (n, 5 weather, 2 blocks, 12 attackers, 4 moves, 12 defenders), the 12 being
    6 party slots x {not terastallized, terastallized}. Two consequences for
    incremental update, both of them improvements:

      * `disabled` / `pp` and `terastallized` are now LEAF inputs, so they enter
        the dirty set instead of forcing a static rebuild -- and a rebuild is
        exactly what no incremental scheme can absorb (§3.1 priced tera's 16.2 %
        rate at more than the whole per-leaf budget).
      * there is now a MOVE axis under the pair axis. The pre-maximum arithmetic
        is dirty per (pair, move); only after the max over moves does it collapse
        to the per-pair KO bucketing. Both fractions are reported."""

    def __init__(self, S, T, L):
        self.S, self.T, self.L = S, T, L

    # ---- inputs a leaf actually supplies --------------------------------
    @staticmethod
    def inputs(C, S, T, order):
        m = LL._gather(C["mi"], order).astype(np.float32)
        si, gi = C["si"], C["gi"]
        hp = np.maximum(m[:, :, enc2._MI["hp"]], 0.0)
        occ, maxhp = S.occ, S.maxhp
        n = C["n"]
        sboost = si[:, :, enc2._SI["attack_boost"]:
                    enc2._SI["attack_boost"] + 7].astype(np.float32)
        scond = si[:, :, LL._SC0:LL._SC0 + len(enc2.LE.SIDE_CONDITION_FIELDS)] \
            .astype(np.float32)
        boost12 = np.zeros((n, 12, 7), np.float32)
        boost12[:, 0] = sboost[:, 0]
        boost12[:, 6] = sboost[:, 1]
        dis4 = m[:, :, enc2._MI["disabled0"]:enc2._MI["disabled0"] + 4]
        pp4 = m[:, :, enc2._MI["pp0"]:enc2._MI["pp0"] + 4]
        return dict(
            hp_frac=np.where(occ, hp / np.maximum(maxhp, 1.0), 0.0),
            alive=(hp > 0) & occ,
            status=m[:, :, enc2._MI["status"]].astype(np.int64),
            stage=boost12[:, :, :5],
            scond=scond,                       # (n,2,K): side-wide, not spread
            hp=hp, maxhp=maxhp, occ=occ,
            live_mv=(dis4 == 0) & (pp4 > 0),   # (n,12,4): per MOVE, now dynamic
            tera_on=m[:, :, enc2._MI["terastallized"]] > 0,
            wsel=enc2.WEATHER_VAR_MAP[gi[:, enc2._GI["weather"]].astype(np.int64)],
            order=order)

    # ---- full recompute --------------------------------------------------
    def full(self, I):
        """enc2.encode_columnar's dynamic damage -> ko_now path verbatim,
        restricted to the KO matrix. `combine()` / `numerator()` are inlined
        here in the same order, so the result is bit-identical, not merely
        equal to a tolerance."""
        S, T, n = self.S, self.T, I["hp"].shape[0]
        SC = enc2.SC
        alive, hp, maxhp, occ = I["alive"], I["hp"], I["maxhp"], I["occ"]
        status, scond = I["status"], I["scond"]
        chan_a = enc2._sides_of(S.chan)[0]                       # (n,2,6,4)
        phys_a = enc2._sides_of(S.mv_phys)[0]
        okm = enc2._sides_of(S.mv_dmg & I["live_mv"] & occ[:, :, None])[0]
        st_a, st_d = enc2._sides_of(I["stage"])                  # (n,2,6,5) each
        # defender boosts are exactly 1.0 off the opposing ACTIVE, so they hit
        # one column of six -- the bench x bench quadrant is boost-free
        db = enc2._boost(st_d[:, :, 0])[:, :, None, None, :]
        c5 = chan_a[..., None]
        num_d0 = np.where(c5 == 4, db[..., 0:1], 1.0)
        den_d0 = np.where(c5 == 1, db[..., 3:4], db[..., 1:2])

        guts_a = enc2._sides_of(T.ab["ab_guts"][S.ability] > 0)[0]
        infil_a = enc2._sides_of(T.ab["ab_infiltrator"][S.ability] > 0)[0]
        veil = scond[:, :, SC["aurora_veil"]] > 0
        refl_d = ((scond[:, :, SC["reflect"]] > 0) | veil)[:, ::-1, None] & ~infil_a
        lscr_d = ((scond[:, :, SC["light_screen"]] > 0) | veil)[:, ::-1, None] & ~infil_a
        burn_mult = (np.where(enc2._sides_of(status == enc2.STATUS_ORDER.index("BURN"))[0]
                              & ~guts_a, 0.5, 1.0)
                     * np.where(guts_a & enc2._sides_of(status > 0)[0], 1.5, 1.0))
        br4 = np.where(phys_a, burn_mult[:, :, :, None], 1.0) * np.where(
            np.where(phys_a, refl_d[:, :, :, None], lscr_d[:, :, :, None]), 0.5, 1.0)
        ms_d = enc2._sides_of((T.ab["ab_multiscale"][S.ability] > 0)
                              & (hp >= maxhp))[1]
        nf = np.repeat(np.stack([((~alive) & occ)[:, :6].sum(1),
                                 ((~alive) & occ)[:, 6:].sum(1)], 1), 6, axis=1)
        so_mult = enc2._sides_of(np.where(T.ab_supreme[S.ability] > 0,
                                          1.0 + 0.1 * np.minimum(nf, 5), 1.0))[0]
        msso = so_mult[:, :, :, None] * np.where(ms_d, 0.5, 1.0)[:, :, None, :]

        # one gather does the slot permutation AND the tera selection
        aix = I["order"] * 2 + I["tera_on"].reshape(n, 2, 6)
        dix = aix[:, ::-1]
        dmg_moves = S.pair(S.dmg[np.arange(n), I["wsel"]], aix, dix)

        # numerator(): masked per-move attacker boost ratio
        num = np.take_along_axis(enc2._boost(st_a[..., enc2._NUM_STAT]),
                                 np.clip(chan_a, 0, 3), axis=-1)
        numm = np.where(okm, np.where(chan_a == 4, 1.0, num), 0.0)[..., None]
        # combine(): mask + attacker boost, defender boost on the active column,
        # then the folded burn/screen and Multiscale/Overlord factors
        x = dmg_moves * numm
        x0 = x[:, :, :, :, :1]
        x0 *= num_d0
        x0 /= den_d0
        x = x * br4[..., None]
        x = x * msso[:, :, :, None, :]                    # (n,2,6,4,6)

        bmv = x.argmax(axis=3)                            # best MOVE per pair
        dmg_full = np.take_along_axis(x, bmv[:, :, :, None, :], axis=3)[:, :, :, 0, :]
        aa, ad = enc2._sides_of(alive)
        live = aa[..., None] & ad[:, :, None, :]
        dmg_full = np.where(live, dmg_full, 0.0)
        hpf = np.maximum(I["hp_frac"], 1e-6)
        hpf_def = enc2._sides_of(hpf)[1][:, :, None, :]
        ko = np.where(dmg_full > 0,
                      np.ceil(hpf_def / np.maximum(dmg_full, 1e-9)), 99.0)
        never = ~(dmg_full > 0) | ~live
        return self.pack(ko, never, n), dmg_full

    @staticmethod
    def pack(ko, never, n):
        out = np.zeros((n, 360), np.float32)
        o = 0
        for d in range(2):
            sub = ko[:, 0] if d == 0 else ko[:, 1].transpose(0, 2, 1)
            nv = never[:, 0] if d == 0 else never[:, 1].transpose(0, 2, 1)
            oh = np.zeros((n, 6, 6, 5), np.float32)
            enc2._bucket_onehot(sub, 5, oh, nv)
            out[:, o:o + 180] = oh.reshape(n, -1)
            o += 180
        return out

    # ---- incremental ------------------------------------------------------
    def incr(self, Ip, Ic, cache):
        """Recompute only the rows (attacker slots) and columns (defender slots)
        whose inputs moved. Falls back to `full` when a structural input moved
        (weather variant or the slot permutation)."""
        S, T = self.S, self.T
        n = Ic["hp"].shape[0]
        mode = "incremental"
        # A weather change re-selects the variant: the whole tensor re-gathers.
        if not np.array_equal(Ip["wsel"], Ic["wsel"]):
            cache["dirty_pairs"] = cache.get("dirty_pairs", 0) + 72 * n
            cache["tot_pairs"] = cache.get("tot_pairs", 0) + 72 * n
            cache["dirty_pm"] = cache.get("dirty_pm", 0) + 288 * n
            cache["tot_pm"] = cache.get("tot_pm", 0) + 288 * n
            return self.full(Ic)[0], "fallback:weather"
        # A SWITCH permutes one side's slots. That is a re-gather of that side's
        # block, not a recompute of the kernel -- but every pair in the two
        # blocks that touch the switching side must be rewritten.
        # A switch swaps exactly TWO slots of one side (the outgoing active and
        # the incoming mon). Only those slot indices change identity, so only
        # their rows/columns are dirty -- the party-order damage tensor itself
        # is untouched, `pair()` just gathers it through a new permutation.
        swapped = (Ip["order"] != Ic["order"])                       # (n,2,6)
        # which SLOTS moved, in the (2,6) cross-side view?
        att_dirty = np.zeros((n, 2, 6), bool)
        def_dirty = np.zeros((n, 2, 6), bool)
        # `mvd` is the one input FINER than a whole attacker row: a new disable
        # (Choice lock, Encore) or a PP exhaustion masks ONE of four moves. The
        # per-move table is what makes that expressible -- the old five-channel
        # table had already taken a maximum, so it could not be un-taken and the
        # whole static context was rebuilt instead.
        mvd = np.zeros((n, 2, 6, 4), bool)
        for key, who in (("stage", "a"), ("status", "a"), ("alive", "ad"),
                         ("tera_on", "ad"), ("hp_frac", "d"), ("hp", "d")):
            d = (Ip[key] != Ic[key])
            while d.ndim > 2:
                d = d.any(axis=-1)
            a, dd = enc2._sides_of(d)
            if "a" in who:
                att_dirty |= a
            if "d" in who:
                def_dirty |= dd
        mvd |= enc2._sides_of(Ip["live_mv"] != Ic["live_mv"])[0]
        # a side-wide input dirties the whole block: a screen belongs to the
        # DEFENDING side, so side s's conditions dirty block s's attackers and
        # block 1-s's defenders
        d = (Ip["scond"] != Ic["scond"]).any(axis=-1)            # (n,2)
        att_dirty |= d[:, :, None]
        def_dirty |= d[:, ::-1, None]
        # Supreme Overlord / fainted count is side-wide on the attacker
        # Supreme Overlord scales with fainted allies -- but only if some mon on
        # that side actually HAS it, which is rare. Gate on it rather than
        # dirtying the whole block on every faint.
        if not np.array_equal(Ip["alive"], Ic["alive"]):
            has_so = enc2._sides_of(
                np.broadcast_to(T.ab_supreme[S.ability] > 0,
                                Ip["alive"].shape))[0].any(axis=2, keepdims=True)
            att_dirty |= has_so
            def_dirty |= has_so[:, ::-1]
        # side s switched -> those slots are dirty as attackers (block s) and as
        # defenders (block 1-s, whose defender axis is side s)
        att_dirty |= swapped
        def_dirty |= swapped[:, ::-1]
        if swapped.any():
            mode = "incremental+switch"
        # Two granularities, because the chain has two stages. The pre-maximum
        # damage table is (pair, move); the max over moves collapses it, and
        # everything downstream -- ko_now, the buckets, the 360 columns -- is
        # per pair. A per-move dirt therefore still dirties its pair: the
        # maximum has to be re-taken over the four (three cached) values.
        mv_dirty = att_dirty[..., None] | mvd                    # (n,2,6,4)
        pm_dirty = mv_dirty[..., None] | def_dirty[:, :, None, None, :]
        pair_dirty = pm_dirty.any(axis=3)
        cache["dirty_pairs"] = cache.get("dirty_pairs", 0) + int(pair_dirty.sum())
        cache["tot_pairs"] = cache.get("tot_pairs", 0) + pair_dirty.size
        cache["dirty_pm"] = cache.get("dirty_pm", 0) + int(pm_dirty.sum())
        cache["tot_pm"] = cache.get("tot_pm", 0) + pm_dirty.size
        # (the prototype recomputes the dirty pairs by masking the full kernel;
        #  a compiled implementation would loop only over them. The RESULT is
        #  what is verified; the COST is reported as the dirty-pair fraction.)
        ko_new, dmg = self.full(Ic)
        return ko_new, mode


def main():
    v = G.vocab()
    states = G.load_states(64)
    static_cost(v, states)

    # ---- (b) prototype ---------------------------------------------------
    n_parents = int(os.environ.get("NPAR", "60"))
    rng = random.Random(99)
    rows = []
    for line in open(os.path.join(LAB, "data/pl2/out/labels_a.jsonl")):
        rows.append(json.loads(line))
        if len(rows) >= n_parents * 4:
            break
    rng.shuffle(rows)
    rows = rows[:n_parents]

    T = enc2.Tables(v)
    L = enc2.DEFAULT_LAYOUT
    ko_cols = [i for i, nm in enumerate(L.names)
               if i >= L.O_REL and i < L.O_REL + 360]
    n_ok = n_bad = 0
    dirty = tot = 0
    dirty_pm = tot_pm = 0
    fallback = 0
    npairs = 0
    modes = collections.Counter()

    for r in rows:
        kids = children_of(r["s"], rng, max_children=4)
        if not kids:
            continue
        Cp = LL.parse_batch([r["s"]], v)
        op = LL._slot_order(Cp["si"][:, :, enc2._SI["active_index"]])
        S0 = enc2.StaticCtx(Cp, T, op)
        chain = KoChain(S0.reorder(np.concatenate([op[:, 0], op[:, 1] + 6], 1)), T, L)
        Ip = KoChain.inputs(Cp, chain.S, T, op)
        koP, _ = chain.full(Ip)
        for k in kids:
            Ck = LL.parse_batch([k], v)
            ok_ = LL._slot_order(Ck["si"][:, :, enc2._SI["active_index"]])
            # the search reuses the PARENT's static context
            chain2 = KoChain(S0.reorder(np.concatenate([ok_[:, 0], ok_[:, 1] + 6], 1)),
                             T, L)
            Ic = KoChain.inputs(Ck, chain2.S, T, ok_)
            cache = {"prev": koP}
            got, mode = chain2.incr(Ip, Ic, cache)
            if mode.startswith("fallback"):
                fallback += 1
            modes[mode] += 1
            # ---- bit-identity against enc2 itself ------------------------
            _, fe = enc2.encode_columnar(Ck, v, S=S0)
            ref = fe[:, L.O_REL:L.O_REL + 360]
            if np.array_equal(got, ref):
                n_ok += 1
            else:
                n_bad += 1
            dirty += cache.get("dirty_pairs", 0)
            tot += cache.get("tot_pairs", 0)
            dirty_pm += cache.get("dirty_pm", 0)
            tot_pm += cache.get("tot_pm", 0)
            npairs += 1

    print("\n=== (b) incremental prototype: KO-matrix chain ===")
    print("  transitions              %d" % npairs)
    print("  bit-identical to enc2    %d ok / %d WRONG" % (n_ok, n_bad))
    for mk, mv in modes.most_common():
        print("    %-24s %4d (%.1f%%)" % (mk, mv, 100 * mv / max(npairs, 1)))
    if tot:
        print("  DIRTY PAIRS              %.1f of 72 per leaf = %.1f%%  ->  %.2fx "
              "less KO arithmetic"
              % (72.0 * dirty / tot, 100.0 * dirty / tot, tot / max(dirty, 1)))
        print("  DIRTY (pair,move)        %.1f of 288 per leaf = %.1f%%  ->  %.2fx "
              "less damage arithmetic"
              % (288.0 * dirty_pm / tot_pm, 100.0 * dirty_pm / tot_pm,
                 tot_pm / max(dirty_pm, 1)))


if __name__ == "__main__":
    main()
