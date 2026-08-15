"""Equivalence + n=1 cost harness for the enc2 fix/speed pass.

`enc2_ref.py` is a FROZEN byte-for-byte copy of enc2.py as it stood before this
pass (sha256 4c45968029d6ad18), with `VOL_COLS` pinned to the old 44-name list
so that changing `rbpool.py` cannot move the reference.

  python enc2_equiv.py bits    bit-identity of every SURVIVING column, exact
                               float equality, on real + pool-wide synthetic
                               states, cold path and share_static path
  python enc2_equiv.py perf    n = 1 SERIAL per-leaf cost, ref vs new (this is
                               the only regime production search runs in)
  python enc2_equiv.py all     both

The column set deliberately moves (volatile membership; the +2 setup level is
gone), so `bits` maps NEW columns back to REF columns BY NAME and asserts:
  * every surviving column is bit-identical (max abs diff exactly 0.0)
  * the columns that disappeared are exactly the expected ones
  * the columns that appeared are exactly the expected ones
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import numpy as np  # noqa: E402
import labenv  # noqa: E402,F401
import enc2  # noqa: E402
import enc2_ref as REF  # noqa: E402
import enc2_gate as G  # noqa: E402
import llencoder as LL  # noqa: E402

# what changes 1 and 2 are ALLOWED to move (everything else must be identical)
EXPECT_ADDED = {"side%d.vol_attract" % s for s in (0, 1)} | \
               {"side%d.vol_charge" % s for s in (0, 1)}
EXPECT_DROPPED = ({"side%d.vol_lightscreen" % s for s in (0, 1)}
                  | {"side%d.vol_reflect" % s for s in (0, 1)}
                  | {"cf_mon%d.%s" % (k, c) for k in range(12)
                     for c in ("s2_ohko_count", "s2_outspeed_count", "s2_sweep",
                               "s2_mtk_1", "s2_mtk_2", "s2_mtk_3",
                               "s2_mtk_4plus", "s2_mtk_never")}
                  | {"cf_side%d.best_sweep_2" % s for s in (0, 1)})


def compare(states, tag, share_static=False, ref_share=None):
    v = G.vocab()
    if ref_share is None:
        ref_share = share_static
    i_ref, f_ref = REF.encode_states(states, v, layout=REF.DEFAULT_LAYOUT,
                                     share_static=ref_share)
    i_new, f_new = enc2.encode_states(states, v, layout=enc2.DEFAULT_LAYOUT,
                                      share_static=share_static)
    ref_names = REF.DEFAULT_LAYOUT.names
    new_names = enc2.DEFAULT_LAYOUT.names
    rix = {n: i for i, n in enumerate(ref_names)}
    added = [n for n in new_names if n not in rix]
    dropped = [n for n in ref_names if n not in set(new_names)]
    surviving = [n for n in new_names if n in rix]
    a = f_new[:, [new_names.index(n) for n in surviving]]
    b = f_ref[:, [rix[n] for n in surviving]]
    eq = (a == b)                                    # EXACT float equality
    nbad = int((~eq).any(axis=0).sum())
    maxd = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if len(surviving) else 0.0
    ids_ok = bool(np.array_equal(i_ref, i_new))
    ok = nbad == 0 and ids_ok and set(added) == EXPECT_ADDED and set(dropped) == EXPECT_DROPPED
    print("  [%-28s] %5d states  %4d/%4d surviving columns bit-identical, "
          "max|d| = %g, ids %s" % (tag, len(states), len(surviving) - nbad,
                                   len(surviving), maxd, "ok" if ids_ok else "DIFFER"))
    if nbad:
        bad = [surviving[i] for i in np.where((~eq).any(axis=0))[0]][:12]
        print("      DIFFERING: %s" % bad)
    if set(added) != EXPECT_ADDED:
        print("      UNEXPECTED added: %s" % sorted(set(added) ^ EXPECT_ADDED))
    if set(dropped) != EXPECT_DROPPED:
        print("      UNEXPECTED dropped: %s" % sorted(set(dropped) ^ EXPECT_DROPPED)[:20])
    return ok, added, dropped


def gate_bits(n_real=5000, n_fuzz=1500, n_shared=512):
    print("=== bit-identity of every surviving column (exact float equality) ===")
    ok = True
    real = G.load_states(n_real)
    o, added, dropped = compare(real, "real labelled corpus")
    ok &= o
    ok &= compare(G.fuzz_states(n_fuzz, seed=11), "pool-wide synthetic")[0]
    # The shared-static path is compared on states that satisfy its precondition
    # (one root's twelve Pokemon), against the REFERENCE'S COLD PATH -- the
    # reference's own shared path bakes the root's tera/disabled/pp in and is
    # stale, which is the bug being fixed, so it is not a ground truth.
    leaves = G.split_states(n_shared)
    ok &= compare(leaves, "one root, share_static", share_static=True,
                  ref_share=False)[0]
    hard = G.split_states(n_shared, seed=5, hard=True)
    ok &= compare(hard, "+ tera/disable/pp moving", share_static=True,
                  ref_share=False)[0]
    v = G.vocab()
    fs = REF.encode_states(hard, v, share_static=True)[1]
    fc = REF.encode_states(hard, v)[1]
    print("  (for scale: on those same states the OLD shared-static path "
          "differs from its own cold path on %d/%d columns -- the stale-context "
          "bug, now gone)" % (int((fs != fc).any(axis=0).sum()), fc.shape[1]))
    print("  columns added   (%d): %s" % (len(added), sorted(added)))
    print("  columns dropped (%d): %s ..." % (len(dropped), sorted(dropped)[:6]))
    print("  ref layout %d -> new layout %d" % (REF.DEFAULT_LAYOUT.N,
                                                enc2.DEFAULT_LAYOUT.N))
    print("BITS: %s" % ("PASS" if ok else "FAIL"))
    return ok


def timeit(fn, reps, inner):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    ts.sort()
    return ts[len(ts) // 2], ts[int(0.1 * len(ts))], ts[min(len(ts) - 1, int(0.9 * len(ts)))]


def paired(fns, reps=60, inner=20):
    """Time several callables INTERLEAVED, one trial each per round, so that a
    drifting machine cannot favour whichever ran first. -> median per callable."""
    ts = {k: [] for k in fns}
    for _ in range(reps):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            for _ in range(inner):
                fn()
            ts[k].append((time.perf_counter() - t0) / inner)
    return {k: sorted(v)[len(v) // 2] for k, v in ts.items()}


def gate_perf(reps=60, inner=20, nbatch=512):
    """n = 1, SERIAL. This is the regime `eval_scratch(&State)` runs in --
    `eval_scratch(&State)` is scalar at every signature and the batched path was
    measured at +3% and deleted, so there is no batch dimension in the search."""
    print("=== n = 1 per-leaf cost (serial, one leaf at a time) ===")
    v = G.vocab()
    states = G.load_states(4096)
    C1 = LL.parse_batch(states[:1], v)
    CB = LL.parse_batch(states[:nbatch], v)
    ctx, fn1, fnB, fnS = {}, {}, {}, {}
    for tag, mod in (("before", REF), ("after", enc2)):
        T = mod.Tables(v)
        order1 = LL._slot_order(C1["si"][:, :, mod._SI["active_index"]])
        S = mod.StaticCtx(C1, T, order1)
        ctx[tag] = (mod, T, S, order1)
        fn1[tag] = (lambda mod=mod, S=S: mod.encode_columnar(C1, v, S=S))
        fnB[tag] = (lambda mod=mod, S=S: mod.encode_columnar(CB, v, S=S))
        fnS[tag] = (lambda mod=mod, T=T, o=order1: mod.StaticCtx(C1, T, o))
        fn1[tag]()
        fnB[tag]()
    d1 = paired(fn1, reps, inner)
    dB = paired(fnB, 9, 1)
    dS = paired(fnS, 15, 1)
    for tag in ("before", "after"):
        print("  %-7s dynamic n=1 %8.1f us/leaf (%5.0f leaves/s) | "
              "batched n=%d %6.2f us/state | static %6.3f ms"
              % (tag, d1[tag] * 1e6, 1 / d1[tag], nbatch, dB[tag] * 1e6 / nbatch,
                 dS[tag] * 1e3))
    print("  speedup: n=1 %.2fx | batched %.2fx | static %.2fx"
          % (d1["before"] / d1["after"], dB["before"] / dB["after"],
             dS["before"] / dS["after"]))
    print("  Rust must still deliver %.0fx against the 8.4 us/leaf budget "
          "(was %.0fx)" % (d1["after"] * 1e6 / 8.4, d1["before"] * 1e6 / 8.4))
    return True


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("bits", "all"):
        ok &= gate_bits()
    if which in ("perf", "all"):
        gate_perf()
    print("\nEQUIV OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
