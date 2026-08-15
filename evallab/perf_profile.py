"""Block-level profile of the dynamic half, at n=1 (a real search leaf) and
batched, with an equivalence check against the uninstrumented encoder first.

Per block:  t(1)            = cost of one leaf, alone
            t(n)/n          = marginal cost per state when batched
            t(1) - t(n)/n   = fixed numpy dispatch overhead for that block
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
import enc2_prof as EP  # noqa: E402
import enc2_gate as G  # noqa: E402
import llencoder as LL  # noqa: E402

NBIG = 512


def main():
    states = G.load_states(4096)
    v = G.vocab()
    C = LL.parse_batch(states, v)
    T = enc2.Tables(v)
    C1 = LL.parse_batch(states[:1], v)
    order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])
    S = enc2.StaticCtx(C1, T, order1)
    SP = EP.StaticCtx(C1, EP.Tables(v), order1)

    # --- equivalence: the instrumented copy must be bit-identical ---------
    Cn = {k: (x[:1024] if isinstance(x, np.ndarray) else 1024) for k, x in C.items()}
    i0, f0 = enc2.encode_columnar(Cn, v, S=S)
    i1, f1 = EP.encode_columnar(Cn, v, S=SP)
    assert np.array_equal(i0, i1), "ids differ"
    assert np.array_equal(f0, f1), "feats differ: max %g" % np.abs(f0 - f1).max()
    print("equivalence: enc2_prof == enc2 on %d real states, %d/%d columns "
          "bit-identical\n" % (Cn["n"], f0.shape[1], f0.shape[1]))

    def run(n, reps):
        Cx = {k: (x[:n] if isinstance(x, np.ndarray) else n) for k, x in C.items()}
        EP.encode_columnar(Cx, v, S=SP)
        EP.prof_reset()
        t0 = time.perf_counter()
        for _ in range(reps):
            EP.encode_columnar(Cx, v, S=SP)
        wall = (time.perf_counter() - t0) / reps
        return {k: x / reps for k, x in EP.PROF.items()}, wall

    p1, w1 = run(1, 300)
    pN, wN = run(NBIG, 8)

    print("=== dynamic half, per-block ===")
    print("%-24s %10s %10s %10s %8s %8s" %
          ("block", "n=1 us", "n=%d us/st" % NBIG, "overhead", "%of n=1", "%of bat"))
    print("-" * 78)
    tot1 = sum(p1.values()) * 1e6
    totN = sum(pN.values()) / NBIG * 1e6
    rows = []
    for k in p1:
        a = p1[k] * 1e6
        b = pN[k] / NBIG * 1e6
        rows.append((k, a, b, a - b))
    for k, a, b, ov in sorted(rows, key=lambda r: -r[1]):
        print("%-24s %10.2f %10.3f %10.2f %7.1f%% %7.1f%%"
              % (k, a, b, ov, 100 * a / tot1, 100 * b / totN))
    print("-" * 78)
    print("%-24s %10.2f %10.3f %10.2f" % ("TOTAL (ticked)", tot1, totN, tot1 - totN))
    print("%-24s %10.2f %10.3f" % ("TOTAL (wall)", w1 * 1e6, wN / NBIG * 1e6))
    print("\noverhead share at n=1: %.1f%%" % (100 * (tot1 - totN) / tot1))
    print("n=1 throughput: %.0f leaves/s   batched: %.0f leaves/s"
          % (1 / w1, NBIG / wN))


if __name__ == "__main__":
    main()
