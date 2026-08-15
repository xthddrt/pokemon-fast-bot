"""Encoder-2 performance bench: batch-size scaling of the dynamic half.

The published 0.083 ms/leaf is measured at n=4096 (enc2_gate.gate_cost), i.e.
a BATCHED per-state cost. A search that evaluates one leaf at a time pays the
n=1 cost. This separates the two and fits t(n) = a + b*n, where `a` is total
fixed numpy call/dispatch overhead and `b` is the marginal per-state work.
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
import enc2_gate as G  # noqa: E402
import llencoder as LL  # noqa: E402


def timeit(fn, reps, inner=1):
    """Return (median, p10, p90) seconds per call over `reps` trials."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    ts.sort()
    return ts[len(ts) // 2], ts[max(0, int(0.1 * len(ts)))], ts[min(len(ts) - 1, int(0.9 * len(ts)))]


def main():
    N = 4096
    states = G.load_states(N)
    v = G.vocab()
    C = LL.parse_batch(states, v)
    T = enc2.Tables(v)
    C1 = LL.parse_batch(states[:1], v)
    order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])
    S = enc2.StaticCtx(C1, T, order1)

    def slice_C(n):
        return {k: (v_[:n] if isinstance(v_, np.ndarray) else n)
                for k, v_ in C.items()}

    print("=== static half (once per search), n=1 ===")
    med, lo, hi = timeit(lambda: enc2.StaticCtx(C1, T, order1), 30)
    print("  static build  %8.3f ms   [p10 %.3f, p90 %.3f]" % (med * 1e3, lo * 1e3, hi * 1e3))

    print("\n=== dynamic half: per-state cost vs batch size ===")
    print("  %6s %12s %12s %12s %14s" % ("n", "total ms", "ms/state", "states/s", "p10-p90 ms"))
    rows = []
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        Cn = slice_C(n)
        enc2.encode_columnar(Cn, v, S=S)   # warm
        reps = 40 if n <= 64 else (15 if n <= 1024 else 7)
        inner = 20 if n <= 8 else (5 if n <= 128 else 1)
        med, lo, hi = timeit(lambda: enc2.encode_columnar(Cn, v, S=S), reps, inner)
        rows.append((n, med))
        print("  %6d %12.4f %12.5f %12.0f   %.4f-%.4f"
              % (n, med * 1e3, med / n * 1e3, n / med, lo * 1e3, hi * 1e3))

    # least-squares fit t(n) = a + b n over the whole range
    ns = np.array([r[0] for r in rows], float)
    ts = np.array([r[1] for r in rows], float)
    A = np.stack([np.ones_like(ns), ns], 1)
    (a, b), *_ = np.linalg.lstsq(A, ts, rcond=None)
    print("\n  fit t(n) = a + b*n over all n:")
    print("    a (fixed dispatch overhead) = %.4f ms" % (a * 1e3))
    print("    b (marginal per state)      = %.5f ms" % (b * 1e3))
    print("    overhead share at n=1       = %.1f%%" % (100 * a / (a + b)))
    print("    overhead share at n=4096    = %.1f%%" % (100 * a / (a + b * 4096)))


if __name__ == "__main__":
    main()
