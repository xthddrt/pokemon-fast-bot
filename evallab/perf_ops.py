"""How much WORK does the dynamic half actually do?

Counts, per block, at n=1 (one search leaf):
  calls  -- array operations dispatched. This is the numpy overhead term; a
            Rust port deletes it entirely (it becomes inlined loop bodies).
  elems  -- output elements written across all those operations. This is the
            arithmetic term; Rust still has to do it, but fused, in registers,
            without materialising a temporary per operation.

Verified bit-identical to enc2 before any number is believed.
"""
import os
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


def main():
    states = G.load_states(64)
    v = G.vocab()
    T = enc2.Tables(v)
    C1 = LL.parse_batch(states[:1], v)
    order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])
    S = enc2.StaticCtx(C1, T, order1)
    SC = EC.StaticCtx(C1, EC.Tables(v), order1)

    # Seed every input as a counting array so that ndarray METHODS
    # (.astype/.sum/.argmax/...) are counted too, not just np.* functions.
    def seed(d):
        return {k: (x.view(npcount.CA) if isinstance(x, np.ndarray) else x)
                for k, x in d.items()}

    i0, f0 = enc2.encode_columnar(C1, v, S=S)
    SC = EC.StaticCtx(seed(C1), EC.Tables(v), order1)
    npcount.reset()
    i1, f1 = EC.encode_columnar(seed(C1), v, S=SC)
    assert np.array_equal(np.asarray(i0), np.asarray(i1)), "ids differ"
    assert np.array_equal(np.asarray(f0), np.asarray(f1)), "feats differ"
    print("equivalence: enc2_count == enc2 at n=1, %d/%d columns bit-identical\n"
          % (f0.shape[1], f0.shape[1]))

    # per-block: snapshot the counters at each timing checkpoint
    snaps = []

    def tk(label, _orig=EC._TK):
        snaps.append((label, npcount.STATS["calls"], npcount.STATS["elems"]))
        _orig(label)

    EC._TK = tk
    npcount.reset()
    EC.encode_columnar(seed(C1), v, S=SC)
    tot_c, tot_e = npcount.STATS["calls"], npcount.STATS["elems"]

    print("=== dynamic half at n=1: work per block ===")
    print("%-24s %8s %10s %10s %9s" % ("block", "calls", "elems", "elem/call", "% elems"))
    print("-" * 66)
    rows = []
    for i in range(1, len(snaps)):
        lab = snaps[i][0]
        dc = snaps[i][1] - snaps[i - 1][1]
        de = snaps[i][2] - snaps[i - 1][2]
        rows.append((lab, dc, de))
    tail_c, tail_e = tot_c - snaps[-1][1], tot_e - snaps[-1][2]
    rows.append(("15_return", tail_c, tail_e))
    for lab, dc, de in sorted(rows, key=lambda r: -r[2]):
        print("%-24s %8d %10d %10.1f %8.1f%%"
              % (lab, dc, de, de / max(dc, 1), 100 * de / max(tot_e, 1)))
    print("-" * 66)
    print("%-24s %8d %10d %10.1f" % ("TOTAL", tot_c, tot_e, tot_e / max(tot_c, 1)))

    print("\n=== top operations by element count ===")
    print("%-22s %8s %10s" % ("op", "calls", "elems"))
    for k, (c, e) in sorted(npcount.BY_KIND.items(), key=lambda x: -x[1][1])[:18]:
        print("%-22s %8d %10d" % (k, c, e))

    print("\n=== the two terms ===")
    print("  numpy call dispatch:  %d calls" % tot_c)
    print("  arithmetic:           %d output elements" % tot_e)
    print("  measured n=1 dynamic: ~1730 us  ->  %.2f us/call, %.1f ns/element"
          % (1730.0 / tot_c, 1730e3 / max(tot_e, 1)))


if __name__ == "__main__":
    main()
