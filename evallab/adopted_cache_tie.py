"""Tie the RUST v8 vector to the ACTUAL cache builder, `enc_adopted.py`.

`adopted_parity.py` re-implements the arm A + setup gather in order to compare
in float32. That leaves one hole: if that re-implementation made the SAME
mistake as the Rust port, both would agree and the gate would pass while the
TRAINING CACHE disagreed with both. So this runs `enc_adopted.py encode` -- the
unmodified builder that produced the published caches -- on a slice of plc1e and
compares its arrays, element for element, against `adopted_dump`'s output.

The cache is float16, so the comparison is `fp16(rust) == cache` -- exact in
fp16 space, and lossless for the ids (int16).

    python adopted_cache_tie.py <src.jsonl.gz> [rows]
"""
import os
import subprocess
import sys

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: E402,F401
import adopted_parity as AP  # noqa: E402

OUT = os.path.join(LAB, "data/adopted/cache_tie")


def main():
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    py = sys.executable
    subprocess.run([py, os.path.join(LAB, "enc_adopted.py"), "encode", src, OUT,
                    "--limit", str(n), "--workers", "2"], check=True)

    states = AP.load_real(src, n)
    _, f_rs, val, _ = AP.rust(states, "cache_tie")
    mon = np.concatenate([f_rs[:, :6 * AP.WMON].reshape(n, 6, AP.WMON),
                          f_rs[:, 6 * AP.WMON + AP.NS:12 * AP.WMON + AP.NS]
                          .reshape(n, 6, AP.WMON)], axis=1)
    sf1 = f_rs[:, 6 * AP.WMON:6 * AP.WMON + AP.NS]
    sf2 = f_rs[:, 12 * AP.WMON + AP.NS:12 * AP.WMON + 2 * AP.NS]
    g = f_rs[:, 12 * AP.WMON + 2 * AP.NS:]

    def ld(k):
        return np.load(os.path.join(OUT, "old_%s.npy" % k), mmap_mode="r")[:n]

    want = {"a1_f": mon[:, 0, :AP.NM], "b1_f": mon[:, 1:6, :AP.NM],
            "a2_f": mon[:, 6, :AP.NM], "b2_f": mon[:, 7:12, :AP.NM],
            "sf1": sf1, "sf2": sf2, "g": g}
    bad = {}
    for k, v in want.items():
        got = np.asarray(ld(k))
        d = got != v.astype(np.float16)
        bad[k] = int(d.sum())
    am = np.asarray(np.load(os.path.join(OUT, "addon_mon.npy"), mmap_mode="r")[:n])
    bad["addon_mon(setup)"] = int((am != mon[:, :, AP.NM:].astype(np.float16)).sum())
    print("rows=%d  fp16-exact mismatches per array: %s" % (n, bad))
    print("TOTAL MISMATCHES: %d  ->  %s" % (sum(bad.values()),
                                            "PASS" if sum(bad.values()) == 0 else "FAIL"))
    return 0 if sum(bad.values()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
