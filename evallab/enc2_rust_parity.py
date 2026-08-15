"""Python/Rust bit-identity gate for enc2 (EVALUATOR_SPEC §8 gate 5).

  python enc2_rust_parity.py [--real 5000] [--fuzz 1500] [--shared 512]

Every one of the 1,413 columns must be bit-identical between `enc2.py` and
`poke-engine/src/genx/enc2.rs`, on real labelled states, on pool-wide synthetic
states, and on the shared-static search path (including a hard set where tera,
`disabled` and PP all move). Exact float equality, not a tolerance.

WHY chunk=1 ON THE COLD PATH. `dmgtab.raw_damage` promotes its base-power array
to float64 if ANY move IN THE BATCH has a base power derived from a `np.where`
over two python floats (Low Kick, Heavy Slam, Rage Fist, Acrobatics, Facade,
Knock Off). At n > 1 that couples states to each other, which the production
search never does -- `build_static` sees one root state. So the reference is
run one state at a time, which is also what the search path does.
"""
import argparse
import os
import subprocess
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
ROOT = os.path.dirname(LAB)

import numpy as np  # noqa: E402
import labenv  # noqa: E402,F401
import enc2  # noqa: E402
import enc2_gate as G  # noqa: E402

BIN = os.path.join(ROOT, "poke-engine/target/release/enc2_bench")
TMP = os.path.join(LAB, "data/enc2")
L = enc2.DEFAULT_LAYOUT


def rust(states, tag, shared=False):
    os.makedirs(TMP, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in tag)
    sp = os.path.join(TMP, "states_%s.txt" % slug)
    op = os.path.join(TMP, "rust_%s.bin" % slug)
    with open(sp, "w") as f:
        f.write("\n".join(states) + "\n")
    subprocess.run([BIN, "dump", sp, op, "shared" if shared else "cold"],
                   check=True, capture_output=True)
    raw = np.fromfile(op, np.uint8)
    n, nid, nf = np.frombuffer(raw[:12].tobytes(), np.uint32)
    assert n == len(states) and nf == L.N, (n, nid, nf)
    o = 12
    ids = np.frombuffer(raw[o:o + 4 * n * nid].tobytes(), np.int32).reshape(n, nid)
    o += 4 * n * nid
    fe = np.frombuffer(raw[o:o + 4 * n * nf].tobytes(), np.float32).reshape(n, nf)
    return ids, fe


def compare(states, tag, shared=False):
    if shared:
        i_py, f_py = enc2.encode_states(states, G.vocab(), share_static=True)
    else:
        # chunk=1: the reference must see one state at a time (see module docstring)
        parts = [enc2.encode_states([s], G.vocab()) for s in states]
        i_py = np.concatenate([p[0] for p in parts])
        f_py = np.concatenate([p[1] for p in parts])
    i_rs, f_rs = rust(states, tag, shared=shared)
    eq = f_py == f_rs                                   # EXACT float equality
    badcol = np.where((~eq).any(axis=0))[0]
    d = np.abs(f_py.astype(np.float64) - f_rs.astype(np.float64))
    maxd = float(d.max())
    ids_ok = bool(np.array_equal(i_py, i_rs))
    ok = len(badcol) == 0 and ids_ok
    print("  [%-30s] %5d states  %4d/%4d columns bit-identical, max|d| = %g, ids %s"
          % (tag, len(states), L.N - len(badcol), L.N, maxd,
             "ok" if ids_ok else "DIFFER"))
    if len(badcol):
        # rank the offenders by how often and how far they differ
        rows = []
        for c in badcol[:400]:
            nrow = int((~eq[:, c]).sum())
            rows.append((float(d[:, c].max()), nrow, L.names[c]))
        rows.sort(reverse=True)
        print("      %d differing columns; worst 15:" % len(badcol))
        for md, nr, nm in rows[:15]:
            print("        %-42s max|d| %.6g  on %d/%d states" % (nm, md, nr, len(states)))
        # block census
        cen = {}
        for c in badcol:
            nm = L.names[c]
            key = nm.split(".")[0]
            key = "".join(ch for ch in key if not ch.isdigit())
            cen[key] = cen.get(key, 0) + 1
        print("      by block: %s" % sorted(cen.items(), key=lambda x: -x[1]))
    if not ids_ok:
        bad = np.where((i_py != i_rs).any(axis=0))[0]
        print("      differing id columns: %s" % bad[:20].tolist())
    return ok, badcol, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=5000)
    ap.add_argument("--fuzz", type=int, default=1500)
    ap.add_argument("--shared", type=int, default=512)
    a = ap.parse_args()
    if not os.path.exists(BIN):
        raise SystemExit("build first: cargo build --release --features terastallization "
                         "--bin enc2_bench")
    print("=== Python/Rust bit-identity, all %d columns, exact float equality ===" % L.N)
    ok = True
    if a.real:
        ok &= compare(G.load_states(a.real), "real labelled corpus")[0]
    if a.fuzz:
        ok &= compare(G.fuzz_states(a.fuzz, seed=11), "pool-wide synthetic")[0]
    if a.shared:
        ok &= compare(G.split_states(a.shared), "one root, shared static",
                      shared=True)[0]
        ok &= compare(G.split_states(a.shared, seed=5, hard=True),
                      "+ tera/disable/PP moving", shared=True)[0]
        # the split property itself, on the RUST side: shared static == cold
        hard = G.split_states(a.shared, seed=5, hard=True)
        _, f_sh = rust(hard, "split_shared", shared=True)
        _, f_cold = rust(hard, "split_cold", shared=False)
        nbad = int((f_sh != f_cold).any(axis=0).sum())
        print("  [rust shared vs rust cold        ] %5d states  %4d/%4d columns identical, "
              "max|d| = %g" % (len(hard), L.N - nbad, L.N,
                               float(np.abs(f_sh.astype(np.float64)
                                            - f_cold.astype(np.float64)).max())))
        ok &= nbad == 0
    print("\nRUST PARITY: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
