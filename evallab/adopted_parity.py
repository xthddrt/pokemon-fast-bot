"""PYTHON/RUST BIT-IDENTITY GATE for the ADOPTED evaluator ("arm A + setup", PKNN v8).

    python adopted_parity.py real  <src.jsonl[.gz]> [n]
    python adopted_parity.py synth [n] [seed]
    python adopted_parity.py awk                      # the hand-built awkward cases
    python adopted_parity.py slots [n]                # LEG C: per-slot adversarial

Every one of the 1,248 packed float columns and 108 embedding ids must be
bit-identical (`f32` bit patterns, not a tolerance) between

  * PYTHON  -- `valuenet/encoder.py:encode_state` under labenv's pinned recipe
    (DROP_TIMES_ATTACKED=1 -> 72 numerics) with enc2's 14-column `cf_mon` block
    gathered into ARM A BENCH ORDER, i.e. `enc_adopted.py:_do_chunk` verbatim,
    and
  * RUST    -- `Network::encode_state` on a v8 net, via `adopted_dump`.

WHY enc2 IS RUN ONE STATE AT A TIME. `dmgtab.raw_damage` promotes its base-power
array to float64 if ANY move IN THE BATCH has a `np.where`-derived base power
(Low Kick, Heavy Slam, Rage Fist, Acrobatics, Facade, Knock Off). At n > 1 that
couples states to each other; the engine never does -- one leaf, one static. So
the reference is chunk=1, which is the semantics the ported code has. `--batched`
additionally reports what the CACHE BUILDER (enc_adopted.py, chunk=2048) would
have produced, because a difference there is a train/inference skew, not a port
bug.
"""
import gzip
import json
import os
import subprocess
import sys
import time

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
ROOT = os.path.dirname(LAB)

import numpy as np  # noqa: E402
import labenv  # noqa: E402,F401
import enc2  # noqa: E402
import lossless_encoder as LE  # noqa: E402

BIN = os.path.join(ROOT, "poke-engine/target/release/adopted_dump")
NET = os.environ.get("ADOPTED_NET", os.path.join(ROOT, "valuenet/nets_v8n/v8n_s1.bin"))
TMP = os.path.join(LAB, "data/adopted")
NM, NS, NG, NID, ND = 72, 99, 18, 9, 14
WMON = NM + ND                       # 86
WIDTH = 12 * WMON + 2 * NS + NG      # 1248
L = enc2.DEFAULT_LAYOUT
CFM = np.stack([np.array([i for i, x in enumerate(L.names)
                          if x.startswith("cf_mon%d." % k)], np.int64) for k in range(12)])
assert CFM.shape == (12, ND), CFM.shape
HPI = np.array([L.names.index("mon%d.hp_frac" % k) for k in range(12)], np.int64)
SETUP_NAMES = [x.split(".", 1)[1] for x in L.names if x.startswith("cf_mon0.")]


# --------------------------------------------------------------------- python
def py_encode(states, batched=False):
    """(ids (n,108) int32, feats (n,1248) float32, perm (n,12) int64) -- the
    ADOPTED vector in the RUST packing: [6 mons][sf1][6 mons][sf2][g]."""
    from encoder import Vocab, encode_state, bench_order
    from poke_engine import State
    voc, ll = Vocab(frozen=True), LE.LosslessVocab()
    n = len(states)

    if batched:
        _, f2 = enc2.encode_states(states, ll, chunk=2048)
    else:
        f2 = np.concatenate([enc2.encode_states([s], ll)[1] for s in states])
    setup = f2[:, CFM.reshape(-1)].reshape(n, 12, ND)
    hp2 = f2[:, HPI]

    ids = np.zeros((n, 12 * NID), np.int32)
    mon = np.zeros((n, 12, WMON), np.float32)
    sf = np.zeros((n, 2, NS), np.float32)
    g = np.zeros((n, NG), np.float32)
    perm = np.zeros((n, 12), np.int64)
    for r, s in enumerate(states):
        st = State.from_string(s)
        d = encode_state(st, voc)
        mon[r, 0, :NM] = d["a1_f"]
        mon[r, 1:6, :NM] = d["b1_f"]
        mon[r, 6, :NM] = d["a2_f"]
        mon[r, 7:12, :NM] = d["b2_f"]
        ids[r, :NID] = d["a1_ids"]
        ids[r, NID:6 * NID] = np.asarray(d["b1_ids"]).reshape(-1)
        ids[r, 6 * NID:7 * NID] = d["a2_ids"]
        ids[r, 7 * NID:] = np.asarray(d["b2_ids"]).reshape(-1)
        sf[r, 0], sf[r, 1], g[r] = d["sf1"], d["sf2"], d["g"]
        for si, side in enumerate((st.side_one, st.side_two)):
            a = int(side.active_index)
            party = [a] + [t[0] for t in bench_order(list(side.pokemon), a, voc)]
            for j, p in enumerate(party):
                perm[r, si * 6 + j] = (0 if p == a else (p + 1 if p < a else p)) + 6 * si
    assert (np.sort(perm, axis=1) == np.arange(12)).all()
    mon[:, :, NM:] = np.take_along_axis(setup, perm[:, :, None], axis=1)

    feats = np.concatenate([mon[:, :6].reshape(n, -1), sf[:, 0],
                            mon[:, 6:].reshape(n, -1), sf[:, 1], g], axis=1)
    assert feats.shape == (n, WIDTH), feats.shape
    return (ids, np.ascontiguousarray(feats, np.float32), perm,
            np.take_along_axis(hp2, perm, axis=1), setup, hp2)


# ----------------------------------------------------------------------- rust
def rust(states, tag, pyfeats=None):
    os.makedirs(TMP, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in tag)
    sp, op = os.path.join(TMP, "s_%s.txt" % slug), os.path.join(TMP, "r_%s.bin" % slug)
    with open(sp, "w") as f:
        f.write("\n".join(states) + "\n")
    cmd = [BIN, NET, sp, op]
    if pyfeats is not None:
        fp = os.path.join(TMP, "pf_%s.bin" % slug)
        pyfeats.astype(np.float32).tofile(fp)
        cmd.append(fp)
    subprocess.run(cmd, check=True, capture_output=True)
    raw = np.fromfile(op, np.uint8)
    n, nid, nf, haspy = np.frombuffer(raw[:16].tobytes(), np.uint32)
    assert (n, nid, nf) == (len(states), 108, WIDTH), (n, nid, nf)
    o = 16
    ids = np.frombuffer(raw[o:o + 4 * n * nid].tobytes(), np.int32).reshape(n, nid)
    o += 4 * n * nid
    fe = np.frombuffer(raw[o:o + 4 * n * nf].tobytes(), np.float32).reshape(n, nf)
    o += 4 * n * nf
    val = np.frombuffer(raw[o:o + 4 * n].tobytes(), np.float32)
    o += 4 * n
    pv = np.frombuffer(raw[o:o + 4 * n].tobytes(), np.float32) if haspy else None
    return ids, fe, val, pv


# -------------------------------------------------------------------- compare
def colname(c):
    """1,248-column index -> the name of that column."""
    if c < 6 * WMON or (6 * WMON + NS) <= c < (12 * WMON + NS):
        b = c if c < 6 * WMON else c - NS
        j, k = divmod(b, WMON)
        who = "s%d.%s" % (1 + j // 6, "active" if j % 6 == 0 else "bench%d" % (j % 6 - 1))
        return "%s.setup.%s" % (who, SETUP_NAMES[k - NM]) if k >= NM else "%s.f[%d]" % (who, k)
    if c < 6 * WMON + NS:
        return "s1.side[%d]" % (c - 6 * WMON)
    c2 = c - (12 * WMON + NS)
    return "s2.side[%d]" % c2 if c2 < NS else "g[%d]" % (c2 - NS)


def sensitivity(f_py, perm):
    """Would a WRONG remap have produced a DIFFERENT vector on these states?

    A whole-vector comparison can only catch the slot-order bug where the two
    gathers disagree in VALUE. `wrong` is the exact Stage-2 mutation: enc2 party
    order (arm A bench slot s <- enc2 slot s+1), no remap. This reports the
    fraction of states on which legs A/B would have caught it -- without which
    "1,248/1,248 columns identical" is compatible with a broken remap.
    """
    n = len(f_py)
    mon = np.concatenate([f_py[:, :6 * WMON].reshape(n, 6, WMON),
                          f_py[:, 6 * WMON + NS:12 * WMON + NS].reshape(n, 6, WMON)],
                         axis=1)
    setup = mon[:, :, NM:]                                       # arm A order
    inv = np.argsort(perm, axis=1)
    e2 = np.take_along_axis(setup, inv[:, :, None], axis=1)       # back to enc2 order
    wrong = np.arange(12)[None, :].repeat(n, 0)                   # party order, no remap
    wrong_gather = np.take_along_axis(e2, wrong[:, :, None], axis=1)
    differs = (wrong_gather != setup).any(axis=(1, 2))
    blocks_distinct = np.array([len({tuple(b) for b in setup[r]}) for r in range(n)])
    return dict(setup_nonzero_frac=float((setup != 0).mean()),
                states_where_wrong_remap_differs=int(differs.sum()),
                states_with_12_distinct_setup_blocks=int((blocks_distinct == 12).sum()),
                mean_distinct_setup_blocks=float(blocks_distinct.mean()))


def compare(states, tag, batched=False, verbose=True):
    t0 = time.time()
    i_py, f_py, perm, hp_g, e2setup, e2hp = py_encode(states, batched=batched)
    t1 = time.time()
    i_rs, f_rs, val, pyval = rust(states, tag, pyfeats=f_py)
    bad = f_py.view(np.uint32) != f_rs.view(np.uint32)     # BIT patterns
    badcol = np.flatnonzero(bad.any(axis=0))
    d = np.abs(f_py.astype(np.float64) - f_rs.astype(np.float64))
    ids_ok = bool(np.array_equal(i_py, i_rs))
    nontrivial = int((perm != np.arange(12)).any(axis=1).sum())
    vmax = float(np.abs(val.astype(np.float64) - pyval.astype(np.float64)).max())
    out = dict(tag=tag, n=len(states), cols=WIDTH, bad_cols=int(len(badcol)),
               bad_values=int(bad.sum()), max_abs_diff=float(d.max()), ids_ok=ids_ok,
               nontrivial_perm=nontrivial, value_max_abs_diff=vmax,
               py_s=round(t1 - t0, 1), rust_s=round(time.time() - t1, 1))
    out.update(sensitivity(f_py, perm))
    # the v8 forward actually ran on every state: no NaN, non-degenerate spread
    out.update(rust_value_min=float(val.min()), rust_value_max=float(val.max()),
               rust_value_nan=int(np.isnan(val).sum()))
    if verbose:
        print("  [%-26s] %6d states  %4d/%d cols and %d/%d ids bit-identical, "
              "max|d| = %g, mismatching values %d, non-identity perms %d, "
              "value max|d| = %g"
              % (tag, len(states), WIDTH - len(badcol), WIDTH,
                 108 if ids_ok else 0, 108, d.max(), int(bad.sum()), nontrivial, vmax))
        if len(badcol):
            rows = sorted(((float(d[:, c].max()), int(bad[:, c].sum()), colname(c), int(c))
                           for c in badcol), reverse=True)
            print("      %d differing columns; worst 20:" % len(badcol))
            for md, nr, nm, c in rows[:20]:
                r0 = int(np.flatnonzero(bad[:, c])[0])
                print("        %-34s col %4d  max|d| %.6g  on %d/%d states  "
                      "(first: state %d, py %.9g vs rs %.9g)"
                      % (nm, c, md, nr, len(states), r0, f_py[r0, c], f_rs[r0, c]))
        if not ids_ok:
            b = np.flatnonzero((i_py != i_rs).any(axis=0))
            print("      differing id columns: %s" % b[:20].tolist())
    out["ok"] = len(badcol) == 0 and ids_ok
    return out, f_py, f_rs, perm, e2setup


# -------------------------------------------------------------------- sources
def leg_c(n=400, seed=7):
    """LEG C -- the per-slot correspondence, asserted DIRECTLY.

    The occupant of arm A slot j is decided from the RAW STATE STRING (each mon
    has a unique hp_frac by construction), never from `encoder.bench_order`. So
    this does not merely compare two vectors that could be self-consistently
    wrong: it names the mon in slot j and demands THAT mon's setup block.

    WHAT IT CATCHES: any permutation error between arm A's per-mon block and
    enc2's `cf_mon` block -- including one that both encoders make identically,
    which legs A/B cannot see. WHAT IT DOES NOT CATCH: an error inside a setup
    column's VALUE (that is legs A/B), an error in arm A's own bench order that
    moves the arm A features AND the setup features together (`hp_frac` moves
    with them, so slot j is simply re-named -- `policy_arm_slots` is the gate for
    that, not this), or anything about the 72 arm A numerics themselves.
    """
    import random

    import synth_pool as SP
    states = SP.distinguishable(random.Random(seed), n)
    print("=== LEG C: %d slot-adversarial states (12 distinct species, distinct "
          "hp_frac, distinct setup block per mon) ===" % len(states))
    out, f_py, f_rs, perm, e2setup = compare(states, "slot adversarial")

    checked = mism = 0
    for r, s in enumerate(states):
        t = s.split("/")
        for si in (0, 1):
            fields = t[si].split("=")
            a = int(fields[6])
            hp = np.array([[float(x.split(",")[6]), float(x.split(",")[7])]
                           for x in fields[:6]])
            frac = (hp[:, 0] / np.maximum(hp[:, 1], 1)).astype(np.float32)
            for j in range(6):
                base = (si * 6 + j) * WMON + (0 if si == 0 else NS)
                got = f_rs[r, base]
                dd = np.abs(frac - got)
                p = int(dd.argmin())
                srt = np.sort(dd)
                assert srt[0] < 1e-6 and srt[1] > 1e-3, (
                    "state %d side %d slot %d: hp_frac %g does not identify one mon "
                    "uniquely (%s) -- the leg C corpus is not distinguishable" %
                    (r, si, j, got, srt[:2]))
                k = 6 * si + (0 if p == a else (p + 1 if p < a else p))
                want = e2setup[r, k]
                blk = f_rs[r, base + NM:base + WMON]
                if not np.array_equal(blk.view(np.uint32), want.view(np.uint32)):
                    mism += 1
                    if mism <= 5:
                        bad = np.flatnonzero(blk != want)
                        print("      MISMATCH state %d side %d armA slot %d holds party "
                              "mon %d (enc2 slot %d): cols %s  rust %s vs enc2 %s"
                              % (r, si, j, p, k, [SETUP_NAMES[c] for c in bad],
                                 blk[bad], want[bad]))
                checked += 1
    out["slot_pairs_checked"] = checked
    out["slot_pairs_mismatched"] = mism
    out["ok"] = out["ok"] and mism == 0
    print("  per-slot correspondence: %d (state, slot) pairs checked, %d mismatched"
          % (checked, mism))
    return out


def load_real(src, n):
    op = gzip.open if src.endswith(".gz") else open
    out = []
    with op(src, "rt") as f:
        for line in f:
            out.append(json.loads(line)["s"])
            if len(out) >= n:
                break
    return out


def main():
    if not os.path.exists(BIN):
        raise SystemExit("build first: cargo build --release --no-default-features "
                         "--features gen9,terastallization --bin adopted_dump")
    mode = sys.argv[1]
    res = []
    if mode == "real":
        src = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
        st = load_real(src, n)
        print("=== LEG A: %d REAL states from %s ===" % (len(st), os.path.basename(src)))
        res.append(compare(st, "real chunk1")[0])
        if os.environ.get("ALSO_BATCHED") == "1":
            res.append(compare(st, "real batched2048", batched=True)[0])
    elif mode == "synth":
        import synth_pool as SP
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 11
        st, cov = SP.build(n, seed)
        print("=== LEG B: %d SYNTHETIC pool-wide states ===" % len(st))
        print("    coverage: %s" % json.dumps(cov))
        res.append(compare(st, "synthetic pool-wide")[0])
        res[-1]["coverage"] = cov
    elif mode == "slots":
        res.append(leg_c(int(sys.argv[2]) if len(sys.argv) > 2 else 400))
    else:
        raise SystemExit(__doc__)
    print(json.dumps(res, indent=1, default=float))
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
