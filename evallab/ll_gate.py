"""ACCEPTANCE GATE for evallab/llencoder.py, on real pair-A states.

Five subcommands, all local, all cheap:

  corpus     extract N distinct pair-A state strings from data/el1/A2k shards
  gate       THE gate. For every state:
                 LE.parse_state(s)        <- the reference parser, itself
                                             validated against the real
                                             poke_engine bindings by
                                             valuenet/roundtrip_gate.py
                 llencoder.encode_states  <- THIS implementation (its own fast
                                             columnar parser + vectorised encode)
                 llencoder.decode         <- the shared reference decoder
             and compares the two field by field with roundtrip_gate.flatten,
             per VALUE, not per state. Because `src` comes from the reference
             parser and `dst` comes from this module's parser, the gate proves
             BOTH the fast parser and the encoder in one pass. Run on both the
             full and the lean variant.
  switchsib  the distinguishability tests (see below)
  bench      encode throughput, both variants
  fp16       what float16 STORAGE costs, measured rather than assumed

SWITCHSIB, the second unimpeachable thing:
  (A) required test -- real successors. For roots with >=2 legal switch arms,
      build s'(switch X) and s'(switch Y) against the same opponent reply and
      assert the two encodings DIFFER, and that decoding each recovers the
      correct `active_index` and `last_used_move` switch target.
  (B) sharper test -- single-field mutations that isolate each CRITICAL FIX.
      Each mutation changes exactly one field of a real state string; the new
      encoder must separate the pair and today's shipped encoder is measured on
      the same pair. This is where "switch siblings must encode differently"
      becomes a statement about a specific column rather than about hp deltas.
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import labenv  # noqa: F401  (pins encoder/engine flags before any import)
import llencoder as ll

sys.path.insert(0, os.path.join(labenv.ROOT, "valuenet"))
import lossless_encoder as LE  # noqa: E402
import roundtrip_gate as RG  # noqa: E402

DEFAULT_CORPUS = os.path.join(labenv.LAB, "data", "el1", "pairA_states.tsv")
SHARDS = os.path.join(labenv.LAB, "data", "el1", "A2k", "shard_*.jsonl.gz")


# ---------------------------------------------------------------------------
def build_corpus(out_path, n=60000, shards=SHARDS):
    """Distinct pair-A state strings, in shard order (no sampling, so the file
    is reproducible). NO generation -- this only reads what already exists."""
    seen, out = set(), []
    for fp in sorted(glob.glob(shards)):
        with gzip.open(fp, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or not row.get("t"):
                    continue
                for t in row["t"]:
                    s = t.get("s")
                    if isinstance(s, str) and s not in seen:
                        seen.add(s)
                        out.append(s)
                if len(out) >= n:
                    break
        if len(out) >= n:
            break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for s in out[:n]:
            f.write("A\t" + s + "\n")
    print("corpus written: %s  states=%d" % (out_path, min(n, len(out))))


def load_states(path, limit):
    return RG.load_corpus(path, limit)


# ---------------------------------------------------------------------------
def run_gate(states, variant, batch=4096):
    """roundtrip_gate's comparison machinery, driven by llencoder."""
    vocab = LE.LosslessVocab()
    ok, tot = Counter(), Counter()
    examples = defaultdict(list)
    rng_viol, rng_max = Counter(), {}
    t_enc = 0.0
    t0 = time.time()
    n_states = 0
    for i in range(0, len(states), batch):
        chunk = states[i:i + batch]
        srcs = [LE.parse_state(s) for s in chunk]
        for src in srcs:
            RG._record_ranges(src, rng_viol, rng_max)
        te = time.time()
        ids, feats = ll.encode_states(chunk, vocab, variant, chunk=batch)
        t_enc += time.time() - te
        for j, src in enumerate(srcs):
            dst = ll.decode(ids[j], feats[j], vocab, variant)
            a, b = RG.flatten(src), RG.flatten(dst)
            for k in set(a) | set(b):
                va, vb = a.get(k, []), b.get(k, [])
                if len(va) != len(vb):
                    tot[k] += max(len(va), len(vb))
                    if len(examples[k]) < 3:
                        examples[k].append(("LENGTH", len(va), len(vb)))
                    continue
                for x, y in zip(va, vb):
                    tot[k] += 1
                    if x == y:
                        ok[k] += 1
                    elif len(examples[k]) < 3:
                        examples[k].append((x, y))
            n_states += 1
    r = {"ok": ok, "tot": tot, "examples": examples, "n": n_states,
         "sec": time.time() - t0, "rng_viol": rng_viol, "rng_max": rng_max,
         "vocab_misses": dict(vocab.misses), "enc_sec": t_enc}
    print("\n#### VARIANT %s  (%d numeric columns, %d ids) ####"
          % (variant.upper(), ll.VARIANTS[variant].N_FEATS, ll.VARIANTS[variant].N_IDS))
    RG.report_roundtrip(r)
    keys = sorted(r["tot"])
    failed = [k for k in keys if r["ok"][k] != r["tot"][k]]
    print("\n-- per-field pass rate (all %d groups) --" % len(keys))
    for k in keys:
        print("  %-44s %9d/%-9d %.6f%%"
              % (k, r["ok"][k], r["tot"][k], 100.0 * r["ok"][k] / max(r["tot"][k], 1)))
    print("\nencode-only wall: %.2fs for %d states (%.4f ms/state, %d states/s)"
          % (r["enc_sec"], n_states, 1000.0 * r["enc_sec"] / max(n_states, 1),
             int(n_states / max(r["enc_sec"], 1e-9))))
    print("GATE %s: %s" % (variant, "PASS" if not failed and not r["vocab_misses"] else "FAIL"))
    return not failed and not r["vocab_misses"]


# ---------------------------------------------------------------------------
# switch-sibling distinguishability
# ---------------------------------------------------------------------------
def _old_vec(state_str, vocab_old, ENC):
    from poke_engine import State
    e = ENC.encode_state(State.from_string(state_str), vocab_old)
    return np.concatenate([e["a1_f"], e["b1_f"].reshape(-1), e["sf1"],
                           e["a2_f"], e["b2_f"].reshape(-1), e["sf2"], e["g"]]), \
        np.concatenate([np.asarray(e["a1_ids"]).reshape(-1),
                        np.asarray(e["b1_ids"]).reshape(-1),
                        np.asarray(e["a2_ids"]).reshape(-1),
                        np.asarray(e["b2_ids"]).reshape(-1)])


def _set_side_field(s, side, idx, val):
    top = s.split("/")
    f = top[side].split("=")
    f[idx] = val
    top[side] = "=".join(f)
    return "/".join(top)


def _set_mon_field(s, side, party, idx, val):
    top = s.split("/")
    f = top[side].split("=")
    p = f[party].split(",")
    p[idx] = val
    f[party] = ",".join(p)
    top[side] = "=".join(f)
    return "/".join(top)


MUTATIONS = [
    ("lum switch target  switch:1 -> switch:3",
     lambda s: (_set_side_field(s, 0, 28, "switch:1"), _set_side_field(s, 0, 28, "switch:3")),
     "encoder.py:497 discards the Switch payload (33.34% of side vectors)"),
    ("banked move identity  NONE-slot -> two moves",
     lambda s: (_set_side_field(s, 0, 23, "UTURN"), _set_side_field(s, 0, 23, "VOLTSWITCH")),
     "encoder.py:539 collapses the move identity to a bool"),
    ("active_move_actions  5 -> 9",
     lambda s: (_set_mon_field(s, 0, 0, 34, "5"), _set_mon_field(s, 0, 0, 34, "9")),
     "encoder.py:376 clips at 2 (14.96% of serve-time slots exceed it)"),
    ("times_attacked  7 -> 12",
     lambda s: (_set_mon_field(s, 0, 0, 31, "7"), _set_mon_field(s, 0, 0, 31, "12")),
     "encoder.py:375 clips at 6 (1.41% at serve)"),
    ("volatile outside the 34-name whitelist  ROOST -> OCTOLOCK",
     lambda s: (_set_side_field(s, 0, 8, "ROOST:"), _set_side_field(s, 0, 8, "OCTOLOCK:")),
     "encoder.py:111-119 drops 73 of 107 variants SILENTLY"),
    ("bench party permutation (mons 2 and 3 swapped)",
     lambda s: (s, _swap_party(s, 0, 2, 3)),
     "BENCH_SORT re-sorts by species vocab id, so party order is unrecoverable"),
    ("wish timer  1 -> 2",
     lambda s: (_set_side_field(s, 0, 18, "1"), _set_side_field(s, 0, 18, "2")),
     "encoder.py:466 collapses the timer to a bool"),
    ("substitute_health without the volatile  0 -> 40",
     lambda s: (_set_side_field(_set_side_field(s, 0, 8, ""), 0, 10, "0"),
                _set_side_field(_set_side_field(s, 0, 8, ""), 0, 10, "40")),
     "encoder.py:491 writes it only when SUBSTITUTE is present"),
]


def _swap_party(s, side, i, j):
    """Swap two NON-ACTIVE party members. Pure permutation: the same six mons,
    a different party order. active_index is left alone (both i and j must
    differ from it), so the position is mechanically identical except for the
    party indices that LastUsedMove::Switch and future_sight.1 point at."""
    top = s.split("/")
    f = top[side].split("=")
    f[i], f[j] = f[j], f[i]
    top[side] = "=".join(f)
    return "/".join(top)


def run_switchsib(states, n_roots=200, shards=SHARDS):
    from poke_engine import State, generate_instructions  # noqa: F401
    import encoder as ENC
    import dataset as DS
    vocab = LE.LosslessVocab()
    vocab_old = ENC.Vocab(frozen=True)

    print("\n== (A) REQUIRED TEST: real switch successors of one root ==")
    print("For each root with >=2 legal switch arms, build the modal successor of")
    print("each switch arm against the SAME opponent reply, then assert the two")
    print("encodings differ and that each decodes back to the right party slot.\n")
    n_pairs, n_diff, n_ai_ok, n_lum_ok, n_roots_used = 0, 0, 0, 0, 0
    n_succ, n_full_ok, old_diff = 0, 0, 0
    for fp in sorted(glob.glob(shards)):
        with gzip.open(fp, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or not row.get("t"):
                    continue
                for t in row["t"]:
                    arms = [a for a in (t.get("q") or {}) if a.lower().startswith("switch ")]
                    if len(arms) < 2 or not t.get("n2"):
                        continue
                    st = State.from_string(t["s"])
                    bstar = max(t["n2"], key=lambda k: t["n2"][k])
                    succ = []
                    for a in arms:
                        s2 = DS.successor(st, a, bstar)
                        if s2 is not None:
                            succ.append((a, s2.to_string()))
                    if len(succ) < 2:
                        continue
                    n_roots_used += 1
                    strs = [x[1] for x in succ]
                    ids, feats = ll.encode_states(strs, vocab, "full")
                    lean_ids, lean_feats = ll.encode_states(strs, vocab, "lean")
                    for x in range(len(succ)):
                        for y in range(x + 1, len(succ)):
                            n_pairs += 1
                            if not np.array_equal(feats[x], feats[y]) or \
                               not np.array_equal(ids[x], ids[y]):
                                n_diff += 1
                            if not np.array_equal(lean_feats[x], lean_feats[y]):
                                pass
                    for x, (arm, sstr) in enumerate(succ):
                        d = ll.decode(ids[x], feats[x], vocab, "full")
                        truth = LE.parse_state(sstr)
                        n_succ += 1
                        if all(d[k]["active_index"] == truth[k]["active_index"]
                               for k in ("side_one", "side_two")):
                            n_ai_ok += 1
                        if all(d[k]["last_used_move"] == truth[k]["last_used_move"]
                               for k in ("side_one", "side_two")):
                            n_lum_ok += 1
                        if RG.flatten(d) == RG.flatten(truth):
                            n_full_ok += 1
                    if n_roots_used >= n_roots:
                        break
                if n_roots_used >= n_roots:
                    break
        if n_roots_used >= n_roots:
            break
    print("roots used                         %d" % n_roots_used)
    print("switch-sibling PAIRS compared      %d" % n_pairs)
    print("pairs with DIFFERENT encodings     %d  (%.4f%%)"
          % (n_diff, 100.0 * n_diff / max(n_pairs, 1)))
    print("successor states encoded           %d" % n_succ)
    print("  active_index decodes correctly (both sides)     %d/%d" % (n_ai_ok, n_succ))
    print("  last_used_move decodes correctly (both sides)   %d/%d" % (n_lum_ok, n_succ))
    print("  ENTIRE state round-trips (all 101 field groups) %d/%d" % (n_full_ok, n_succ))
    assert n_pairs > 0, "no switch-sibling pairs found"
    assert n_diff == n_pairs, "SWITCH SIBLINGS COLLIDE: %d/%d" % (n_pairs - n_diff, n_pairs)
    assert n_ai_ok == n_succ and n_lum_ok == n_succ and n_full_ok == n_succ
    print("RESULT (A): PASS -- every switch-sibling pair encodes differently, and")
    print("            every successor decodes back to the exact source state.")

    print("\n== (B) SHARPER TEST: one-field mutations, new encoder vs today's ==")
    print("%-52s %-9s %-9s" % ("mutated field", "NEW", "TODAY"))
    base = states[0]
    rows = []
    for name, mut, why in MUTATIONS:
        try:
            s1, s2 = mut(base)
            i1, f1 = ll.encode_states([s1], vocab, "full")
            i2, f2 = ll.encode_states([s2], vocab, "full")
            new_diff = (not np.array_equal(f1, f2)) or (not np.array_equal(i1, i2))
            o1 = _old_vec(s1, vocab_old, ENC)
            o2 = _old_vec(s2, vocab_old, ENC)
            old_d = (not np.array_equal(o1[0], o2[0])) or (not np.array_equal(o1[1], o2[1]))
        except Exception as e:
            rows.append((name, "ERR:%s" % e, "", why))
            continue
        rows.append((name, "DIFFERS" if new_diff else "COLLIDES",
                     "DIFFERS" if old_d else "COLLIDES", why))
        old_diff += int(old_d)
    for name, a, b, why in rows:
        print("  %-50s %-9s %-9s   %s" % (name, a, b, why))
    bad = [r for r in rows if r[1] != "DIFFERS"]
    print("RESULT (B): %s -- %d/%d mutations separated by the new encoder, "
          "%d/%d by today's." % ("PASS" if not bad else "FAIL",
                                 len(rows) - len(bad), len(rows), old_diff, len(rows)))
    assert not bad, "new encoder collides on: %s" % [r[0] for r in bad]
    return True


# ---------------------------------------------------------------------------
def run_bench(states, reps=3):
    vocab = LE.LosslessVocab()
    print("\n== ENCODE THROUGHPUT (single process, %d states) ==" % len(states))
    # reference implementation, for the ratio
    ref_n = min(2000, len(states))
    t0 = time.time()
    for s in states[:ref_n]:
        LE.encode(LE.parse_state(s), vocab)
    ref = (time.time() - t0) / ref_n
    print("  %-26s %8.4f ms/state  %9d states/s   (parse+encode, reference)"
          % ("valuenet/lossless_encoder", 1000 * ref, int(1 / ref)))
    for variant in ("full", "lean"):
        V = ll.VARIANTS[variant]
        # PRODUCTION path: chunk, write into a preallocated array, drop the
        # chunk. This is exactly what lldataset.py does and it is the number
        # that governs "how long to encode a million positions".
        best, t_parse, t_enc = None, 0.0, 0.0
        for rep in range(reps):
            out = np.zeros((len(states), V.N_FEATS), np.float16)
            tp = te = 0.0
            t0 = time.time()
            for i in range(0, len(states), 4096):
                ch = states[i:i + 4096]
                a = time.time()
                C = ll.parse_batch(ch, vocab)
                b = time.time()
                bi, bf = ll.encode_columnar(C, vocab, variant)
                c = time.time()
                out[i:i + len(ch)] = bf
                tp += b - a
                te += c - b
            dt = (time.time() - t0) / len(states)
            if best is None or dt < best:
                best, t_parse, t_enc = dt, tp / len(states), te / len(states)
            del out
        print("  %-26s %8.4f ms/state  %9d states/s   (parse %.4f + encode %.4f)  "
              "%d cols, %.0f MB/1M rows f32"
              % ("llencoder [" + variant + "]", 1000 * best, int(1 / best),
                 1000 * t_parse, 1000 * t_enc, V.N_FEATS, V.N_FEATS * 4e6 / 1e6))
        print("  %-26s %8.2fx faster; 1M positions in %.1f min single-core"
              % ("", ref / best, best * 1e6 / 60.0))
    return True


def run_fp16(states):
    """float16 is the STORAGE dtype of the training cache (memory). Measure what
    it costs the round-trip rather than assuming it costs nothing."""
    vocab = LE.LosslessVocab()
    print("\n== float16 STORAGE COST (encoding itself stays float32) ==")
    for variant in ("full", "lean"):
        ids, feats = ll.encode_states(states, vocab, variant)
        f16 = feats.astype(np.float16).astype(np.float32)
        ok, tot = Counter(), Counter()
        for j, s in enumerate(states):
            src = LE.parse_state(s)
            dst = ll.decode(ids[j], f16[j], vocab, variant)
            a, b = RG.flatten(src), RG.flatten(dst)
            for k in set(a) | set(b):
                for x, y in zip(a.get(k, []), b.get(k, [])):
                    tot[k] += 1
                    ok[k] += int(x == y)
        failed = [k for k in sorted(tot) if ok[k] != tot[k]]
        print("  [%s] %d states, fields failing under float16: %d"
              % (variant, len(states), len(failed)))
        for k in failed:
            print("      %-40s %d/%d  (%.4f%%)"
                  % (k, ok[k], tot[k], 100.0 * ok[k] / tot[k]))
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus", "gate", "switchsib", "bench", "fp16", "all"])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("-n", type=int, default=50000)
    ap.add_argument("--variant", default="both")
    a = ap.parse_args()

    if a.cmd == "corpus":
        build_corpus(a.corpus, a.n)
        return
    if not os.path.exists(a.corpus):
        build_corpus(a.corpus, max(a.n, 60000))
    states = load_states(a.corpus, a.n)
    print("pair-A states loaded: %d  (distinct: %d)" % (len(states), len(set(states))))
    variants = ("full", "lean") if a.variant == "both" else (a.variant,)
    ok = True
    if a.cmd in ("gate", "all"):
        for v in variants:
            ok &= run_gate(states, v)
    if a.cmd in ("switchsib", "all"):
        ok &= run_switchsib(states)
    if a.cmd in ("bench", "all"):
        run_bench(states)
    if a.cmd in ("fp16", "all"):
        run_fp16(states[:2000])
    print("\nOVERALL: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
