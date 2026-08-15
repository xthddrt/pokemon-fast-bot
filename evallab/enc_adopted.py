"""ADOPTED ENCODER cache builder -- "arm A + setup", the layout ENCODER_FINAL adopts.

WHAT IT PRODUCES, and why in exactly this file layout
-----------------------------------------------------
`vt_lib.Arm` already knows how to mmap and batch a cache; it wants

    old_{a1,a2}_{f,ids}.npy   old_{b1,b2}_{f,ids}.npy   old_{sf1,sf2}.npy
    old_g.npy   addon_mon.npy   addon_layout.json   meta.npz

so this writes precisely that and nothing else.  Nothing downstream changes.

  * ARM A  -- `valuenet/encoder.py:encode_state` under labenv's pinned recipe
    (BENCH_SORT=1, PP_TRUE_MAX=1, REVEAL_MASKS=0, **DROP_TIMES_ATTACKED=1**),
    i.e. 72 per-mon numerics, 9 embedding ids, 99 per-side, 18 global.
  * SETUP  -- enc2's 14-column per-mon counterfactual block (`cf_mon%d.*`),
    written to `addon_mon.npy` so `vt_lib.addon_cols("setup")` picks it up and
    it flows through arm A's SHARED per-mon MLP.

THE ONE HARD PART -- SLOT ALIGNMENT (lifted verbatim from vt_addon.py)
----------------------------------------------------------------------
arm A orders the bench by (species vocab id, party index); enc2 orders it by
party index.  Two different, both deterministic, permutations.  Gluing enc2's
setup block onto arm A's per-mon vector without the remap pairs each mon's
arm-A features with a DIFFERENT mon's setup features -- silently.  So per row
we build

    perm[j] = enc2 slot holding the mon arm A puts in slot j

from `encoder.bench_order`, the ONE definition of arm A's bench permutation,
and gather with it.  `verify` PROVES the remap with a feature both encoders
compute independently: hp_frac (arm A per-mon col 0, enc2 `mon%d.hp_frac`).

dtypes
------
float16 for every numeric, int16 for every embedding id.  Both are checked, not
assumed: `--check` measures the fp32->fp16 round-trip error on the real rows and
asserts every id fits in int16.  (The 2x saving is 2.7 GB vs 5.4 GB on plc1 and
21 GB vs 43 GB on the pretrain set; `vt_lib.Arm.batch` upcasts to
float32/int64 on the way into the net either way.)

USAGE
    python enc_adopted.py encode <src.jsonl[.gz]> <outdir> [--meta meta.npz]
                                 [--workers N] [--limit N] [--check K]
    python enc_adopted.py verify <outdir> <src.jsonl[.gz]> [K]

`src` is one json object per line with a state string under "s" (plc1
positions.jsonl) or under "s" alongside "y"/"w" (pretrain_select output).  Row r
of every output array IS line r of `src` -- that is the join contract, and
`--meta` makes it checkable: the per-row game id must equal meta["g"].
"""
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labenv  # noqa: F401,E402

CHUNK = 2048
KEYS_F = ("a1_f", "b1_f", "sf1", "a2_f", "b2_f", "sf2", "g")
KEYS_I = ("a1_ids", "b1_ids", "a2_ids", "b2_ids")
SETUP_D = 14


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def src_files(src):
    """A file, or a directory of *.jsonl.gz read in SORTED NAME ORDER.

    Sorted order is the join contract for the pretraining set: the selector
    writes one gz per input shard, and row r of the cache is line r of that
    concatenation."""
    if os.path.isdir(src):
        fs = sorted(os.path.join(src, f) for f in os.listdir(src)
                    if f.endswith(".jsonl.gz") or f.endswith(".jsonl"))
        assert fs, "no *.jsonl[.gz] under %s" % src
        return fs
    return [src]


def iter_rows(src, limit=0):
    """Stream json rows in row order. NEVER materialises the state strings: at
    ~3.5 kB each, the 7.9M-row pretraining set is ~28 GB of them."""
    n = 0
    for p in src_files(src):
        with _open(p) as f:
            for line in f:
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def read_states(src, limit=0):
    """Materialising reader -- CANARY / fp16-check sizes only."""
    states, gids = [], []
    for r in iter_rows(src, limit):
        states.append(r["s"])
        gids.append(r.get("g", ""))
    return states, gids


def shapes(n, nm, n_ids, ns, ng):
    return {"a1_f": (n, nm), "a2_f": (n, nm), "b1_f": (n, 5, nm), "b2_f": (n, 5, nm),
            "sf1": (n, ns), "sf2": (n, ns), "g": (n, ng),
            "a1_ids": (n, n_ids), "a2_ids": (n, n_ids),
            "b1_ids": (n, 5, n_ids), "b2_ids": (n, 5, n_ids)}


class MetaCols:
    """Every non-state field of the source, column by column, in row order.

    For plc1 the labels already exist as a prebuilt meta.npz and this is only
    used to CHECK the join; for the pretraining set it IS the label file
    (target `y`, weight `w`, and the provenance the selector attached)."""

    def __init__(self):
        self.f, self.s, self.keys = {}, {}, None

    def add(self, r):
        if self.keys is None:
            self.keys = [k for k in r if k != "s"]
            for k in self.keys:
                (self.f if isinstance(r[k], (int, float)) or r[k] is None
                 else self.s)[k] = []
        for k in self.keys:
            v = r.get(k)
            if k in self.f:
                self.f[k].append(np.nan if v is None else float(v))
            else:
                self.s[k].append("" if v is None else str(v))

    def save(self, path):
        d = {k: np.array(v, np.float64) for k, v in self.f.items()}
        d.update({k: np.array(v) for k, v in self.s.items()})
        d["i"] = np.arange(len(next(iter(d.values()))), dtype=np.int64)
        np.savez_compressed(path, **d)


def _flush(pool, batch, w):
    if pool is None:
        return sum(_do_chunk(t)[1] for t in batch)
    return sum(k for _, k in pool.map(_do_chunk, batch, chunksize=1))


# --------------------------------------------------------------------- worker
_G = {}


def _init(outdir, n, nm, n_ids, ns, ng):
    import enc2
    import lossless_encoder as LE
    from encoder import Vocab
    from poke_engine import State  # noqa: F401
    L = enc2.DEFAULT_LAYOUT
    cf = np.stack([np.array([i for i, x in enumerate(L.names)
                             if x.startswith("cf_mon%d." % k)], np.int64)
                   for k in range(12)])
    assert cf.shape == (12, SETUP_D), cf.shape
    hp = np.array([L.names.index("mon%d.hp_frac" % k) for k in range(12)], np.int64)
    sp = shapes(n, nm, n_ids, ns, ng)
    _G.update(enc2=enc2, ll=LE.LosslessVocab(), voc=Vocab(frozen=True), cf=cf, hp=hp,
              arr={k: np.lib.format.open_memmap(os.path.join(outdir, "old_%s.npy" % k),
                                                mode="r+") for k in sp},
              am=np.lib.format.open_memmap(os.path.join(outdir, "addon_mon.npy"), mode="r+"),
              hpchk=np.lib.format.open_memmap(os.path.join(outdir, "_hpchk.npy"), mode="r+"))


def _do_chunk(task):
    off, states = task
    from encoder import encode_state, bench_order
    from poke_engine import State
    enc2, voc = _G["enc2"], _G["voc"]
    k = len(states)

    # --- enc2: the 14-column setup block, in ENC2 slot order -----------------
    _, f2 = enc2.encode_states(states, _G["ll"], chunk=CHUNK)
    assert np.isfinite(f2).all(), "non-finite enc2 feature at row %d" % off
    setup = f2[:, _G["cf"].reshape(-1)].reshape(k, 12, SETUP_D)
    hp2 = f2[:, _G["hp"]]                                    # (k, 12) enc2 order

    # --- arm A, and the enc2->armA slot permutation, from ONE parse ----------
    A = _G["arr"]
    perm = np.zeros((k, 12), np.int64)
    for r, s in enumerate(states):
        st = State.from_string(s)
        d = encode_state(st, voc)
        i = off + r
        A["a1_f"][i], A["a1_ids"][i] = d["a1_f"], d["a1_ids"]
        A["b1_f"][i], A["b1_ids"][i] = d["b1_f"], d["b1_ids"]
        A["a2_f"][i], A["a2_ids"][i] = d["a2_f"], d["a2_ids"]
        A["b2_f"][i], A["b2_ids"][i] = d["b2_f"], d["b2_ids"]
        A["sf1"][i], A["sf2"][i], A["g"][i] = d["sf1"], d["sf2"], d["g"]
        for si, side in enumerate((st.side_one, st.side_two)):
            a = int(side.active_index)
            mons = list(side.pokemon)
            party = [a] + [t[0] for t in bench_order(mons, a, voc)]
            for j, p in enumerate(party):
                perm[r, si * 6 + j] = (0 if p == a else (p + 1 if p < a else p)) + 6 * si

    assert (np.sort(perm, axis=1) == np.arange(12)).all(), "perm is not a permutation"
    assert (perm[:, :6] < 6).all() and (perm[:, 6:] >= 6).all(), "perm crosses sides"
    _G["am"][off:off + k] = np.take_along_axis(setup, perm[:, :, None], axis=1)
    # alignment witness: enc2 hp_frac gathered through perm, to be compared
    # against arm A's own hp_frac (col 0) by `verify`.
    _G["hpchk"][off:off + k] = np.take_along_axis(hp2, perm, axis=1)
    return off, k


# --------------------------------------------------------------------- encode
def encode(src, outdir, meta_path="", workers=0, limit=0, check=0, rows=0):
    from encoder import NUM_MON_FEATS, NUM_SIDE_FEATS, NUM_GLOBAL_FEATS, N_IDS
    import enc2
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    nm, n_ids, ns, ng = NUM_MON_FEATS, N_IDS, NUM_SIDE_FEATS, NUM_GLOBAL_FEATS
    assert nm == 72, "expected the adopted 72-wide per-mon block, got %d" % nm

    m = np.load(meta_path, allow_pickle=False) if meta_path else None
    if m is not None and not rows:
        rows = len(m["g"])
    n = rows or sum(1 for _ in iter_rows(src, limit))
    if limit:
        n = min(n, limit)
    print("rows=%d (counted in %.0fs)  mon=%d ids=%d side=%d glob=%d setup=%d"
          % (n, time.time() - t0, nm, n_ids, ns, ng, SETUP_D), flush=True)

    sp = shapes(n, nm, n_ids, ns, ng)
    for k, s in sp.items():
        np.lib.format.open_memmap(os.path.join(outdir, "old_%s.npy" % k), mode="w+",
                                  dtype=np.int16 if "ids" in k else np.float16, shape=s)
    np.lib.format.open_memmap(os.path.join(outdir, "addon_mon.npy"), mode="w+",
                              dtype=np.float16, shape=(n, 12, SETUP_D))
    np.lib.format.open_memmap(os.path.join(outdir, "_hpchk.npy"), mode="w+",
                              dtype=np.float16, shape=(n, 12))

    L = enc2.DEFAULT_LAYOUT
    json.dump({"D_mon": SETUP_D, "D_rest": 0, "groups": {"mon.setup": [0, SETUP_D]},
               "mon_names": [x for x in L.names if x.startswith("cf_mon0.")],
               "rest_names": []},
              open(os.path.join(outdir, "addon_layout.json"), "w"))

    w = workers or max(1, (os.cpu_count() or 2) // 2)
    ctx = mp.get_context("fork")
    pool = (ctx.Pool(w, initializer=_init, initargs=(outdir, n, nm, n_ids, ns, ng))
            if w > 1 else None)
    if pool is None:
        _init(outdir, n, nm, n_ids, ns, ng)

    # STREAMING, in bounded super-batches: the parent never holds more than
    # ~4*w chunks of state strings at once, and the workers get one chunk each.
    cols, t1, done, off, batch, buf = MetaCols(), time.time(), 0, 0, [], []
    for r in iter_rows(src, limit):
        buf.append(r["s"])
        cols.add(r)
        if len(buf) == CHUNK:
            batch.append((off, buf))
            off, buf = off + CHUNK, []
        if len(batch) >= 4 * w:
            done += _flush(pool, batch, w)
            batch = []
            el = time.time() - t1
            print("  %d/%d rows  %.0fs  %.1f states/s (%.1f/core-s)"
                  % (done, n, el, done / max(el, 1e-9), done / max(el, 1e-9) / w),
                  flush=True)
    if buf:
        batch.append((off, buf))
    if batch:
        done += _flush(pool, batch, w)
    if pool is not None:
        pool.close()
        pool.join()
    assert done == n, "encoded %d rows but allocated %d" % (done, n)
    el = time.time() - t1

    if m is not None:
        mg = np.asarray(m["g"]).astype(str)
        got = np.array(cols.s["g"], dtype=mg.dtype)
        bad = np.flatnonzero(mg != got)
        assert len(bad) == 0, "JOIN BROKEN: %d rows disagree, first at %d (%r vs %r)" % (
            len(bad), bad[0], got[bad[0]], mg[bad[0]])
        print("join verified: all %d row game ids match meta.npz" % n, flush=True)
    else:
        cols.save(os.path.join(outdir, "meta.npz"))
        print("meta.npz written from the source rows (%d numeric + %d string cols)"
              % (len(cols.f), len(cols.s)), flush=True)
    rate = n / max(el, 1e-9)
    stats = {"rows": n, "workers": w, "encode_s": round(el, 1),
             "states_per_s": round(rate, 1), "states_per_core_s": round(rate / w, 1),
             "mon_feats": nm, "n_ids": n_ids, "side_feats": ns, "glob_feats": ng,
             "setup_d": SETUP_D, "src": os.path.basename(src)}
    if check:
        stats.update(selfcheck(outdir, src, min(check, n)))
        # the alignment witness has served its purpose; it is 24 B/row of pure
        # scaffolding and must not ship with the cache.
        if os.environ.get("KEEP_HPCHK") != "1":
            os.remove(os.path.join(outdir, "_hpchk.npy"))
    stats["bytes"] = {f: os.path.getsize(os.path.join(outdir, f))
                      for f in sorted(os.listdir(outdir)) if f.endswith(".npy")}
    stats["bytes_total"] = sum(stats["bytes"].values())
    json.dump(stats, open(os.path.join(outdir, "ENC_STATS.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "bytes"}, indent=1), flush=True)
    return stats


# --------------------------------------------------------------------- verify
def selfcheck(outdir, src, k=5000):
    """Everything that can be wrong and is cheap to catch, on the first k rows."""
    out = {}
    A = {x: np.load(os.path.join(outdir, "old_%s.npy" % x), mmap_mode="r")
         for x in KEYS_F + KEYS_I}
    am = np.load(os.path.join(outdir, "addon_mon.npy"), mmap_mode="r")
    hpchk = np.load(os.path.join(outdir, "_hpchk.npy"), mmap_mode="r")
    n = A["a1_f"].shape[0]
    k = min(k, n)

    nan = {x: int(np.isnan(np.asarray(A[x][:k], np.float32)).sum()) for x in KEYS_F}
    nan["addon_mon"] = int(np.isnan(np.asarray(am[:k], np.float32)).sum())
    out["nan_counts"] = nan
    assert sum(nan.values()) == 0, nan
    out["id_max"] = int(max(int(np.asarray(A[x][:k]).max()) for x in KEYS_I))
    out["id_min"] = int(min(int(np.asarray(A[x][:k]).min()) for x in KEYS_I))
    assert 0 <= out["id_min"] and out["id_max"] < 32767, "ids do not fit int16"

    # SLOT ALIGNMENT PROOF: arm A's own hp_frac (per-mon col 0) vs enc2's
    # hp_frac gathered through the same permutation. Two independent encoders.
    a = np.concatenate([np.asarray(A["a1_f"][:k, :1], np.float32),
                        np.asarray(A["b1_f"][:k, :, 0], np.float32),
                        np.asarray(A["a2_f"][:k, :1], np.float32),
                        np.asarray(A["b2_f"][:k, :, 0], np.float32)], axis=1)
    d = np.abs(a - np.asarray(hpchk[:k], np.float32))
    out["hp_align_max_abs_diff"] = float(d.max())
    out["hp_align_rows"] = int(k)
    assert d.max() < 2e-3, "SLOT MISALIGNMENT: arm A vs enc2 hp_frac differ by %g" % d.max()

    # fp16 losslessness, measured on the REAL rows (re-encoding in fp32)
    out.update(fp16_error(src, k))

    # not-all-zero: a silently dead block is the failure this catches
    out["setup_nonzero_frac"] = float((np.asarray(am[:k], np.float32) != 0).mean())
    out["setup_const_cols"] = int((np.asarray(am[:k], np.float32)
                                  .reshape(-1, SETUP_D).std(axis=0) == 0).sum())
    return out


def fp16_error(src, k):
    """Encode k rows again in fp32 and measure what float16 storage costs."""
    from encoder import Vocab, encode_state
    from poke_engine import State
    import enc2
    import lossless_encoder as LE
    states, _ = read_states(src, k)
    voc = Vocab(frozen=True)
    f32 = []
    for s in states:
        d = encode_state(State.from_string(s), voc)
        f32.append(np.concatenate([np.asarray(d[x], np.float32).reshape(-1)
                                   for x in ("a1_f", "b1_f", "sf1", "a2_f", "b2_f",
                                             "sf2", "g")]))
    x = np.stack(f32)
    L = enc2.DEFAULT_LAYOUT
    cf = np.array([i for i, nm in enumerate(L.names)
                   if nm.startswith("cf_mon")], np.int64)
    _, f2 = enc2.encode_states(states, LE.LosslessVocab(), chunk=CHUNK)
    x = np.concatenate([x, np.asarray(f2[:, cf], np.float32)], axis=1)
    y = x.astype(np.float16).astype(np.float32)
    err = np.abs(x - y)
    nz = np.abs(x) > 0
    rel = np.zeros_like(err)
    rel[nz] = err[nz] / np.abs(x[nz])
    return {"fp16_rows": int(len(x)), "fp16_cols": int(x.shape[1]),
            "fp16_max_abs_err": float(err.max()),
            "fp16_max_rel_err": float(rel.max()),
            "fp16_mean_abs_err": float(err.mean()),
            "fp16_exact_frac": float((err == 0).mean()),
            "fp16_max_abs_value": float(np.abs(x).max())}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "encode":
        src, outdir = sys.argv[2], sys.argv[3]
        kw = {}
        rest = sys.argv[4:]
        for i in range(0, len(rest) - 1, 2):
            kw[rest[i].lstrip("-")] = rest[i + 1]
        encode(src, outdir, meta_path=kw.get("meta", ""),
               workers=int(kw.get("workers", 0)), limit=int(kw.get("limit", 0)),
               check=int(kw.get("check", 0)), rows=int(kw.get("rows", 0)))
    elif cmd == "verify":
        print(json.dumps(selfcheck(sys.argv[2], sys.argv[3],
                                   int(sys.argv[4]) if len(sys.argv) > 4 else 5000),
                         indent=1))
    else:
        raise SystemExit(__doc__)
