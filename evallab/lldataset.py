"""Shards -> lossless-feature .npz cache, the same cached shape `dataset.py`
produces but with `llencoder`'s flat vector instead of the incumbent encoder's
per-block tensors.

  <prefix>_ll<variant>_pos.npz        train games
  <prefix>_ll<variant>_hold_pos.npz   held-out games
  keys: ids int16[N, 184]  feats float16[N, 1863|1206]  y  gid  ply  nlegal

WHY THIS IS FAST
  `dataset.py` calls `State.from_string` and then encodes one position at a
  time. Nothing here needs the engine: the state STRING is the whole input, so
  a shard becomes a list of strings and `llencoder.encode_states` turns 8,192 of
  them into one array with vectorised numpy. Sibling successors DO need the
  engine and are deliberately not built here -- the playout-averaging agent owns
  the real labels; this cache exists to prove the plumbing and to be the
  substrate those labels attach to.

STORAGE DTYPE, stated plainly
  Features are stored float16 to fit the 8.6 GB box, exactly as `dataset.py`
  does. The ENCODER emits float32 and the round-trip gate runs on float32.
  `ll_gate.py fp16` measures precisely which fields float16 storage costs, and
  ENCODER_BUILD.md reports it. Set LL_DTYPE=float32 to keep the cache exact at
  2x the memory.

USAGE
  python lldataset.py <shard_glob> <out_prefix> [--variant full|lean]
ENV
  POS_STRIDE  keep every Nth decision (default 1)
  LL_DTYPE    float16 (default) | float32
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

import labenv  # noqa: F401
import llencoder as ll

sys.path.insert(0, os.path.join(labenv.ROOT, "valuenet"))
import lossless_encoder as LE  # noqa: E402

POS_STRIDE = int(os.environ.get("POS_STRIDE", "1"))
DTYPE = np.float16 if os.environ.get("LL_DTYPE", "float16") == "float16" else np.float32
CHUNK = 8192


def stream_games(paths, holdout):
    for p in sorted(paths):
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                if labenv.is_holdout(row["g"]) != holdout:
                    continue
                yield row


def build(paths, out_prefix, variant, holdout):
    V = ll.VARIANTS[variant]
    vocab = LE.LosslessVocab()
    t0 = time.time()

    n_games, n_pos = 0, 0
    for g in stream_games(paths, holdout):
        n_games += 1
        n_pos += len(range(0, len(g["t"]), POS_STRIDE))
    print("pass1: games=%d (holdout=%s) positions=%d (stride %d)  %.0fs"
          % (n_games, holdout, n_pos, POS_STRIDE, time.time() - t0), flush=True)
    if n_pos == 0:
        return None

    feats = np.zeros((n_pos, V.N_FEATS), DTYPE)
    ids = np.zeros((n_pos, V.N_IDS), np.int16)
    y = np.zeros(n_pos, np.float32)
    gid = np.zeros(n_pos, np.int32)
    ply = np.zeros(n_pos, np.int32)
    nlegal = np.zeros(n_pos, np.int16)

    buf_s, buf_meta, i = [], [], 0

    def flush():
        nonlocal i, buf_s, buf_meta
        if not buf_s:
            return
        bi, bf = ll.encode_states(buf_s, vocab, variant, chunk=CHUNK)
        assert np.abs(bi).max() < 32767, "id exceeds int16"
        k = len(buf_s)
        ids[i:i + k], feats[i:i + k] = bi, bf.astype(DTYPE)
        for j, (oy, og, op, on) in enumerate(buf_meta):
            y[i + j], gid[i + j], ply[i + j], nlegal[i + j] = oy, og, op, on
        i += k
        buf_s, buf_meta = [], []

    for gi, gme in enumerate(stream_games(paths, holdout)):
        out = float(gme["outcome"])
        for pi in range(0, len(gme["t"]), POS_STRIDE):
            t = gme["t"][pi]
            buf_s.append(t["s"])
            buf_meta.append((out, gi, pi, len(t.get("q", {}))))
            if len(buf_s) >= CHUNK:
                flush()
        if (gi + 1) % 2000 == 0:
            print("  %d/%d games  %.0fs  pos=%d" % (gi + 1, n_games, time.time() - t0, i),
                  flush=True)
    flush()

    path = "%s_ll%s%s_pos.npz" % (out_prefix, variant, "_hold" if holdout else "")
    np.savez(path, ids=ids[:i], feats=feats[:i], y=y[:i], gid=gid[:i],
             ply=ply[:i], nlegal=nlegal[:i])
    print("wrote %s  rows=%d cols=%d  %.2f GB  %.0fs  (%.0f pos/s)"
          % (path, i, V.N_FEATS, os.path.getsize(path) / 1e9, time.time() - t0,
             i / max(time.time() - t0, 1e-9)), flush=True)
    if vocab.misses:
        print("VOCAB MISSES (each is real information loss):", dict(list(vocab.misses)[:20]))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("out_prefix")
    ap.add_argument("--variant", default="full", choices=["full", "lean"])
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))
    assert paths, "no shards matched %r" % a.shards
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    build(paths, a.out_prefix, a.variant, holdout=False)
    build(paths, a.out_prefix, a.variant, holdout=True)
