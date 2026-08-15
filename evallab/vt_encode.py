"""ENCODER VALUE TEST -- cache builder.

Turns one labels_*.jsonl shard into the two arms' feature caches:

  ARM A (BASELINE, what we ship): valuenet/encoder.py `encode_state` under the
        v6 champion recipe pinned by labenv (BENCH_SORT=1, PP_TRUE_MAX=1,
        REVEAL_MASKS=0) -- i.e. the SHIPPED encoder, not the lossless one --
        stored in the per-block shape `labmodel.LabValueNet` consumes.
  ARM B (NEW): evallab/enc2.py, 1,413 numeric columns + 112 embedding ids,
        stored flat.

Both arms are built from the SAME state string, in the SAME row order as the
jsonl, so `meta.npz` (labels, band, game id) joins to both by row index.

  python vt_encode.py <labels_shard.jsonl> <out_dir> <tag>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labenv  # noqa: F401,E402
import lossless_encoder as LE  # noqa: E402
import enc2  # noqa: E402
from encoder import Vocab, encode_state  # noqa: E402
from poke_engine import State  # noqa: E402

OLD_KEYS = ("a1_ids", "a1_f", "b1_ids", "b1_f", "sf1",
            "a2_ids", "a2_f", "b2_ids", "b2_f", "sf2", "g")
CHUNK = 1024


def main():
    src, out_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    states = []
    for line in open(src):
        r = json.loads(line)
        if "label_p" not in r:
            continue
        states.append(r["s"])
    n = len(states)
    t0 = time.time()

    # ---- ARM B: enc2 (cold path -- no shared static; the reference path) ----
    ll_vocab = LE.LosslessVocab()
    f2 = np.zeros((n, 1413), np.float16)
    i2 = np.zeros((n, 112), np.int16)
    for i in range(0, n, CHUNK):
        ids, f = enc2.encode_states(states[i:i + CHUNK], ll_vocab, chunk=CHUNK)
        assert np.abs(ids).max() < 32767
        i2[i:i + len(ids)], f2[i:i + len(f)] = ids, f.astype(np.float16)
    np.savez(os.path.join(out_dir, "enc2_%s.npz" % tag), feats=f2, ids=i2)
    print("[%s] enc2 done %d rows %.0fs" % (tag, n, time.time() - t0), flush=True)
    del f2, i2

    # ---- ARM A: shipped encoder ----
    v = Vocab(frozen=True)
    buf = {}
    for i, s in enumerate(states):
        d = encode_state(State.from_string(s), v)
        if not buf:
            for k in OLD_KEYS:
                a = np.asarray(d[k])
                buf[k] = np.zeros((n,) + a.shape, np.int16 if "ids" in k else np.float16)
        for k in OLD_KEYS:
            buf[k][i] = d[k]
    np.savez(os.path.join(out_dir, "old_%s.npz" % tag), **buf)
    print("[%s] old done %d rows %.0fs  unk=%s"
          % (tag, n, time.time() - t0, dict(list(v.unk_hits.items())[:5])), flush=True)


if __name__ == "__main__":
    main()
