"""Build plc1's meta.npz -- the label side of the cache -- IN `i` ORDER.

`i` is the 0-based line number of the position in
s3://<bucket>/evallab/plc1/positions.jsonl.gz, so sorting the 39 local label
shards by `i` makes row r of meta.npz correspond to line r of the state file.
That equality is the whole join contract, and the encoder asserts it row by row
against the game id it reads out of the state file itself.

    python plc1_meta.py <labels_dir> <out.npz>
"""
import json
import os
import sys

import numpy as np

FIELDS_F = ("label_p", "q_search", "y_single", "se")
FIELDS_I = ("ply", "wins", "losses", "ties", "trunc", "n_playouts")


def main(src_dir, out):
    files = sorted(f for f in os.listdir(src_dir)
                   if f.startswith("labels_") and f.endswith(".jsonl"))
    assert files, src_dir
    rows = []
    for fn in files:
        with open(os.path.join(src_dir, fn)) as f:
            for line in f:
                rows.append(json.loads(line))
    n = len(rows)
    i = np.array([r["i"] for r in rows], np.int64)
    o = np.argsort(i, kind="stable")
    assert np.array_equal(i[o], np.arange(n)), "i is not a 0..n-1 permutation"
    rows = [rows[k] for k in o]
    d = {"i": np.arange(n, dtype=np.int64),
         "g": np.array([r["g"] for r in rows]),
         "pair": np.array([r["pair"] for r in rows]),
         "band": np.array([r["band"] for r in rows]),
         "src": np.array([r["src"] for r in rows])}
    for k in FIELDS_F:
        d[k] = np.array([r[k] for r in rows], np.float64)
    for k in FIELDS_I:
        d[k] = np.array([r[k] for r in rows], np.int32)
    np.savez_compressed(out, **d)
    print("meta rows=%d  label_p mean=%.4f  bands=%s"
          % (n, d["label_p"].mean(),
             {b: int((d["band"] == b).sum()) for b in ("early", "mid", "late")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
