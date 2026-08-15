"""ENCODER V2 -- re-encode ARM B only, after the enc2 repairs.

Arm A's cache does not move (its encoder, its columns and its rows are all
unchanged), so this rebuilds the enc2 half alone and PROVES the row order still
matches the existing `meta.npz` -- the join between the two arms and the labels
is by row index, and a silent re-order would invalidate every comparison.

  python vt_encode2.py shard <labels_shard.jsonl> <out_dir> <tag>
  python vt_encode2.py join  <out_dir>          # a,b,c,d -> enc2_feats/ids.npy
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

CHUNK = 1024
TAGS = ("a", "b", "c", "d")


def shard(src, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    L = enc2.DEFAULT_LAYOUT
    states, lab, band, gid = [], [], [], []
    for line in open(src):
        r = json.loads(line)
        if "label_p" not in r:
            continue
        states.append(r["s"])
        lab.append(r["label_p"])
        band.append(r["band"])
        gid.append(r["game"] if "game" in r else r.get("g", ""))
    n = len(states)
    t0 = time.time()
    ll_vocab = LE.LosslessVocab()
    f2 = np.zeros((n, L.N), np.float16)
    i2 = np.zeros((n, L.N_IDS), np.int16)
    for i in range(0, n, CHUNK):
        ids, f = enc2.encode_states(states[i:i + CHUNK], ll_vocab, chunk=CHUNK)
        assert np.abs(ids).max() < 32767
        assert np.isfinite(f).all(), "non-finite feature in shard %s at %d" % (tag, i)
        i2[i:i + len(ids)], f2[i:i + len(f)] = ids, f.astype(np.float16)
    np.savez(os.path.join(out_dir, "enc2b_%s.npz" % tag), feats=f2, ids=i2,
             label_p=np.array(lab, np.float64),
             band=np.array(band), gid=np.array(gid))
    print("[%s] %d rows %.0fs (%.2f ms/state) %d cols"
          % (tag, n, time.time() - t0, 1000 * (time.time() - t0) / n, L.N), flush=True)


def join(out_dir):
    L = enc2.DEFAULT_LAYOUT
    meta = np.load(os.path.join(out_dir, "meta.npz"), allow_pickle=False)
    parts = [np.load(os.path.join(out_dir, "enc2b_%s.npz" % t)) for t in TAGS]
    lab = np.concatenate([p["label_p"] for p in parts])
    band = np.concatenate([p["band"] for p in parts])
    # ROW-ORDER PROOF: the labels and bands rebuilt from the jsonl shards, in
    # this order, must equal the ones meta.npz was built from, row for row.
    assert lab.shape == meta["label_p"].shape, (lab.shape, meta["label_p"].shape)
    assert np.array_equal(lab, meta["label_p"]), "label_p row order moved"
    assert np.array_equal(band, meta["band"]), "band row order moved"
    n = len(lab)
    feats = np.lib.format.open_memmap(os.path.join(out_dir, "enc2_feats.npy"),
                                      mode="w+", dtype=np.float16, shape=(n, L.N))
    ids = np.lib.format.open_memmap(os.path.join(out_dir, "enc2_ids.npy"),
                                    mode="w+", dtype=np.int16, shape=(n, L.N_IDS))
    o = 0
    for p in parts:
        k = len(p["feats"])
        feats[o:o + k], ids[o:o + k] = p["feats"], p["ids"]
        o += k
    assert o == n
    feats.flush(), ids.flush()
    json.dump({"N": L.N, "N_IDS": L.N_IDS, "NM": L.NM, "NS": L.NS, "NR": L.NR,
               "NCM": L.NCM, "NCS": L.NCS, "NG": L.NG, "O_MON": L.O_MON,
               "O_SIDE": L.O_SIDE, "O_REL": L.O_REL, "O_CFM": L.O_CFM,
               "O_CFS": L.O_CFS, "O_GLOB": L.O_GLOB, "names": L.names},
              open(os.path.join(out_dir, "enc2_layout.json"), "w"))
    print("joined %d rows x %d cols; row order verified against meta.npz" % (n, L.N))


if __name__ == "__main__":
    if sys.argv[1] == "shard":
        shard(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        join(sys.argv[2])
