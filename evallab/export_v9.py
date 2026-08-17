"""Write a PKNN **v9** .bin from a 3-trunk PhaseNet checkpoint.

Copy-adapted from export_v8.py (kept untouched — it is the production v8
exporter). The checkpoint is corrections/phase_probe.py's
`{"sd": {"shared": <ArmAPlusNet sd>, "trunks": <ModuleList of 3 trunks>}}`:
`shared` is EXACTLY the net export_v8 serializes (its own `base.trunk.*` is
the unused warm-start copy and is dropped here); the trunks are three copies
of the v8 trunk shape, in blend order [early, mid, late].

Layout = v8's byte for byte, except VERSION 9 and: everywhere v8 writes one
trunk's tensors this file writes THREE, grouped per trunk at each position
(mirrored by Network::load in poke-engine/src/genx/evaluate_nn.rs):
  header(64) | vocab | mon0.w mon0.b mon2.w mon2.b [t0.w t0.b]x3
  | u32 attn_d=0 | u32 n_slots=0 | u32 n_blocks=0 u32 flags
  | u32 npp + pp bytes | [trunk2.w trunk2.b trunk4.w trunk4.b]x3

  python export_v9.py <ckpt.pt> <out.bin>
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: F401,E402
from encoder import (  # noqa: E402
    MAX_PP, NUM_GLOBAL_FEATS, NUM_MON_FEATS, NUM_SIDE_FEATS,
)

ROOT = os.path.dirname(LAB)
VOCAB = os.path.join(ROOT, "valuenet/vocab.json")
TABLES = ["species", "item", "ability", "move", "teratype"]
EMB_DIMS = {"species": 32, "item": 12, "ability": 12, "move": 16, "teratype": 8}
EXPECTED_ROWS = {"species": 1444, "item": 240, "ability": 321, "move": 886,
                 "teratype": 21}
N_MON_EMB = 140
N_SETUP = 14
MON_HID, TRUNK = 128, 256              # evaluate_nn.rs:28-29, hard-checked there
VERSION = 9
N_TRUNKS = 3
# v9 mon width == v8's: encoder numerics (72 under DROP_TIMES_ATTACKED=1) + setup.
MON_IN = N_MON_EMB + NUM_MON_FEATS + N_SETUP                       # 226
TRUNK_IN = 4 * MON_HID + 2 * NUM_SIDE_FEATS + NUM_GLOBAL_FEATS     # 728


def main(ckpt_path, out):
    assert NUM_MON_FEATS == 72, (
        "labenv must pin DROP_TIMES_ATTACKED=1 (got NUM_MON_FEATS=%d); the v9 "
        "mon block is 72 numerics + 14 setup" % NUM_MON_FEATS)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    shared = {k[5:] if k.startswith("base.") else k: v
              for k, v in ck["sd"]["shared"].items()}
    trunks = dict(ck["sd"]["trunks"])
    vocab = json.load(open(VOCAB))

    got_mon_in = int(shared["mon.0.weight"].shape[1])
    got_trunk_in = int(trunks["0.0.weight"].shape[1])
    assert got_mon_in == MON_IN, (got_mon_in, MON_IN)
    assert got_trunk_in == TRUNK_IN, (got_trunk_in, TRUNK_IN)
    assert int(shared["mon.0.weight"].shape[0]) == MON_HID, \
        shared["mon.0.weight"].shape
    assert int(trunks["0.0.weight"].shape[0]) == TRUNK, trunks["0.0.weight"].shape

    MAIN = [("mon.0.weight", (MON_HID, MON_IN)), ("mon.0.bias", (MON_HID,)),
            ("mon.2.weight", (MON_HID, MON_HID)), ("mon.2.bias", (MON_HID,))]
    # shared's own trunk is the warm-start leftover PhaseNet never runs
    # (phase_probe.PhaseNet.__call__ only uses embed_mon/pool); the real
    # trunks live under "trunks". Dropped, but accounted for below.
    DROP = ["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
            "trunk.4.weight", "trunk.4.bias"]
    T0 = [[("%d.0.weight" % k, (TRUNK, TRUNK_IN)), ("%d.0.bias" % k, (TRUNK,))]
          for k in range(N_TRUNKS)]
    TAIL = [[("%d.2.weight" % k, (TRUNK, TRUNK)), ("%d.2.bias" % k, (TRUNK,)),
             ("%d.4.weight" % k, (1, TRUNK)), ("%d.4.bias" % k, (1,))]
            for k in range(N_TRUNKS)]

    # two-sided on BOTH dicts: no key silently dropped, none silently missing
    consumed = ({"emb.%s.weight" % t for t in TABLES}
                | {n for n, _ in MAIN} | set(DROP))
    missing, extra = consumed - set(shared), set(shared) - consumed
    assert not (missing or extra), (sorted(missing), sorted(extra))
    consumed_t = {n for grp in T0 + TAIL for n, _ in grp}
    missing, extra = consumed_t - set(trunks), set(trunks) - consumed_t
    assert not (missing or extra), (sorted(missing), sorted(extra))

    rows = {}
    for t in TABLES:
        table = vocab[t]
        assert len(table) == EXPECTED_ROWS[t], (t, len(table))
        assert sorted(table.values()) == list(range(len(table))), t
        assert table.get("UNK") == 0, t
        rows[t] = len(table)

    tensors = []
    for t in TABLES:
        w = shared["emb.%s.weight" % t].numpy()
        assert w.dtype == np.float32 and w.shape[1] == EMB_DIMS[t] \
            and w.shape[0] >= rows[t], (t, w.shape)
        assert not w[0].any(), "%s row 0 (padding) not zero" % t
        tensors.append(np.ascontiguousarray(w[:rows[t]]))

    def take(sd, names):
        out_ = []
        for name, shape in names:
            w = sd[name].numpy()
            assert w.dtype == np.float32 and w.shape == shape, (name, w.shape)
            out_.append(np.ascontiguousarray(w))
        return out_

    tensors += take(shared, MAIN)
    for grp in T0:
        tensors += take(trunks, grp)
    tail_tensors = []
    for grp in TAIL:
        tail_tensors += take(trunks, grp)

    dims = [rows["species"], 32, rows["item"], 12, rows["ability"], 12,
            rows["move"], 16, rows["teratype"], 8, MON_IN, MON_HID,
            TRUNK_IN, TRUNK]
    header = b"PKNN" + struct.pack("<15I", VERSION, *dims)
    assert len(header) == 64

    vocab_bytes = bytearray()
    for t in TABLES:
        by_id = sorted(vocab[t].items(), key=lambda kv: kv[1])
        vocab_bytes += struct.pack("<I", rows[t])
        for name, _ in by_id:
            b = name.upper().encode("ascii")
            vocab_bytes += struct.pack("<H", len(b)) + b

    # max-PP table: PP_TRUE_MAX=1 is pinned by labenv, so the net was trained on
    # pp/true_max and the engine MUST get the table or it re-derives pp/64.
    by_id = sorted(vocab["move"].items(), key=lambda kv: kv[1])
    vals = []
    for name, _ in by_id:
        mp = MAX_PP.get(name.upper())
        assert mp is not None and 0 <= mp <= 255, (name, mp)
        vals.append(mp)
    assert len(vals) == rows["move"]
    pp_bytes = struct.pack("<I", len(vals)) + bytes(vals)

    bench_sort, pp_true_max = 1, 1      # labenv's pinned recipe
    with open(out, "wb") as f:
        f.write(header)
        f.write(vocab_bytes)
        for a in tensors:
            f.write(a.tobytes())
        f.write(struct.pack("<I", 0))               # attn_d = 0
        f.write(struct.pack("<I", 0))               # policy slots = 0
        f.write(struct.pack("<II", 0,               # n_blocks = 0
                            (bench_sort & 1) | ((pp_true_max & 1) << 1)))
        f.write(pp_bytes)
        for a in tail_tensors:
            f.write(a.tobytes())

    data = open(out, "rb").read()
    expect = (64 + len(vocab_bytes) + 4 * sum(a.size for a in tensors)
              + 4 + 4 + 8 + len(pp_bytes)
              + 4 * sum(a.size for a in tail_tensors))
    assert len(data) == expect, (len(data), expect)
    assert struct.unpack("<15I", data[4:64]) == (VERSION, *dims)
    sha = hashlib.sha256(data).hexdigest()
    meta = {"version": VERSION, "phase_trunks": N_TRUNKS, "mon_in": MON_IN,
            "trunk_in": TRUNK_IN, "mon_hid": MON_HID, "trunk": TRUNK,
            "bench_mult": 1, "bench_sort": bench_sort,
            "pp_true_max": pp_true_max, "attn_d": 0, "policy_slots": 0,
            "n_blocks": 0, "bytes": len(data), "sha256": sha,
            "from_ckpt": os.path.basename(ckpt_path),
            "ckpt_cfg": {k: v for k, v in cfg.items()
                         if isinstance(v, (str, int, float))}}
    print(json.dumps(meta, indent=1))
    json.dump(meta, open(os.path.splitext(out)[0] + ".meta.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
