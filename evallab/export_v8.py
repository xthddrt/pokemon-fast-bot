"""Write a PKNN **v8** .bin from an evallab "arm A + setup" checkpoint.

`valuenet/export_weights.py` is the reference producer, but it consumes
`valuenet/train.py`'s checkpoint dict and its version ladder stops at 7. The
adopted evaluator is trained in the lab (`vt_n.py` -> `vt_lib.ArmAPlusNet`
wrapping `labmodel.LabValueNet`, which `labmodel.assert_baseline_parity` proves
is `train.ValueNet` bit for bit), so the ONLY differences are

  * tensor keys carry a "base." prefix (the wrapper),
  * mon.0 is 226 wide = 140 emb + 72 numerics + 14 setup  -> version 8,
  * everything else -- section order, vocab block, max-PP table, the v6 tail
    trailer -- is byte-for-byte the v6/v7 layout that `Network::load` reads.

Layout consumed by poke-engine/src/genx/evaluate_nn.rs:
  header(64) | vocab | mon0.w mon0.b mon2.w mon2.b t0.w t0.b | u32 attn_d=0
  | u32 n_slots=0 | u32 n_blocks=0 u32 flags | u32 npp + pp bytes
  | trunk2.w trunk2.b trunk4.w trunk4.b

  python export_v8.py <ckpt.pt> <out.bin>
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
VERSION = 8
# v8 mon width: encoder numerics (72 under DROP_TIMES_ATTACKED=1) + setup block.
MON_IN = N_MON_EMB + NUM_MON_FEATS + N_SETUP                       # 226
TRUNK_IN = 4 * MON_HID + 2 * NUM_SIDE_FEATS + NUM_GLOBAL_FEATS     # 728


def main(ckpt_path, out):
    assert NUM_MON_FEATS == 72, (
        "labenv must pin DROP_TIMES_ATTACKED=1 (got NUM_MON_FEATS=%d); the v8 "
        "mon block is 72 numerics + 14 setup" % NUM_MON_FEATS)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["sd"] if "sd" in ck else ck["model"]
    cfg = ck.get("cfg", {})
    sd = {k[5:] if k.startswith("base.") else k: v for k, v in sd.items()}
    vocab = json.load(open(VOCAB))

    got_mon_in = int(sd["mon.0.weight"].shape[1])
    got_trunk_in = int(sd["trunk.0.weight"].shape[1])
    assert got_mon_in == MON_IN, (got_mon_in, MON_IN)
    assert got_trunk_in == TRUNK_IN, (got_trunk_in, TRUNK_IN)
    assert int(sd["mon.0.weight"].shape[0]) == MON_HID, sd["mon.0.weight"].shape
    assert int(sd["trunk.0.weight"].shape[0]) == TRUNK, sd["trunk.0.weight"].shape

    MAIN = [("mon.0.weight", (MON_HID, MON_IN)), ("mon.0.bias", (MON_HID,)),
            ("mon.2.weight", (MON_HID, MON_HID)), ("mon.2.bias", (MON_HID,)),
            ("trunk.0.weight", (TRUNK, TRUNK_IN)), ("trunk.0.bias", (TRUNK,))]
    MID = [("trunk.2.weight", (TRUNK, TRUNK)), ("trunk.2.bias", (TRUNK,))]
    TAIL = [("trunk.4.weight", (1, TRUNK)), ("trunk.4.bias", (1,))]

    # two-sided: no key silently dropped, none silently missing
    consumed = ({"emb.%s.weight" % t for t in TABLES}
                | {n for n, _ in MAIN + MID + TAIL})
    missing, extra = consumed - set(sd), set(sd) - consumed
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
        w = sd["emb.%s.weight" % t].numpy()
        assert w.dtype == np.float32 and w.shape[1] == EMB_DIMS[t] \
            and w.shape[0] >= rows[t], (t, w.shape)
        assert not w[0].any(), "%s row 0 (padding) not zero" % t
        tensors.append(np.ascontiguousarray(w[:rows[t]]))

    def take(names):
        out_ = []
        for name, shape in names:
            w = sd[name].numpy()
            assert w.dtype == np.float32 and w.shape == shape, (name, w.shape)
            out_.append(np.ascontiguousarray(w))
        return out_

    tensors += take(MAIN)
    tail_tensors = take(MID + TAIL)

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
    meta = {"version": VERSION, "mon_in": MON_IN, "trunk_in": TRUNK_IN,
            "mon_hid": MON_HID, "trunk": TRUNK, "bench_mult": 1,
            "bench_sort": bench_sort, "pp_true_max": pp_true_max,
            "attn_d": 0, "policy_slots": 0, "n_blocks": 0,
            "bytes": len(data), "sha256": sha,
            "from_ckpt": os.path.basename(ckpt_path),
            "ckpt_cap": cfg.get("cap"), "ckpt_seed": cfg.get("seed"),
            "ckpt_brier_all": (cfg.get("brier") or {}).get("all")}
    print(json.dumps(meta, indent=1))
    json.dump(meta, open(os.path.splitext(out)[0] + ".meta.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
