"""Fast .pt -> .bin export for mid-hammer flip tests: runs the REAL
valuenet/export_weights.py writer but stubs out the constants derivation
(the ~minutes MCTS-trajectory step) with the BASE net's sidecar constants.

The .bin bytes are IDENTICAL to a full export of the same checkpoint (the
sidecar constants live in a separate json, never in the bin), so a candidate
that passes the flip test on this bin ships those exact bytes. The sidecar it
writes is marked "stale_fast_export": true and is replaced by ship.py's real
derive.

  .venv python _fast_export.py <ckpt.pt> <out.bin> <base_constants.json>
"""

import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
VALUENET = os.path.join(os.path.dirname(HERE), "valuenet")


def main():
    ckpt, out, consts_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # encoder flags must match the checkpoint BEFORE export_weights imports it
    sys.path.insert(0, HERE)
    import common

    ck = common.load_ckpt(ckpt)
    common.apply_env(common.ckpt_env(ck))
    del sys.modules["common"]

    base = json.load(open(consts_path))
    consts = {k: base[k] for k in
              ("games", "iters", "lnN", "delta", "sigma", "k", "tau", "r")
              if k in base}
    consts.update({k: v for k, v in base.items() if k.startswith("PE_TUNE_")})
    consts["stale_fast_export"] = True
    consts["stale_constants_from"] = os.path.basename(consts_path)

    fake = types.ModuleType("derive_constants")
    fake.derive = lambda shard, ckpt_path: dict(consts)
    sys.modules["derive_constants"] = fake

    # export_weights parses sys.argv at IMPORT time. --shard only needs an
    # existing file (the stub never reads it).
    sys.argv = ["export_weights.py", ckpt, out, "--shard", consts_path]
    sys.path.insert(0, VALUENET)
    import export_weights

    return export_weights.main()


if __name__ == "__main__":
    sys.exit(main())
