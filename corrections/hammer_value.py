"""VALUE HAMMER — conform the net's eval on ruled positions to playout truth.

    foul-play/.venv/bin/python corrections/hammer_value.py \
        [--net valuenet/nets_v8b/v8b_s1.pt] [--conform 0.2] [--lr 3e-6] \
        [--cap 30000] [--tag h1] [--anchor 10310] [--anchor-w 100] \
        [--anchor-bs 512]

Spec + full process: corrections/VALUE_HAMMER.md (defaults here = the
2026-08-15 sweep winner documented there).

Sally's loop (2026-08-15, the value variant of HAMMER_SPEC Part 2): when a
loss exposes a position the evaluator misprices (e.g. 2665399837 turn 18:
eval 0.39, playout truth 0.096 over 104 playouts), the position's world
states + measured target are appended to corrections/value_ledger.jsonl and
the CURRENT champion checkpoint is fine-tuned on ALL ledger entries
cumulatively until the net's eval of every ruled state reaches its ruled
band (<= --conform for losing positions). Overfitting is accepted by design:
each new loss's examples ride along forever, and the growing ledger dilutes
per-example overfit over time.

Ledger row: {id, game, decision, target, conform, states: [state strings],
             n_playouts, note, ts}

Pipeline per run:
  1. encode ALL ledger states via evallab/enc_adopted.py (arm A + setup)
  2. fine-tune the ckpt (BCE toward each entry's target) until the NET-level
     eval of every state is inside its band, or --cap steps
  3. export via evallab/export_v8.py -> <stem>_<tag>.bin (+ sidecar copied
     from the source net: the search constants are a property of the reward
     geometry, which a small value correction does not move materially)
  2b. ANCHOR (anti-collateral): the ruled BCE alone shifts the net's GLOBAL
     output bias (measured: 8 same-target rows moved mean |dlogit| 1.29 over
     10,310 parity states in 3 steps — that is not local overfit, it is
     scale destruction). A self-distillation term pins the net to the SOURCE
     net's own logits on --anchor real states (from the parity corpus), so
     the exception is carved locally while everything else stays put.
  4. gates, via the REAL engine (leaf_prof logits):
       a. every ruled state's engine-side eval inside its band
       b. collateral report: |delta logit| mean/p99 over the 10,310 parity
          states vs the source net (reported, not gating — Sally's call)
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")
PY = os.path.join(ROOT, "foul-play", ".venv", "bin", "python")
LEDGER = os.path.join(HERE, "value_ledger.jsonl")
LEAF_PROF = os.path.join(ROOT, "poke-engine", "target", "release", "leaf_prof")
PARITY = "/private/tmp/claude-501/-Users-sallyliu-pokemon-fast-bot/fcc1b52c-d38e-40bd-b9db-95a07c1d94b7/scratchpad/parity_states.txt"


def engine_logits(bin_path, states_file):
    env = dict(os.environ, PE_NN_WEIGHTS=bin_path)
    out = subprocess.run([LEAF_PROF, "logits", states_file], env=env,
                         capture_output=True, text=True, check=True).stdout
    return [float(l.split("\t")[1]) for l in out.splitlines() if "\t" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(ROOT, "valuenet/nets_v8b/v8b_s1.pt"))
    ap.add_argument("--conform", type=float, default=0.2)
    # defaults = the 2026-08-15 minibatch sweep winner (VALUE_HAMMER.md
    # §2/§4): all parity states as anchors, heavy pin, minibatch 512,
    # lr 3e-6 — 70s end-to-end at drift equal to the best measured. If a
    # future multi-ruling hammer misses the drift bars, drop lr to 1e-6.
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--tag", default="h1")
    ap.add_argument("--anchor", type=int, default=10310)
    ap.add_argument("--anchor-w", type=float, default=100.0)
    # anchor minibatch per step (cycled in shuffled epochs over the full
    # anchor set — same constraint in expectation, ~10x less compute per
    # step; the end-of-run gates still check every state). 0 = full batch.
    ap.add_argument("--anchor-bs", type=int, default=512)
    a = ap.parse_args()

    entries = [json.loads(l) for l in open(LEDGER)]
    if not entries:
        raise SystemExit("empty ledger")
    states, targets, bands = [], [], []
    for e in entries:
        for s in e["states"]:
            states.append(s)
            targets.append(float(e["target"]))
            bands.append(float(e.get("conform", a.conform)))
    print(f"{len(entries)} ruling(s), {len(states)} states")

    # 1. encode
    work = os.path.join(HERE, "_hammer_work")
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "states.jsonl")
    with open(src, "w") as f:
        for st, t in zip(states, targets):
            f.write(json.dumps({"s": st, "y": t}) + "\n")
    enc = os.path.join(work, "enc")
    subprocess.run([PY, os.path.join(LAB, "enc_adopted.py"), "encode", src, enc],
                   cwd=LAB, check=True, capture_output=True, text=True)

    # anchor states: encoded once per anchor count, reused by future runs
    anchor_enc = os.path.join(work, f"anchor_enc_{a.anchor}")
    if a.anchor and not os.path.isdir(anchor_enc):
        import random as _r
        _r.seed(7)
        alines = [l.strip() for l in open(PARITY) if l.strip()]
        asample = _r.sample(alines, min(a.anchor, len(alines)))
        asrc = os.path.join(work, "anchor.jsonl")
        with open(asrc, "w") as f:
            for st in asample:
                f.write(json.dumps({"s": st, "y": 0.0}) + "\n")
        subprocess.run([PY, os.path.join(LAB, "enc_adopted.py"), "encode", asrc, anchor_enc],
                       cwd=LAB, check=True, capture_output=True, text=True)

    # 2. fine-tune
    sys.path.insert(0, LAB)
    os.environ["VT_ENC"] = enc
    import labenv  # noqa: F401
    import numpy as np
    import torch
    import vt_lib
    torch.set_num_threads(4)
    net = vt_lib.build_net("old", (128, 256), 1, add="setup")
    ck = torch.load(a.net, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["sd"] if "sd" in ck else ck["model"])
    arm = vt_lib.Arm("old", add="setup")
    idx = np.arange(len(states))
    batch = arm.batch(idx)
    tgt = torch.tensor(targets, dtype=torch.float32)
    band = torch.tensor(bands, dtype=torch.float32)

    aarm = None
    if a.anchor:
        os.environ["VT_ENC"] = anchor_enc
        import importlib
        importlib.reload(vt_lib)
        aarm = vt_lib.Arm("old", add="setup")
        n_anchor = np.load(os.path.join(anchor_enc, "old_a1_f.npy"), mmap_mode="r").shape[0]
        with torch.no_grad():
            net.eval()
            anchor_ref = torch.cat([
                net(aarm.batch(np.arange(i, min(i + 4096, n_anchor)))).detach()
                for i in range(0, n_anchor, 4096)])
        bs = a.anchor_bs if a.anchor_bs else n_anchor
        rng = np.random.default_rng(7)
        perm, pos = rng.permutation(n_anchor), 0

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    step = 0
    for step in range(1, a.cap + 1):
        opt.zero_grad()
        loss = lossf(net(batch), tgt)
        if aarm is not None:
            if pos + bs > n_anchor:
                perm, pos = rng.permutation(n_anchor), 0
            sel = perm[pos:pos + bs]
            pos += bs
            loss = loss + a.anchor_w * torch.nn.functional.mse_loss(
                net(aarm.batch(sel)), anchor_ref[torch.as_tensor(sel, dtype=torch.long)])
        loss.backward()
        opt.step()
        with torch.no_grad():
            evals = torch.sigmoid(net(batch))
        if bool((evals <= band).all()):
            break
    with torch.no_grad():
        net.eval()
        drift = (torch.cat([
            net(aarm.batch(np.arange(i, min(i + 4096, n_anchor))))
            for i in range(0, n_anchor, 4096)]) - anchor_ref).abs() \
            if aarm is not None else torch.zeros(1)
    print(f"fine-tune: {step} steps, ruled evals: "
          f"{[round(float(v), 3) for v in evals]}, "
          f"anchor drift mean {float(drift.mean()):.4f} max {float(drift.max()):.4f}")

    # 3. save + export + sidecar
    stem = os.path.splitext(a.net)[0].rsplit("_s", 1)[0]
    out_pt = f"{stem}_{a.tag}.pt"
    out_bin = f"{stem}_{a.tag}.bin"
    torch.save({"sd": net.state_dict(), "cfg": ck.get("cfg", {})}, out_pt)
    subprocess.run([PY, os.path.join(LAB, "export_v8.py"), out_pt, out_bin],
                   cwd=LAB, check=True, capture_output=True, text=True)
    side_src = os.path.splitext(a.net)[0] + ".constants.json"
    side_out = os.path.splitext(out_bin)[0] + ".constants.json"
    sc = json.load(open(side_src))
    sc["_hammered"] = {"tag": a.tag, "rulings": [e["id"] for e in entries],
                       "ts": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    json.dump(sc, open(side_out, "w"), indent=1)

    # 4. gates through the real engine
    import math
    ruled_file = os.path.join(work, "ruled.txt")
    with open(ruled_file, "w") as f:
        for s in states:
            f.write(s + "\n")
    new_ruled = engine_logits(out_bin, ruled_file)
    ruled_evals = [1 / (1 + math.exp(-x)) for x in new_ruled]
    # +0.002: f32 export rounding can land ~0.0005 above a band the fp32
    # net exactly met (observed in the anchor sweep)
    ok = all(v <= b + 0.002 for v, b in zip(ruled_evals, bands))
    print(f"ENGINE GATE (ruled states): {[round(v, 3) for v in ruled_evals]} "
          f"-> {'PASS' if ok else 'FAIL'} (bands {sorted(set(bands))})")
    if os.path.isfile(PARITY):
        # source-net parity logits never change for a given source bin: cache
        src_bin = a.net.replace(".pt", ".bin")
        lcache = os.path.join(work, f"parity_logits_{os.path.basename(src_bin)}.json")
        if os.path.isfile(lcache):
            old = json.load(open(lcache))
        else:
            old = engine_logits(src_bin, PARITY)
            json.dump(old, open(lcache, "w"))
        new = engine_logits(out_bin, PARITY)
        d = sorted(abs(x - y) for x, y in zip(old, new))
        print(f"COLLATERAL over {len(d)} parity states: "
              f"mean |dlogit| {sum(d)/len(d):.4f}, p99 {d[int(0.99*len(d))]:.4f}, "
              f"max {d[-1]:.4f}")
    print("shipped candidate:", out_bin)
    return out_bin if ok else None


if __name__ == "__main__":
    main()
