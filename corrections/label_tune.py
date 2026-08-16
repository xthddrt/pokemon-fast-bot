"""LABEL MINI-RETRAIN — the hammer as a warm-started fine-tune on corrected
labels (Sally 2026-08-16). One objective everywhere: BCE toward measured
truth. No anchor-to-teacher, no force balance, no escalation.

  - corpus rows keep their labels; the measured pairs' rows get NEW labels
    (40-playout truths, both seatings: mirror labels for swap-aug too)
  - ledger states (incl. non-corpus mining-era rows) are oversampled ~50/50
    against the uniform corpus so corrections move fast
  - warm start from the champion ckpt, cosine anneal to zero, holdout
    early-stop; bands are checked AFTER as an exit exam, not driven during

    python label_tune.py --ckpt v8c_s1.pt --steps 5000 --tag lt1
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")
LEDGER = os.path.join(HERE, "value_ledger_v8c2.jsonl")
CORPUS_ENC = os.path.join(HERE, "_hammer_work", "corpus_enc")
LEDGER_ENC = os.path.join(HERE, "_hammer_work", "enc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="lt1")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-emb", type=float, default=3.3e-6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--mix", type=float, default=0.5,
                    help="fraction of each batch drawn from ledger rows")
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()

    sys.path.insert(0, LAB)
    os.environ["VT_ENC"] = CORPUS_ENC
    import labenv  # noqa: F401
    import numpy as np
    import torch
    import vt_lib
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    net = vt_lib.build_net("old", (128, 256), 1, add="setup")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["sd"] if "sd" in ck else ck["model"])
    net.to(dev)

    # ---- corpus on GPU, labels patched ----
    arm = vt_lib.Arm("old", add="setup")
    n = np.load(os.path.join(CORPUS_ENC, "old_a1_f.npy"), mmap_mode="r").shape[0]
    gpu = {}
    for k in vt_lib.OLD_KEYS:
        gpu[k] = torch.from_numpy(np.asarray(arm.a[k])).to(
            dev, torch.int64 if "ids" in k else torch.float32)
    gpu_am = torch.from_numpy(np.asarray(arm.am[:, :, arm.mcol], np.float32)).to(dev)

    def cbatch(sel):
        s = torch.as_tensor(np.asarray(sel), dtype=torch.long, device=dev)
        b = {k: gpu[k][s] for k in vt_lib.OLD_KEYS}
        b["am"] = gpu_am[s]
        return b

    meta = np.load(os.path.join(CORPUS_ENC, "meta.npz"), allow_pickle=False)
    lab = meta["label_p"].astype(np.float32).copy()
    lab_mir = 1.0 - lab  # swap-aug default; measured mirrors override below
    entries = [json.loads(l) for l in open(LEDGER)]
    n_patch = 0
    for e in entries:
        ri = e.get("row_i")
        if ri is None:
            continue
        if e["id"].endswith("-mir"):
            lab_mir[ri] = e["target"]
        else:
            lab[ri] = e["target"]
        n_patch += 1
    print(f"corpus rows: {n}, label patches applied: {n_patch}", flush=True)
    y = torch.from_numpy(lab).to(dev)
    y_m = torch.from_numpy(lab_mir).to(dev)

    hold = np.load(os.path.join(CORPUS_ENC, "holdout_i.npy")).astype(np.int64)
    mask = np.ones(n, bool); mask[hold] = False
    pool = np.flatnonzero(mask)
    y_h = torch.from_numpy(lab[hold]).to(dev)

    # ---- ledger states (all 13,390, incl. the 88 non-corpus) on GPU ----
    os.environ["VT_ENC"] = LEDGER_ENC
    import importlib
    importlib.reload(vt_lib)
    larm = vt_lib.Arm("old", add="setup")
    nl = np.load(os.path.join(LEDGER_ENC, "old_a1_f.npy"), mmap_mode="r").shape[0]
    lg = {}
    for k in vt_lib.OLD_KEYS:
        lg[k] = torch.from_numpy(np.asarray(larm.a[k])).to(
            dev, torch.int64 if "ids" in k else torch.float32)
    lg_am = torch.from_numpy(np.asarray(larm.am[:, :, larm.mcol], np.float32)).to(dev)
    tgt_l = torch.tensor([float(e["target"]) for e in entries for _ in e["states"]],
                         dtype=torch.float32, device=dev)
    assert tgt_l.shape[0] == nl, (tgt_l.shape[0], nl)

    def lbatch(sel):
        s = torch.as_tensor(np.asarray(sel), dtype=torch.long, device=dev)
        b = {k: lg[k][s] for k in vt_lib.OLD_KEYS}
        b["am"] = lg_am[s]
        return b

    # ---- warm tune ----
    emb, dense = [], []
    for nm, p in net.named_parameters():
        (emb if "emb" in nm else dense).append(p)
    opt = torch.optim.AdamW(
        [{"params": dense, "lr": a.lr, "weight_decay": 0.0},
         {"params": emb, "lr": a.lr_emb, "weight_decay": 0.0}])
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(11)
    n_led = int(a.batch * a.mix)
    n_cor = a.batch - n_led
    best = (1e9, None, -1)
    t0 = time.time()
    import math
    for step in range(1, a.steps + 1):
        mlt = 0.5 * (1 + math.cos(math.pi * step / a.steps))
        opt.param_groups[0]["lr"] = a.lr * mlt
        opt.param_groups[1]["lr"] = a.lr_emb * mlt
        ci = pool[rng.integers(0, len(pool), n_cor)]
        li = rng.integers(0, nl, n_led)
        cb = cbatch(ci)
        # swap-aug on the corpus half (ledger rows carry explicit mirrors)
        sm = torch.from_numpy(rng.random(n_cor) < 0.5).to(dev)
        vt_lib.swap_rows_(cb, sm)
        yc = torch.where(sm, y_m[torch.as_tensor(ci, dtype=torch.long, device=dev)],
                         y[torch.as_tensor(ci, dtype=torch.long, device=dev)])
        net.train()
        z_c = net(cb)
        z_l = net(lbatch(li))
        loss = 0.5 * lossf(z_c, yc) + 0.5 * lossf(z_l, tgt_l[
            torch.as_tensor(li, dtype=torch.long, device=dev)])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                ph = []
                for i in range(0, len(hold), 8192):
                    ph.append(torch.sigmoid(net(cbatch(hold[i:i+8192]))))
                ph = torch.cat(ph)
                hb = float(((ph - y_h) ** 2).mean())
            print(f"  step {step}: loss {float(loss):.4f} holdout {hb:.6f} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if hb < best[0]:
                best = (hb, {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()}, step)
    net.load_state_dict(best[1])
    print(f"best holdout {best[0]:.6f} @ step {best[2]}", flush=True)

    net.to("cpu")
    out_pt = os.path.join(os.path.dirname(a.ckpt), f"v8c_{a.tag}.pt")
    torch.save({"sd": net.state_dict(), "cfg": ck.get("cfg", {})}, out_pt)
    import subprocess
    out_bin = out_pt.replace(".pt", ".bin")
    subprocess.run([sys.executable, os.path.join(LAB, "export_v8.py"), out_pt, out_bin],
                   cwd=LAB, check=True, capture_output=True, text=True)
    print(f"exported {out_bin}", flush=True)


if __name__ == "__main__":
    main()
