"""V9 TRAINING — 3-trunk phase net on the pooled corpus (Sally 2026-08-17).

Data: enc_plc12 (2M, v8-era labels) + fresh v9 corpus (2.0M, 1-per-bucket
decorrelated, n=8/8/10 labels by v8c_s1). Phase-balanced batches (equal thirds
by hp-mass band across BOTH corpora), swap-aug 50%, per-trunk weight decay
with 2x on the late trunk (the pp1 lesson). From-scratch seeds; selection =
paired Brier on the FRESH game-level holdout; bench_v1 reported for
continuity.

    python v9_train.py --seed 0 --steps 60000 --tag v9s0
"""
import argparse
import copy
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")
OLD_ENC = "/root/v9/work/enc_plc12"
FRESH_ENC = "/root/v9/work/enc_fresh"
HOLD_ENC = "/root/v9/work/enc_fresh_holdout"
BENCH = os.path.join(HERE, "bench_v1.jsonl")
BENCH_ENC = "/root/v9/work/enc_bench"

sys.path.insert(0, HERE)
from phase_probe import PhaseNet, hat_weights, batch_phase  # noqa: E402


def load_enc(vt_lib, np, torch, dev, enc_dir):
    os.environ["VT_ENC"] = enc_dir
    import importlib
    importlib.reload(vt_lib)
    arm = vt_lib.Arm("old", add="setup")
    n = np.load(os.path.join(enc_dir, "old_a1_f.npy"), mmap_mode="r").shape[0]
    g = {}
    for k in vt_lib.OLD_KEYS:
        g[k] = torch.from_numpy(np.asarray(arm.a[k])).to(
            dev, torch.int64 if "ids" in k else torch.float32)
    g["am"] = torch.from_numpy(
        np.asarray(arm.am[:, :, arm.mcol], np.float32)).to(dev)
    return g, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v9s0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=0.002)
    ap.add_argument("--wd-late-mult", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--arch", choices=["phase3", "single"], default="phase3",
                    help="phase3 = 3-trunk hp-mass blend; single = one plain "
                         "trunk, IDENTICAL data/batching/schedule, the control "
                         "for the 3-net-vs-1-net Brier comparison")
    ap.add_argument("--rows-per-corpus", type=int, default=0,
                    help="data-scaling probe: subsample EACH corpus to this "
                         "many rows (preserves the 50/50 old/fresh mix); "
                         "everything else identical, so Brier-vs-rows is the "
                         "pure data-quantity curve")
    a = ap.parse_args()

    sys.path.insert(0, LAB)
    os.environ["VT_ENC"] = OLD_ENC
    import labenv  # noqa: F401
    import numpy as np
    import torch
    import vt_lib
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)

    # fresh-init base net for this seed, then the 3-trunk wrapper
    base = vt_lib.build_net("old", (128, 256), a.seed + 1, add="setup")
    if a.arch == "phase3":
        net = PhaseNet(base).to(dev)
        net.phase_var = "hp"
    else:
        net = base.to(dev)

    corp = []
    for enc_dir, label_src in ((OLD_ENC, "meta"), (FRESH_ENC, "jsonl")):
        g, n = load_enc(vt_lib, np, torch, dev, enc_dir)
        if label_src == "meta":
            lab = np.load(os.path.join(enc_dir, "meta.npz"))["label_p"].astype(np.float32)
        else:
            import gzip
            lab = np.array([json.loads(l)["y"] for l in
                            gzip.open("/root/v9/data/fresh_train.jsonl.gz", "rt")],
                           np.float32)
        assert len(lab) == n, (enc_dir, len(lab), n)
        y = torch.from_numpy(lab).to(dev)
        if a.rows_per_corpus and a.rows_per_corpus < n:
            sub = np.sort(np.random.default_rng(
                20260817 + len(corp)).choice(n, a.rows_per_corpus, False))
            st = torch.from_numpy(sub).to(dev)
            g = {k: g[k][st] for k in g}
            y = y[st]
            n = a.rows_per_corpus
        corp.append({"g": g, "n": n, "y": y})
        print(f"corpus {enc_dir}: {n} rows", flush=True)

    def cbatch(ci, oi):
        """ci: corpus index tensor rows... build a mixed batch from (corpus, row)."""
        b = {}
        for k in list(corp[0]["g"].keys()):
            b[k] = torch.cat([corp[c]["g"][k][idx] for c, idx in oi], 0)
        return b

    # phase bands per corpus row (computed once, on gpu, in slabs)
    def phases_of(c):
        out = []
        for i in range(0, corp[c]["n"], 65536):
            idx = torch.arange(i, min(i + 65536, corp[c]["n"]), device=dev)
            b = {k: corp[c]["g"][k][idx] for k in corp[c]["g"]}
            out.append(batch_phase(b))
        return torch.cat(out)
    bands = []
    for c in range(2):
        p = phases_of(c)
        bands.append(torch.bucketize(p, torch.tensor([1 / 3, 2 / 3], device=dev)))
        print(f"corpus {c} band mix:", [int((bands[c] == k).sum()) for k in range(3)],
              flush=True)
    pools = []  # per band: (corpus, rows) index tensors
    for k in range(3):
        pools.append([(c, torch.nonzero(bands[c] == k).squeeze(1)) for c in range(2)])

    # holdout (fresh, game-level) + old holdout for reference
    hg, hn = load_enc(vt_lib, np, torch, dev, HOLD_ENC)
    import gzip
    hy = torch.tensor([json.loads(l)["y"] for l in
                       gzip.open("/root/v9/data/fresh_holdout.jsonl.gz", "rt")],
                      dtype=torch.float32, device=dev)
    assert hn == len(hy)

    def brier(model, g, n, ys, slices=False):
        with torch.no_grad():
            ps, phs = [], []
            for i in range(0, n, 8192):
                idx = torch.arange(i, min(i + 8192, n), device=dev)
                b = {k: g[k][idx] for k in g}
                ps.append(torch.sigmoid(model(b)))
                phs.append(batch_phase(b))
        e2 = (torch.cat(ps) - ys) ** 2
        if not slices:
            return float(e2.mean())
        ph = torch.cat(phs)
        out = {"all": float(e2.mean())}
        for nm, lo, hi in (("early", 0, 1/3), ("mid", 1/3, 2/3), ("late", 2/3, 1.01)):
            m = (ph >= lo) & (ph < hi)
            out[nm] = (round(float(e2[m].mean()), 6), int(m.sum()))
        return out

    # optimizer: shared params standard; per-trunk groups so late gets wd*mult
    emb, shared, trunks = [], [], [[], [], []]
    for nm, p in net.named_parameters():
        if nm.startswith("trunks."):
            trunks[int(nm.split(".")[1])].append(p)
        elif "emb" in nm:
            emb.append(p)
        else:
            shared.append(p)
    groups = [{"params": shared, "lr": a.lr, "weight_decay": a.wd},
              {"params": emb, "lr": a.lr, "weight_decay": 0.0}]
    if a.arch == "phase3":
        groups += [
            {"params": trunks[0], "lr": a.lr, "weight_decay": a.wd},
            {"params": trunks[1], "lr": a.lr, "weight_decay": a.wd},
            {"params": trunks[2], "lr": a.lr, "weight_decay": a.wd * a.wd_late_mult}]
    opt = torch.optim.AdamW(groups)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(a.seed * 7919 + 13)
    # FULL-COVERAGE EPOCHS (Sally 2026-08-17): no sampling. One epoch = every
    # row of BOTH corpora exactly once in its original seating AND exactly once
    # mirrored, shuffled. Virtual index: [0, 2*(n0+n1)); high bit = mirror.
    n0, n1 = corp[0]["n"], corp[1]["n"]
    ntot = n0 + n1
    steps_per_epoch = (2 * ntot) // a.batch
    max_steps = a.steps
    print(f"epoch = {2*ntot} samples (corpus+mirror) = {steps_per_epoch} steps; "
          f"max {max_steps} steps ≈ {max_steps/steps_per_epoch:.1f} epochs",
          flush=True)
    best = (1e9, None, -1)
    t0 = time.time()
    perm = None
    for step in range(1, max_steps + 1):
        mlt = step / a.warmup if step <= a.warmup else 0.5 * (
            1 + math.cos(math.pi * (step - a.warmup) / (max_steps - a.warmup)))
        for grp in opt.param_groups:
            grp["lr"] = a.lr * mlt
        pos = ((step - 1) % steps_per_epoch) * a.batch
        if pos == 0:
            perm = torch.from_numpy(
                rng.permutation(2 * ntot)).to(dev)
        v = perm[pos:pos + a.batch]
        mirror = v >= ntot
        base_i = torch.where(mirror, v - ntot, v)
        in_c1 = base_i >= n0
        oi = [(0, base_i[~in_c1]), (1, base_i[in_c1] - n0)]
        b = cbatch(None, oi)
        yb = torch.cat([corp[0]["y"][oi[0][1]], corp[1]["y"][oi[1][1]]])
        sm = torch.cat([mirror[~in_c1], mirror[in_c1]])
        vt_lib.swap_rows_(b, sm)
        yb = torch.where(sm, 1.0 - yb, yb)
        net.train()
        loss = lossf(net(b), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == max_steps:
            net.eval()
            hb = brier(net, hg, hn, hy)
            print(f"  step {step}: loss {float(loss):.4f} "
                  f"fresh-holdout {hb:.6f} {time.time()-t0:.0f}s", flush=True)
            if hb < best[0]:
                sd = net.state_dict()
                if a.arch == "phase3":
                    keep = {k: {kk: vv.detach().cpu().clone()
                                for kk, vv in v.items()} for k, v in sd.items()}
                else:
                    keep = {k: v.detach().cpu().clone() for k, v in sd.items()}
                best = (hb, keep, step)
    net.load_state_dict(best[1])
    net.eval()
    print(f"BEST fresh-holdout {best[0]:.6f} @ step {best[2]}", flush=True)
    print(f"SLICES {a.tag}:", brier(net, hg, hn, hy, slices=True), flush=True)

    # bench_v1 through the identical python path
    if os.path.isdir(BENCH_ENC):
        bg, bn = load_enc(vt_lib, np, torch, dev, BENCH_ENC)
        rows = [json.loads(l) for l in open(BENCH)]
        bt = torch.tensor([r["truth"] for r in rows], dtype=torch.float32,
                          device=dev)
        print(f"BENCH_v1 {a.tag}: {brier(net, bg, bn, bt):.4f}", flush=True)

    out = f"/root/v9/v9_{a.tag}.pt"
    torch.save({"sd": net.state_dict(),
                "cfg": {"arch": a.arch, "blend": "hat(p) knots 0/0.5/1",
                        "p": "1-sum(hp_frac)/12", "seed": a.seed}}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
