"""PHASE-SPECIALIZATION PROBE — does per-phase trunk capacity lower Brier?

Architecture decision (Sally 2026-08-17): shared embeddings + shared per-mon
MLP, trunk replicated x3 (early/mid/late), outputs blended at the LOGIT level
with piecewise-linear hat weights over a continuous phase variable

    p = 1 - (sum of hp_frac over all 12 mons) / 12          (0 start -> ~1 end)

Boundary calibration is solved BY CONSTRUCTION: the blend is part of the
trained function (every sample trains the 1-2 trunks its p activates, jointly,
end to end), so there are no hard switches and no separately-calibrated heads
anywhere. The engine sees one continuous function of the state.

Probe protocol: warm-start everything from the champion base ckpt (all three
trunks = its trunk), train on the full corpus + swap-aug with the standard BCE
objective, holdout early-stop. Verdict = python-side Brier (holdout + frozen
bench), overall and per phase slice, against v8c_s1 evaluated through the
IDENTICAL path. No engine export: that work is only justified if this wins.

    python phase_probe.py --ckpt v8c_s1.pt --tag pp1 --steps 30000
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")
CORPUS_ENC = os.path.join(HERE, "_hammer_work", "corpus_enc")
BENCH = os.path.join(HERE, "bench_v1.jsonl")
BENCH_ENC = os.path.join(HERE, "_hammer_work", "bench_enc")


def hat_weights_alive(b):
    """Blend weights from TOTAL ALIVE MONS, knots at 12/8/4 (Sally 2026-08-17).
    Early peaks at 12 alive, mid at 8, late owns everything at or below 4.
    Piecewise linear, sums to 1 everywhere; only adjacent trunks are active."""
    import torch
    a = ((b["a1_f"][:, 0] > 0).float() + (b["a2_f"][:, 0] > 0).float()
         + (b["b1_f"][:, :, 0] > 0).float().sum(1)
         + (b["b2_f"][:, :, 0] > 0).float().sum(1))
    w_e = ((a - 8) / 4).clamp(0, 1)
    w_l = ((8 - a) / 4).clamp(0, 1)
    w_m = (1 - w_e - w_l).clamp(min=0)
    return torch.stack([w_e, w_m, w_l])


def hat_weights(p):
    """(B,) -> (3,B) piecewise-linear blend: knots at p=0, 0.5, 1."""
    import torch
    w_e = (1 - 2 * p).clamp(min=0)
    w_m = 1 - (2 * p - 1).abs()
    w_m = w_m.clamp(min=0)
    w_l = (2 * p - 1).clamp(min=0)
    return torch.stack([w_e, w_m, w_l])


def batch_phase(b):
    hp = (b["a1_f"][:, 0] + b["a2_f"][:, 0]
          + b["b1_f"][:, :, 0].sum(1) + b["b2_f"][:, :, 0].sum(1))
    return (1 - hp / 12.0).clamp(0, 1)


class PhaseNet:
    """Wrapper: shared embed/mon from an ArmAPlusNet, three trunk copies."""

    phase_var = "hp"

    def __init__(self, base_net):
        import torch.nn as nn
        self.net = base_net                      # embeddings + mon MLP (shared)
        self.trunks = nn.ModuleList(
            [copy.deepcopy(base_net.base.trunk) for _ in range(3)])

    def to(self, dev):
        self.net.to(dev); self.trunks.to(dev); return self

    def features(self, b):
        import torch
        n, B = self.net, self.net.base
        am = b.get("am")

        def mon(ids, f, sl):
            return n.embed_mon(ids, f if am is None else
                               torch.cat([f, am[:, sl]], dim=-1))
        return torch.cat([
            mon(b["a1_ids"], b["a1_f"], 0),
            B.pool(mon(b["b1_ids"], b["b1_f"], slice(1, 6))),
            mon(b["a2_ids"], b["a2_f"], 6),
            B.pool(mon(b["b2_ids"], b["b2_f"], slice(7, 12))),
            b["sf1"], b["sf2"], b["g"]], dim=-1)

    def __call__(self, b):
        import torch
        x = self.features(b)
        w = (hat_weights_alive(b) if self.phase_var == "alive"
             else hat_weights(batch_phase(b)))                # (3,B)
        z = torch.stack([t(x).squeeze(-1) for t in self.trunks])
        return (w * z).sum(0)

    def parameters(self):
        seen, out = set(), []
        for m in (self.net, self.trunks):
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); out.append(p)
        return out

    def named_parameters(self):
        for nm, p in self.net.named_parameters():
            yield "shared." + nm, p
        for nm, p in self.trunks.named_parameters():
            yield "trunks." + nm, p

    def state_dict(self):
        return {"shared": self.net.state_dict(),
                "trunks": self.trunks.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["shared"])
        self.trunks.load_state_dict(sd["trunks"])

    def train(self): self.net.train(); self.trunks.train()
    def eval(self): self.net.eval(); self.trunks.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="pp1")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-emb", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--wd", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--phase-var", choices=["hp", "alive"], default="hp",
                    help="blend variable: hp mass p, or alive count w/ knots "
                         "12/8/4. Report slices stay hp-based in BOTH modes "
                         "so the A/B compares like with like.")
    a = ap.parse_args()

    sys.path.insert(0, LAB)
    os.environ["VT_ENC"] = CORPUS_ENC
    import labenv  # noqa: F401
    import numpy as np
    import torch
    import vt_lib
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    base = vt_lib.build_net("old", (128, 256), 1, add="setup")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    base.load_state_dict(ck["sd"] if "sd" in ck else ck["model"])
    control = copy.deepcopy(base).to(dev)          # untouched v8c_s1
    net = PhaseNet(base).to(dev)
    net.phase_var = a.phase_var
    print(f"phase variable: {a.phase_var}", flush=True)

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
    lab = meta["label_p"].astype(np.float32)
    y = torch.from_numpy(lab).to(dev)
    y_m = 1.0 - y
    hold = np.load(os.path.join(CORPUS_ENC, "holdout_i.npy")).astype(np.int64)
    mask = np.ones(n, bool); mask[hold] = False
    pool = np.flatnonzero(mask)
    y_h = y[torch.as_tensor(hold, dtype=torch.long, device=dev)]

    def brier(model, idx, ys):
        with torch.no_grad():
            ps, phs = [], []
            for i in range(0, len(idx), 8192):
                b = cbatch(idx[i:i + 8192])
                ps.append(torch.sigmoid(model(b)))
                phs.append(batch_phase(b))
            ps, phs = torch.cat(ps), torch.cat(phs)
        e2 = (ps - ys) ** 2
        out = {"all": float(e2.mean())}
        for nm, lo, hi in (("early", 0, 1/3), ("mid", 1/3, 2/3), ("late", 2/3, 1.01)):
            m = (phs >= lo) & (phs < hi)
            out[nm] = (float(e2[m].mean()), int(m.sum()))
        return out

    print("baseline (v8c_s1, identical path):", flush=True)
    b0 = brier(control, hold, y_h)
    print(f"  holdout {b0['all']:.6f} | early {b0['early']} mid {b0['mid']} "
          f"late {b0['late']}", flush=True)

    emb, rest = [], []
    for nm, p in net.named_parameters():
        (emb if "emb" in nm else rest).append(p)
    opt = torch.optim.AdamW(
        [{"params": rest, "lr": a.lr, "weight_decay": a.wd},
         {"params": emb, "lr": a.lr_emb, "weight_decay": 0.0}])
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(13)
    best = (1e9, None, -1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        mlt = step / a.warmup if step <= a.warmup else 0.5 * (
            1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        opt.param_groups[0]["lr"] = a.lr * mlt
        opt.param_groups[1]["lr"] = a.lr_emb * mlt
        ci = pool[rng.integers(0, len(pool), a.batch)]
        b = cbatch(ci)
        sm = torch.from_numpy(rng.random(a.batch) < 0.5).to(dev)
        vt_lib.swap_rows_(b, sm)
        ti = torch.as_tensor(ci, dtype=torch.long, device=dev)
        yc = torch.where(sm, y_m[ti], y[ti])
        net.train()
        loss = lossf(net(b), yc)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            net.eval()
            hb = brier(net, hold, y_h)
            print(f"  step {step}: loss {float(loss):.4f} holdout {hb['all']:.6f} "
                  f"(e {hb['early'][0]:.4f} m {hb['mid'][0]:.4f} "
                  f"l {hb['late'][0]:.4f}) {time.time()-t0:.0f}s", flush=True)
            if hb["all"] < best[0]:
                best = (hb["all"], {k: {kk: vv.detach().cpu().clone()
                                        for kk, vv in v.items()}
                                    for k, v in net.state_dict().items()}, step)
    net.load_state_dict(best[1])
    net.eval()
    print(f"best holdout {best[0]:.6f} @ step {best[2]}", flush=True)

    # ---- frozen bench through the identical python path ----
    if not os.path.isdir(BENCH_ENC):
        src = os.path.join(HERE, "_hammer_work", "bench_states.jsonl")
        with open(src, "w") as f:
            for line in open(BENCH):
                r = json.loads(line)
                f.write(json.dumps({"s": r["s"], "y": r["truth"]}) + "\n")
        subprocess.run([sys.executable, os.path.join(LAB, "enc_adopted.py"),
                        "encode", src, BENCH_ENC],
                       cwd=LAB, check=True, capture_output=True, text=True)
    rows = [json.loads(l) for l in open(BENCH)]
    os.environ["VT_ENC"] = BENCH_ENC
    import importlib
    importlib.reload(vt_lib)
    barm = vt_lib.Arm("old", add="setup")
    nb = len(rows)
    bg = {}
    for k in vt_lib.OLD_KEYS:
        bg[k] = torch.from_numpy(np.asarray(barm.a[k])).to(
            dev, torch.int64 if "ids" in k else torch.float32)
    bg_am = torch.from_numpy(np.asarray(barm.am[:, :, barm.mcol], np.float32)).to(dev)
    bb = {k: bg[k] for k in vt_lib.OLD_KEYS}; bb["am"] = bg_am
    truth = torch.tensor([r["truth"] for r in rows], dtype=torch.float32, device=dev)
    with torch.no_grad():
        ph = batch_phase(bb)
        for nm, model in (("v8c_s1", control), (f"phase-{a.tag}", net)):
            pv = torch.sigmoid(model(bb))
            e2 = (pv - truth) ** 2
            parts = []
            for snm, lo, hi in (("early", 0, 1/3), ("mid", 1/3, 2/3),
                                ("late", 2/3, 1.01)):
                m = (ph >= lo) & (ph < hi)
                parts.append(f"{snm} {float(e2[m].mean()):.4f} (n={int(m.sum())})")
            print(f"BENCH {nm}: all {float(e2.mean()):.4f} | " + " | ".join(parts),
                  flush=True)

    out_pt = os.path.join(os.path.dirname(a.ckpt), f"v8c_{a.tag}.pt")
    torch.save({"sd": net.state_dict(), "cfg": {"arch": "phase3",
               "blend": "hat(p),knots 0/0.5/1", "p": "1-sum(hp_frac)/12"}}, out_pt)
    print(f"saved {out_pt} (python-only; engine export needs phase support)",
          flush=True)


if __name__ == "__main__":
    main()
