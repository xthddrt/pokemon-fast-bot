"""v9 PARITY BATTERY — the acceptance gate for 3-trunk phase-net support.

Builds a RANDOM-weight PhaseNet (trunks deliberately perturbed apart, so an
identical-trunk blend cannot pass vacuously), exports it with
evallab/export_v9.py, and compares python logits (phase_probe.PhaseNet through
the enc_adopted/vt_lib path) against engine logits (leaf_prof `logits`) on
ladder states covering all phase-blend regimes (p < 1/3, both sides of 0.5,
p > 2/3). Engine side runs twice: plain forward (PE_NN_MON_MEMO=0
PE_NN_LEAF_CACHE=0) and the default memo/cache path.

PASS = max |python_logit - engine_logit| <= 1e-3 on both engine configs, AND
the v8 regression: valuenet/nets_v8c/v8c_s1.bin under its sidecar constants
still reproduces the seeded-search baseline (monte_carlo_tree_search(state, 0,
20000, 1, 424242) on the first worlds.jsonl state, side_one [move, visits]
sorted by visits desc) EXACTLY.

PREREQUISITE for the v8 regression: the poke_engine wheel in foul-play/.venv
must be rebuilt from the current poke-engine tree first (see README wheel
rebuild command) — a stale wheel would pass vacuously.

  foul-play/.venv/bin/python corrections/v9_parity.py
"""
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "evallab")
WORK = os.path.join(HERE, "_v9_work")
ENGINE = os.path.join(ROOT, "poke-engine")
LEAF_PROF = os.path.join(ENGINE, "target/release/leaf_prof")
WORLDS = os.path.join(
    ROOT, "ladder-games/games/2026-08-16_2665943126_xmasblitz_L/worlds.jsonl")
V8_BIN = os.path.join(ROOT, "valuenet/nets_v8c/v8c_s1.bin")
BASELINE = os.environ.get(
    "V9_PARITY_BASELINE",
    "/private/tmp/claude-501/-Users-sallyliu-pokemon-fast-bot/"
    "fcc1b52c-d38e-40bd-b9db-95a07c1d94b7/scratchpad/parity_baseline.json")
PER_BUCKET = 10
TOL = 1e-3


def phase_of(s):
    """p = 1 - mean hp_frac over the 12 mons, parsed straight off the state
    string: comma-fields 6/7 (hp/maxhp) of the first 6 '='-separated mon
    substrings of each of the first two '/'-separated side segments."""
    hp = 0.0
    for seg in s.split("/")[:2]:
        for m in seg.split("=")[:6]:
            f = m.split(",")
            hp += int(f[6]) / int(f[7]) if int(f[7]) else 0.0
    return 1 - hp / 12.0


def bucket_of(p):
    return 0 if p < 1 / 3 else 1 if p < 0.5 else 2 if p < 2 / 3 else 3


def select_states():
    buckets = [[] for _ in range(4)]
    seen = set()
    for line in open(WORLDS):
        s = json.loads(line)["state"]
        if s in seen:
            continue
        seen.add(s)
        b = bucket_of(phase_of(s))
        if len(buckets[b]) < PER_BUCKET:
            buckets[b].append(s)
    names = ["p<1/3", "1/3<=p<0.5", "0.5<=p<2/3", "p>2/3"]
    for nm, b in zip(names, buckets):
        assert len(b) >= 4, "phase coverage: bucket %s has only %d states" % (
            nm, len(b))
    states = [s for b in buckets for s in b]
    assert len(states) >= 32, len(states)
    print("states: %d  (%s)" % (
        len(states), ", ".join("%s: %d" % (nm, len(b))
                               for nm, b in zip(names, buckets))))
    return states


def v8_regression_child():
    """Runs in a SUBPROCESS whose env net_env(v8c_s1.bin) already set — the
    engine's LazyLock reads PE_* once, so this cannot share the parent."""
    import poke_engine as pe
    line = open(WORLDS).readline()
    st = pe.State.from_string(json.loads(line)["state"])
    res = pe.monte_carlo_tree_search(st, 0, 20000, 1, 424242)
    got = sorted(([o.move_choice, o.visits] for o in res.side_one),
                 key=lambda x: -x[1])
    base = [list(x) for x in json.load(open(BASELINE))]
    if got != base:
        print("v8 REGRESSION FAIL\n got:  %s\n base: %s" % (got, base))
        return 1
    print("v8 regression: seeded search visits IDENTICAL to baseline "
          "(%d arms, top %s)" % (len(got), got[0]))
    return 0


def main():
    os.makedirs(WORK, exist_ok=True)
    states = select_states()

    # ---- encode (must exist before vt_lib import: it reads VT_ENC) ----
    src = os.path.join(WORK, "states.jsonl")
    with open(src, "w") as f:
        for s in states:
            f.write(json.dumps({"s": s, "y": 0.5}) + "\n")
    enc = os.path.join(WORK, "enc")
    r = subprocess.run([sys.executable, os.path.join(LAB, "enc_adopted.py"),
                        "encode", src, enc],
                       cwd=LAB, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    # ---- random-weight PhaseNet, trunks perturbed APART ----
    sys.path.insert(0, LAB)
    sys.path.insert(0, HERE)
    os.environ["VT_ENC"] = enc
    import labenv  # noqa: F401
    import numpy as np
    import torch
    import vt_lib
    from phase_probe import PhaseNet, batch_phase

    torch.manual_seed(9)
    base = vt_lib.build_net("old", (128, 256), 1, add="setup")
    net = PhaseNet(base)
    for k, trunk in enumerate(net.trunks):
        g = torch.Generator().manual_seed(101 + k)
        for p_ in trunk.parameters():
            p_.data.add_(torch.randn(p_.shape, generator=g) * 0.1)
    net.eval()
    ckpt = os.path.join(WORK, "rand_phase.pt")
    torch.save({"sd": net.state_dict(),
                "cfg": {"arch": "phase3", "note": "random-weight parity net"}},
               ckpt)

    # ---- export ----
    v9_bin = os.path.join(WORK, "rand_phase_v9.bin")
    r = subprocess.run([sys.executable, os.path.join(LAB, "export_v9.py"),
                        ckpt, v9_bin], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    assert struct.unpack("<I", open(v9_bin, "rb").read(8)[4:])[0] == 9

    # ---- python logits (phase_probe's bench-block pattern) ----
    arm = vt_lib.Arm("old", add="setup")
    bb = {k: torch.from_numpy(np.asarray(arm.a[k])).to(
        torch.int64 if "ids" in k else torch.float32) for k in vt_lib.OLD_KEYS}
    bb["am"] = torch.from_numpy(np.asarray(arm.am[:, :, arm.mcol], np.float32))
    n = bb["a1_f"].shape[0]
    assert n == len(states), (n, len(states))
    with torch.no_grad():
        py = net(bb).numpy().astype(np.float64)
        p_enc = batch_phase(bb).numpy()
    # enc_adopted stores features float16, so the encoded hp_frac (and hence
    # python's blend variable) carries ~2^-11 quantization the engine's f32
    # features do not — same situation as every v8 python-vs-engine compare,
    # covered by the 1e-3 logit tolerance. This is a parse-recipe sanity
    # check, not the gate.
    p_str = np.array([phase_of(s) for s in states])
    assert np.abs(p_enc - p_str).max() < 1e-3, \
        "phase parse disagrees with encoded hp_frac"

    # ---- engine logits via leaf_prof, plain forward AND memo/cache path ----
    r = subprocess.run(["cargo", "build", "--release", "--bin", "leaf_prof",
                        "--features", "terastallization",
                        "--no-default-features"],
                       cwd=ENGINE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    stf = os.path.join(WORK, "states.txt")
    with open(stf, "w") as f:
        f.write("\n".join(states) + "\n")

    names = ["p<1/3", "1/3<=p<0.5", "0.5<=p<2/3", "p>2/3"]
    ok = True
    for label, extra in (("plain-forward",
                          {"PE_NN_MON_MEMO": "0", "PE_NN_LEAF_CACHE": "0"}),
                         ("memo+cache", {})):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PE_")}
        env["PE_NN_WEIGHTS"] = v9_bin
        env.update(extra)
        r = subprocess.run([LEAF_PROF, "logits", stf],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        eng = np.array([float(l.split("\t")[1])
                        for l in r.stdout.strip().split("\n")])
        assert len(eng) == len(states), (len(eng), len(states))
        d = np.abs(py - eng)
        print("engine[%s]: max|py-eng| = %.3e  (%s)" % (
            label, d.max(),
            ", ".join("%s: %.1e" % (nm, d[[bucket_of(p) == i
                                           for p in p_str]].max())
                      for i, nm in enumerate(names))))
        ok &= d.max() <= TOL

    # ---- v8 regression in a fresh process with the sidecar env ----
    sys.path.insert(0, HERE)
    from mine_value import net_env
    r = subprocess.run([sys.executable, __file__, "--v8-regression-child"],
                       env=net_env(V8_BIN))
    ok &= r.returncode == 0

    print("V9 PARITY BATTERY: %s (tolerance %.0e)" %
          ("PASS" if ok else "FAIL", TOL))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--v8-regression-child" in sys.argv:
        sys.exit(v8_regression_child())
    sys.exit(main())
