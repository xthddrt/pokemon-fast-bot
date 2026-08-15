"""The metrics that actually matter.

(a) VALUE CE  -- binary cross-entropy of the net's win probability against the
    realised outcome, on held-out GAMES. This is the number the production
    pipeline optimises, and the one the audit already knows how to read.

(b) SIBLING DISCRIMINATION -- the number nobody has measured. Over each oracle
    decision's sibling successor states (one per legal arm, built exactly as
    `dataset.py` builds them):
      spread   sd of the net's win probability across the siblings. A net whose
               siblings all score the same CANNOT choose a move, whatever its CE.
      top1     the net's argmax sibling is the oracle's best move
      top3     the net's argmax is in the oracle's top 3 by value
      spearman rank correlation of net sibling values with oracle per-move values
      regret   oracle_q[oracle best] - oracle_q[net's pick], in win probability.
               The decision-theoretic bottom line: what picking by this net
               costs per decision against a 5M-iteration search.
    Two references bracket every cell: RANDOM (uniform arm) and ORACLE (0 by
    construction). A net that does not beat RANDOM on regret is not steering.

(c) MEMORISATION vs GENERALISATION -- the same metrics on pair A (trained),
    pair B (half its species shared with A) and pair C (disjoint). The A->C gap
    is how much of the net is memory; the A->B->C ladder says how fast that
    memory decays with roster overlap.

USAGE
  python evaluate.py <ckpt.pt> <oracle.jsonl> <hold_pos_glob> [out.json]
"""

import glob
import json
import os
import sys

import numpy as np
import torch

import labenv  # noqa: F401
import labmodel
from dataset import KEYS as POS_KEYS, EXTRA, encode_full, successor  # noqa: E402
from encoder import Vocab  # noqa: E402
from poke_engine import State  # noqa: E402

ALL_KEYS = POS_KEYS + EXTRA
THREADS = int(os.environ.get("EVALLAB_THREADS", "4"))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def batchify(rows):
    return {k: torch.tensor(np.stack([r[k] for r in rows]),
                            dtype=torch.long if "ids" in k else torch.float32)
            for k in ALL_KEYS}


def npz_batch(z, sl):
    return {k: torch.tensor(z[k][sl].astype("int64" if "ids" in k else "float32"))
            for k in ALL_KEYS}


def load_net(path):
    """Accepts a lab checkpoint OR a shipped valuenet/train.py checkpoint.

    The shipped nets are loadable because LabValueNet at REL=0 is bit-identical
    to train.ValueNet (labmodel.assert_baseline_parity) and their `vocab` is the
    same table as the local vocab.json -- both asserted here, not assumed, so
    scoring the ladder incumbent on this metric is a one-argument change rather
    than a second harness that could disagree.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocab(frozen=True)
    if "sd" in ck:
        net = labmodel.LabValueNet(vocab.sizes(), use_rel=bool(ck["rel"]))
        net.load_state_dict(ck["sd"])
    else:
        assert ck["vocab"] == vocab.d, "shipped checkpoint vocab != local vocab.json"
        assert ck.get("attn_d", 0) == 0 and ck.get("bench_pool", "sum") == "sum" \
            and ck.get("trunk_blocks", 0) == 0, "shipped net is not the baseline architecture"
        net = labmodel.LabValueNet(vocab.sizes(), use_rel=False)
        net.load_state_dict(ck["model"])
        ck = {"rel": 0, "rank_w": -1.0, "val_ce": ck.get("val", float("nan")),
              "shipped": os.path.basename(path)}
    net.eval()
    return net, ck, vocab


def value_ce(net, paths):
    """(a) BCE on held-out games, per pair."""
    out = {}
    for p in paths:
        z = np.load(p)
        if len(z["y"]) == 0:
            continue
        name = os.path.basename(p).split("_")[0]
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(z["y"]), 8192):
                b = npz_batch(z, slice(i, i + 8192))
                y = torch.tensor(z["y"][i:i + 8192], dtype=torch.float32)
                zz = net(b)
                tot += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    zz, y, reduction="sum"))
                n += len(y)
        base = float(z["y"].mean())
        const = -(base * np.log(max(base, 1e-9)) + (1 - base) * np.log(max(1 - base, 1e-9)))
        out[name] = {"ce": tot / n, "n": n, "base_rate": base, "const_ce": float(const)}
    return out


def sibling_metrics(net, oracle_path, vocab, max_positions=None):
    """(b)+(c). One row per oracle decision, aggregated by pair."""
    rows = [json.loads(l) for l in open(oracle_path)]
    if max_positions:
        rows = rows[:max_positions]
    per_pair = {}
    skipped = 0
    for r in rows:
        st = State.from_string(r["s"])
        bstar = r["best2"] or max(r["n2"], key=lambda k: r["n2"][k])
        arms, qs, feats = [], [], []
        for arm, q in r["q"].items():
            s2 = successor(st, arm, bstar)
            if s2 is None:
                continue
            arms.append(arm)
            qs.append(float(q))
            feats.append(encode_full(s2, vocab))
        if len(arms) < 3:
            skipped += 1
            continue
        with torch.no_grad():
            v = torch.sigmoid(net(batchify(feats))).numpy()
        qs = np.asarray(qs)
        order = np.argsort(-qs)
        best_arm = arms[int(order[0])]
        top3 = {arms[int(i)] for i in order[:3]}
        pick = arms[int(np.argmax(v))]
        d = per_pair.setdefault(r["pair"], {"n": 0, "spread": [], "top1": [], "top3": [],
                                            "rho": [], "regret": [], "rand_regret": [],
                                            "gap": [], "narms": []})
        d["n"] += 1
        d["spread"].append(float(v.std()))
        d["top1"].append(1.0 if pick == best_arm else 0.0)
        d["top3"].append(1.0 if pick in top3 else 0.0)
        d["rho"].append(spearman(v, qs))
        d["regret"].append(float(qs.max() - qs[arms.index(pick)]))
        d["rand_regret"].append(float(qs.max() - qs.mean()))
        d["gap"].append(float(qs.max() - qs.min()))
        d["narms"].append(len(arms))
    agg = {}
    for k, d in per_pair.items():
        agg[k] = {"n": d["n"], "narms": float(np.mean(d["narms"])),
                  "spread": float(np.mean(d["spread"])),
                  "top1": float(np.mean(d["top1"])), "top3": float(np.mean(d["top3"])),
                  "spearman": float(np.nanmean(d["rho"])),
                  "regret": float(np.mean(d["regret"])),
                  "random_regret": float(np.mean(d["rand_regret"])),
                  "oracle_gap": float(np.mean(d["gap"])),
                  "top1_se": float(np.std(d["top1"]) / max(np.sqrt(d["n"]), 1)),
                  "regret_se": float(np.std(d["regret"]) / max(np.sqrt(d["n"]), 1))}
    if skipped:
        print("  (skipped %d oracle positions with <3 constructible successors)" % skipped)
    return agg


def main():
    torch.set_num_threads(THREADS)
    ckpt, oracle_path, hold_glob = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    net, ck, vocab = load_net(ckpt)
    res = {"ckpt": ckpt, "rel": int(ck["rel"]), "rank_w": ck["rank_w"],
           "game_frac": ck.get("game_frac", 1.0), "train_val_ce": ck["val_ce"]}
    res["value_ce"] = value_ce(net, sorted(glob.glob(hold_glob)))
    res["sibling"] = sibling_metrics(net, oracle_path, vocab)
    print(json.dumps(res, indent=1))
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
