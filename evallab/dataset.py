"""Shards -> tensors.

Two products, both built from the SAME encoder path so a training row and an
evaluation row are the same kind of object:

  pos.npz   recorded decisions' ROOT states, labelled with the game outcome
            (the value head's BCE target), plus `gid`/`ply` so a split can be
            made by GAME (never by position -- positions inside one game are
            heavily correlated and a position-level split leaks the outcome).

  sib.npz   SIBLING SUCCESSOR states: for a subsample of decisions, one encoded
            state per legal arm, obtained by applying that arm against the
            opponent's reference move and taking the MODAL branch.  Carries the
            search's per-arm avg_score `q` as the ranking target and `grp` as
            the decision id.  This is the object the ranking loss trains on and
            the object the sibling-discrimination metric measures.

SUCCESSOR CONSTRUCTION (identical at train and eval time, on purpose)
    b* = opponent's argmax-visit arm in the recorded search
    s'(a) = modal branch of generate_instructions(s, a, b*)
The recorded `q[a]` is MCTS's root Q for arm a, marginalised over the
opponent's root strategy, so s'(a) is a 1-ply projection of it rather than the
same object.  It is the projection that actually governs move choice when a
value net is used at depth, and using one construction everywhere is what makes
the metric comparable across cells.

MEMORY (the reason this is a streaming two-pass and not a list of dicts)
  The local box has 8.6 GB. 440k encoded positions as float32 in python lists
  peaks near 6 GB and thrashes. So: pass 1 counts rows by streaming the shards,
  pass 2 fills PREALLOCATED arrays, floats are stored as float16 and ids as
  int16, and the trainer upcasts per batch. float16 costs ~1e-3 relative
  precision on features that are all O(1) ratios -- immaterial, and checked by
  `--verify` which re-encodes a sample in float32 and compares.

PLAYOUT LABELS (optional)
  Set PLAYOUT_LABELS=<playout.py labels jsonl> and the value set is built from
  the LABELLED positions only, with `y` = the playout-averaged win probability
  instead of the single game outcome, plus two new arrays: `npl` (playouts
  behind each label) and `se` (its standard error). `y` stays a side-one win
  probability in [0,1], which is what `binary_cross_entropy_with_logits`
  already consumes, so no trainer change is required to use it -- and `se` is
  there for anyone who wants to weight or filter by label precision.
  POS_STRIDE is ignored in this mode: the label file IS the selection.

USAGE
  python dataset.py <shard_glob> <out_prefix> [sib_decisions]
ENV
  POS_STRIDE   keep every Nth decision for the value set (default 1)
  PLAYOUT_LABELS  path to playout.py output; switches y to playout labels
"""

import glob
import gzip
import json
import os
import sys
import time

import numpy as np

import labenv  # noqa: F401
import relfeat
from encoder import Vocab, bench_order, encode_state  # noqa: E402
from poke_engine import State, generate_instructions  # noqa: E402

KEYS = ("a1_ids", "a1_f", "b1_ids", "b1_f", "sf1",
        "a2_ids", "a2_f", "b2_ids", "b2_f", "sf2", "g")
EXTRA = ("gx", "sx1", "sx2")
ALL = KEYS + EXTRA
POS_STRIDE = int(os.environ.get("POS_STRIDE", "1"))
PLAYOUT_LABELS = os.environ.get("PLAYOUT_LABELS", "")


def load_labels(path):
    """playout.py output -> {(game_id, ply): (label_p, n_playouts, se)}.

    Keyed by (game, ply) and not by state string: the shard is the one source
    of the state, so the join cannot disagree with it, and the label file stays
    small enough to read into memory at any corpus size."""
    lab = {}
    for p in sorted(glob.glob(path)) or [path]:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "label_p" not in r:
                    continue
                lab[(r["g"], int(r["ply"]))] = (
                    float(r["label_p"]), int(r["n_playouts"]), float(r["se"]))
    print("playout labels: %d positions from %s" % (len(lab), path), flush=True)
    return lab


def encode_full(state, vocab):
    """encoder.encode_state + the relational blocks."""
    d = encode_state(state, vocab)
    d["gx"] = relfeat.global_extra(state)
    s1, s2 = state.side_one, state.side_two
    m1, m2 = list(s1.pokemon), list(s2.pokemon)
    d["sx1"] = relfeat.side_extra(state, s1, s2, bench_order(m1, int(s1.active_index), vocab))
    d["sx2"] = relfeat.side_extra(state, s2, s1, bench_order(m2, int(s2.active_index), vocab))
    return d


def arm_to_move(a):
    a = a.lower()
    if a in ("no move", "nomove"):
        return "none"
    return a[7:] if a.startswith("switch ") else a


def successor(state, a, b):
    """Modal branch after (a, b). None if the engine rejects the pair."""
    try:
        br = [x for x in generate_instructions(state, arm_to_move(a), arm_to_move(b)) if x.percentage > 0]
    except Exception:
        return None
    if not br:
        return None
    return state.apply_instructions(max(br, key=lambda x: x.percentage))


def stream_games(paths, holdout):
    for p in sorted(paths):
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                if labenv.is_holdout(row["g"]) != holdout:
                    continue
                yield row


def _alloc(n, sample):
    out = {}
    for k in ALL:
        a = np.asarray(sample[k])
        dt = np.int16 if "ids" in k else np.float16
        out[k] = np.zeros((n,) + a.shape, dtype=dt)
    return out


def build(paths, out_prefix, sib_decisions=30000, seed=0, holdout=False):
    """holdout=False -> the TRAIN games; holdout=True -> the held-out games.
    The split is `labenv.is_holdout` on the game id and nothing else, so the
    oracle sampler and the evaluator land on exactly the same games without any
    seed being passed between stages."""
    vocab = Vocab(frozen=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    labels = load_labels(PLAYOUT_LABELS) if PLAYOUT_LABELS else None

    # ---- pass 1: count (streaming, so no shard is ever fully resident)
    n_games, n_dec, n_pos = 0, 0, 0
    for g in stream_games(paths, holdout):
        n_games += 1
        n_dec += len(g["t"])
        if labels is not None:
            n_pos += sum((g["g"], pi) in labels for pi in range(len(g["t"])))
        else:
            n_pos += len(range(0, len(g["t"]), POS_STRIDE))
    p_sib = min(1.0, sib_decisions / max(n_dec, 1)) if sib_decisions else 0.0
    print("pass1: games=%d (holdout=%s) decisions=%d positions=%d (stride %d) sib p=%.3f  %.0fs"
          % (n_games, holdout, n_dec, n_pos, POS_STRIDE, p_sib, time.time() - t0), flush=True)
    if n_pos == 0:
        return

    # ---- pass 2: fill
    pos = None
    y = np.zeros(n_pos, dtype=np.float32)
    gid = np.zeros(n_pos, dtype=np.int32)
    ply = np.zeros(n_pos, dtype=np.int32)
    nlegal = np.zeros(n_pos, dtype=np.int16)
    npl = np.zeros(n_pos, dtype=np.int16)
    lse = np.zeros(n_pos, dtype=np.float32)
    sib_chunks, sq, sgrp, sbest = [], [], [], []
    i, grp = 0, 0
    for gi, gme in enumerate(stream_games(paths, holdout)):
        out = float(gme["outcome"])
        for pi, t in enumerate(gme["t"]):
            lab = labels.get((gme["g"], pi)) if labels is not None else None
            take_pos = lab is not None if labels is not None else (pi % POS_STRIDE == 0)
            take_sib = p_sib > 0 and len(t.get("q", {})) >= 3 and rng.random() < p_sib
            if not (take_pos or take_sib):
                continue
            st = State.from_string(t["s"])
            if take_pos:
                d = encode_full(st, vocab)
                if pos is None:
                    pos = _alloc(n_pos, d)
                for k in ALL:
                    pos[k][i] = d[k]
                y[i], gid[i], ply[i], nlegal[i] = out, gi, pi, len(t["q"])
                if lab is not None:
                    # THE POINT OF ALL THIS: y becomes the playout-averaged win
                    # probability instead of one coin flip shared by the whole
                    # game. npl/se travel with it so a consumer can weight or
                    # filter by how precise each label actually is.
                    y[i], npl[i], lse[i] = lab
                i += 1
            if take_sib:
                bstar = max(t["n2"], key=lambda k: t["n2"][k]) if t["n2"] else None
                if bstar is None:
                    continue
                rows, qs = [], []
                for arm, q in t["q"].items():
                    s2 = successor(st, arm, bstar)
                    if s2 is None:
                        continue
                    rows.append(encode_full(s2, vocab))
                    qs.append(float(q))
                if len(rows) < 3:
                    continue
                chunk = {k: np.stack([r[k] for r in rows]).astype(
                    np.int16 if "ids" in k else np.float16) for k in ALL}
                sib_chunks.append(chunk)
                best = int(np.argmax(qs))
                sq += qs
                sgrp += [grp] * len(qs)
                sbest += [1.0 if j == best else 0.0 for j in range(len(qs))]
                grp += 1
        if (gi + 1) % 500 == 0:
            print("  %d/%d games  %.0fs  pos=%d sib_groups=%d"
                  % (gi + 1, n_games, time.time() - t0, i, grp), flush=True)

    o = {k: pos[k][:i] for k in ALL}
    o.update({"y": y[:i], "gid": gid[:i], "ply": ply[:i], "nlegal": nlegal[:i]})
    if labels is not None:
        o.update({"npl": npl[:i], "se": lse[:i]})
        print("playout-labelled: %d rows, mean y=%.4f mean se=%.4f"
              % (i, float(y[:i].mean()), float(lse[:i].mean())), flush=True)
    np.savez(out_prefix + "_pos.npz", **o)
    del o, pos
    if sib_chunks:
        s = {k: np.concatenate([c[k] for c in sib_chunks]) for k in ALL}
    else:
        s = {k: np.zeros((0,), dtype=np.float16) for k in ALL}
    s.update({"q": np.asarray(sq, dtype=np.float32), "grp": np.asarray(sgrp, dtype=np.int32),
              "best": np.asarray(sbest, dtype=np.float32)})
    np.savez(out_prefix + "_sib.npz", **s)
    print("wrote %s_pos.npz (%d rows) %s_sib.npz (%d rows / %d decisions)  %.0fs"
          % (out_prefix, i, out_prefix, len(sq), grp, time.time() - t0), flush=True)
    if vocab.unk_hits:
        print("VOCAB UNK HITS:", dict(list(vocab.unk_hits.items())[:20]), flush=True)


if __name__ == "__main__":
    paths = sorted(glob.glob(sys.argv[1]))
    assert paths, "no shards matched %r" % sys.argv[1]
    out = sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n_sib = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    build(paths, out, n_sib, holdout=False)
    build(paths, out + "_hold", 0, holdout=True)
