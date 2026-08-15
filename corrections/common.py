"""Shared plumbing for the hammer correction loop (HAMMER_SPEC.md Part 2).

Everything here is speed-critical path for a <2min capture->hammer->verify
cycle, so imports of torch / valuenet modules are deferred into functions and
the encoder env flags are set from the TARGET checkpoint before those imports
(they are read at import time and size every array).

Run everything with foul-play/.venv/bin/python (the wheel that loads PKNN v6+v7).
"""

import gzip
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = os.path.join(ROOT, "corrections")
LEDGER = os.path.join(CORR, "ledger.jsonl")
VALUENET = os.path.join(ROOT, "valuenet")
FOULPLAY = os.path.join(ROOT, "foul-play")
GAMES = os.path.join(ROOT, "ladder-games", "games")
M4 = os.path.join(VALUENET, "m4_artifacts")
VENV_PY = os.path.join(FOULPLAY, ".venv", "bin", "python")
OOS_SHARD = os.path.join(CORR, "oos.jsonl.gz")

# Live production selection values (run_game.sh flags on every current game).
TERA_GATE = {"per_mon": 0.0015, "visit_frac": 0.3333, "opp_tera_bonus": 0.003}

OPP_TEAM_SIZE = 6


# ---------------------------------------------------------------------------
# move-name normalisation
# ---------------------------------------------------------------------------

def norm_choice(s):
    """Sally's input -> selection.py choice format ('suckerpunch',
    'suckerpunch-tera', 'switch gyarados')."""
    s = s.strip().lower()
    if s.startswith("switch"):
        rest = re.sub(r"[\s\-'.]", "", s[len("switch"):])
        return "switch " + rest
    tera = False
    for suf in ("-tera", " tera"):
        if s.endswith(suf):
            s, tera = s[: -len(suf)], True
    return re.sub(r"[\s\-'.]", "", s) + ("-tera" if tera else "")


def engine_move(choice):
    """selection.py choice -> generate_instructions move string.
    MoveChoice::from_string wants a BARE species id for switches."""
    return choice[len("switch "):] if choice.startswith("switch ") else choice


# ---------------------------------------------------------------------------
# game archive parsing
# ---------------------------------------------------------------------------

def resolve_game_dir(game):
    for cand in (game, os.path.join(GAMES, game), os.path.join(ROOT, game)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise SystemExit("game dir not found: %s" % game)


def parse_decisions(game_dir):
    """battle.log.gz -> [{'decision': 1-based idx, 'turn': N, 'choice': str}].

    Search blocks ('Searching for a move using MCTS...') interleave with
    'Turn: N' markers; the Nth block is worlds.jsonl decision N and belongs to
    the most recent turn marker.
    """
    out, cur, turn = [], None, 0
    with gzip.open(os.path.join(game_dir, "battle.log.gz"), "rt",
                   errors="replace") as f:
        for line in f:
            m = re.search(r"Turn: (\d+)\s*$", line)
            if m:
                turn = int(m.group(1))
                continue
            if "Searching for a move using MCTS" in line:
                cur = {"decision": len(out) + 1, "turn": turn, "choice": None}
                out.append(cur)
                continue
            m = re.search(r"Choice: (.+?)\s*$", line)
            if m and cur is not None and cur["choice"] is None:
                cur["choice"] = m.group(1).strip()
    return out


def load_worlds(game_dir):
    """worlds.jsonl -> {decision: [row, ...] sorted by world}."""
    byd = {}
    with open(os.path.join(game_dir, "worlds.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            byd.setdefault(r["decision"], []).append(r)
    for v in byd.values():
        v.sort(key=lambda r: r["world"])
    return byd


def game_net(game_dir):
    """--nn-weights basename from the archived flags (index.jsonl/meta.json)."""
    meta = json.load(open(os.path.join(game_dir, "meta.json")))
    m = re.search(r"--nn-weights\s+(\S+)", meta.get("flags") or "")
    return os.path.basename(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# engine helpers (hand-eval process: PE_NN_WEIGHTS must NOT be set here)
# ---------------------------------------------------------------------------

def root_options(state_str, iterations=64):
    """Legal options for both sides, straight from the engine's root
    enumeration (handles forced switches, locked moves, tera variants)."""
    import poke_engine as pe

    st = pe.State.from_string(state_str)
    res = pe.monte_carlo_tree_search(st, 10, iterations=iterations, threads=1,
                                     seed=7)
    return ([o.move_choice for o in res.side_one],
            [o.move_choice for o in res.side_two])


def opp_context(state_str):
    """(opp_alive, opp_unrevealed, opp_tera_used) exactly as fp/search/main.py
    computes them for the tera gate, derived from the recorded world state
    (fainted mons are always real+revealed; unrevealed ones are sampled)."""
    import poke_engine as pe

    mons = pe.State.from_string(state_str).side_two.pokemon
    fainted = sum(1 for p in mons if p.hp <= 0 and p.revealed)
    unrevealed = sum(1 for p in mons if not p.revealed)
    tera_used = any(p.terastallized for p in mons)
    return {"opp_alive": OPP_TEAM_SIZE - fainted,
            "opp_unrevealed": unrevealed,
            "opp_tera_used": tera_used}


def build_successors(entry, options=None, iterations=64):
    """Apply each of our options against EVERY legal opponent reply in every
    recorded world; return {option: {'states': [State], 'weights': [float]}}.

    weights = world_chance * uniform-over-replies * branch_percentage, i.e.
    the ruling's 'irrespective of what the opponent does' measure. Training
    uses the states unweighted (every successor is labelled 1.0); the weights
    drive the fast precheck's mean-value ranking.
    """
    import poke_engine as pe

    per_option = {}
    for wrow in entry["states"]:
        st = pe.State.from_string(wrow["state"])
        chance = wrow["chance"]
        res = pe.monte_carlo_tree_search(st, 10, iterations=iterations,
                                         threads=1, seed=7)
        s1_opts = [o.move_choice for o in res.side_one]
        s2_opts = [o.move_choice for o in res.side_two]
        for s1 in (options if options is not None else s1_opts):
            if s1 not in s1_opts:
                continue  # option not legal in this world (rare)
            slot = per_option.setdefault(s1, {"states": [], "weights": []})
            for s2 in s2_opts:
                branches = pe.generate_instructions(st, engine_move(s1),
                                                    engine_move(s2))
                tot = sum(b.percentage for b in branches if b.percentage > 0)
                if tot <= 0:
                    continue
                for b in branches:
                    if b.percentage <= 0:
                        continue
                    slot["states"].append(st.apply_instructions(b))
                    slot["weights"].append(
                        chance * (1.0 / len(s2_opts)) * (b.percentage / tot))
    return per_option


# ---------------------------------------------------------------------------
# target net loading (env flags first, torch/valuenet imports after)
# ---------------------------------------------------------------------------

def load_ckpt(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit(
            "%s is not a train.py checkpoint (.pt with 'model'). The hammer "
            "needs the .pt; a bin-only net must first be reconstructed to a "
            "checkpoint." % path)
    return ck


def ckpt_env(ck):
    mon_in = ck["model"]["mon.0.weight"].shape[1]
    return {
        "REVEAL_MASKS": "1" if mon_in == 227 else "0",
        "BENCH_SORT": str(ck.get("bench_sort", "1")),
        "PP_TRUE_MAX": str(ck.get("pp_true_max", "1")),
        "POLICY_WEIGHT": str(ck.get("policy_weight", 0.0) or 0),
        "BENCH_POOL": ck.get("bench_pool", "sum"),
        "TRUNK_BLOCKS": str(ck.get("trunk_blocks", 0)),
        "ATTN_D": str(ck.get("attn_d", 0) or 0),
        "MON_HID": str(ck.get("mon_hid", 128)),
        "TRUNK": str(ck.get("trunk", 256)),
    }


def apply_env(env):
    os.environ.update(env)


def build_net(ck):
    """train.py ValueNet from a checkpoint. apply_env(ckpt_env(ck)) MUST have
    run before this (import-time flags). This is the audit agent's verified
    bit-exact torch harness loader (scratchpad audit2_parity.py)."""
    if VALUENET not in sys.path:
        sys.path.insert(0, VALUENET)
    import train

    net = train.ValueNet({k: len(v) for k, v in ck["vocab"].items()})
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def frozen_vocab():
    if VALUENET not in sys.path:
        sys.path.insert(0, VALUENET)
    from encoder import Vocab

    return Vocab(frozen=True)


def encode_batch(pe_states, vocab):
    """[poke_engine.State] -> {key: np.ndarray[batch,...]} for ValueNet."""
    import numpy as np

    if VALUENET not in sys.path:
        sys.path.insert(0, VALUENET)
    from encoder import encode_state

    encs = [encode_state(s, vocab) for s in pe_states]
    return {k: np.stack([e[k] for e in encs]) for k in encs[0]}


def to_torch(batch, device="cpu"):
    import torch

    return {k: torch.as_tensor(v).long().to(device) if "ids" in k
            else torch.as_tensor(v).float().to(device)
            for k, v in batch.items()}


def net_values(net, tbatch, bs=4096):
    """sigmoid(value logit) per row, no grad."""
    import torch

    n = next(iter(tbatch.values())).shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, n, bs):
            b = {k: v[s:s + bs] for k, v in tbatch.items()}
            o = net(b)
            v = o[0] if isinstance(o, tuple) else o
            out.append(torch.sigmoid(v))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def read_ledger(path=LEDGER):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def append_ledger(row, path=LEDGER):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
