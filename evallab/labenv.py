"""Shared environment/bootstrap for every evallab entry point.

ONE place that pins the encoder flags and the engine's NN weights, because the
lab's whole point is that two runs differ only in the thing under test. Import
this BEFORE `poke_engine` or `valuenet.encoder` anywhere in the lab: poke-engine
reads its PE_* knobs once per process (Rust LazyLock) and encoder.py reads its
flags at import time, so a late import silently gets defaults.

Encoder flags are the v6 champion recipe (valuenet_v6nopol.constants.json ->
ckpt): BENCH_SORT=1, PP_TRUE_MAX=1, no reveal masks (the lab is full-information
by construction, so every mask bit would be a constant 1 -- see README).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "evallab")

# --- encoder recipe (must be set before `import valuenet.encoder`) ------------
os.environ.setdefault("BENCH_SORT", "1")
os.environ.setdefault("PP_TRUE_MAX", "1")
os.environ.setdefault("REVEAL_MASKS", "0")
os.environ.setdefault("MASK_SPLICE_SPECIES", "0")
# ADOPTED ENCODER (CORPUS_ADEQUACY.md fix 2): drop `times_attacked`. It is
# non-zero on 1.41 % of corpus mons (max 6) and 45.6 % at inference (max 36), so
# the learned weight is applied out of distribution at nearly every live leaf.
# Removed, not zeroed -- a constant column is an invisible bias. NUM_MON_FEATS
# becomes 72, so any cache built before this flag is stale.
os.environ.setdefault("DROP_TIMES_ATTACKED", "1")

# --- engine recipe (must be set before `import poke_engine`) -----------------
NET = os.environ.get("EVALLAB_NET", os.path.join(ROOT, "valuenet/m4_artifacts/valuenet_v6nopol.bin"))
if os.path.isfile(NET):
    os.environ.setdefault("PE_NN_WEIGHTS", NET)
    side = os.path.splitext(NET)[0] + ".constants.json"
    if os.path.isfile(side):
        for k, v in json.load(open(side)).items():
            if k.startswith("PE_"):
                os.environ.setdefault(k, str(v))

for p in (os.path.join(ROOT, "foul-play"), os.path.join(ROOT, "valuenet"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

SETS_PATH = os.path.join(ROOT, "foul-play/data/pkmn_sets_cache/gen9randombattle.json")
MOVES_PATH = os.path.join(ROOT, "foul-play/data/moves.json")
DEX_PATH = os.path.join(ROOT, "foul-play/data/pokedex.json")
CHART_PATH = os.path.join(LAB, "type_chart.json")
DATA = os.environ.get("EVALLAB_DATA", os.path.join(LAB, "data"))

# ---- the ONE definition of the train/holdout split -------------------------
# Split by GAME index, deterministically, so every stage (dataset builder,
# oracle sampler, evaluator) agrees without passing seeds around. Positions
# inside one game share an outcome label, so anything finer than a game-level
# split leaks the label.
HOLDOUT_MOD = 10


def is_holdout(game_id):
    """game_id is generate.py's 'A0000123' form."""
    return int(str(game_id)[1:]) % HOLDOUT_MOD == 0
