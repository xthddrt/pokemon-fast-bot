"""FIXED-PAIR, FULL-INFORMATION self-play corpus generator for the eval lab.

Differences from `valuenet/generate_selfplay.py`, all deliberate:

* **One fixed team pair per corpus.** No cross-pairing. The point of the lab is
  to hold the roster constant so "did the net learn this position set" is a
  question with an answer.
* **Full information.** Both teams are built explicitly and the search runs on
  the true state, so there is no determinizer and no hidden-info sampler. Any
  evaluation difference we measure is the value function's, not the sampler's.
* **Iteration-budgeted search**, not millisecond-budgeted: a fixed iteration
  count makes the corpus reproducible across machines and makes the "search
  quality" axis a number rather than a property of the box.
* **avg_scores are recorded, not just visits.** `q` = total_score/visits per arm
  is the sibling-ranking target; visits alone cannot supply one (arms differing
  by 1 visit can differ hugely in value, and zero-visit arms have no rank).
* **Explicit diversity injection.** Argmax self-play on ONE fixed pair collapses
  to a handful of trajectories. Three independent knobs widen it (below), and
  `stats.py` reports the diversity actually achieved.

DIVERSITY (why three knobs and not one)
  lead permutation  every game shuffles each side's party order, so the 36
                    (lead x lead) openings are all sampled instead of one.
  forced-random     the first `RAND_PLIES` decisions (per game, drawn
                    0..RAND_PLIES) are uniform over the legal set on BOTH
                    sides -- this is what actually leaves the opening book.
  temperature       the next `TEMP_PLIES` decisions sample ~ visits.
  epsilon           every later decision is uniform-random with prob `EPS`.
  Root Dirichlet noise is NOT available: it lives inside the Rust search and
  this lab does not touch Rust. The knobs above are the python-side substitute
  and are strictly stronger at the root (a forced-random ply is a Dirichlet
  draw at alpha -> 0). Stated as a limitation, not hidden.

WHAT IS RECORDED (gzipped JSONL, one line per game; line 1 is a header)
  {"g": game_id, "outcome": 1|0|0.5, "pair": "A", "lead": [i, j],
   "t": [{"s": state_string,
          "n": {arm: visits},  "q": {arm: avg_score},   # side_one
          "n2": {...},         "q2": {...},             # side_two
          "it": iterations, "m": played_arm, "m2": played_arm_side_two,
          "e": ""|"rand"|"temp"|"eps"}, ...]}           # how m was chosen
`q` is the search's own value for that arm in [0,1] = side_one win probability.

USAGE
  python generate.py <pair> <n_games> <out_dir> [start] [iters]
ENV
  EVALLAB_WORKERS   process pool size (default 4)
  EVALLAB_ITERS     MCTS iterations per decision (default 25000)
  RAND_PLIES TEMP_PLIES EPS   diversity knobs (defaults 3 / 6 / 0.08)
"""

import concurrent.futures as cf
import datetime
import gzip
import json
import os
import random
import sys
import zlib

import labenv  # noqa: F401  (pins encoder + engine env; must be first)
import labteams
from fp.battle import Battle, Move, Pokemon  # noqa: E402
from fp.helpers import normalize_name  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from poke_engine import State, generate_instructions, monte_carlo_tree_search  # noqa: E402

MAX_STEPS = 200
ITERS = int(os.environ.get("EVALLAB_ITERS", "25000"))
RAND_PLIES = int(os.environ.get("RAND_PLIES", "3"))
TEMP_PLIES = int(os.environ.get("TEMP_PLIES", "6"))
EPS = float(os.environ.get("EPS", "0.08"))


def build_team(specs):
    mons = []
    for spec in specs:
        p = Pokemon(normalize_name(spec["species"]), spec["level"])
        p.ability = normalize_name(spec.get("ability") or "")
        item = normalize_name(spec.get("item") or "")
        p.item = item or None
        p.moves = [Move(normalize_name(m)) for m in spec["moves"]]
        if spec.get("teraType"):
            p.tera_type = normalize_name(spec["teraType"])
        mons.append(p)
    return mons


def arm_to_move(a):
    a = a.lower()
    if a in ("no move", "nomove"):
        return "none"
    return a[7:] if a.startswith("switch ") else a


def start_state(pair, rng):
    """Fresh full-information state with a randomised lead on both sides."""
    t1 = build_team(labteams.team_spec(pair, 0))
    t2 = build_team(labteams.team_spec(pair, 1))
    # Permuting the whole party (not just picking a lead) also varies the
    # party-index order, which the engine's switch arms are named by. The
    # ENCODER sorts the bench by species id, so this does not change any
    # feature vector -- only which mon starts on the field.
    rng.shuffle(t1)
    rng.shuffle(t2)
    b = Battle("g")
    b.battle_type = None
    b.user.active, b.user.reserve = t1[0], t1[1:]
    b.opponent.active, b.opponent.reserve = t2[0], t2[1:]
    return State.from_string(battle_to_poke_engine_state(b).to_string()), (t1[0].name, t2[0].name)


def run_game(args):
    gid, pair, seed, iters = args
    rng = random.Random(seed)
    try:
        state, lead = start_state(pair, rng)
    except Exception as e:  # pragma: no cover - build failures are loud, not silent
        return {"g": gid, "error": repr(e)[:300]}

    n_rand = rng.randint(0, RAND_PLIES)
    turns = []
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        # ONE simultaneous-move search yields both sides' root statistics, so a
        # self-play decision costs one search, not two.
        res = monte_carlo_tree_search(state, 0, iters, 1, (seed * 7919 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        n_map = {m.move_choice: m.visits for m in res.side_one}
        n2_map = {m.move_choice: m.visits for m in res.side_two}
        q_map = {m.move_choice: round(m.total_score / m.visits, 5) for m in res.side_one if m.visits > 0}
        q2_map = {m.move_choice: round(m.total_score / m.visits, 5) for m in res.side_two if m.visits > 0}

        legal1 = list(n_map)
        legal2 = list(n2_map)
        if step < n_rand:
            tag = "rand"
            p1 = rng.choice(legal1)
            p2 = rng.choice(legal2)
        elif step < n_rand + TEMP_PLIES:
            tag = "temp"
            p1 = rng.choices([m.move_choice for m in s1], weights=[m.visits for m in s1])[0]
            p2 = rng.choices([m.move_choice for m in s2], weights=[m.visits for m in s2])[0]
        else:
            tag = ""
            p1 = max(s1, key=lambda m: m.visits).move_choice
            p2 = max(s2, key=lambda m: m.visits).move_choice
            if rng.random() < EPS:
                tag = "eps"
                p1 = rng.choice(legal1)
            if rng.random() < EPS:
                p2 = rng.choice(legal2)
                tag = tag or "eps"

        turns.append({"s": state.to_string(), "n": n_map, "q": q_map,
                      "n2": n2_map, "q2": q2_map, "it": res.total_visits,
                      "m": p1, "m2": p2, "e": tag})
        try:
            branches = [b for b in generate_instructions(state, arm_to_move(p1), arm_to_move(p2))
                        if b.percentage > 0]
        except Exception:
            break
        if not branches:
            break
        state = state.apply_instructions(rng.choices(branches, weights=[b.percentage for b in branches])[0])

    a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
    outcome = 1.0 if (a1 > 0 and a2 == 0) else 0.0 if (a2 > 0 and a1 == 0) else 0.5
    return {"g": gid, "pair": pair, "outcome": outcome, "lead": list(lead), "t": turns}


def main():
    pair = sys.argv[1]
    n_games = int(sys.argv[2])
    out_dir = sys.argv[3]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else ITERS
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shard_%s_%07d.jsonl.gz" % (pair, start))

    # Resume: keep the parseable prefix, rewrite the shard to it, append after.
    done, clean, has_header = set(), [], False
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt", errors="replace") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("kind") == "header":
                        if not has_header:
                            has_header = True
                            clean.insert(0, line)
                    elif "g" in row:
                        done.add(row["g"])
                        clean.append(line)
        except (EOFError, OSError, zlib.error):
            pass
        with gzip.open(path + ".tmp", "wt") as out:
            out.writelines(clean)
        os.replace(path + ".tmp", path)

    tasks = [("%s%07d" % (pair, i), pair, 5_000_000 + 977 * i, iters)
             for i in range(start, start + n_games) if "%s%07d" % (pair, i) not in done]
    print("shard %s: %d done, %d to run, iters=%d" % (path, len(done), len(tasks), iters), flush=True)
    workers = int(os.environ.get("EVALLAB_WORKERS", "4"))
    n = 0
    with gzip.open(path, "at") as out, cf.ProcessPoolExecutor(max_workers=workers) as ex:
        if not has_header:
            out.write(json.dumps({
                "kind": "header", "schema": 1, "pair": pair, "iters": iters,
                "start": start, "n_games": n_games, "full_info": True,
                "rand_plies": RAND_PLIES, "temp_plies": TEMP_PLIES, "eps": EPS,
                "net": os.environ.get("PE_NN_WEIGHTS"),
                "teams": {"0": labteams.PAIRS[pair][0], "1": labteams.PAIRS[pair][1]},
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }) + "\n")
            out.flush()
        for row in ex.map(run_game, tasks, chunksize=1):
            out.write(json.dumps(row) + "\n")
            n += 1
            if n % 50 == 0:
                out.flush()
                print("games: %d/%d" % (n, len(tasks)), flush=True)
    print("shard complete: %d games" % n, flush=True)


if __name__ == "__main__":
    main()
