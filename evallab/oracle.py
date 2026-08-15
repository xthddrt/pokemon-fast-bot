"""Near-ground-truth per-move values for a held-out position set.

The corpus's own 25k-iteration searches are the TRAINING signal; they are not a
yardstick for it. This runs a much deeper search (default 5M iterations, full
information, same engine and same net) on a stratified sample of decisions from
HELD-OUT games only, and that is the evaluation standard everything in
`evaluate.py` is scored against.

STRATIFICATION
  Positions are sampled across three phases by remaining material
  (12, 11-9, <=8 alive across both sides) and require >= 3 legal arms -- a
  decision with two arms cannot discriminate anything, and an unstratified
  sample is dominated by mid-game states.

OUTPUT (jsonl, one line per position)
  {"pair", "g", "ply", "s", "it", "alive": [a1, a2],
   "q": {arm: avg_score}, "n": {arm: visits},          # side one
   "q2": {...}, "n2": {...},                           # side two
   "best": arm, "best2": arm}
`best` is the argmax of `q` among arms with a meaningful visit share, not the
argmax of visits: at 5M iterations two arms routinely end up with near-equal
visit counts while their values differ, and it is the VALUE that defines the
right move.

USAGE
  python oracle.py <shard_glob> <out.jsonl> <n_positions> [iters]
ENV  EVALLAB_WORKERS
"""

import concurrent.futures as cf
import glob
import gzip
import json
import os
import random
import sys
import time

import labenv  # noqa: F401
from poke_engine import State, monte_carlo_tree_search  # noqa: E402

ITERS = int(os.environ.get("ORACLE_ITERS", "5000000"))
MIN_VISIT_SHARE = 0.02


def phase(alive_total):
    return 0 if alive_total >= 12 else 1 if alive_total >= 9 else 2


def pick_positions(paths, n_positions, seed=11):
    rng = random.Random(seed)
    cands = {0: [], 1: [], 2: []}
    for p in sorted(paths):
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                if not labenv.is_holdout(row["g"]):
                    continue
                for pi, t in enumerate(row["t"]):
                    if len(t.get("q", {})) < 3:
                        continue
                    st = State.from_string(t["s"])
                    a1 = sum(x.hp > 0 for x in st.side_one.pokemon)
                    a2 = sum(x.hp > 0 for x in st.side_two.pokemon)
                    cands[phase(a1 + a2)].append(
                        {"pair": row["pair"], "g": row["g"], "ply": pi, "s": t["s"],
                         "alive": [a1, a2]})
    out = []
    per = max(1, n_positions // 3)
    for k in (0, 1, 2):
        rng.shuffle(cands[k])
        out += cands[k][:per]
    # top up from whichever phase still has spare candidates
    spare = [c for k in (0, 1, 2) for c in cands[k][per:]]
    rng.shuffle(spare)
    out += spare[:max(0, n_positions - len(out))]
    rng.shuffle(out)
    print("oracle candidates by phase: %s -> selected %d"
          % ({k: len(v) for k, v in cands.items()}, len(out)), flush=True)
    return out


def deepen(task):
    pos, iters, seed = task
    st = State.from_string(pos["s"])
    r = monte_carlo_tree_search(st, 0, iters, 1, seed)

    def side(arms):
        tv = sum(m.visits for m in arms) or 1
        n = {m.move_choice: m.visits for m in arms}
        q = {m.move_choice: round(m.total_score / m.visits, 6) for m in arms if m.visits > 0}
        live = [k for k in q if n[k] / tv >= MIN_VISIT_SHARE] or list(q)
        best = max(live, key=lambda k: q[k]) if live else None
        return n, q, best

    n1, q1, b1 = side(r.side_one)
    n2, q2, b2 = side(r.side_two)
    out = dict(pos)
    out.update({"it": r.total_visits, "n": n1, "q": q1, "best": b1,
                "n2": n2, "q2": q2, "best2": b2})
    return out


def main():
    paths = sorted(glob.glob(sys.argv[1]))
    assert paths, "no shards matched %r" % sys.argv[1]
    out_path = sys.argv[2]
    n_positions = int(sys.argv[3])
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else ITERS

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["g"], r["ply"]))
                except Exception:
                    pass
    picks = [p for p in pick_positions(paths, n_positions) if (p["g"], p["ply"]) not in done]
    print("oracle: %d done, %d to run at %d iterations" % (len(done), len(picks), iters), flush=True)
    tasks = [(p, iters, 31337 + i) for i, p in enumerate(picks)]
    workers = int(os.environ.get("EVALLAB_WORKERS", "4"))
    t0 = time.time()
    with open(out_path, "a") as out, cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(deepen, tasks, chunksize=1)):
            out.write(json.dumps(row) + "\n")
            if (i + 1) % 10 == 0:
                out.flush()
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("oracle complete: %d positions in %.0fs" % (len(tasks), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
