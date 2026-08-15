"""PLAYOUT-AVERAGED VALUE LABELS for the eval lab.

WHY THIS EXISTS
    Today every training position carries ONE label: the outcome of the single
    game it happened to sit in. That label is a Bernoulli draw with variance up
    to 0.25 around the position's true value, and it is the same draw for every
    position in the game -- a 40-ply game contributes 40 positions all labelled
    with one coin flip. The value head can only learn the features that survive
    that much noise, which is why it converges to material counting.

    This module replaces that label with an EMPIRICAL WIN RATE: from the
    position, play the game to termination N times under a FIXED policy on BOTH
    sides, and average. Label variance falls ~N-fold, and the label becomes an
    estimate of V^pi(s) -- the value of the position under the policy that is
    actually played -- instead of a sample of it.

WHAT "FIXED POLICY" MEANS HERE, EXACTLY
    Every decision in a playout is `monte_carlo_tree_search(s, 0, ITERS, 1, seed)`
    followed by argmax-visits, on BOTH sides. NO forced-random plies, NO visit
    temperature, NO epsilon -- EPS=0 everywhere. The diversity knobs that
    generate.py uses to spread a corpus are exactly the label corruption being
    removed here: a position whose continuation contains an 8%-epsilon blunder
    is not being labelled with its own value.

    The randomness that REMAINS is the game's own: `generate_instructions`
    branches (damage rolls, accuracy, crits, secondary effects) are sampled by
    their true probabilities, and the MCTS seed varies per playout so the
    policy's own search noise is averaged over rather than frozen at one draw.
    That is the distribution the label is supposed to be an expectation over.

LABEL CONVENTION (matches generate.py and dataset.py, deliberately)
    p = P(side_one wins) + 0.5 * P(tie), from the SAME alive-count rule
    generate.py uses. dataset.py's `y` is a side-one-win indicator, so
    `label_p` is a drop-in replacement for it and needs no sign convention
    anywhere else in the lab.

    `se` is the standard error of the MEAN over the N playout outcomes
    (ddof=1), not sqrt(p(1-p)/N): ties score 0.5, so the outcome variable is not
    Bernoulli and the binomial formula would misstate it.

OUTPUT (JSONL, one line per labelled position; the STATE lives in the shard and
in positions.jsonl under the same (game id, ply) key and is not duplicated here)
    {"i": idx, "g": game_id, "ply": ply, "band": "early|mid|late",
     "label_p": float, "n_playouts": int, "se": float,
     "y_single": the single game outcome this position carries TODAY,
     "q_search": the generating search's own root value at this position,
     "wins": int, "losses": int, "ties": int, "trunc": int,
     "steps": mean playout length, "cs": core-seconds spent on this position}

USAGE
    python playout.py sample <shard_glob> <out.jsonl> <n_positions> [seed]
    python playout.py run    <positions.jsonl> <out.jsonl> <n_playouts> [start] [count] [iters]
ENV
    EVALLAB_WORKERS   process pool size (default 4)
"""

import concurrent.futures as cf
import glob
import gzip
import json
import os
import random
import sys
import time
import zlib

import labenv  # noqa: F401  (pins encoder + engine env; must be first)
from poke_engine import State, generate_instructions, monte_carlo_tree_search  # noqa: E402

MAX_STEPS = 200
ITERS = int(os.environ.get("EVALLAB_ITERS", "2000"))

# Ply bands. Sally's priority is EARLY and MIDGAME, so the allocation is
# deliberately not uniform over decisions (the corpus itself is 27/39/34).
BANDS = (("early", 0, 10), ("mid", 10, 25), ("late", 25, 10 ** 9))
ALLOC = {"early": 0.40, "mid": 0.40, "late": 0.20}
# Per-game, PER-BAND caps. A global per-game cap cannot hit a band quota that
# exceeds the band's natural share of decisions (early plies are 27% of the
# corpus but 40% of the target), so the cap is per band and the pool is filled
# band-wise. Positions inside one game are correlated, which is what these caps
# limit; they are sized to leave every band's pool comfortably above its quota.
PER_GAME_CAP = {"early": 5, "mid": 6, "late": 4}


def band_of(ply):
    for name, lo, hi in BANDS:
        if lo <= ply < hi:
            return name
    return "late"


def arm_to_move(a):
    a = a.lower()
    if a in ("no move", "nomove"):
        return "none"
    return a[7:] if a.startswith("switch ") else a


def root_value(t):
    """The generating search's own value for side one at this position:
    visits-weighted mean of the per-arm avg_scores. Recorded for free from the
    shard, and it is the natural third point of comparison against the single
    outcome and the playout average."""
    n, q = t.get("n") or {}, t.get("q") or {}
    tot = sum(n.get(a, 0) for a in q)
    if tot <= 0:
        return None
    return round(sum(n.get(a, 0) * q[a] for a in q) / tot, 5)


# --------------------------------------------------------------------------
# 1. position selection
# --------------------------------------------------------------------------

def stream_shard(paths):
    for p in sorted(paths):
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                yield row


def sample(paths, out_path, n_positions, seed=0):
    """Stratified-by-ply selection over the corpus.

    TWO passes, because the state strings are ~2.5 kB each and 200,000 of them
    do not belong in memory on an 8.6 GB box: pass 1 chooses (game, ply) KEYS
    only, pass 2 streams the shard again and materialises just those rows.

    Games are NOT filtered by the train/holdout split -- `labenv.is_holdout`
    still applies downstream on the recorded game id, so ~10% of the labelled
    positions land in the holdout set and give the lab a low-noise EVALUATION
    target as well as a training one.

    The final file is written in SHUFFLED order (via a byte-offset rewrite, so
    memory stays O(keys) not O(states)). That is load-bearing: boxes are given
    contiguous index slices, and the canary is the first 200 rows -- both are
    only representative if neighbouring rows are not from the same game.
    """
    rng = random.Random(seed)
    quota = {b: int(round(ALLOC[b] * n_positions)) for b in ALLOC}
    pools = {b[0]: [] for b in BANDS}
    n_games = 0
    for row in stream_shard(paths):
        n_games += 1
        turns = row["t"]
        per = {b[0]: [] for b in BANDS}
        for k, t in enumerate(turns):
            if len(t.get("q") or {}) < 2:
                continue  # not a real decision: nothing to be uncertain about
            per[band_of(k)].append(k)
        for b, ks in per.items():
            rng.shuffle(ks)
            for k in ks[:PER_GAME_CAP[b]]:
                pools[b].append((row["g"], k))
    chosen = []
    for b, pool in pools.items():
        rng.shuffle(pool)
        if len(pool) < quota[b]:
            print("WARNING: band %s pool %d < quota %d (per-game cap too tight)"
                  % (b, len(pool), quota[b]), flush=True)
        chosen += pool[:quota[b]]
    want = set(chosen)
    print("pass1: %d games, pools=%s, chose %d keys"
          % (n_games, {b: len(pools[b]) for b in pools}, len(want)), flush=True)

    # pass 2: materialise, corpus order, remembering each row's byte offset
    tmp = out_path + ".unshuffled"
    offs = {}
    with open(tmp, "w") as f:
        for row in stream_shard(paths):
            for k, t in enumerate(row["t"]):
                if (row["g"], k) not in want:
                    continue
                offs[(row["g"], k)] = f.tell()
                f.write(json.dumps({
                    "g": row["g"], "ply": k, "band": band_of(k), "s": t["s"],
                    "y_single": float(row["outcome"]), "q_search": root_value(t),
                }) + "\n")

    order = [k for k in chosen if k in offs]
    rng.shuffle(order)
    from collections import Counter
    bands = Counter()
    with open(tmp) as src, open(out_path, "w") as out:
        for i, k in enumerate(order):
            src.seek(offs[k])
            r = json.loads(src.readline())
            r["i"] = i
            bands[r["band"]] += 1
            out.write(json.dumps(r) + "\n")
    os.remove(tmp)
    print("sampled %d positions from %d games -> %s  bands=%s"
          % (len(order), n_games, out_path, dict(bands)), flush=True)


# --------------------------------------------------------------------------
# 2. playouts
# --------------------------------------------------------------------------

def one_playout(state_str, base_seed, iters):
    """Play to termination, fixed argmax-visit policy on both sides, EPS=0.
    Returns (outcome, steps, truncated)."""
    rng = random.Random(base_seed)
    state = State.from_string(state_str)
    trunc = 1
    step = 0
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            trunc = 0
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            trunc = 0
            break
        res = monte_carlo_tree_search(state, 0, iters, 1, (base_seed * 7919 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        p1 = max(s1, key=lambda m: m.visits).move_choice
        p2 = max(s2, key=lambda m: m.visits).move_choice
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
    return outcome, step + 1, trunc


def label_one(args):
    """N playouts from one position -> mean, se, and the diagnostics."""
    row, n_playouts, iters = args
    t0 = time.time()
    try:
        outs, steps, trunc = [], 0, 0
        for j in range(n_playouts):
            base = (int(row["i"]) * 1_000_003 + j * 7919 + 11) & 0x7FFFFFFF
            o, st, tr = one_playout(row["s"], base, iters)
            outs.append(o)
            steps += st
            trunc += tr
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # pragma: no cover
        # BaseException, not Exception: pyo3 raises PanicException (a
        # BaseException) when the Rust engine panics -- e.g. the reachable
        # "Invalid rest_turns value: -1" at generate_instructions.rs:10487.
        # Caught as Exception it escapes the worker, then fails to pickle back
        # to the parent ("Can't pickle <class 'pyo3_runtime.PanicException'>"),
        # which tears down the whole ProcessPoolExecutor and kills the box.
        # One bad position must cost one position, not a 64-core slice.
        # repr() is stringified here so the returned row is always picklable.
        return {"i": row["i"], "error": repr(e)[:300]}
    n = len(outs)
    mean = sum(outs) / n
    var = sum((o - mean) ** 2 for o in outs) / (n - 1) if n > 1 else 0.0
    # The state string is NOT echoed back: it is 2.5 kB and it already exists,
    # byte-identical, in the shard and in positions.jsonl, both keyed by the
    # same (game id, ply). Echoing it would make this file 40x larger for
    # nothing. dataset.py joins on that key.
    out = {k: v for k, v in row.items() if k != "s"}
    out.update({
        "label_p": round(mean, 6),
        "n_playouts": n,
        "se": round((var / n) ** 0.5, 6),
        "wins": sum(1 for o in outs if o == 1.0),
        "losses": sum(1 for o in outs if o == 0.0),
        "ties": sum(1 for o in outs if o == 0.5),
        "trunc": trunc,
        "steps": round(steps / n, 2),
        "cs": round(time.time() - t0, 2),
    })
    return out


def run(pos_path, out_path, n_playouts, start=0, count=None, iters=ITERS):
    # Parse ONLY the assigned slice. A 1M-position file is ~3.5 GB of state
    # strings; materialising all of it and then forking a 64-way pool from the
    # parent is an OOM, and the rows outside the slice are never used. Line k
    # of the file is position i=k by construction (corpus_select.py / sample()
    # both write in shuffled order and index after shuffling).
    end = None if count is None else start + count
    rows = []
    with open(pos_path) as f:
        for k, line in enumerate(f):
            if k < start:
                continue
            if end is not None and k >= end:
                break
            rows.append(json.loads(line))
    end = start + len(rows)

    # Resume: keep the parseable prefix, rewrite, append after. Same contract as
    # generate.py -- a killed box costs only the positions it had in flight.
    done = set()
    if os.path.exists(out_path):
        clean = []
        try:
            with open(out_path, errors="replace") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "i" in r and "label_p" in r:
                        done.add(r["i"])
                        clean.append(line)
        except (OSError, zlib.error):
            pass
        with open(out_path + ".tmp", "w") as f:
            f.writelines(clean)
        os.replace(out_path + ".tmp", out_path)
    tasks = [(r, n_playouts, iters) for r in rows if r["i"] not in done]
    print("positions %d..%d: %d done, %d to run, N=%d iters=%d"
          % (start, end, len(done), len(tasks), n_playouts, iters), flush=True)

    workers = int(os.environ.get("EVALLAB_WORKERS", "4"))
    t0 = time.time()
    n, cs, bad = 0, 0.0, 0
    with open(out_path, "a") as f, cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(label_one, tasks, chunksize=1):
            f.write(json.dumps(r) + "\n")
            n += 1
            bad += "error" in r
            cs += r.get("cs", 0.0)
            if n % 25 == 0:
                f.flush()
                el = time.time() - t0
                print("  %d/%d  %.0fs elapsed  %.3f core-s/playout  eta %.0fs  errors=%d"
                      % (n, len(tasks), el, cs / max(n * n_playouts, 1),
                         el / n * (len(tasks) - n), bad), flush=True)
    print("DONE %d positions  %.0fs wall  %.1f core-h  %.4f core-s/playout  errors=%d"
          % (n, time.time() - t0, cs / 3600.0, cs / max(n * n_playouts, 1), bad), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        paths = sorted(glob.glob(sys.argv[2]))
        assert paths, "no shards matched %r" % sys.argv[2]
        sample(paths, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
            int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None,
            int(sys.argv[7]) if len(sys.argv) > 7 else ITERS)
    else:
        raise SystemExit(__doc__)
