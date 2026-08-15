"""Two-tier flip verification (HAMMER_SPEC.md Part 2, corrections/verify_flip.py).

Tier (a) FAST PRECHECK (~ms, net-level, importable): mean net value of the
ruled move's successor states vs every alternative option's successors across
the recorded worlds; flip candidate when the ruled move ranks #1.

Tier (b) REAL TEST (seconds): the actual production search — poke_engine MCTS
on each recorded world at a FIXED iteration budget (threads=1, fixed seed,
pool of 4 spawned workers so each fresh process reads PE_NN_WEIGHTS/PE_TUNE_*
once) — pooled and selected by the REAL foul-play pipeline:
fp.search.selection.select_move_from_mcts_results with --selection-argmax-only
semantics + the live tera gate (0.0015/0.3333/0.003) and its fallback rule
(the actual _apply_argmax_tera_gate, not a reimplementation).
PASS = final choice == ruled_move.

CLI (real test):
  .venv python verify_flip.py --bin <weights.bin> [--constants sidecar.json]
      [--ledger PATH] [--ids id1,id2] [--iters 100000] [--workers 4] [--seed 7]
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time

import common


# ---------------------------------------------------------------------------
# tier (a): fast precheck
# ---------------------------------------------------------------------------

def build_precheck_cache(entries, vocab, device="cpu"):
    """One-time successor fan + encode for every ledger entry and every legal
    option (ruled + alternatives). Returns
    {entry_id: {'ruled': str, 'options': {opt: (torch_batch, weights)}}}."""
    import numpy as np

    caches = {}
    for e in entries:
        per_opt = common.build_successors(e, options=None)
        if e["ruled_move"] not in per_opt:
            raise SystemExit("entry %s: ruled move %r not legal in any "
                             "recorded world (options: %s)"
                             % (e["id"], e["ruled_move"], sorted(per_opt)))
        opts = {}
        for opt, slot in per_opt.items():
            tb = common.to_torch(common.encode_batch(slot["states"], vocab),
                                 device)
            w = np.asarray(slot["weights"], dtype=np.float64)
            opts[opt] = (tb, w / w.sum())
        caches[e["id"]] = {"ruled": e["ruled_move"], "options": opts}
    return caches


def precheck(net, caches):
    """{entry_id: {'flip': bool, 'values': {opt: weighted mean value}}}."""
    out = {}
    for eid, c in caches.items():
        vals = {}
        for opt, (tb, w) in c["options"].items():
            v = common.net_values(net, tb).cpu().numpy()
            vals[opt] = float((v * w).sum())
        ruled_v = vals[c["ruled"]]
        flip = all(ruled_v >= v for o, v in vals.items() if o != c["ruled"])
        out[eid] = {"flip": flip, "values": vals}
    return out


# ---------------------------------------------------------------------------
# tier (b): real production-search flip test
# ---------------------------------------------------------------------------

def _search_worker(task):
    """Runs in a SPAWNED process: poke_engine resolves PE_NN_WEIGHTS and the
    PE_TUNE_* constants from the env once, per process."""
    state_str, iters, seed = task
    import poke_engine as pe

    st = pe.State.from_string(state_str)
    res = pe.monte_carlo_tree_search(st, 10, iterations=iters, threads=1,
                                     seed=seed)
    return {
        "total_visits": res.total_visits,
        "side_one": [(o.move_choice, o.visits, o.total_score,
                      o.total_score_sq) for o in res.side_one],
        "side_two": [(o.move_choice, o.visits, o.total_score,
                      o.total_score_sq) for o in res.side_two],
        "root_pairs": [[(int(v), float(t)) for v, t in row]
                       for row in res.root_pairs],
    }


class _Opt:
    __slots__ = ("move_choice", "visits", "total_score", "total_score_sq")

    def __init__(self, t):
        (self.move_choice, self.visits, self.total_score,
         self.total_score_sq) = t


class _Res:
    __slots__ = ("total_visits", "side_one", "side_two", "root_pairs")

    def __init__(self, d):
        self.total_visits = d["total_visits"]
        self.side_one = [_Opt(t) for t in d["side_one"]]
        self.side_two = [_Opt(t) for t in d["side_two"]]
        self.root_pairs = d["root_pairs"]


def production_select(world_results, entry):
    """The REAL selection pipeline: pooled visit-share argmax + the live
    argmax tera gate, via fp.search.selection (imported, not reimplemented)."""
    if common.FOULPLAY not in sys.path:
        sys.path.insert(0, common.FOULPLAY)
    from config import FoulPlayConfig
    from fp.search.selection import select_move_from_mcts_results

    FoulPlayConfig.selection_argmax_only = True
    FoulPlayConfig.tera_gate_score_per_mon = common.TERA_GATE["per_mon"]
    FoulPlayConfig.tera_gate_visit_frac = common.TERA_GATE["visit_frac"]
    FoulPlayConfig.tera_gate_opp_tera_bonus = common.TERA_GATE["opp_tera_bonus"]
    return select_move_from_mcts_results(
        world_results,
        revealed_opponent_names=None,
        opp_alive=entry.get("opp_alive", common.OPP_TEAM_SIZE),
        opp_unrevealed=entry.get("opp_unrevealed", 0),
        opp_tera_used=entry.get("opp_tera_used", False),
    )


def flip_test(bin_path, constants, entries, iters=600000, workers=4, seed=7,
              quiet=False):
    """Run the real test for every entry against one weights .bin.
    constants: dict whose PE_TUNE_* keys become the engine's search constants
    (pass the net's sidecar contents). Returns ({entry_id: result}, wall_s)."""
    bin_path = os.path.abspath(bin_path)
    if not os.path.isfile(bin_path):
        raise SystemExit("flip_test: no such bin: %s" % bin_path)
    os.environ["PE_NN_WEIGHTS"] = bin_path
    for k, v in constants.items():
        if k.startswith("PE_TUNE_"):
            os.environ[k] = str(v)

    tasks, owner = [], []
    for e in entries:
        for i, w in enumerate(e["states"]):
            tasks.append((w["state"], iters, seed + i))
            owner.append(e["id"])

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        outs = pool.map(_search_worker, tasks, chunksize=1)
    wall = time.time() - t0

    results = {}
    for e in entries:
        world_results = []
        idx = 0
        for i, (oid, out) in enumerate(zip(owner, outs)):
            if oid != e["id"]:
                continue
            world_results.append((_Res(out), e["states"][idx]["chance"], idx))
            idx += 1
        choice = production_select(world_results, e)
        results[e["id"]] = {
            "pass": choice == e["ruled_move"],
            "choice": choice,
            "ruled": e["ruled_move"],
            "iters": iters,
            "worlds": len(world_results),
        }
        if not quiet:
            print("[flip] %s: chose %r vs ruled %r -> %s"
                  % (e["id"], choice, e["ruled_move"],
                     "PASS" if choice == e["ruled_move"] else "FAIL"))
    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--constants", default=None,
                    help="sidecar json; default: <bin stem>.constants.json")
    ap.add_argument("--ledger", default=common.LEDGER)
    ap.add_argument("--ids", default=None)
    # 600k ~= the production 4500ms budget (measured ~130-160k iters/s/worker);
    # at 100k the unhammered v6nopol already picked the yraf-T16 ruled move.
    ap.add_argument("--iters", type=int, default=600000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the selection pipeline's own log lines")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    consts_path = args.constants or (
        os.path.splitext(args.bin)[0] + ".constants.json")
    if not os.path.isfile(consts_path):
        raise SystemExit("no constants sidecar at %s (pass --constants); "
                         "running the engine's default constants would test "
                         "the wrong search" % consts_path)
    constants = json.load(open(consts_path))

    entries = common.read_ledger(args.ledger)
    if args.ids:
        want = set(args.ids.split(","))
        entries = [e for e in entries if e["id"] in want]
    if not entries:
        raise SystemExit("no ledger entries to verify (%s)" % args.ledger)

    results, wall = flip_test(args.bin, constants, entries, args.iters,
                              args.workers, args.seed)
    npass = sum(r["pass"] for r in results.values())
    print("verify_flip: %d/%d PASS in %.1fs (%d iters/world, %d workers)"
          % (npass, len(results), wall, args.iters, args.workers))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
