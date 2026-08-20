"""PER-TURN EVALUATOR AUDIT — what the bot believed vs what actually happens.

Sally 2026-08-16: for every decision of an archived ladder game, play N
bot-vs-bot playouts from that decision's recorded world states and compare the
measured win rate to the evaluator's number.

The comparison is apples-to-apples by construction: playouts start from the
SAME sampled worlds the search saw (chance-weighted), so both numbers answer
"expected result from here, given what we believed about the opponent's team".

    foul-play/.venv/bin/python corrections/audit_game.py <game-dir> \
        [--n 20] [--ms 100] [--net valuenet/nets_v8c/v8c_hz18.bin]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ladder-games", "analysis"))


def one_playout(args):
    """Self-play from a world state to terminal; 1.0 = we win. Mirrors
    mine_value.one_playout, but time-budgeted instead of iteration-budgeted.

    `forced` (5th element, optional) overrides OUR move on the first step —
    the opponent still plays their own search argmax, which is the correct
    simultaneous-move model: they cannot see our deviation. Returns None when
    the forced arm is not legal in this world."""
    state_str, seed, ms, max_steps = args[:4]
    forced = args[4] if len(args) > 4 else None
    opp_sample = args[5] if len(args) > 5 else True
    from poke_engine import State, generate_instructions, monte_carlo_tree_search
    import mine_value as mv
    rng = random.Random(seed)
    state = State.from_string(state_str)
    for step in range(max_steps):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        res = monte_carlo_tree_search(state, ms, 0, 1,
                                      (seed * 7919 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        if opp_sample:
            # Ladder opponents are not argmax machines, and in a SIMULTANEOUS
            # game argmax-vs-argmax isn't even an equilibrium — the solution
            # concept is a mixed strategy. Sampling the opponent in proportion
            # to their visit share plays that mixture. Its own RNG stream keyed
            # by (seed, step) so the draw is identical across audited arms
            # (common random numbers survive the extra randomness).
            orng = random.Random((seed * 31 + step) & 0x7FFFFFFF)
            p2 = orng.choices([m.move_choice for m in s2],
                              weights=[m.visits for m in s2])[0]
        else:
            p2 = max(s2, key=lambda m: m.visits).move_choice
        if step == 0 and forced is not None:
            hit = [m for m in res.side_one if m.move_choice.lower() == forced.lower()]
            if not hit:
                return None          # arm illegal in this world
            p1 = hit[0].move_choice
        else:
            p1 = max(s1, key=lambda m: m.visits).move_choice
        try:
            branches = [b for b in generate_instructions(
                state, mv.arm_to_move(p1), mv.arm_to_move(p2))
                if b.percentage > 0]
        except Exception:
            break
        if not branches:
            break
        pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        nxt = state.apply_instructions(pick)
        if (not any(p.hp > 0 for p in nxt.side_one.pokemon)
                and not any(p.hp > 0 for p in nxt.side_two.pokemon)):
            w = mv.double_ko_winner(state, pick)
            return w if w is not None else 0.5
        state = nxt
    a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
    return 1.0 if (a1 > 0 and a2 == 0) else 0.0 if (a2 > 0 and a1 == 0) else 0.5


def allocate(chances, n):
    """UNIFORM playouts per world (Sally 2026-08-16), i.e. stratified sampling:
    every world gets the same number of draws and the chance weights are applied
    at aggregation instead of at allocation. Equal precision per stratum, and no
    world is left unmeasured. `n` is the per-decision target, so a 16-world
    first-turn search gets 2 each where an 8-world search gets 4."""
    k = max(1, int(round(n / max(len(chances), 1))))
    return {w: k for w in chances}


def arms_audit(a, game_dir, keys, decisions, worlds, chances, tt):
    """POLICY audit: was the played move actually the best of the top arms?

    Every arm of a decision is played out from the SAME worlds with the SAME
    seeds (common random numbers), so arm-vs-arm differences are paired: the
    shared randomness (damage rolls, crits, opponent choices) cancels instead
    of drowning the signal, which is what makes n=40 enough to see a real
    mistake."""
    import statistics
    from concurrent.futures import ProcessPoolExecutor

    plan, tasks, meta = [], [], []
    for i, d in enumerate(keys):
        if i >= len(decisions):
            break
        dec = decisions[i]
        pooled = [(n, sh, sc) for n, sh, sc
                  in tt.pool(dec["ws"], dec["chance"], top=a.arms)
                  if n != "No Move"]
        ranked = [n for n, _, _ in pooled]
        played = dec.get("choice")
        if played and played not in ranked:
            ranked.append(played)
            pooled.append((played, 0.0, float("nan")))
        if not ranked:
            continue
        info = {n: (sh, sc) for n, sh, sc in pooled}
        plan.append((i, d, ranked, played or ranked[0], info))
        alloc = allocate(chances[d], a.n)
        for arm in ranked:
            for w, cnt in alloc.items():
                for j in range(cnt):
                    # seed is independent of the arm: CRN pairing
                    seed = (d * 1000003 + w * 7919 + j * 104729) & 0x7FFFFFFF
                    tasks.append((worlds[d][w], seed, a.ms, a.max_steps,
                                  arm, a.opp == "sample"))
                    meta.append((i, arm, w, j))

    print(f"{len(plan)} decisions x up to {a.arms} arms x ~{a.n} playouts = "
          f"{len(tasks)} playouts @ {a.ms}ms, {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(one_playout, tasks, chunksize=1))
    el = time.time() - t0
    print(f"done in {el:.0f}s ({el/max(len(tasks),1):.2f}s each)", flush=True)

    res = defaultdict(lambda: defaultdict(dict))   # (i,arm) -> world -> j -> out
    for (i, arm, w, j), o in zip(meta, outs):
        if o is not None:
            res[(i, arm)][w][j] = o

    def weighted(i, arm, d):
        tot = num = var = cov = 0.0
        for w, per_j in res[(i, arm)].items():
            if not per_j:
                continue
            c = chances[d][w]
            p = sum(per_j.values()) / len(per_j)
            num += c * p
            var += (c ** 2) * max(p * (1 - p), 0.25 / len(per_j)) / len(per_j)
            tot += c
            cov += c
        ctot = sum(chances[d].values()) or 1.0
        return (num / tot if tot else float("nan"),
                (var ** 0.5) / tot if tot else float("nan"), cov / ctot)

    def paired(i, arm, ref, d):
        """mean(arm - ref) over shared (world, seed) draws."""
        num = den = var = 0.0
        for w in set(res[(i, arm)]) & set(res[(i, ref)]):
            A, C = res[(i, arm)][w], res[(i, ref)][w]
            ds = [A[j] - C[j] for j in set(A) & set(C)]
            if not ds:
                continue
            c = chances[d][w]
            v = statistics.variance(ds) if len(ds) > 1 else 0.0
            num += c * (sum(ds) / len(ds))
            var += (c ** 2) * max(v, 0.25) / len(ds)
            den += c
        if not den:
            return float("nan"), float("nan")
        return num / den, (var ** 0.5) / den

    rows, flags = [], []
    for i, d, ranked, played, info in plan:
        dec = decisions[i]
        arm_rows = []
        for arm in ranked:
            t, se, cov = weighted(i, arm, d)
            dlt, dse = paired(i, arm, played, d)
            sh, sc = info.get(arm, (0.0, float("nan")))
            arm_rows.append({"arm": arm, "truth": t, "se": se, "cov": cov,
                             "delta": dlt, "dse": dse, "share": sh, "eval": sc,
                             "played": arm == played})
        rows.append({"turn": dec["turn"], "decision": d, "played": played,
                     "arms": arm_rows})
        alts = [r for r in arm_rows if not r["played"] and r["delta"] == r["delta"]]
        if alts:
            best = max(alts, key=lambda r: r["delta"])
            z = best["delta"] / best["dse"] if best["dse"] else 0.0
            if best["delta"] > 0 and z >= 2.0:
                flags.append((dec["turn"], played, best, z))

    print(f"\n# policy audit — {os.path.basename(game_dir)}")
    print(f"# {a.n} paired playouts/arm @ {a.ms}ms, {os.path.basename(a.net)}, "
          f"opponent={a.opp}\n")
    print("| Turn | Arm | Visit% | Eval | Playout truth | Δ vs played | z |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        for ar in sorted(r["arms"], key=lambda x: -(x["truth"] if x["truth"] == x["truth"] else -9)):
            mark = " ◀ played" if ar["played"] else ""
            dl = "—" if ar["played"] else f"{ar['delta']:+.2f} ±{ar['dse']:.2f}"
            z = "—" if ar["played"] else f"{abs(ar['delta']/ar['dse']) if ar['dse'] else 0:.1f}"
            ev = "—" if ar["eval"] != ar["eval"] else f"{ar['eval']:.2f}"
            print(f"| {r['turn']} | {ar['arm']}{mark} | {ar['share']*100:.0f}% "
                  f"| {ev} | {ar['truth']:.2f} ±{ar['se']:.2f} | {dl} | {z} |")

    print(f"\n## bad choices (alternative better, z>=2): {len(flags)}")
    if flags:
        print("\n| Turn | Played | Better arm | Δ win rate | z |")
        print("|---|---|---|---|---|")
        for turn, played, best, z in sorted(flags, key=lambda f: -f[3]):
            print(f"| {turn} | {played} | {best['arm']} | "
                  f"{best['delta']:+.2f} ±{best['dse']:.2f} | {z:.1f} |")
    tot_loss = sum(f[2]["delta"] for f in flags)
    print(f"\ntotal win-rate left on the table across flagged decisions: "
          f"{tot_loss:+.2f}")
    regs = []
    for r in rows:
        alts = [x["delta"] for x in r["arms"]
                if not x["played"] and x["delta"] == x["delta"]]
        regs.append(max([d for d in alts if d > 0], default=0.0))
    if regs:
        print(f"raw regret (best alt minus played, clamped at 0): "
              f"mean {sum(regs)/len(regs):+.3f}/turn over {len(regs)} decisions")

    out = a.out or os.path.join(game_dir, "policy_audit.json")
    json.dump(rows, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--n", type=int, default=32,
                    help="playouts per decision, spread UNIFORMLY over its "
                         "worlds (32 -> 4 each at 8 worlds, 2 each at 16)")
    ap.add_argument("--ms", type=int, default=100, help="search ms per step")
    ap.add_argument("--net", default=os.path.join(ROOT, "valuenet/nets_v8c/v8c_hz18.bin"))
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AUDIT_CONCURRENT", "4")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--opp", choices=["sample", "argmax"], default="argmax",
                    help="opponent policy inside playouts: 'sample' draws in "
                         "proportion to visit share (ladder-realistic, and the "
                         "mixed-strategy solution concept); 'argmax' is the "
                         "old best-move-always model")
    ap.add_argument("--collapsed-only", action="store_true",
                    help="audit only decisions where every sampled world "
                         "agrees on the full opponent roster (all 6 revealed)")
    ap.add_argument("--arms", type=int, default=0,
                    help="POLICY audit: measure the top-K arms of every "
                         "decision, not just the position (0 = value audit)")
    a = ap.parse_args()

    import mine_value as mv
    os.environ.update(mv.net_env(a.net))  # inherited by pool workers

    import turn_table as tt
    game_dir = a.game if os.path.isdir(a.game) else tt.find_game(a.game)
    decisions, players, _ = tt.parse(game_dir)

    worlds = defaultdict(dict)
    chances = defaultdict(dict)
    for line in open(os.path.join(game_dir, "worlds.jsonl")):
        r = json.loads(line)
        worlds[r["decision"]][r["world"]] = r["state"]
        chances[r["decision"]][r["world"]] = r["chance"]
    keys = sorted(worlds)
    if len(keys) != len(decisions):
        print(f"WARNING: {len(keys)} world decisions vs {len(decisions)} log "
              f"decisions — pairing by order", file=sys.stderr)
    if a.collapsed_only:
        def roster(st):
            return tuple(m.split(",")[0] for m in st.split("/")[1].split("=")[:6])
        sel = [i for i, d in enumerate(keys) if i < len(decisions)
               and len({roster(st) for st in worlds[d].values()}) == 1]
        keys = [keys[i] for i in sel]
        decisions = [decisions[i] for i in sel]
        print(f"collapsed-only: {len(keys)} of {len(sel) and max(sel)+1} "
              f"decisions have the full roster known", flush=True)

    if a.arms:
        return arms_audit(a, game_dir, keys, decisions, worlds, chances, tt)

    tasks, meta = [], []
    dec_of = decisions
    for i, d in enumerate(keys):
        if i >= len(decisions):
            break
        alloc = allocate(chances[d], a.n)
        ranked = tt.pool(dec_of[i]["ws"], dec_of[i]["chance"], top=1)
        pooled_argmax = dec_of[i].get("choice") or (ranked[0][0] if ranked else None)
        for w, cnt in alloc.items():
            for j in range(cnt):
                seed = (d * 1000003 + w * 7919 + j * 104729) & 0x7FFFFFFF
                tasks.append((worlds[d][w], seed, a.ms, a.max_steps,
                              pooled_argmax, a.opp == "sample"))
                meta.append((i, w))

    print(f"{len(keys)} decisions x ~{a.n} playouts = {len(tasks)} playouts "
          f"@ {a.ms}ms, {a.workers} workers, net={os.path.basename(a.net)}",
          flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(one_playout, tasks, chunksize=1))
    print(f"done in {time.time()-t0:.0f}s ({(time.time()-t0)/len(tasks):.2f}s each)",
          flush=True)

    per = defaultdict(lambda: defaultdict(list))
    for (i, w), o in zip(meta, outs):
        per[i][w].append(o)

    rows = []
    for i, d in enumerate(keys):
        if i not in per:
            continue
        dec = decisions[i]
        ranked = tt.pool(dec["ws"], dec["chance"], top=1)
        ev = ranked[0][2] if ranked else float("nan")
        move = ranked[0][0] if ranked else "?"
        ctot = sum(chances[d].values()) or 1.0
        truth = var = 0.0
        for w, outs_w in per[i].items():
            c = chances[d][w] / ctot
            p = sum(outs_w) / len(outs_w)
            truth += c * p
            var += (c ** 2) * max(p * (1 - p), 0.25 / len(outs_w)) / len(outs_w)
        se = var ** 0.5
        z = abs(ev - truth) / se if se > 0 else 0.0
        rows.append({"decision": d, "turn": dec["turn"], "move": move,
                     "eval": round(ev, 3), "truth": round(truth, 3),
                     "diff": round(truth - ev, 3), "se": round(se, 3),
                     "z": round(z, 1), "n": sum(len(v) for v in per[i].values())})

    print(f"\n# evaluator audit — {os.path.basename(game_dir)}")
    print(f"# {a.n} playouts/decision @ {a.ms}ms self-play, "
          f"{os.path.basename(a.net)}\n")
    print("| Turn | Top move | Eval | Playout truth | Diff | z |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        flag = " **" if r["z"] >= 4 else ""
        print(f"| {r['turn']} | {r['move']} | {r['eval']:.2f} | "
              f"{r['truth']:.2f} ±{r['se']:.2f} | {r['diff']:+.2f}{flag} | "
              f"{r['z']:.1f} |")
    mae = sum(abs(r["diff"]) for r in rows) / max(len(rows), 1)
    bias = sum(r["diff"] for r in rows) / max(len(rows), 1)
    print(f"\nmean |eval - truth| = {mae:.3f} · mean signed = {bias:+.3f} "
          f"· decisions with z>=4: {sum(r['z'] >= 4 for r in rows)}/{len(rows)}")

    out = a.out or os.path.join(game_dir, "audit.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
