"""PHANTOM-CUTOFF DUEL — real hidden-information random battles (Sally 2026-08-17).

Agent A: the launched search behavior (pooled worlds over belief-sampled
opponents, argmax) at duel settings. Agent B: identical, plus the engine's
phantom cutoff — any node where a sampled never-revealed mon is active is a
leaf. Both play under TRUE partial information: the harness holds the real
state; each agent sees its own team, the opponent's revealed info, and a
belief sampler. Mirrored team pairs (same matchup, sides swapped).

Settings per Sally: 4 worlds x 1000ms, argmax both sides. The tera gate is
omitted on BOTH sides (symmetric; the duel measures the phantom-cut delta).

  python opp_duel.py --pairs 2 --ms 150 --tag smoke        # local canary
  python opp_duel.py --pairs 200 --ms 1000 --tag duel1     # box run
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

MAX_TURNS = 300
MAX_CONCURRENT = int(os.environ.get("MINE_CONCURRENT", "4"))


def _agent_choice(true_state, side, revealed_opp, worlds, ms, phantom, rng):
    """Pooled argmax over belief-sampled worlds. side = 1 or 2 (which side
    this agent plays). revealed_opp = what the OPPONENT has revealed to us."""
    import opp_model_lab as lab
    from poke_engine import State, monte_carlo_tree_search
    opp_side = 2 if side == 1 else 1
    s1, s2, tail = lab.split_state(true_state)
    votes = {}
    for w in range(worlds):
        seed = rng.randrange(1 << 30)
        if opp_side == 1:
            mons, mask = lab.blind_side_masked(s1[0], revealed_opp, seed)
            world = lab.join_state((mons, s1[1]), s2, tail)
            kw = {"phantom_side_one": mask} if phantom else {}
        else:
            mons, mask = lab.blind_side_masked(s2[0], revealed_opp, seed)
            world = lab.join_state(s1, (mons, s2[1]), tail)
            kw = {"phantom_side_two": mask} if phantom else {}
        res = monte_carlo_tree_search(State.from_string(world), ms, 0, 1,
                                      rng.randrange(1 << 30), **kw)
        arms = res.side_one if side == 1 else res.side_two
        for m in arms:
            if m.visits > 0:
                votes[m.move_choice] = votes.get(m.move_choice, 0) + m.visits
    return max(votes, key=votes.get)


def _reveal_from_choice(rev, true_side_mons, active_species, choice, lab):
    c = choice.lower()
    if c.startswith("switch "):
        want = c[7:].replace("-", "").replace(" ", "")
        for m in true_side_mons:
            sp = lab.mon_species(m)
            if sp.lower().replace("-", "") == want:
                rev.switch_in(sp)
                return
    elif c not in ("none", "no move"):
        base = c[:-5] if c.endswith("-tera") else c
        rev.used_move(active_species, base)
        if c.endswith("-tera"):
            for m in true_side_mons:
                if lab.mon_species(m) == active_species:
                    rev.terastallized(active_species,
                                      lab.mon_fields(m)[lab.MON_TERA_TYPE])


def play_game(args):
    (pair_seed, swap, worlds, ms) = args
    import opp_model_lab as lab
    import mine_value as mv
    from poke_engine import State, generate_instructions
    import run_duels as rd
    from fp.search import ps_teams
    import tempfile
    rng = random.Random(pair_seed * 31 + swap)
    lab.fill_pool(n_teams=15, seed=pair_seed * 7 + 3)

    teams = {}
    for k, salt in (("p1", 0), ("p2", 1)):
        ps_teams.seed(pair_seed * 977 + salt)
        teams[k] = {"team": ps_teams.random_team()}
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"teams": teams}, tf)
    tf.close()
    true_state = rd.opening_state(tf.name, "p1", tf.name, "p2")
    os.unlink(tf.name)

    # swap=1: agent B takes side 1 (mirror seating)
    b_side = 1 if swap else 2
    rev = {1: lab.Revealed(), 2: lab.Revealed()}
    state = State.from_string(true_state)
    for t in range(MAX_TURNS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        ss = state.to_string()
        s1, s2, _ = lab.split_state(ss)
        act1 = lab.mon_species(s1[0][int(str(state.side_one.active_index))]
                               if str(state.side_one.active_index).isdigit()
                               else s1[0][0])
        act2 = lab.mon_species(s2[0][int(str(state.side_two.active_index))]
                               if str(state.side_two.active_index).isdigit()
                               else s2[0][0])
        # anything on the field has been seen
        rev[1].switch_in(act1)
        rev[2].switch_in(act2)
        choices = {}
        for side in (1, 2):
            choices[side] = _agent_choice(
                ss, side, rev[2 if side == 1 else 1], worlds, ms,
                phantom=(side == b_side), rng=rng)
        _reveal_from_choice(rev[1], s1[0], act1, choices[1], lab)
        _reveal_from_choice(rev[2], s2[0], act2, choices[2], lab)
        try:
            branches = [b for b in generate_instructions(
                state, mv.arm_to_move(choices[1]), mv.arm_to_move(choices[2]))
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
            s1_win = w if w is not None else 0.5
            return {"pair": pair_seed, "swap": swap,
                    "b_score": s1_win if b_side == 1 else 1 - s1_win, "turns": t}
        state = nxt
    a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
    s1_win = 1.0 if (a1 > 0 and a2 == 0) else 0.0 if (a2 > 0 and a1 == 0) else 0.5
    return {"pair": pair_seed, "swap": swap,
            "b_score": s1_win if b_side == 1 else 1 - s1_win, "turns": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--ms", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=5_000_001)
    ap.add_argument("--tag", default="oppduel")
    ap.add_argument("--label-bin",
                    default=os.path.join(ROOT, "valuenet/nets_v8c/v8c_s1.bin"))
    a = ap.parse_args()
    import mine_value as mv
    if "PE_NN_WEIGHTS" not in os.environ:
        for k, v in mv.net_env(a.label_bin).items():
            if k.startswith("PE_"):
                os.environ[k] = v
    import subprocess as sp
    chk = sp.run([mv.LEAF_PROF, "logits", "/dev/null"], env=dict(os.environ),
                 capture_output=True, text=True)
    if "valuenet: loaded" not in (chk.stderr + chk.stdout):
        raise SystemExit("FATAL: net failed to load")
    print("net verified", flush=True)

    tasks = [(a.seed_base + i, sw, a.worlds, a.ms)
             for i in range(a.pairs) for sw in (0, 1)]
    t0 = time.time()
    import concurrent.futures as cf
    out = []
    with cf.ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        for r in pool.map(play_game, tasks, chunksize=1):
            out.append(r)
            n = len(out)
            bs = sum(x["b_score"] for x in out)
            print(f"  {n}/{len(tasks)} games, B score {bs:.1f} "
                  f"({bs/n*100:.1f}%) {time.time()-t0:.0f}s", flush=True)
    work = os.path.join(HERE, "_mine_work", a.tag)
    os.makedirs(work, exist_ok=True)
    json.dump(out, open(os.path.join(work, "duel.json"), "w"), indent=1)
    n = len(out)
    bs = sum(x["b_score"] for x in out)
    import math
    se = math.sqrt(0.25 / n)
    print(f"\nRESULT B(phantom-cut) vs A: {bs:.1f}/{n} = {bs/n*100:.2f}% "
          f"(+-{se*100:.1f}%) elo {400*math.log10((bs/n)/(1-bs/n+1e-9)):+.0f}",
          flush=True)


if __name__ == "__main__":
    main()
