"""15-MINUTE TEST: can CHEAP rollouts beat the NN-driven search value?

The search's value q_search correlates ~0.11 with truth in the early game --
nearly uninformative. This asks whether a rollout under a trivial policy beats
it, which would mean a lever exists that needs no training at all.

Two cheap policies, both with ZERO MCTS iterations:
  random  -- uniform over legal arms, both sides
  greedy  -- highest-damage arm by a 1-ply lookahead, both sides

Scored the only way that counts: Brier against the MEASURED win rate
(label_p), per phase, against the same bars as everything else.
"""
import json, os, random, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labenv  # noqa: F401  (pins encoder flags + PE_NN_WEIGHTS before import)
from poke_engine import State, generate_instructions, monte_carlo_tree_search  # noqa: E402
from playout import arm_to_move  # same arm->move conversion as the labeller

MAX_STEPS = 60


def legal_arms(state, seed):
    """Legal arms for both sides, from a 1-iteration search (cheapest way to
    get the engine's own legal-arm enumeration)."""
    res = monte_carlo_tree_search(state, 0, 1, 1, seed)
    return [m.move_choice for m in res.side_one], [m.move_choice for m in res.side_two]


def rollout(state_str, seed, policy):
    rng = random.Random(seed)
    state = State.from_string(state_str)
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        try:
            a1, a2 = legal_arms(state, (seed * 7919 + step) & 0x7FFFFFFF)
        except Exception:
            break
        if not a1 or not a2:
            break
        if policy == "random":
            p1, p2 = rng.choice(a1), rng.choice(a2)
        else:  # greedy: 1-ply, pick the arm whose expected branch does most damage
            p1, p2 = pick_greedy(state, a1, a2, rng)
        try:
            br = [b for b in generate_instructions(state, arm_to_move(p1), arm_to_move(p2))
                  if b.percentage > 0]
        except Exception:
            break
        if not br:
            break
        state = state.apply_instructions(rng.choices(br, weights=[b.percentage for b in br])[0])
    a1n = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2n = sum(p.hp > 0 for p in state.side_two.pokemon)
    return 1.0 if (a1n > 0 and a2n == 0) else 0.0 if (a2n > 0 and a1n == 0) else 0.5


def pick_greedy(state, a1, a2, rng):
    """Cheapest possible 'good' policy: each side picks the arm that maximises
    opponent hp lost in the modal branch, opponent held at a random arm."""
    def best(arms, other_arms, mine_is_one):
        ref = rng.choice(other_arms)
        best_arm, best_dmg = arms[0], -1.0
        for arm in arms:
            try:
                m1, m2 = (arm, ref) if mine_is_one else (ref, arm)
                br = [b for b in generate_instructions(state, arm_to_move(m1), arm_to_move(m2))
                      if b.percentage > 0]
                if not br:
                    continue
                b = max(br, key=lambda x: x.percentage)
                ns = state.apply_instructions(b)
                foe = ns.side_two if mine_is_one else ns.side_one
                cur = state.side_two if mine_is_one else state.side_one
                dmg = sum(p.hp for p in cur.pokemon) - sum(p.hp for p in foe.pokemon)
                state.reverse_instructions(b) if hasattr(state, "reverse_instructions") else None
                if dmg > best_dmg:
                    best_arm, best_dmg = arm, dmg
            except Exception:
                continue
        return best_arm
    return best(a1, a2, True), best(a2, a1, False)


def work(args):
    row, n, policy = args
    t0 = time.time()
    outs = []
    for j in range(n):
        try:
            outs.append(rollout(row["s"], (row["i"] * 1_000_003 + j * 7919 + 5) & 0x7FFFFFFF, policy))
        except Exception:
            pass
    if not outs:
        return None
    return {"band": row["band"], "p": sum(outs) / len(outs), "truth": row["label_p"],
            "q": row["q_search"], "cs": time.time() - t0}


def main():
    n_pos = int(os.environ.get("NPOS", "180"))
    n_roll = int(os.environ.get("NROLL", "6"))
    policy = os.environ.get("POLICY", "random")
    src = "/Users/sallyliu/pokemon-fast-bot/evallab/data/pl2/out/labels_a.jsonl"
    per_band, rows = n_pos // 3, []
    counts = {"early": 0, "mid": 0, "late": 0}
    for line in open(src):
        d = json.loads(line)
        if counts.get(d["band"], 0) < per_band and d.get("q_search") is not None:
            rows.append(d); counts[d["band"]] += 1
        if all(v >= per_band for v in counts.values()):
            break
    print(f"positions={len(rows)} rollouts={n_roll} policy={policy}", flush=True)
    with mp.Pool(4) as pool:
        res = [r for r in pool.map(work, [(r, n_roll, policy) for r in rows]) if r]
    print(f"{'band':<7}{'n':>5}{'Brier rollout':>15}{'Brier search':>14}{'constant':>10}{'cost/pos':>10}")
    for b in ("early", "mid", "late"):
        R = [r for r in res if r["band"] == b]
        if not R:
            continue
        mt = sum(r["truth"] for r in R) / len(R)
        br_r = sum((r["p"] - r["truth"]) ** 2 for r in R) / len(R)
        br_q = sum((r["q"] - r["truth"]) ** 2 for r in R) / len(R)
        br_c = sum((mt - r["truth"]) ** 2 for r in R) / len(R)
        cs = sum(r["cs"] for r in R) / len(R)
        print(f"{b:<7}{len(R):>5}{br_r:>15.4f}{br_q:>14.4f}{br_c:>10.4f}{cs:>9.2f}s")


if __name__ == "__main__":
    main()
