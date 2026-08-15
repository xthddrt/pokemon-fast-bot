"""SPSA Tier-1 parameter optimizer for poke-engine's eval + search constants.

One iteration (user-approved plan):
  - draw a random sign vector D in {-1,+1}^37
  - build two arms: theta+ = theta + c_k*scale*D, theta- = theta - c_k*scale*D
    (scale_i = |default_i|, c_0 = 10%, c_k = c0/(k+1)^0.101)
  - spawn two worker pools, each pinned to one arm via PE_TUNE_* env vars
    (read once per process by LazyLock statics in the engine)
  - play `--pairs` corpus team-pairs x 2 orientations = mirrored games at
    `--ms` search time; per turn, each side's move comes from its own pool
  - update every parameter at once:
    theta_i += a_k * net * scale_i * D_i   (net = point diff / games in [-1,1])
    a_k = A0/(k+1+A_STAB)^0.602, clamped to [0.2*default, 3*default]

Opponent-model params (veto keeps, OPPONENT_UCB_EXPLORATION, entry-intent)
are structurally excluded: they were never given PE_TUNE_* env hooks, because
self-play would optimize the human model away. BRANCH_DEPTH is discrete ->
Phase 3.

State (theta, k) persists to spsa_state.json every iteration; per-iteration
rows append to spsa_log.jsonl. Kill and rerun any time - it resumes.

Run:  foul-play/.venv/bin/python spsa_tier1.py            (full: 2500 iters)
      foul-play/.venv/bin/python spsa_tier1.py --smoke    (1 iter, 2 games)
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "foul-play"))

from fp.battle import Battle, Move, Pokemon  # noqa: E402
from fp.helpers import normalize_name  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from poke_engine import State, generate_instructions, monte_carlo_tree_search  # noqa: E402

CORPUS = "/Users/sallyliu/pokemon-ai/synthetic-corpus"
STATE_PATH = os.path.join(ROOT, "spsa_state.json")
LOG_PATH = os.path.join(ROOT, "spsa_log.jsonl")
MAX_STEPS = 120

# (env var, default). scale = |default|; clamp = sorted([0.2*d, 3*d]).
PARAMS = [
    ("PE_TUNE_UCB_EXPLORATION", 1.5197),
    ("PE_TUNE_POKEMON_ALIVE", 30.2166),
    ("PE_TUNE_POKEMON_ALIVE_KNOWN", 42.89),
    ("PE_TUNE_POKEMON_HP", 145.7585),
    ("PE_TUNE_KNOWN_HP_MULTIPLIER", 1.1193),
    ("PE_TUNE_USED_TERA_BASE", 9.9103),
    ("PE_TUNE_USED_TERA_PER_UNREVEALED", 8.1049),
    ("PE_TUNE_USED_TERA_PER_ALIVE", 20.7818),
    ("PE_TUNE_POKEMON_HIDDEN", 21.0012),
    ("PE_TUNE_NO_ANSWER_PENALTY_BASE", 20.1614),
    ("PE_TUNE_NO_ANSWER_PENALTY_HP", 36.9323),
    ("PE_TUNE_ACTIVE_MATCHUP", 32.754),
    ("PE_TUNE_PHANTOM_MATCHUP_WEIGHT", 0.3367),
    ("PE_TUNE_POKEMON_ATTACK_BOOST", 28.3367),
    ("PE_TUNE_POKEMON_DEFENSE_BOOST", 23.1857),
    ("PE_TUNE_POKEMON_SPECIAL_ATTACK_BOOST", 33.0136),
    ("PE_TUNE_POKEMON_SPECIAL_DEFENSE_BOOST", 34.8338),
    ("PE_TUNE_POKEMON_SPEED_BOOST", 32.8303),
    ("PE_TUNE_OPPONENT_BOOST_ASYMMETRY", 1.3888),
    ("PE_TUNE_SUBSTITUTE", 47.3667),
    ("PE_TUNE_SUB_TABLE_SCALE", 1.0759),
    ("PE_TUNE_REFLECT", 22.4585),
    ("PE_TUNE_LIGHT_SCREEN", 23.4615),
    ("PE_TUNE_AURORA_VEIL", 39.9129),
    ("PE_TUNE_TAILWIND", 27.8904),
    ("PE_TUNE_POKEMON_FROZEN", -40.2208),
    ("PE_TUNE_POKEMON_ASLEEP", -39.0348),
    ("PE_TUNE_POKEMON_PARALYZED", -28.3425),
    ("PE_TUNE_POKEMON_TOXIC", -29.3584),
    ("PE_TUNE_POKEMON_POISONED", -9.9491),
    ("PE_TUNE_POKEMON_BURNED", -29.5744),
    ("PE_TUNE_LEECH_SEED", -33.0764),
    ("PE_TUNE_CONFUSION", -22.3871),
    ("PE_TUNE_STEALTH_ROCK", -11.5339),
    ("PE_TUNE_SPIKES", -7.5742),
    ("PE_TUNE_TOXIC_SPIKES", -5.2556),
    ("PE_TUNE_STICKY_WEB", -44.9404),
    ("PE_TUNE_PP_HEAL_SLOPE_BELOW", 4.5673),
    ("PE_TUNE_PP_HEAL_SLOPE_ABOVE", 0.7999),
    ("PE_TUNE_PP_STALL_SLOPE", 0.6817),
    ("PE_TUNE_FUNDAMENTAL_SCALE", 1.0525),
    ("PE_TUNE_ITEM_TABLE_SCALE", 1.1155),
    ("PE_TUNE_REVIVAL_BLESSING", 44.94),
    ("PE_TUNE_SUB_BYPASS_DISCOUNT", 0.6261),
    ("PE_TUNE_BOOST_MULT_1", 1.0488),
    ("PE_TUNE_BOOST_MULT_2", 1.9841),
    ("PE_TUNE_BOOST_MULT_3", 3.5207),
    ("PE_TUNE_BOOST_MULT_4", 3.5686),
    ("PE_TUNE_BOOST_MULT_5", 5.0483),
    ("PE_TUNE_BOOST_MULT_6", 6.6809),
    ("PE_TUNE_BOOST_MULT_NEG_1", -0.9658),
    ("PE_TUNE_BOOST_MULT_NEG_2", -2.157),
    ("PE_TUNE_BOOST_MULT_NEG_3", -2.6769),
    ("PE_TUNE_BOOST_MULT_NEG_4", -4.0305),
    ("PE_TUNE_BOOST_MULT_NEG_5", -4.959),
    ("PE_TUNE_BOOST_MULT_NEG_6", -5.9005),
    ("PE_TUNE_ALL_FAINTED", 633.7376),
    ("PE_TUNE_SAFE_GUARD", 4.58),
    ("PE_TUNE_HEALING_WISH", 24.3641),
    ("PE_TUNE_ILLUSION_INTACT", 7.4016),
    ("PE_TUNE_SIGMOID_SCALE", 0.0172),
    ("PE_TUNE_DUEL_MARGIN", 0.0812),
]

C0 = 0.10       # starting perturbation: 10% of |default|
C_GAMMA = 0.101
A0 = 0.4        # step ~= 0.4 * c0 * scale * net early on
A_STAB = 50
A_ALPHA = 0.602


def _pool_init(env_map):
    os.environ.update(env_map)


def search_best(state_str, search_ms):
    """Return (side_one best move, side_two best move) under this pool's arm."""
    state = State.from_string(state_str)
    res = monte_carlo_tree_search(state, search_ms, threads=1)
    out = []
    for arms in (res.side_one, res.side_two):
        live = [m for m in arms if m.visits > 0]
        if not live:
            return None
        out.append(arm_to_move_string(max(live, key=lambda m: m.visits).move_choice))
    return tuple(out)


def arm_to_move_string(arm):
    s = arm.lower()
    if s in ("no move", "nomove"):
        return "none"
    if s.startswith("switch "):
        return s[len("switch "):]
    return arm


def build_team(team_spec):
    mons = []
    for spec in team_spec:
        pkmn = Pokemon(normalize_name(spec["species"]), spec["level"])
        pkmn.ability = normalize_name(spec.get("ability") or "")
        item = normalize_name(spec.get("item") or "")
        pkmn.item = item or None
        pkmn.moves = [Move(normalize_name(m)) for m in spec["moves"]]
        if spec.get("teraType"):
            pkmn.tera_type = normalize_name(spec["teraType"])
        mons.append(pkmn)
    return mons


def initial_state_string(sidecar_path):
    data = json.load(open(sidecar_path))
    p1 = build_team(data["teams"]["p1"]["team"])
    p2 = build_team(data["teams"]["p2"]["team"])
    battle = Battle("g")
    battle.battle_type = None
    battle.user.active = p1[0]
    battle.user.reserve = p1[1:]
    battle.opponent.active = p2[0]
    battle.opponent.reserve = p2[1:]
    return battle_to_poke_engine_state(battle).to_string()


def alive_counts(state):
    return (
        sum(p.hp > 0 for p in state.side_one.pokemon),
        sum(p.hp > 0 for p in state.side_two.pokemon),
    )


def play_game(pool_plus, pool_minus, state_str, plus_is_side_one, search_ms, seed):
    """Return points for the PLUS arm: 1 win, 0.5 draw/cap, 0 loss."""
    rng = random.Random(seed)
    for _ in range(MAX_STEPS):
        state = State.from_string(state_str)
        a1, a2 = alive_counts(state)
        if a1 == 0 or a2 == 0:
            break
        fut_p = pool_plus.submit(search_best, state_str, search_ms)
        fut_m = pool_minus.submit(search_best, state_str, search_ms)
        res_p, res_m = fut_p.result(), fut_m.result()
        if res_p is None or res_m is None:
            break
        s1_move = res_p[0] if plus_is_side_one else res_m[0]
        s2_move = res_m[1] if plus_is_side_one else res_p[1]
        try:
            branches = generate_instructions(state, s1_move, s2_move)
        except Exception:
            break
        weights = [max(b.percentage, 0.0) for b in branches]
        if not branches or sum(weights) <= 0:
            break
        state = state.apply_instructions(rng.choices(branches, weights=weights)[0])
        state_str = state.to_string()
    a1, a2 = alive_counts(State.from_string(state_str))
    if a1 > 0 and a2 == 0:
        return 1.0 if plus_is_side_one else 0.0
    if a2 > 0 and a1 == 0:
        return 0.0 if plus_is_side_one else 1.0
    return 0.5


def arm_env(theta, c_map, signs, direction):
    return {
        name: repr(theta[name] + direction * c_map[name] * abs(default) * signs[name])
        for name, default in PARAMS
    }


def load_state():
    if os.path.exists(STATE_PATH):
        d = json.load(open(STATE_PATH))
        theta, joined = d["theta"], d.get("joined", {})
        frozen = set(d.get("frozen", []))
        # params added after a run started join at their default, with a
        # fresh per-parameter schedule starting at their join iteration
        for name, default in PARAMS:
            if name not in theta:
                theta[name] = default
                joined[name] = d["k"]
            joined.setdefault(name, 0)
        return d["k"], theta, joined, frozen
    return (0, {name: default for name, default in PARAMS},
            {name: 0 for name, _ in PARAMS}, set())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2500)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--ms", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--files", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iterations, args.pairs, args.workers = 1, 1, 2

    for name, _ in PARAMS:
        assert name not in os.environ, "parent env must be clean: " + name

    if args.files:
        files = [l.strip() for l in open(args.files) if l.strip()]
    else:
        files = sorted(glob.glob(CORPUS + "/*.teams.json"))
    assert files, "corpus sidecars not found"
    k, theta, joined, frozen = load_state()
    start_k = k
    print("params: %d | resuming at k=%d | sidecars: %d" % (len(PARAMS), k, len(files)), flush=True)

    while k < args.iterations:
        it_rng = random.Random(910_000 + k)
        signs = {name: it_rng.choice((-1, 1)) for name, _ in PARAMS}
        # per-parameter schedules: each decays by its own age since joining,
        # so late additions explore at full perturbation while converged
        # parameters keep their small mature steps
        age = {name: k - joined[name] for name, _ in PARAMS}
        c_map = {name: C0 / (age[name] + 1) ** C_GAMMA for name, _ in PARAMS}
        a_map = {name: A0 / (age[name] + 1 + A_STAB) ** A_ALPHA for name, _ in PARAMS}
        # frozen parameters: pinned at theta, no perturbation, no update
        for name in frozen:
            c_map[name] = 0.0
            a_map[name] = 0.0
        c_k = min(c_map.values())  # logged: mature-cohort perturbation
        pair_files = it_rng.sample(files, args.pairs)

        env_plus = arm_env(theta, c_map, signs, +1)
        env_minus = arm_env(theta, c_map, signs, -1)
        points = 0.0
        games = args.pairs * 2
        with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_pool_init,
                                    initargs=(env_plus,)) as pool_plus, \
             cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_pool_init,
                                    initargs=(env_minus,)) as pool_minus, \
             cf.ThreadPoolExecutor(max_workers=games) as driver:
            futs = []
            for i, f in enumerate(pair_files):
                s = initial_state_string(f)
                for orient in (True, False):  # mirrored: plus plays both sides
                    futs.append(driver.submit(
                        play_game, pool_plus, pool_minus, s, orient,
                        args.ms, 77_000 + k * 100 + i * 2 + orient))
            points = sum(f.result() for f in futs)

        net = (points - (games - points)) / games  # in [-1, 1]
        for name, default in PARAMS:
            lo, hi = sorted((0.2 * default, 3.0 * default))
            theta[name] = min(
                hi, max(lo, theta[name] + a_map[name] * net * abs(default) * signs[name]))

        k += 1
        json.dump({"k": k, "theta": theta, "joined": joined, "frozen": sorted(frozen)},
                  open(STATE_PATH, "w"), indent=1)
        with open(LOG_PATH, "a") as log:
            log.write(json.dumps({
                "k": k, "c_k": round(c_k, 5), "c_new": round(max(c_map.values()), 5), "net": net,
                "points_plus": points, "games": games,
                "files": [os.path.basename(f) for f in pair_files],
                "signs": signs, "theta": {n: round(v, 4) for n, v in theta.items()},
            }) + "\n")
        if k % 25 == 0 or args.smoke:
            print("iter %d/%d | net %+0.2f | c_k %.3f" % (k, args.iterations, net, c_k), flush=True)

    print("done: iterations %d -> %d, state in %s" % (start_k, k, STATE_PATH))


if __name__ == "__main__":
    main()
