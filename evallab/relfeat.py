"""PRECOMPUTED RELATIONAL FEATURES -- the thing the audit says the net cannot
compute for itself.

WHY THESE COLUMNS GO WHERE THEY GO
  The per-mon block is fed through a SHARED MLP and then SUM-POOLED over the
  bench (`train.py:pool`, BENCH_POOL=sum).  Sum-pooling is exactly the operation
  that destroys a relation: it can represent "how much bench is left" and can
  never represent "which bench mon answers their active".  So every column here
  lands in the SIDE block (per side, concatenated, never pooled) or the GLOBAL
  block (one per position).  Nothing is added to the per-mon block on purpose.

TWO BLOCKS
  gx  16 global columns, all strictly BETWEEN the two active Pokemon
  sx  16 side columns, computed twice (ours, theirs) exactly like `sf`

SIDE-SWAP SAFETY
  Both blocks are laid out so that swapping the two sides is a pure column
  permutation plus one sign flip (`GX_SWAP`, `gx_swap()`).  Every "us" column
  has an explicit "them" partner -- that is why, e.g., `we_first` and
  `they_first` are two columns rather than one bit and its complement.

COST
  ~0.5 ms per position, dominated by the 8 `calculate_damage` calls the KO
  threat needs (52 us each, measured).  Everything else is table lookups.
  Measured end-to-end in `dataset.py` and reported in EVALLAB_RESULTS.md.

APPROXIMATIONS (stated, not hidden)
  * Speed: boosts, paralysis (with the Quick Feet exception), Choice Scarf,
    Tailwind, Trick Room, the five weather/terrain speed abilities, Unburden,
    Slow Start and the paradox speed volatiles.  NOT modelled: Iron Ball,
    Gravity/Smack Down grounding changes, Sticky Web's speed drop is only seen
    through the speed boost stage the engine already applied.
  * KO threat uses the engine's own damage calculator with the opponent's first
    move held fixed, so ordering-dependent modifiers (Knock Off's item bonus)
    are evaluated against one fixed partner move rather than all four.
  * Hazard cost is HP fraction only (Stealth Rock + Spikes, with Heavy-Duty
    Boots / Magic Guard / grounding).  Toxic Spikes and Sticky Web are status
    and speed effects, already present as raw counts in the side block.
"""

import json
import os

import numpy as np

import labenv
from poke_engine import calculate_damage

# ---------------------------------------------------------------- static data
_c = json.load(open(labenv.CHART_PATH))
TYPES = _c["types"]                      # index-aligned with poke-engine's enum
CHART = np.array(_c["chart"], dtype=np.float32)
TIX = {t: i for i, t in enumerate(TYPES)}
MOVES = {k.upper(): v for k, v in json.load(open(labenv.MOVES_PATH)).items()}

N_GX = 16
N_SX = 16

# gx column names, in order. The "us"/"them" pairing is load-bearing: see
# GX_SWAP below.
GX_NAMES = [
    "spd_margin", "we_first", "they_first", "speed_tie",
    "off_us", "off_them", "stab_us", "stab_them",
    "dmg_us", "dmg_them", "ko_us", "ko_them",
    "ko_first_us", "ko_first_them", "race_us", "race_them",
]
# permutation applied when side_one and side_two are exchanged
GX_SWAP = np.array([0, 2, 1, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14])
GX_NEGATE = np.zeros(N_GX, dtype=np.float32)
GX_NEGATE[0] = 1.0                       # spd_margin is signed


def gx_swap(gx):
    """Mirror a gx block (or a [B,16] batch of them) to the other perspective."""
    out = gx[..., GX_SWAP].copy()
    out[..., 0] = -out[..., 0]
    return out


SX_NAMES = (["hazcost_b%d" % i for i in range(5)]
            + ["benchoff_b%d" % i for i in range(5)]
            + ["benchfast_b%d" % i for i in range(5)]
            + ["bench_answer"])

WEATHER_SPEED_ABILITY = {
    "CHLOROPHYLL": ("SUN", "HARSHSUN"),
    "SWIFTSWIM": ("RAIN", "HEAVYRAIN"),
    "SANDRUSH": ("SAND",),
    "SLUSHRUSH": ("SNOW", "HAIL"),
}
BOOST_MULT = {-6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3,
              0: 1.0, 1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2}


# ------------------------------------------------------------------ typing
def eff_types(p):
    """DEFENSIVE typing, tera-aware, matching encoder.py's F_TYPES_EFF."""
    tera = str(p.tera_type).upper()
    if p.terastallized and tera != "STELLAR":
        return (tera,)
    return (str(p.types[0]).upper(), str(p.types[1]).upper())


def stab_types(p):
    """Types that get STAB. A tera'd mon keeps STAB on its original types and
    gains it on the tera type (gen9 sim/pokemon.ts)."""
    ts = {str(p.types[0]).upper(), str(p.types[1]).upper()}
    if p.terastallized:
        ts.add(str(p.tera_type).upper())
    return ts


def type_mult(atk, defender_types):
    i = TIX.get(atk)
    if i is None:
        return 1.0
    m = 1.0
    for d in defender_types:
        j = TIX.get(d)
        if j is not None:
            m *= float(CHART[i, j])
    return m


def _damaging(p):
    """[(move_id, type, category, priority, basePower)] for damaging moves."""
    out = []
    for mv in p.moves:
        mid = str(mv.id).upper()
        d = MOVES.get(mid)
        if not d or not d.get("basePower"):
            continue
        if d.get("category") not in ("physical", "special"):
            continue
        if getattr(mv, "disabled", False):
            continue
        if getattr(mv, "pp", 1) <= 0:
            continue
        out.append((mid, str(d["type"]).upper(), d["category"], int(d.get("priority") or 0),
                    float(d["basePower"])))
    return out


# Defensive ABILITY modifiers on the type multiplier. Omitting these is not a
# rounding error: Levitate turns a 2x Earthquake into 0x, and pair A's Rotom-Wash
# has it. The engine's damage calculator already applies them (so the KO columns
# were right), but the pure type-multiplier columns would have been wrong.
ABILITY_TYPE_MOD = {
    "LEVITATE": {"GROUND": 0.0},
    "EARTHEATER": {"GROUND": 0.0},
    "FLASHFIRE": {"FIRE": 0.0},
    "WELLBAKEDBODY": {"FIRE": 0.0},
    "WATERABSORB": {"WATER": 0.0},
    "STORMDRAIN": {"WATER": 0.0},
    "DRYSKIN": {"WATER": 0.0, "FIRE": 1.25},
    "VOLTABSORB": {"ELECTRIC": 0.0},
    "LIGHTNINGROD": {"ELECTRIC": 0.0},
    "MOTORDRIVE": {"ELECTRIC": 0.0},
    "SAPSIPPER": {"GRASS": 0.0},
    "THICKFAT": {"FIRE": 0.5, "ICE": 0.5},
    "HEATPROOF": {"FIRE": 0.5},
    "WATERBUBBLE": {"FIRE": 0.5},
    "PURIFYINGSALT": {"GHOST": 0.5},
    "FLUFFY": {"FIRE": 2.0},
}
MOLD_BREAKERS = {"MOLDBREAKER", "TERAVOLT", "TURBOBLAZE"}


def defensive_mult(att, dfn, move_type):
    """Type multiplier of `move_type` into `dfn`, including the defender's
    ability and Air Balloon, and the attacker's Mold Breaker override."""
    m = type_mult(move_type, eff_types(dfn))
    if str(att.ability).upper() in MOLD_BREAKERS:
        return m
    m *= ABILITY_TYPE_MOD.get(str(dfn.ability).upper(), {}).get(move_type, 1.0)
    if move_type == "GROUND" and str(dfn.item).upper() == "AIRBALLOON":
        m = 0.0
    return m


def offense(att, dfn):
    """(best type multiplier, best STAB multiplier) of att's damaging moves
    into dfn. STAB multiplier = the type multiplier of the best STAB move."""
    st = stab_types(att)
    best, best_stab = 0.0, 0.0
    for _, mt, _, _, _ in _damaging(att):
        m = defensive_mult(att, dfn, mt)
        best = max(best, m)
        if mt in st:
            best_stab = max(best_stab, m)
    return best, best_stab


# ------------------------------------------------------------------- speed
def eff_speed(state, side, p):
    """Effective speed of `p` as `side`'s active, before Trick Room."""
    spd = float(p.speed)
    spd *= BOOST_MULT.get(int(side.speed_boost), 1.0)
    ability = str(p.ability).upper()
    status = str(p.status).upper()
    vs = {str(v).upper() for v in side.volatile_statuses}
    item = str(p.item).upper()
    weather = str(state.weather).upper()
    terrain = str(state.terrain).upper()

    if status == "PARALYZE":
        spd *= 1.5 if ability == "QUICKFEET" else 0.5
    elif status != "NONE" and ability == "QUICKFEET":
        spd *= 1.5
    if item == "CHOICESCARF":
        spd *= 1.5
    if side.side_conditions.tailwind > 0:
        spd *= 2.0
    w = WEATHER_SPEED_ABILITY.get(ability)
    if w and weather in w:
        spd *= 2.0
    if ability == "SURGESURFER" and terrain == "ELECTRICTERRAIN":
        spd *= 2.0
    if ability == "SLOWSTART" and "SLOWSTART" in vs:
        spd *= 0.5
    if "UNBURDEN" in vs:
        spd *= 2.0
    if "PROTOSYNTHESISSPE" in vs or "QUARKDRIVESPE" in vs:
        spd *= 1.5
    return max(spd, 1.0)


def faster(state, spd_a, spd_b):
    """(a_first, b_first, tie) under Trick Room."""
    if spd_a == spd_b:
        return 0.0, 0.0, 1.0
    a = spd_a > spd_b
    if state.trick_room:
        a = not a
    return (1.0, 0.0, 0.0) if a else (0.0, 1.0, 0.0)


# ------------------------------------------------------------------ hazards
def grounded(p):
    ts = eff_types(p)
    if "FLYING" in ts:
        return False
    if str(p.ability).upper() == "LEVITATE":
        return False
    if str(p.item).upper() == "AIRBALLOON":
        return False
    return True


def hazard_cost(p, side_conditions):
    """Fraction of maxhp `p` loses on switching in, HP effects only."""
    if str(p.item).upper() == "HEAVYDUTYBOOTS" or str(p.ability).upper() == "MAGICGUARD":
        return 0.0
    cost = 0.0
    if side_conditions.stealth_rock:
        cost += type_mult("ROCK", eff_types(p)) / 8.0
    sp = int(side_conditions.spikes)
    if sp and grounded(p):
        cost += (0.0, 1 / 8, 1 / 6, 1 / 4)[min(sp, 3)]
    return min(cost, 1.0)


# ----------------------------------------------------------------- KO threat
def _best_damage(state, us, them, us_is_side_one, us_first):
    """(best max-roll damage we deal, priority of that move). One engine
    calculate_damage call per candidate move, opponent move held fixed."""
    partner = None
    for mv in them.moves:
        if str(mv.id).upper() != "NONE":
            partner = str(mv.id)
            break
    if partner is None:
        return 0.0, 0
    best, best_prio = 0.0, 0
    for mid, _mt, _cat, prio, _bp in _damaging(us):
        try:
            if us_is_side_one:
                r = calculate_damage(state, mid, partner, us_first)[0]
            else:
                r = calculate_damage(state, partner, mid, not us_first)[1]
        except Exception:
            continue
        if not r:
            continue
        # calculate_damage returns [max_noncrit, max_crit] per side (the engine's
        # legacy binding shape). The KO threat uses the MAX NON-CRIT roll: "can
        # this move KO on a high roll", not "can it KO on a crit", which would
        # fire on ~60% of positions and stop discriminating anything.
        d = float(r[0])
        if d > best:
            best, best_prio = d, prio
    return best, best_prio


# ------------------------------------------------------------------ blocks
def global_extra(state):
    """gx: the 16 relational columns between the two actives."""
    gx = np.zeros(N_GX, dtype=np.float32)
    s1, s2 = state.side_one, state.side_two
    a1 = list(s1.pokemon)[int(s1.active_index)]
    a2 = list(s2.pokemon)[int(s2.active_index)]
    if a1 is None or a2 is None or a1.maxhp == 0 or a2.maxhp == 0:
        return gx

    v1, v2 = eff_speed(state, s1, a1), eff_speed(state, s2, a2)
    first1, first2, tie = faster(state, v1, v2)
    gx[0] = np.tanh(np.log(v1 / v2))
    gx[1], gx[2], gx[3] = first1, first2, tie

    off_us, stab_us = offense(a1, a2)
    off_them, stab_them = offense(a2, a1)
    gx[4], gx[5] = min(off_us, 4.0) / 4.0, min(off_them, 4.0) / 4.0
    gx[6], gx[7] = min(stab_us, 4.0) / 4.0, min(stab_them, 4.0) / 4.0

    d_us, prio_us = _best_damage(state, a1, a2, True, first1 >= first2)
    d_them, prio_them = _best_damage(state, a2, a1, False, first2 > first1)
    hp1, hp2 = max(a1.hp, 1), max(a2.hp, 1)
    gx[8] = min(d_us / hp2, 1.5) / 1.5
    gx[9] = min(d_them / hp1, 1.5) / 1.5
    ko_us = 1.0 if d_us >= a2.hp > 0 else 0.0
    ko_them = 1.0 if d_them >= a1.hp > 0 else 0.0
    gx[10], gx[11] = ko_us, ko_them
    # who lands their KO first: move priority dominates, speed breaks the tie
    if prio_us != prio_them:
        us_first = prio_us > prio_them
    else:
        us_first = first1 > 0 or (tie > 0 and v1 >= v2)
    gx[12] = 1.0 if (ko_us and us_first) else 0.0
    gx[13] = 1.0 if (ko_them and not us_first) else 0.0
    # race: we KO and they do not KO us first (and symmetrically)
    gx[14] = 1.0 if (ko_us and (not ko_them or us_first)) else 0.0
    gx[15] = 1.0 if (ko_them and (not ko_us or not us_first)) else 0.0
    return gx


def side_extra(state, side, opp_side, bench):
    """sx for one side. `bench` is encoder.bench_order()'s [(idx, mon)] list, so
    slot k here is the SAME Pokemon as slot k of the per-mon bench block."""
    sx = np.zeros(N_SX, dtype=np.float32)
    opp_active = list(opp_side.pokemon)[int(opp_side.active_index)]
    if opp_active is None or opp_active.maxhp == 0:
        return sx
    v_opp = eff_speed(state, opp_side, opp_active)
    n_alive, n_answer = 0, 0
    for k, (_ti, p) in enumerate(bench[:5]):
        if p is None or p.maxhp == 0 or p.hp <= 0:
            continue
        n_alive += 1
        sx[k] = hazard_cost(p, side.side_conditions)
        off, _ = offense(p, opp_active)
        sx[5 + k] = min(off, 4.0) / 4.0
        # a benched mon has no boosts/volatiles of its own; the side's speed
        # boost belongs to the ACTIVE, so a bench speed comparison uses the raw
        # stat plus the side-wide modifiers (tailwind, trick room) only.
        spd = float(p.speed)
        if side.side_conditions.tailwind > 0:
            spd *= 2.0
        if str(p.item).upper() == "CHOICESCARF":
            spd *= 1.5
        if str(p.status).upper() == "PARALYZE":
            spd *= 0.5
        f, _, t = faster(state, spd, v_opp)
        sx[10 + k] = f
        if off >= 2.0 and f > 0:
            n_answer += 1
    sx[15] = n_answer / max(n_alive, 1)
    return sx
