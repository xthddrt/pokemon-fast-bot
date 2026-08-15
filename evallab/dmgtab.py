"""Vectorised gen9 damage kernel + the static per-set-pair damage table (spec §3).

WHAT THIS IS FOR
  EVALUATOR_SPEC §3 wants every relational feature computed from a damage table
  so that runtime is "one division per pair plus multipliers". This file is that
  table and the kernel that fills it. The SAME kernel serves the encoder's
  per-search static half, so the table and the live path can never disagree
  about mechanics -- only about which inputs they were handed.

  Everything is numpy and broadcasts: `raw_damage` takes dicts of arrays for
  attacker / defender / move / field, all mutually broadcastable, and returns
  damage at the MAXIMUM roll. One call does 5851x4x5851 set pairs or 12x4x12
  battle pairs with the same code.

THE FORMULA
    base = floor(floor(floor(2L/5 + 2) * BP * A / D) / 50) + 2
    dmg  = base * STAB * type * weather * burn * screen * item * ability * roll

APPROXIMATIONS, STATED (all measured against the live engine by `enc2_gate.py
dmg`, which is spec gate 3 -- the numbers are in ENCODER2_BUILD.md)
  * Showdown chains modifiers as 12-bit fixed point with a rounding step between
    each; this multiplies them in float64 and rounds once. Worst case is 1 HP.
  * Damage is reported at the MAXIMUM roll (the engine's `calculate_damage`
    returns the same), because "can this move kill" is the question. The min
    roll is derived as `MIN_ROLL * max` where needed (the sweep conjunction uses
    it, so a "sweep" means a GUARANTEED sweep).
  * Multi-hit moves use their EXPECTED hit count (3.167 for [2,5], 4.5 with
    Loaded Dice), not a distribution.
  * Abilities/items outside `AB_*`/`IT_*` below are treated as no modifier. The
    tables cover every item in the randbats pool and every ability in it that
    changes damage; the gate reports what that leaves on the table.
  * Accuracy is NOT modelled. A 70%-accurate OHKO reads as an OHKO. The spec's
    KO matrix asks "best-KO count", not "expected turns to kill".
"""

import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.dirname(os.path.abspath(__file__))
MOVES_PATH = os.path.join(ROOT, "foul-play/data/moves.json")
DEX_PATH = os.path.join(ROOT, "foul-play/data/pokedex.json")
CHART_PATH = os.path.join(LAB, "type_chart.json")

MIN_ROLL = 0.85          # damage roll floor; max roll is 1.0

# --------------------------------------------------------------- type chart
_c = json.load(open(CHART_PATH))
TYPES = _c["types"]                                   # 19, poke-engine order
CHART = np.array(_c["chart"], np.float32)             # CHART[atk, def]
TIX = {t: i for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES)
TYPELESS = N_TYPES                                    # sentinel column
# CHART padded with a neutral row/col for TYPELESS, so a missing type is 1.0x
CHART_P = np.ones((N_TYPES + 1, N_TYPES + 1), np.float32)
CHART_P[:N_TYPES, :N_TYPES] = CHART


def tix(name):
    return TIX.get(str(name).upper(), TYPELESS)


# ------------------------------------------------------------- move features
# bp_kind: how base power is determined. 0 = the constant in MV_BP.
BP_FIXED, BP_LEVEL, BP_HALFHP, BP_TGTWEIGHT, BP_WEIGHTRATIO = 0, 1, 2, 3, 4
BP_BOOSTS, BP_TIMESATK, BP_ACROBATICS, BP_FACADE, BP_TERABLAST = 5, 6, 7, 8, 9
BP_WEATHERBALL, BP_ENDEAVOR, BP_KNOCKOFF, BP_COLLISION = 10, 11, 12, 13

BP_KIND = {
    "SEISMICTOSS": BP_LEVEL, "NIGHTSHADE": BP_LEVEL,
    "SUPERFANG": BP_HALFHP, "RUINATION": BP_HALFHP,
    "LOWKICK": BP_TGTWEIGHT, "GRASSKNOT": BP_TGTWEIGHT,
    "HEAVYSLAM": BP_WEIGHTRATIO, "HEATCRASH": BP_WEIGHTRATIO,
    "STOREDPOWER": BP_BOOSTS, "POWERTRIP": BP_BOOSTS,
    "RAGEFIST": BP_TIMESATK, "ACROBATICS": BP_ACROBATICS,
    "FACADE": BP_FACADE, "TERABLAST": BP_TERABLAST,
    "WEATHERBALL": BP_WEATHERBALL, "ENDEAVOR": BP_ENDEAVOR,
    "KNOCKOFF": BP_KNOCKOFF,
    "COLLISIONCOURSE": BP_COLLISION, "ELECTRODRIFT": BP_COLLISION,
}
# Moves whose damage this kernel cannot model at all -> treated as 0 damage and
# COUNTED, so the gate can report how much of the pool is unmodelled.
UNMODELLED = {"BEATUP", "MIRRORCOAT", "COUNTER", "METALBURST", "COMEUPPANCE",
              "FINALGAMBIT", "NATURESMADNESS", "PSYWAVE"}

# offensive stat override: move -> "DEF" (Body Press uses the user's Defense)
OVERRIDE_OFF_STAT = {"BODYPRESS": "DEF"}

FLAG_BITS = {"contact": 1, "punch": 2, "bite": 4, "sound": 8, "slicing": 16,
             "bullet": 32, "pulse": 64, "recoil": 128, "wind": 256}

# ------------------------------------------------------------------ abilities
# Unconditional stat multipliers (applied to the ATTACKER's offensive stat).
AB_ATK_MULT = {"HUGEPOWER": 2.0, "PUREPOWER": 2.0, "HUSTLE": 1.5,
               "GORILLATACTICS": 1.5}
AB_SPA_MULT = {}
# defender's defensive stat
AB_DEF_MULT = {"FURCOAT": 2.0}
AB_SPD_MULT = {"ICESCALES": 2.0}
# flat damage multipliers on the attacker's side
AB_TYPE_BOOST = {"TRANSISTOR": ("ELECTRIC", 1.3), "DRAGONSMAW": ("DRAGON", 1.5),
                 "STEELWORKER": ("STEEL", 1.5), "STEELYSPIRIT": ("STEEL", 1.5),
                 "ROCKYPAYLOAD": ("ROCK", 1.5), "WATERBUBBLE": ("WATER", 2.0)}
AB_FLAG_BOOST = {"IRONFIST": ("punch", 1.2), "STRONGJAW": ("bite", 1.5),
                 "TOUGHCLAWS": ("contact", 1.3), "SHARPNESS": ("slicing", 1.5),
                 "PUNKROCK": ("sound", 1.3), "MEGALAUNCHER": ("pulse", 1.5),
                 "RECKLESS": ("recoil", 1.2)}
AB_ADAPTABILITY = {"ADAPTABILITY"}
AB_TECHNICIAN = {"TECHNICIAN"}                  # BP <= 60 -> x1.5
AB_SHEERFORCE = {"SHEERFORCE"}                  # has a secondary -> x1.3
AB_TINTEDLENS = {"TINTEDLENS"}                  # resisted -> x2
AB_SE_BOOST = {"NEUROFORCE": 1.25}              # super-effective -> x1.25
AB_SE_RESIST = {"FILTER": 0.75, "SOLIDROCK": 0.75, "PRISMARMOR": 0.75}
AB_MULTISCALE = {"MULTISCALE", "SHADOWSHIELD"}  # defender at full HP -> x0.5
AB_GUTS = {"GUTS"}                              # statused -> atk x1.5, no burn cut
AB_PIXILATE = {"PIXILATE": "FAIRY", "AERILATE": "FLYING", "REFRIGERATE": "ICE",
               "GALVANIZE": "ELECTRIC", "NORMALIZE": "NORMAL"}
MOLD_BREAKERS = {"MOLDBREAKER", "TERAVOLT", "TURBOBLAZE"}
# type multipliers the defender's ability imposes (relfeat.py's table, which
# caught the Levitate bug; extended with the rest of the randbats pool)
AB_TYPE_MOD = {
    "LEVITATE": {"GROUND": 0.0}, "EARTHEATER": {"GROUND": 0.0},
    "FLASHFIRE": {"FIRE": 0.0}, "WELLBAKEDBODY": {"FIRE": 0.0},
    "WATERABSORB": {"WATER": 0.0}, "STORMDRAIN": {"WATER": 0.0},
    "DRYSKIN": {"WATER": 0.0, "FIRE": 1.25},
    "VOLTABSORB": {"ELECTRIC": 0.0}, "LIGHTNINGROD": {"ELECTRIC": 0.0},
    "MOTORDRIVE": {"ELECTRIC": 0.0}, "SAPSIPPER": {"GRASS": 0.0},
    "THICKFAT": {"FIRE": 0.5, "ICE": 0.5}, "HEATPROOF": {"FIRE": 0.5},
    "WATERBUBBLE": {"FIRE": 0.5}, "PURIFYINGSALT": {"GHOST": 0.5},
    "FLUFFY": {"FIRE": 2.0},
}

# -------------------------------------------------------------------- items
IT_ATK_MULT = {"CHOICEBAND": 1.5}
IT_SPA_MULT = {"CHOICESPECS": 1.5}
IT_DEF_MULT = {"EVIOLITE": 1.5}
IT_SPD_MULT = {"ASSAULTVEST": 1.5, "EVIOLITE": 1.5}
IT_ALL_MULT = {"LIFEORB": 1.3, "MUSCLEBAND": 1.1, "WISEGLASSES": 1.1}
IT_SE_MULT = {"EXPERTBELT": 1.2}
IT_TYPE_BOOST = {                       # 1.2x type-boosting held items
    "FLAMEPLATE": "FIRE", "SPLASHPLATE": "WATER", "ZAPPLATE": "ELECTRIC",
    "MEADOWPLATE": "GRASS", "ICICLEPLATE": "ICE", "FISTPLATE": "FIGHTING",
    "TOXICPLATE": "POISON", "EARTHPLATE": "GROUND", "SKYPLATE": "FLYING",
    "MINDPLATE": "PSYCHIC", "INSECTPLATE": "BUG", "STONEPLATE": "ROCK",
    "SPOOKYPLATE": "GHOST", "DRACOPLATE": "DRAGON", "DREADPLATE": "DARK",
    "IRONPLATE": "STEEL", "PIXIEPLATE": "FAIRY", "SILKSCARF": "NORMAL",
    "MAGNET": "ELECTRIC", "SILVERPOWDER": "BUG", "CHARCOAL": "FIRE",
    "MYSTICWATER": "WATER", "MIRACLESEED": "GRASS", "NEVERMELTICE": "ICE",
    "BLACKBELT": "FIGHTING", "POISONBARB": "POISON", "SOFTSAND": "GROUND",
    "SHARPBEAK": "FLYING", "TWISTEDSPOON": "PSYCHIC", "HARDSTONE": "ROCK",
    "SPELLTAG": "GHOST", "BLACKGLASSES": "DARK", "METALCOAT": "STEEL",
    "WELLSPRINGMASK": "WATER", "HEARTHFLAMEMASK": "FIRE",
    "CORNERSTONEMASK": "ROCK",
}
IT_ORB_BOOST = {                       # 1.2x, species-locked signature orbs
    "ADAMANTCRYSTAL": ("DRAGON", "STEEL"), "LUSTROUSGLOBE": ("DRAGON", "WATER"),
    "GRISEOUSCORE": ("DRAGON", "GHOST"), "LUSTROUSORB": ("DRAGON", "WATER"),
    "SOULDEW": ("DRAGON", "PSYCHIC"),
}
IT_LOADED_DICE = "LOADEDDICE"


class MoveTable:
    """Move features as parallel arrays, indexed by a dense local move index.
    `ix` maps an UPPER-CASE move id to that index; index 0 is the null move."""

    def __init__(self, path=MOVES_PATH):
        raw = json.load(open(path))
        names = ["NONE"] + sorted(k.upper() for k in raw)
        self.names = names
        self.ix = {n: i for i, n in enumerate(names)}
        n = len(names)
        z = lambda dt=np.float32: np.zeros(n, dt)  # noqa: E731
        self.bp, self.prio = z(), z(np.int16)
        self.type = np.full(n, TYPELESS, np.int16)
        self.cat = np.full(n, 2, np.int8)          # 0 phys, 1 spec, 2 status
        self.hits = np.ones(n, np.float32)
        self.hits_dice = np.ones(n, np.float32)    # with Loaded Dice
        self.flags = z(np.int32)
        self.secondary = z(np.int8)
        self.bp_kind = z(np.int8)
        self.unmodelled = z(np.int8)
        self.off_is_def = z(np.int8)               # Body Press
        self.def_is_phys = z(np.int8)              # Psyshock: hit physical Def
        self.off_is_target = z(np.int8)            # Foul Play
        self.pp = np.full(n, 1.0, np.float32)
        self.boosts = np.zeros((n, 7), np.int8)    # self-boosts: atk def spa spd spe acc eva
        self.heals = z(np.int8)
        self.vol = ["" for _ in range(n)]
        self.side = [() for _ in range(n)]
        self.status = ["" for _ in range(n)]
        for k, d in raw.items():
            i = self.ix[k.upper()]
            self.bp[i] = float(d.get("basePower") or 0)
            self.prio[i] = int(d.get("priority") or 0)
            self.type[i] = tix(d.get("type"))
            self.cat[i] = {"physical": 0, "special": 1}.get(d.get("category"), 2)
            self.pp[i] = float(d.get("pp") or 1) * 1.6      # gen9 max PP (x8/5)
            mh = d.get("multihit")
            if isinstance(mh, list):
                self.hits[i] = 3.167                        # gen9 [2,5] average
                self.hits_dice[i] = 4.5
            elif mh:
                self.hits[i] = self.hits_dice[i] = float(mh)
            f = d.get("flags") or {}
            self.flags[i] = sum(b for k2, b in FLAG_BITS.items() if f.get(k2))
            if d.get("recoil"):
                self.flags[i] |= FLAG_BITS["recoil"]
            self.secondary[i] = 1 if d.get("secondary") else 0
            self.bp_kind[i] = BP_KIND.get(k.upper(), BP_FIXED)
            self.unmodelled[i] = 1 if k.upper() in UNMODELLED else 0
            self.off_is_def[i] = 1 if OVERRIDE_OFF_STAT.get(k.upper()) == "DEF" else 0
            self.def_is_phys[i] = 1 if d.get("overrideDefensiveStat") == "defense" else 0
            self.off_is_target[i] = 1 if d.get("overrideOffensivePokemon") == "target" else 0
            for src, tgt_self in ((d.get("boosts") or {}, d.get("target") == "self"),
                                  ((d.get("self") or {}).get("boosts") or {}, True)):
                if not tgt_self:
                    continue
                for s, j in (("atk", 0), ("attack", 0), ("def", 1), ("defense", 1),
                             ("spa", 2), ("special-attack", 2), ("spd", 3),
                             ("special-defense", 3), ("spe", 4), ("speed", 4),
                             ("accuracy", 5), ("evasion", 6)):
                    if s in src:
                        self.boosts[i, j] = int(src[s])
            self.heals[i] = 1 if (d.get("heal") or d.get("drain")) else 0
            self.vol[i] = str(d.get("volatileStatus") or "")
            self.side[i] = tuple((d.get("side_conditions") or {}))
            self.status[i] = str(d.get("status") or "")
        self.damaging = (self.cat < 2) & ((self.bp > 0) | (self.bp_kind > 0))

    def idx(self, names):
        """UPPER-CASE names -> local indices (0 for anything unknown)."""
        return np.array([self.ix.get(str(x).upper(), 0) for x in names], np.int32)


MT = None


def moves():
    global MT
    if MT is None:
        MT = MoveTable()
    return MT


# --------------------------------------------------------------- the kernel
def _boost_mult(stage):
    """Standard stage multiplier, vectorised over integer stages in [-6, 6]."""
    s = np.clip(stage, -6, 6).astype(np.float32)
    return np.where(s >= 0, (2.0 + np.maximum(s, 0)) / 2.0,
                    2.0 / (2.0 - np.minimum(s, 0)))


BOOST_MULT_TABLE = _boost_mult(np.arange(-6, 7))


def _lookup(d, keys, default=1.0, dtype=np.float32):
    """dict -> array aligned with `keys` (a list of UPPER-CASE names)."""
    return np.array([d.get(k, default) for k in keys], dtype)


def type_multiplier(move_type, def_t1, def_t2):
    """CHART lookup for a (move type, defender type pair). All args broadcast."""
    return CHART_P[move_type, def_t1] * CHART_P[move_type, def_t2]


def raw_damage(A, D, M, F=None):
    """Damage at the MAX roll. A, D, M, F are dicts of broadcastable arrays.

    A (attacker): level, atk, spa, defense (for Body Press), spe, weight,
        stab1, stab2 (types that get STAB), ab (ability-code arrays, see
        `ability_codes`), it (item-code arrays), boost_atk/boost_spa (stages),
        burned, statused, terastallized, tera_type.
    D (defender): defense, spd, maxhp, hp, t1, t2, weight, ab, it,
        boost_def/boost_spd.
    M (move): bp, type, cat, flags, secondary, hits, bp_kind, prio, unmodelled,
        off_is_def, def_is_phys, off_is_target.
    F (field): reflect, light_screen, aurora_veil, weather (type index or -1),
        n_positive_boosts (for Stored Power), times_attacked (Rage Fist).

    Returns damage in HP points, float32, 0 where the move cannot damage.
    """
    F = F or {}
    lvl = A["level"]
    cat = M["cat"]
    phys = cat == 0
    spec = cat == 1
    damaging = cat < 2

    # ---- base power ------------------------------------------------------
    bp = M["bp"].astype(np.float32)
    kind = M["bp_kind"]
    if np.any(kind == BP_TGTWEIGHT):
        w = D["weight"]
        wbp = np.where(w >= 200, 120.0, np.where(w >= 100, 100.0,
              np.where(w >= 50, 80.0, np.where(w >= 25, 60.0,
              np.where(w >= 10, 40.0, 20.0)))))
        bp = np.where(kind == BP_TGTWEIGHT, wbp, bp)
    if np.any(kind == BP_WEIGHTRATIO):
        r = A["weight"] / np.maximum(D["weight"], 0.1)
        rbp = np.where(r >= 5, 120.0, np.where(r >= 4, 100.0,
              np.where(r >= 3, 80.0, np.where(r >= 2, 60.0, 40.0))))
        bp = np.where(kind == BP_WEIGHTRATIO, rbp, bp)
    if np.any(kind == BP_BOOSTS):
        bp = np.where(kind == BP_BOOSTS,
                      20.0 + 20.0 * F.get("n_positive_boosts", 0), bp)
    if np.any(kind == BP_TIMESATK):
        bp = np.where(kind == BP_TIMESATK,
                      np.minimum(350.0, 50.0 + 50.0 * F.get("times_attacked", 0)), bp)
    if np.any(kind == BP_ACROBATICS):
        bp = np.where(kind == BP_ACROBATICS,
                      np.where(A.get("has_item", 1) > 0, 55.0, 110.0), bp)
    if np.any(kind == BP_FACADE):
        bp = np.where(kind == BP_FACADE,
                      np.where(A.get("statused", 0) > 0, 140.0, 70.0), bp)
    if np.any(kind == BP_KNOCKOFF):
        bp = np.where(kind == BP_KNOCKOFF,
                      np.where(D.get("has_item", 0) > 0, 97.5, 65.0), bp)

    # ---- move type (tera / -ate abilities) -------------------------------
    mt = M["type"].astype(np.int16)
    if np.any(kind == BP_TERABLAST):
        mt = np.where((kind == BP_TERABLAST) & (A.get("terastallized", 0) > 0),
                      A["tera_type"], mt)
    ate = A.get("ab_ate", None)
    if ate is not None:
        mt = np.where((ate >= 0) & (mt == TIX["NORMAL"]), ate, mt)
        bp = np.where((ate >= 0) & (M["type"] == TIX["NORMAL"]), bp * 1.2, bp)
    if np.any(kind == BP_WEATHERBALL):
        wt = F.get("weather_type", np.int16(TYPELESS))
        mt = np.where(kind == BP_WEATHERBALL, np.where(wt < N_TYPES, wt, mt), mt)
        bp = np.where((kind == BP_WEATHERBALL) & (wt < N_TYPES), 100.0, bp)

    # ---- offensive / defensive stats -------------------------------------
    atk = A["atk"] * _boost_stage(A.get("boost_atk", 0))
    spa = A["spa"] * _boost_stage(A.get("boost_spa", 0))
    atk = atk * _lookup_arr(A, "ab_atk_mult") * _lookup_arr(A, "it_atk_mult")
    spa = spa * _lookup_arr(A, "ab_spa_mult") * _lookup_arr(A, "it_spa_mult")
    # Guts: 1.5x Attack when statused (and the burn cut is skipped below)
    guts = (A.get("ab_guts", 0) > 0) & (A.get("statused", 0) > 0)
    atk = np.where(guts, atk * 1.5, atk)
    # Body Press uses the user's (boosted) Defense as the attack stat
    if np.any(M["off_is_def"]):
        own_def = A["defense"] * _boost_stage(A.get("boost_def", 0))
        atk = np.where(M["off_is_def"] > 0, own_def, atk)
    # Foul Play uses the TARGET's Attack
    if np.any(M["off_is_target"]):
        atk = np.where(M["off_is_target"] > 0,
                       D["atk"] * _boost_stage(D.get("boost_atk", 0)), atk)

    dfd = D["defense"] * _boost_stage(D.get("boost_def", 0))
    dsp = D["spd"] * _boost_stage(D.get("boost_spd", 0))
    dfd = dfd * _lookup_arr(D, "ab_def_mult") * _lookup_arr(D, "it_def_mult")
    dsp = dsp * _lookup_arr(D, "ab_spd_mult") * _lookup_arr(D, "it_spd_mult")
    # Sandstorm gives Rock types +50% SpD
    dsp = dsp * np.where(F.get("sand", 0) > 0,
                         np.where((D["t1"] == TIX["ROCK"]) | (D["t2"] == TIX["ROCK"]),
                                  1.5, 1.0), 1.0)
    # Snow gives Ice types +50% Def
    dfd = dfd * np.where(F.get("snow", 0) > 0,
                         np.where((D["t1"] == TIX["ICE"]) | (D["t2"] == TIX["ICE"]),
                                  1.5, 1.0), 1.0)

    Aeff = np.where(phys, atk, spa)
    Deff = np.where(phys | (M["def_is_phys"] > 0), dfd, dsp)
    Deff = np.maximum(Deff, 1.0)

    # ---- base damage (the integer chain) ---------------------------------
    lvl_term = np.floor(2.0 * lvl / 5.0 + 2.0)
    base = np.floor(np.floor(np.floor(lvl_term * bp * Aeff / Deff) / 50.0)) + 2.0

    # ---- multipliers -----------------------------------------------------
    eff = type_multiplier(mt, D["t1"], D["t2"])
    # defender ability type modifiers, unless the attacker breaks moulds
    abmod = _ability_type_mod(A, D, mt)
    eff = np.where(A.get("ab_moldbreaker", 0) > 0, eff, eff * abmod)
    # Air Balloon
    eff = np.where((D.get("it_airballoon", 0) > 0) & (mt == TIX["GROUND"]), 0.0, eff)

    stab = np.where((mt == A["stab1"]) | (mt == A["stab2"]),
                    np.where(A.get("ab_adaptability", 0) > 0, 2.0, 1.5), 1.0)
    # a terastallized mon gets STAB on its tera type too (2.0 if it was already
    # one of its base types)
    tera_on = A.get("terastallized", 0) > 0
    tera_hit = tera_on & (mt == A["tera_type"])
    stab = np.where(tera_hit,
                    np.where((mt == A["stab1"]) | (mt == A["stab2"]), 2.0, 1.5),
                    stab)

    mult = np.ones_like(base)
    mult = mult * eff * stab
    # weather
    w = F.get("weather_type", TYPELESS)
    mult = mult * np.where(F.get("sun", 0) > 0,
                           np.where(mt == TIX["FIRE"], 1.5,
                                    np.where(mt == TIX["WATER"], 0.5, 1.0)), 1.0)
    mult = mult * np.where(F.get("rain", 0) > 0,
                           np.where(mt == TIX["WATER"], 1.5,
                                    np.where(mt == TIX["FIRE"], 0.5, 1.0)), 1.0)
    # burn halves physical damage (Guts and Facade excepted)
    mult = mult * np.where((A.get("burned", 0) > 0) & phys & ~guts, 0.5, 1.0)
    # screens
    scr = np.where(phys, F.get("reflect", 0), F.get("light_screen", 0))
    scr = np.maximum(scr, F.get("aurora_veil", 0))
    mult = mult * np.where((scr > 0) & (A.get("ab_infiltrator", 0) == 0), 0.5, 1.0)
    # items
    mult = mult * _lookup_arr(A, "it_all_mult")
    mult = mult * np.where(eff > 1.0, _lookup_arr(A, "it_se_mult"), 1.0)
    tb = A.get("it_type_boost", None)
    if tb is not None:
        mult = mult * np.where(mt == tb, 1.2, 1.0)
    # abilities
    tybo = A.get("ab_type_boost_type", None)
    if tybo is not None:
        mult = mult * np.where(mt == tybo, A["ab_type_boost_mult"], 1.0)
    flbo = A.get("ab_flag_boost_bit", None)
    if flbo is not None:
        mult = mult * np.where((M["flags"] & flbo) > 0, A["ab_flag_boost_mult"], 1.0)
    mult = mult * np.where((A.get("ab_technician", 0) > 0) & (bp <= 60), 1.5, 1.0)
    mult = mult * np.where((A.get("ab_sheerforce", 0) > 0) & (M["secondary"] > 0), 1.3, 1.0)
    mult = mult * np.where((A.get("ab_tintedlens", 0) > 0) & (eff < 1.0) & (eff > 0), 2.0, 1.0)
    mult = mult * np.where(eff > 1.0, _lookup_arr(A, "ab_se_boost"), 1.0)
    mult = mult * np.where(eff > 1.0, _lookup_arr(D, "ab_se_resist"), 1.0)
    # Multiscale / Shadow Shield: defender at full HP takes half
    mult = mult * np.where((D.get("ab_multiscale", 0) > 0)
                           & (D["hp"] >= D["maxhp"]), 0.5, 1.0)

    dmg = base * mult * M["hits"]

    # ---- fixed-damage and HP-proportional moves override everything ------
    if np.any(kind == BP_LEVEL):
        dmg = np.where(kind == BP_LEVEL, np.where(eff > 0, lvl, 0.0), dmg)
    if np.any(kind == BP_HALFHP):
        dmg = np.where(kind == BP_HALFHP, np.where(eff > 0, D["hp"] / 2.0, 0.0), dmg)
    if np.any(kind == BP_ENDEAVOR):
        dmg = np.where(kind == BP_ENDEAVOR,
                       np.where(eff > 0, np.maximum(D["hp"] - A.get("hp", 0), 0.0), 0.0), dmg)
    dmg = np.where(M["unmodelled"] > 0, 0.0, dmg)
    return np.where(damaging, np.maximum(dmg, 0.0), 0.0).astype(np.float32)


def _boost_stage(stage):
    if np.isscalar(stage) and stage == 0:
        return np.float32(1.0)
    return BOOST_MULT_TABLE[np.clip(np.asarray(stage), -6, 6) + 6]


def _lookup_arr(d, key, default=1.0):
    v = d.get(key)
    return np.float32(default) if v is None else v


def _ability_type_mod(A, D, mt):
    """Defender-ability multiplier on the type chart, e.g. Levitate x0."""
    t = D.get("ab_typemod_type", None)
    if t is None:
        return np.float32(1.0)
    m = np.where(mt == t, D["ab_typemod_mult"], 1.0)
    t2 = D.get("ab_typemod_type2", None)
    if t2 is not None:
        m = m * np.where(mt == t2, D["ab_typemod_mult2"], 1.0)
    return m


# ------------------------------------------------- ability / item code arrays
def ability_codes(names):
    """UPPER-CASE ability names -> the dict of arrays `raw_damage` wants under
    the attacker's and defender's `ab_*` keys."""
    n = len(names)
    out = {
        "ab_atk_mult": _lookup(AB_ATK_MULT, names),
        "ab_spa_mult": _lookup(AB_SPA_MULT, names),
        "ab_def_mult": _lookup(AB_DEF_MULT, names),
        "ab_spd_mult": _lookup(AB_SPD_MULT, names),
        "ab_adaptability": np.array([x in AB_ADAPTABILITY for x in names], np.int8),
        "ab_technician": np.array([x in AB_TECHNICIAN for x in names], np.int8),
        "ab_sheerforce": np.array([x in AB_SHEERFORCE for x in names], np.int8),
        "ab_tintedlens": np.array([x in AB_TINTEDLENS for x in names], np.int8),
        "ab_guts": np.array([x in AB_GUTS for x in names], np.int8),
        "ab_multiscale": np.array([x in AB_MULTISCALE for x in names], np.int8),
        "ab_moldbreaker": np.array([x in MOLD_BREAKERS for x in names], np.int8),
        "ab_infiltrator": np.array([x == "INFILTRATOR" for x in names], np.int8),
        "ab_se_boost": _lookup(AB_SE_BOOST, names),
        "ab_se_resist": _lookup(AB_SE_RESIST, names),
        "ab_ate": np.array([tix(AB_PIXILATE[x]) if x in AB_PIXILATE else -1
                            for x in names], np.int16),
    }
    tb = np.full(n, -1, np.int16)
    tm = np.ones(n, np.float32)
    for i, x in enumerate(names):
        if x in AB_TYPE_BOOST:
            tb[i], tm[i] = tix(AB_TYPE_BOOST[x][0]), AB_TYPE_BOOST[x][1]
    out["ab_type_boost_type"], out["ab_type_boost_mult"] = tb, tm
    fb = np.zeros(n, np.int32)
    fm = np.ones(n, np.float32)
    for i, x in enumerate(names):
        if x in AB_FLAG_BOOST:
            fb[i], fm[i] = FLAG_BITS[AB_FLAG_BOOST[x][0]], AB_FLAG_BOOST[x][1]
    out["ab_flag_boost_bit"], out["ab_flag_boost_mult"] = fb, fm
    # defender type modifiers: at most two entries per ability in the pool
    t1 = np.full(n, -1, np.int16)
    m1 = np.ones(n, np.float32)
    t2 = np.full(n, -1, np.int16)
    m2 = np.ones(n, np.float32)
    for i, x in enumerate(names):
        d = AB_TYPE_MOD.get(x)
        if d:
            items = list(d.items())
            t1[i], m1[i] = tix(items[0][0]), items[0][1]
            if len(items) > 1:
                t2[i], m2[i] = tix(items[1][0]), items[1][1]
    out["ab_typemod_type"], out["ab_typemod_mult"] = t1, m1
    out["ab_typemod_type2"], out["ab_typemod_mult2"] = t2, m2
    return out


def item_codes(names):
    """UPPER-CASE item names -> the `it_*` arrays."""
    n = len(names)
    out = {
        "it_atk_mult": _lookup(IT_ATK_MULT, names),
        "it_spa_mult": _lookup(IT_SPA_MULT, names),
        "it_def_mult": _lookup(IT_DEF_MULT, names),
        "it_spd_mult": _lookup(IT_SPD_MULT, names),
        "it_all_mult": _lookup(IT_ALL_MULT, names),
        "it_se_mult": _lookup(IT_SE_MULT, names),
        "it_airballoon": np.array([x == "AIRBALLOON" for x in names], np.int8),
        "it_loaded_dice": np.array([x == IT_LOADED_DICE for x in names], np.int8),
        "has_item": np.array([x not in ("", "NONE") for x in names], np.int8),
    }
    tb = np.full(n, -1, np.int16)
    for i, x in enumerate(names):
        if x in IT_TYPE_BOOST:
            tb[i] = tix(IT_TYPE_BOOST[x])
        elif x in IT_ORB_BOOST:
            tb[i] = tix(IT_ORB_BOOST[x][0])     # first of the two boosted types
    out["it_type_boost"] = tb
    return out


# =========================================================================
# the static per-set-pair table (spec §3)
# =========================================================================
def _stat(base, level, nature=1.0):
    return np.floor((np.floor((2 * base + 31 + 21) * level / 100.0) + 5) * nature)


def _hp(base, level):
    return np.floor((2 * base + 31 + 21) * level / 100.0) + level + 10


class SetTable:
    """Every gen9 randbats set as parallel arrays: the row space of the static
    damage table. A "set" is the format's own key
    (level,item,ability,m0..m3,tera), so two sets of one species with different
    items are different rows -- which is the point, since the item changes the
    damage."""

    def __init__(self):
        import rbpool
        pool = rbpool.Pool()
        dex = {k.replace("-", "").replace(" ", "").replace(".", "").replace("'", "").lower(): v
               for k, v in json.load(open(DEX_PATH)).items()}
        mt = moves()
        keep = []
        for s in pool.sets:
            if s[0].lower() in dex:
                keep.append(s)
        self.sets = keep
        n = len(keep)
        self.n = n
        lvl = np.array([s[1] for s in keep], np.float32)
        bs = np.array([[dex[s[0].lower()]["baseStats"][k] for k in
                        ("hp", "attack", "defense", "special-attack",
                         "special-defense", "speed")] for s in keep], np.float32)
        self.level = lvl
        self.maxhp = _hp(bs[:, 0], lvl)
        self.atk = _stat(bs[:, 1], lvl)
        self.defense = _stat(bs[:, 2], lvl)
        self.spa = _stat(bs[:, 3], lvl)
        self.spd = _stat(bs[:, 4], lvl)
        self.spe = _stat(bs[:, 5], lvl)
        self.weight = np.array([dex[s[0].lower()].get("weightkg", 50.0)
                                for s in keep], np.float32)
        tt = [[tix(t) for t in dex[s[0].lower()]["types"]] for s in keep]
        self.t1 = np.array([t[0] for t in tt], np.int16)
        self.t2 = np.array([t[1] if len(t) > 1 else TYPELESS for t in tt], np.int16)
        self.tera = np.array([tix(s[5]) for s in keep], np.int16)
        self.mv = np.stack([mt.idx([s[4][i] if i < len(s[4]) else "NONE"
                                    for s in keep]) for i in range(4)], 1)
        abil = [s[3].upper() for s in keep]
        item = [s[2].upper() for s in keep]
        self.ab = ability_codes(abil)
        self.it = item_codes(item)
        self.species = [s[0] for s in keep]
        self.key = ["%s|%s|%s|%s|%s" % (s[0], s[1], s[2], s[3], ",".join(sorted(s[4])))
                    for s in keep]
        self.key_ix = {}
        for i, k in enumerate(self.key):
            self.key_ix.setdefault(k, i)

    def attacker(self, i):
        """Attacker-side dict for `raw_damage`, for a slice/array of set rows."""
        A = {"level": self.level[i], "atk": self.atk[i], "spa": self.spa[i],
             "defense": self.defense[i], "weight": self.weight[i],
             "stab1": self.t1[i], "stab2": self.t2[i], "tera_type": self.tera[i],
             "terastallized": np.int8(0), "hp": self.maxhp[i]}
        for k, v in self.ab.items():
            A[k] = v[i]
        for k, v in self.it.items():
            A[k] = v[i]
        return A

    def defender(self, i):
        D = {"defense": self.defense[i], "spd": self.spd[i], "atk": self.atk[i],
             "maxhp": self.maxhp[i], "hp": self.maxhp[i], "t1": self.t1[i],
             "t2": self.t2[i], "weight": self.weight[i]}
        for k, v in self.ab.items():
            D[k] = v[i]
        for k, v in self.it.items():
            D[k] = v[i]
        return D

    def move_block(self, i):
        """Move arrays for attacker rows `i`, shaped (len(i), 4)."""
        mt = moves()
        mi = self.mv[i]
        dice = self.it["it_loaded_dice"][i][:, None] > 0
        return {"bp": mt.bp[mi], "type": mt.type[mi], "cat": mt.cat[mi],
                "flags": mt.flags[mi], "secondary": mt.secondary[mi],
                "hits": np.where(dice, mt.hits_dice[mi], mt.hits[mi]),
                "bp_kind": mt.bp_kind[mi], "prio": mt.prio[mi],
                "unmodelled": mt.unmodelled[mi], "off_is_def": mt.off_is_def[mi],
                "def_is_phys": mt.def_is_phys[mi],
                "off_is_target": mt.off_is_target[mi]}


def build_pair_table(out_path=None, chunk=64, progress=None):
    """The static all-pairs table: best damage as a FRACTION of the defender's
    full HP, split by category so runtime multipliers (Atk vs SpA boosts,
    Reflect vs Light Screen, burn) can be applied to the right half.

    -> dict with `phys` and `spec`, each float16[n_sets, n_sets], and stats.
    """
    st = SetTable()
    n = st.n
    phys = np.zeros((n, n), np.float16)
    spec = np.zeros((n, n), np.float16)
    t0 = time.time()
    dall = np.arange(n)
    D = st.defender(dall)
    Dd = {k: (v[None, None, :] if isinstance(v, np.ndarray) and v.ndim == 1 else v)
          for k, v in D.items()}
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ai = np.arange(lo, hi)
        A = st.attacker(ai)
        Ad = {k: (v[:, None, None] if isinstance(v, np.ndarray) and v.ndim == 1 else v)
              for k, v in A.items()}
        M = st.move_block(ai)
        Md = {k: v[:, :, None] for k, v in M.items()}
        d = raw_damage(Ad, Dd, Md)                     # (chunk, 4, n)
        frac = d / st.maxhp[None, None, :]
        isphys = (Md["cat"] == 0)
        phys[lo:hi] = np.where(isphys, frac, 0.0).max(axis=1).astype(np.float16)
        spec[lo:hi] = np.where(Md["cat"] == 1, frac, 0.0).max(axis=1).astype(np.float16)
        if progress and (lo // chunk) % 10 == 0:
            progress(lo, n, time.time() - t0)
    stats = {"n_sets": n, "build_s": time.time() - t0,
             "bytes": phys.nbytes + spec.nbytes}
    if out_path:
        np.savez(out_path, phys=phys, spec=spec)
        stats["disk_bytes"] = os.path.getsize(out_path + ".npz"
                                              if not out_path.endswith(".npz") else out_path)
    return {"phys": phys, "spec": spec, "sets": st, "stats": stats}


if __name__ == "__main__":
    sys.path.insert(0, LAB)
    import labenv  # noqa: F401
    st = SetTable()
    print("sets", st.n)
    r = build_pair_table(progress=lambda lo, n, t: print("  %d/%d  %.1fs" % (lo, n, t)))
    s = r["stats"]
    print("built %d x %d in %.1fs, %.1f MB in memory"
          % (s["n_sets"], s["n_sets"], s["build_s"], s["bytes"] / 1e6))
