"""SYNTHETIC POOL-WIDE state generator for the adopted-encoder parity gate (LEG B).

The last Rust port shipped two live bugs -- Loaded Dice not selecting the
multi-hit count, and an "-ate" ability boost computed in f64 where numpy uses
f32 -- that FIVE THOUSAND REAL STATES passed. Both were found only by states
that exercised the whole randbats pool. So real-state parity is not evidence;
this file is.

Four generators, in decreasing order of what they guarantee:

  1 `sweep_sets`  -- every one of the 5,851 gen9 randbats SETS appears, 12 per
    state. That is a hard guarantee of all 509 species, all 60 items, all 203
    abilities, all 349 moves and all 19 tera types, not a sampling hope.
  2 `cross`       -- every item on species that never carry it, and every
    ability on species that never have it. Breaks the (species, item, ability)
    correlation the pool has, which is what hides a table-indexing bug.
  3 `targeted`    -- the awkward mechanics by name: Loaded Dice multi-hit,
    the five "-ate" abilities on Normal moves, Protean/Libero, Terapagos +
    Stellar, Palafin/Zero to Hero, Ditto/Imposter, Mimikyu/Disguise,
    Illusion, Revival Blessing, Rage Fist (`times_attacked` -- the PRUNED
    column), and every setup move the move table classifies as one.
  4 `fuzz`        -- `enc2_gate.fuzz_states`, for volatile/hazard/field breadth.
"""
import json
import os
import random
import sys

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: E402,F401
import lossless_encoder as LE  # noqa: E402
import dmgtab as DT  # noqa: E402
import rbpool  # noqa: E402
import enc2_gate as G  # noqa: E402

STATUSES = ["NONE"] * 6 + ["BURN", "SLEEP", "PARALYZE", "POISON", "TOXIC", "FREEZE"]
MULTIHIT = ["BULLETSEED", "ROCKBLAST", "ICICLESPEAR", "PINMISSILE", "TAILSLAP",
            "SCALESHOT", "POPULATIONBOMB", "WATERSHURIKEN", "ARMTHRUST", "BONERUSH",
            "DOUBLESLAP", "FURYSWIPES", "SURGINGSTRIKES", "TRIPLEAXEL"]
NORMAL_MOVES = ["HYPERVOICE", "BOOMBURST", "BODYSLAM", "DOUBLEEDGE", "TACKLE",
                "SWIFT", "FACADE", "EXTREMESPEED", "QUICKATTACK", "TRIATTACK"]
ATE = ["PIXILATE", "AERILATE", "REFRIGERATE", "GALVANIZE", "NORMALIZE"]


def _pool():
    if not hasattr(_pool, "p"):
        _pool.p = rbpool.Pool()
        _pool.dex = {k.replace("-", "").replace(" ", "").replace(".", "")
                     .replace("'", "").lower(): v
                     for k, v in json.load(open(DT.DEX_PATH)).items()}
        _pool.sets = [s for s in _pool.p.sets if s[0].lower() in _pool.dex]
    return _pool.p


def setup_moves():
    """Every move the encoder classifies as setup: a STATUS move (cat 2) that
    raises its own user's atk / spa / spe. Read off `dmgtab.moves()` rather than
    hand-listed, so it cannot drift from what `enc2.mv_is_setup` computes."""
    mt = DT.moves()
    off = mt.boosts[:, [0, 2, 4]].max(axis=1)
    ok = (mt.cat == 2) & (off >= 1)
    inv = {v: k for k, v in mt.ix.items()}
    return sorted(inv[i] for i in np.flatnonzero(ok) if i in inv and i != 0)


def mon(sp, lvl, it0, ab0, mv, tera, rng, **over):
    d = _pool.dex[sp.lower()]
    b = d["baseStats"]
    t1 = d["types"][0].upper()
    t2 = d["types"][1].upper() if len(d["types"]) > 1 else "TYPELESS"

    def stat(k):
        return int(np.floor((2 * b[k] + 52) * lvl / 100.0) + 5)

    mhp = int(np.floor((2 * b["hp"] + 52) * lvl / 100.0) + lvl + 10)
    out = dict(species=sp.upper(), level=lvl, type1=t1, type2=t2, btype1=t1, btype2=t2,
               hp=rng.choice([mhp, mhp, int(mhp * rng.random()), 0]), maxhp=mhp,
               ability=ab0.upper(), bability=ab0.upper(), item=(it0 or "NONE").upper(),
               status=rng.choice(STATUSES), attack=stat("attack"), defense=stat("defense"),
               spa=stat("special-attack"), spd=stat("special-defense"), speed=stat("speed"),
               weight=d.get("weightkg", 50), rest_turns=rng.randint(0, 3),
               sleep_turns=rng.randint(0, 3),
               terastallized="true" if rng.random() < 0.10 else "false",
               tera_type=tera.upper(), times_attacked=rng.randint(0, 6),
               reveal_mask=rng.randint(0, 255), nature="SERIOUS")
    for i in range(4):
        out["move%d" % i] = "%s;%s;%d" % (mv[i].upper() if i < len(mv) else "NONE",
                                          "true" if rng.random() < 0.05 else "false",
                                          rng.randint(0, 32))
    out.update(over)
    return out


def wrap(mons12, rng, tame_vol=False):
    """12 mon dicts -> one state string, with the field, hazards, boosts,
    volatiles and active index randomised the way `fuzz_states` does."""
    vols = rbpool.reachable_list()
    weathers = [w for w in LE.WEATHER_ORDER if w not in ("HARSHSUN", "HEAVYRAIN")]
    s = G.flat_state(mons12[:6], mons12[6:],
                     weather=rng.choice(weathers), terrain=rng.choice(LE.TERRAIN_ORDER),
                     trick_room=rng.random() < 0.15)
    st = G.St(s)
    for side in (0, 1):
        sc = [0] * 19
        for k in ("stealth_rock", "sticky_web", "reflect", "light_screen",
                  "aurora_veil", "tailwind", "safeguard", "mist"):
            if rng.random() < 0.2:
                sc[LE.SIDE_CONDITION_FIELDS.index(k)] = rng.randint(1, 5)
        sc[LE.SIDE_CONDITION_FIELDS.index("spikes")] = rng.randint(0, 3)
        sc[LE.SIDE_CONDITION_FIELDS.index("toxic_spikes")] = rng.randint(0, 2)
        sc[LE.SIDE_CONDITION_FIELDS.index("toxic_count")] = rng.randint(0, 8)
        sc[LE.SIDE_CONDITION_FIELDS.index("protect")] = rng.randint(0, 2)
        # the active slot must not be an empty one
        alive = [i for i in range(6) if mons12[side * 6 + i] is not None
                 and mons12[side * 6 + i].get("hp", 1) > 0] or [0]
        st.side(side, active_index=rng.choice(alive),
                side_conditions=";".join(str(x) for x in sc),
                volatiles=":".join([] if tame_vol else
                                   [v for v in vols if rng.random() < 0.05]),
                durations=";".join(str(rng.randint(0, 3)) for _ in range(13)),
                substitute_health=rng.choice([0, 0, 40]),
                boost_atk=rng.randint(-2, 4), boost_def=rng.randint(-2, 2),
                boost_spa=rng.randint(-2, 4), boost_spd=rng.randint(-2, 2),
                boost_spe=rng.randint(-2, 4), boost_acc=rng.randint(-1, 1),
                boost_eva=rng.randint(-1, 1),
                wish0=rng.randint(0, 2), wish1=rng.randint(0, 150),
                fs0=rng.randint(0, 3), fs1=rng.randint(0, 5),
                force_switch=rng.choice(["true", "false"]),
                revival_blessing=rng.choice(["true", "false", "false"]),
                force_trapped=rng.choice(["true", "false"]),
                last_used_move=rng.choice(["move:none", "move:0", "switch:2"]),
                times_revived=rng.randint(0, 2),
                last_move_failed=rng.choice(["true", "false"]))
    return str(st)


# ------------------------------------------------------------------ generators
def sweep_sets(rng, passes=1):
    """EVERY randbats set, `passes` times, 12 per state."""
    sets = list(_pool().sets)
    out = []
    for p in range(passes):
        order = list(sets)
        if p:
            rng.shuffle(order)
        for i in range(0, len(order), 12):
            blk = order[i:i + 12]
            while len(blk) < 12:
                blk.append(order[len(blk) % len(order)])
            # the pool is in species order, which is close to vocab-id order, so
            # without this arm A's bench sort is a no-op and the sweep would not
            # exercise the enc2 -> arm A remap at all
            rng.shuffle(blk)
            out.append(wrap([mon(s[0], s[1], s[2], s[3], s[4], s[5], rng) for s in blk], rng))
    return out


def cross(rng):
    """Every ITEM and every ABILITY forced onto species that never carry it."""
    p = _pool()
    sets = list(_pool.sets)
    out = []
    for field, universe in (("item", p.items), ("ability", p.abilities)):
        for off in range(3):
            u = list(universe)
            for i in range(0, len(u), 12):
                blk = u[i:i + 12]
                while len(blk) < 12:
                    blk.append(u[len(blk) % len(u)])
                rng.shuffle(blk)
                mons = []
                for j, v in enumerate(blk):
                    s = sets[(i * 7 + j * 101 + off * 1531) % len(sets)]
                    over = {field: v.upper()}
                    if field == "ability":
                        over["bability"] = v.upper()
                    mons.append(mon(s[0], s[1], s[2], s[3], s[4], s[5], rng, **over))
                out.append(wrap(mons, rng))
    return out


def targeted(rng):
    """The awkward mechanics, by name."""
    p = _pool()
    sets = list(_pool.sets)
    by_sp = {}
    for s in sets:
        by_sp.setdefault(s[0].lower(), s)
    out = []

    def pick(i):
        return sets[i % len(sets)]

    def state(specs, tame=False):
        mons = []
        for j in range(12):
            s = pick(j * 313 + len(out) * 17)
            base = mon(s[0], s[1], s[2], s[3], s[4], s[5], rng)
            if j < len(specs) and specs[j]:
                base.update(specs[j])
                base["hp"] = base["maxhp"]          # awkward mons must be ALIVE
            mons.append(base)
        out.append(wrap(mons, rng, tame_vol=tame))

    # -- Loaded Dice x every multi-hit move, with and without the item ---------
    for hold in ("LOADEDDICE", "NONE"):
        for i in range(0, len(MULTIHIT), 12):
            blk = MULTIHIT[i:i + 12]
            state([dict(item=hold, move0=m, move1="BULLETSEED", move2="SWORDSDANCE",
                        move3="PROTECT") for m in blk])
    # -- the five "-ate" abilities on Normal moves (the f32/f64 repeat) --------
    state([dict(ability=a, bability=a, move0=m, move1="TACKLE", move2="NASTYPLOT",
                move3="RETURN")
           for a in ATE for m in NORMAL_MOVES[:2]][:12])
    state([dict(ability=a, bability=a, move0=m, move1="BOOMBURST", move2="SWORDSDANCE",
                move3="DOUBLEEDGE")
           for a in ATE for m in NORMAL_MOVES[2:4]][:12])
    # -- Protean / Libero -----------------------------------------------------
    state([dict(ability=a, bability=a, move0="TACKLE", move1="SURF", move2="FLAMETHROWER",
                move3="SWORDSDANCE") for a in ("PROTEAN", "LIBERO")] * 6)
    # -- Terapagos / Stellar --------------------------------------------------
    tp = [dict(species=sp.upper(), tera_type="STELLAR", terastallized=t, stellar=st)
          for sp in ("terapagos", "terapagosterastal", "terapagosstellar")
          if sp in by_sp for t in ("true", "false") for st in (0, 1, 255)]
    state((tp + [dict(tera_type="STELLAR", terastallized="true")] * 12)[:12])
    # -- Palafin / Zero to Hero ----------------------------------------------
    state([dict(species=sp.upper(), ability="ZEROTOHERO", bability="ZEROTOHERO")
           for sp in ("palafin", "palafinhero") if sp in by_sp] * 6 or None)
    # -- Ditto / Imposter -----------------------------------------------------
    state([dict(species="DITTO", ability="IMPOSTER", bability="IMPOSTER",
                move0="TRANSFORM")] * 12)
    # -- Mimikyu / Disguise ---------------------------------------------------
    state([dict(species="MIMIKYU", ability="DISGUISE", bability="DISGUISE",
                illusion_broken=b) for b in ("true", "false")] * 6)
    # -- Illusion -------------------------------------------------------------
    state([dict(species=sp.upper(), ability="ILLUSION", bability="ILLUSION",
                illusion_broken=b)
           for sp in ("zoroark", "zoroarkhisui", "zorua", "zoruahisui") if sp in by_sp
           for b in ("true", "false")][:12])
    # -- Revival Blessing -----------------------------------------------------
    state([dict(move0="REVIVALBLESSING", move1="PROTECT", move2="SWORDSDANCE",
                move3="TACKLE", hp=0)] * 3 +
          [dict(move0="REVIVALBLESSING", move1="RECOVER")] * 9)
    # -- Rage Fist: the ONLY user of times_attacked, the PRUNED column ---------
    for ta in (0, 1, 2, 3, 6, 12, 36):
        state([dict(move0="RAGEFIST", move1="BULKUP", move2="DRAINPUNCH",
                    move3="TAUNT", times_attacked=ta)] * 12)
    # -- every setup move -----------------------------------------------------
    sm = [m for m in setup_moves() if m in DT.moves().ix]
    for i in range(0, len(sm), 12):
        blk = sm[i:i + 12]
        state([dict(move0=m, move1="TACKLE", move2="RECOVER", move3="PROTECT")
               for m in blk])
    return out


def distinguishable(rng, n=400):
    """LEG C states: every one of the 12 mons individually identifiable in BOTH
    an arm A column and the setup block.

    * 12 DISTINCT species, each with a DISTINCT `hp_frac` (arm A per-mon col 0),
      so "which mon is in arm A slot j" is decidable from the raw state string
      without asking either encoder.
    * a DISTINCT setup move per mon (distinct boost profile -> distinct
      `setup_boost_*`) and a distinct speed/offence profile (distinct
      `s1_outspeed_count` / `s1_ohko_count` / `s1_mtk_*`), so the 14-column setup
      block differs from mon to mon. A remap bug therefore MOVES VALUES, which is
      the only condition under which any whole-vector comparison can see it.
    * nothing is fainted and no volatile is set, so no slot is zeroed out and
      nothing can accidentally equalise two blocks.
    """
    sm = [m for m in setup_moves() if m in DT.moves().ix]
    hits = ["EARTHQUAKE", "SURF", "FLAMETHROWER", "THUNDERBOLT", "ICEBEAM", "PSYCHIC",
            "SHADOWBALL", "MOONBLAST", "CLOSECOMBAT", "LEAFSTORM", "DRACOMETEOR",
            "IRONHEAD", "PLAYROUGH", "AQUAJET", "BULLETPUNCH", "SUCKERPUNCH"]
    _pool()
    sets, out = list(_pool.sets), []
    for r in range(n):
        chosen, seen = [], set()
        i = r * 37
        while len(chosen) < 12:
            s = sets[i % len(sets)]
            i += 1
            if s[0] in seen:
                continue
            seen.add(s[0])
            chosen.append(s)
        # Species are drawn in POOL order, which is close to vocab-id order, so
        # arm A's bench sort would be a no-op and the remap would never be
        # exercised. Shuffling is what makes this corpus adversarial at all.
        rng.shuffle(chosen)
        mons = []
        for j, s in enumerate(chosen):
            m = mon(s[0], 100, s[2], s[3], s[4], s[5], rng,
                    move0=sm[(r * 12 + j) % len(sm)],
                    move1=hits[(r + j) % len(hits)],
                    move2=hits[(r + j * 3 + 5) % len(hits)],
                    move3="PROTECT",
                    status="NONE", terastallized="false", times_attacked=j)
            # distinct hp_frac: j+1 thirteenths of maxhp, never 0 and never equal
            m["hp"] = max(1, int(round(m["maxhp"] * (j + 1) / 13.0)))
            m["speed"] = 50 + 11 * j          # distinct speed order
            m["attack"] = 80 + 13 * j
            m["spa"] = 80 + 7 * j
            mons.append(m)
        st = G.St(G.flat_state(mons[:6], mons[6:], weather="NONE", terrain="NONE",
                               trick_room=False))
        for side in (0, 1):
            st.side(side, active_index=(r + side) % 6, side_conditions=";".join(["0"] * 19),
                    volatiles="", durations=";".join(["0"] * 13), substitute_health=0,
                    boost_atk=0, boost_def=0, boost_spa=0, boost_spd=0, boost_spe=0,
                    boost_acc=0, boost_eva=0, wish0=0, wish1=0, fs0=0, fs1=0,
                    force_switch="false", revival_blessing="false", force_trapped="false",
                    last_used_move="move:none", times_revived=0, last_move_failed="false")
        out.append(str(st))
    return out


# --------------------------------------------------------------------- driver
def build(n=0, seed=11):
    """`sweep + cross + targeted`, then `fuzz` up to `n` states (0 = no fuzz)."""
    rng = random.Random(seed)
    _pool()
    st = sweep_sets(rng) + cross(rng) + targeted(rng)
    if n and len(st) < n:
        st += G.fuzz_states(n - len(st), seed=seed)
    return st, coverage(st)


def coverage(states):
    """What the state set ACTUALLY reached, parsed back out of the strings."""
    sp, it, ab, te, mv, stat = (set() for _ in range(6))
    tera_on = ta_nz = 0
    for s in states:
        t = s.split("/")
        for si in (0, 1):
            for slot in t[si].split("=")[:6]:
                p = slot.split(",")
                if len(p) < 32 or p[G.MON["maxhp"]] == "0":
                    continue
                sp.add(p[G.MON["species"]])
                it.add(p[G.MON["item"]])
                ab.add(p[G.MON["ability"]])
                te.add(p[G.MON["tera_type"]])
                stat.add(p[G.MON["status"]])
                tera_on += p[G.MON["terastallized"]] == "true"
                ta_nz += int(p[G.MON["times_attacked"]]) > 0
                for k in ("move0", "move1", "move2", "move3"):
                    mv.add(p[G.MON[k]].split(";")[0])
    p = _pool()
    return dict(states=len(states), species=len(sp - {"NONE"}), species_pool=len(p.species),
                items=len(it - {"NONE"}), items_pool=len(p.items),
                abilities=len(ab - {"NONE"}), abilities_pool=len(p.abilities),
                moves=len(mv - {"NONE"}), moves_pool=len(p.moves),
                tera_types=len(te), tera_types_pool=len(p.tera_types),
                statuses=len(stat), terastallized_mons=tera_on,
                times_attacked_nonzero_mons=ta_nz)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    s, c = build(n)
    print(json.dumps(c, indent=1))
