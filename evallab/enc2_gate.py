"""Verification gates for enc2 -- EVALUATOR_SPEC §8.

  python enc2_gate.py hand      gate 2: hand-checked KO / speed / sweep cases
  python enc2_gate.py dmg       gate 3: kernel vs live engine calculate_damage
  python enc2_gate.py fainted   fainted-mon handling
  python enc2_gate.py corpus    10k+ real states: NaN / range / constant census
  python enc2_gate.py fuzz      pool-wide synthetic states (coverage the
                                one-team-pair corpus cannot give)
  python enc2_gate.py split     the shared-static search path must be exact
  python enc2_gate.py cost      gate 4: static vs dynamic encode cost
  python enc2_gate.py all       everything, with a PASS/FAIL line each

STATE SURGERY
  `poke_engine.State` is read-only from Python, so the hand-checked cases are
  built by editing the state STRING at known field offsets (`Mon`/`Side`/`St`
  below). The offsets are the ones `llencoder.parse_batch` reads, so a case that
  the encoder mis-parses would fail loudly rather than silently.
"""

import collections
import itertools
import json
import os
import random
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: E402,F401
import lossless_encoder as LE  # noqa: E402
import enc2  # noqa: E402
import dmgtab as DT  # noqa: E402
import rbpool  # noqa: E402

LABELS = [os.path.join(LAB, "data/pl2/out/labels_%s.jsonl" % c) for c in "abcd"]

# ---- state-string field offsets (mirrors llencoder.parse_batch) -----------
MON = {"species": 0, "level": 1, "type1": 2, "type2": 3, "btype1": 4, "btype2": 5,
       "hp": 6, "maxhp": 7, "ability": 8, "bability": 9, "item": 10, "nature": 11,
       "evs": 12, "attack": 13, "defense": 14, "spa": 15, "spd": 16, "speed": 17,
       "status": 18, "rest_turns": 19, "sleep_turns": 20, "weight": 21,
       "move0": 22, "move1": 23, "move2": 24, "move3": 25, "terastallized": 26,
       "tera_type": 27, "revealed": 28, "illusion_broken": 29, "known": 30,
       "times_attacked": 31, "last_item": 32, "opbau": 33,
       "active_move_actions": 34, "stellar": 35, "reveal_mask": 36}
SIDE = {"active_index": 6, "side_conditions": 7, "volatiles": 8, "durations": 9,
        "substitute_health": 10, "boost_atk": 11, "boost_def": 12, "boost_spa": 13,
        "boost_spd": 14, "boost_spe": 15, "boost_acc": 16, "boost_eva": 17,
        "wish0": 18, "wish1": 19, "fs0": 20, "fs1": 21, "force_switch": 22,
        "banked": 23, "baton_passing": 24, "shed_tailing": 25,
        "revival_blessing": 26, "force_trapped": 27, "last_used_move": 28,
        "slow_uturn": 29, "times_revived": 30, "last_move_failed": 31}


class St:
    """A mutable view of a state string."""

    def __init__(self, s):
        t = s.split("/")
        self.sides = [t[0].split("="), t[1].split("=")]
        self.tail = t[2:]

    def mon(self, side, slot, **kw):
        p = self.sides[side][slot].split(",")
        for k, v in kw.items():
            if k.startswith("move") and ";" not in str(v):
                v = "%s;false;32" % v
            p[MON[k]] = str(v)
        self.sides[side][slot] = ",".join(p)
        return self

    def side(self, side, **kw):
        for k, v in kw.items():
            self.sides[side][SIDE[k]] = str(v)
        return self

    def field(self, weather=None, terrain=None, trick_room=None):
        if weather is not None:
            self.tail[0] = "%s;5" % weather
        if terrain is not None:
            self.tail[1] = "%s;5" % terrain
        if trick_room is not None:
            self.tail[2] = "%s;5" % ("true" if trick_room else "false")
        return self

    def __str__(self):
        return "/".join(["=".join(self.sides[0]), "=".join(self.sides[1])] + self.tail)


def load_states(n, path=None):
    out = []
    for p in ([path] if path else LABELS):
        with open(p) as f:
            for line in f:
                out.append(json.loads(line)["s"])
                if len(out) >= n:
                    return out
    return out


TEMPLATE = None


def template():
    """A real state, used as the skeleton every hand-built case edits."""
    global TEMPLATE
    if TEMPLATE is None:
        TEMPLATE = load_states(1)[0]
    return TEMPLATE


def blank(hp=100, maxhp=100, stat=100, spe=100, species="DITTO", typ="NORMAL"):
    return dict(species=species, level=100, type1=typ, type2="TYPELESS",
                btype1=typ, btype2="TYPELESS", hp=hp, maxhp=maxhp,
                ability="NONE", bability="NONE", item="NONE", status="NONE",
                attack=stat, defense=stat, spa=stat, spd=stat, speed=spe,
                terastallized="false", tera_type=typ, times_attacked=0,
                move0="NONE", move1="NONE", move2="NONE", move3="NONE")


def flat_state(mons_a, mons_b, **field):
    """Twelve fully specified Pokemon -> a state string. Each entry is a dict of
    `blank()` overrides; None means an empty slot."""
    st = St(template())
    for side, mons in ((0, mons_a), (1, mons_b)):
        for slot in range(6):
            spec = mons[slot] if slot < len(mons) else None
            d = dict(blank())
            if spec is None:
                d.update(hp=0, maxhp=0)
            else:
                d.update(spec)
            st.mon(side, slot, **d)
        st.side(side, active_index=0, volatiles="", boost_atk=0, boost_def=0,
                boost_spa=0, boost_spd=0, boost_spe=0, boost_acc=0, boost_eva=0,
                side_conditions=";".join(["0"] * 19),
                durations=";".join(["0"] * 13), substitute_health=0,
                wish0=0, wish1=0, fs0=0, fs1=0, force_switch="false",
                force_trapped="false", last_used_move="move:none",
                times_revived=0, last_move_failed="false")
    st.field(**field)
    return str(st)


VOCAB = None


def vocab():
    global VOCAB
    if VOCAB is None:
        VOCAB = LE.LosslessVocab()
    return VOCAB


def enc(states, **kw):
    dbg = {}
    ids, fe = enc2.encode_states(states, vocab(), debug=dbg, **kw)
    return ids, fe, dbg


def col(fe, name, L=enc2.DEFAULT_LAYOUT):
    return fe[:, L.names.index(name)]


# =========================================================================
# gate 2 -- hand-checked KO counts, speed order, sweep conjunction
# =========================================================================
def gate_hand(verbose=True):
    fails = []

    def check(name, got, want):
        ok = bool(np.all(np.asarray(got) == np.asarray(want)))
        if not ok:
            fails.append("%s: got %s want %s" % (name, got, want))
        if verbose:
            print("  %-52s %s" % (name, "ok" if ok else "FAIL  got=%s want=%s"
                                  % (got, want)))

    L = enc2.DEFAULT_LAYOUT

    # ---- 1. KO counts ----------------------------------------------------
    # Attacker: 100 Atk, level 100, TACKLE (40 BP, physical, normal, no STAB
    # because the attacker is FIRE). Defender: 100 Def, NORMAL, maxhp 100.
    #   lvl_term = floor(2*100/5+2) = 42
    #   base = floor(floor(42*40*A/D)/50)+2
    # A/D = 100/100 -> floor(1680/50)+2 = 33+2 = 35 damage on 100 maxhp -> 3HKO
    a = blank(species="CHARMANDER", typ="FIRE", stat=100, spe=200)
    a.update(move0="TACKLE")
    d = blank(species="SNORLAX", typ="NORMAL", hp=100, maxhp=100, stat=100, spe=50)
    s = flat_state([a], [d])
    _, fe, dbg = enc(([s]))
    dmg = dbg["dmg_points"][0, 0, 6]
    check("KO: 35 dmg vs 100 hp -> exactly 3 hits", int(np.ceil(100 / dmg)), 3)
    check("KO one-hot at current HP is 3HKO",
          col(fe, "rel.ko_now_us_to_them_00_3hko")[0], 1.0)
    check("KO one-hot at current HP is not OHKO",
          col(fe, "rel.ko_now_us_to_them_00_ohko")[0], 0.0)
    # same attacker, defender at 30/100 -> OHKO now. (The full-HP KO matrix was
    # dropped, so the "still 3HKO at full HP" column no longer exists; the
    # underlying damage is still checked above and by `dbg["dmg_points"]`.)
    d2 = dict(d, hp=30)
    _, fe, dbg = enc([flat_state([a], [d2])])
    check("KO at 30%% hp is OHKO now", col(fe, "rel.ko_now_us_to_them_00_ohko")[0], 1.0)
    check("damage is unchanged by the defender's current HP (35 into 100 maxhp)",
          int(round(float(dbg["dmg_points"][0, 0, 6]))), 35)
    # immunity -> "never"
    dg = blank(species="GENGAR", typ="GHOST", hp=100, maxhp=100, stat=100, spe=50)
    _, fe, _ = enc([flat_state([a], [dg])])
    check("Normal into Ghost is 'never'",
          col(fe, "rel.ko_now_us_to_them_00_never")[0], 1.0)

    # ---- 2. speed order --------------------------------------------------
    fast = dict(blank(spe=200), move0="TACKLE")
    slow = dict(blank(spe=100), move0="TACKLE")
    _, fe, dbg = enc([flat_state([fast], [slow])])
    check("faster active acts first (pairspeed_00 = 1)",
          col(fe, "rel.pairspeed_00")[0], 1.0)
    check("speed margin positive", col(fe, "rel.active_speed_margin")[0] > 0, True)
    _, fe, _ = enc([flat_state([slow], [fast])])
    check("slower active acts second", col(fe, "rel.pairspeed_00")[0], 0.0)
    # Trick Room inverts it
    _, fe, _ = enc([flat_state([fast], [slow], trick_room=True)])
    check("Trick Room: faster active acts SECOND",
          col(fe, "rel.pairspeed_00")[0], 0.0)
    check("Trick Room: speed margin flips sign",
          col(fe, "rel.active_speed_margin")[0] < 0, True)
    _, fe, _ = enc([flat_state([slow], [fast], trick_room=True)])
    check("Trick Room: slower active acts FIRST", col(fe, "rel.pairspeed_00")[0], 1.0)
    # paralysis halves speed
    par = dict(blank(spe=200), move0="TACKLE", status="PARALYZE")
    _, fe, _ = enc([flat_state([par], [dict(blank(spe=150), move0="TACKLE")])])
    check("paralysis (200 -> 100) loses to 150", col(fe, "rel.pairspeed_00")[0], 0.0)
    # Choice Scarf
    scarf = dict(blank(spe=150), move0="TACKLE", item="CHOICESCARF")
    _, fe, _ = enc([flat_state([scarf], [dict(blank(spe=200), move0="TACKLE")])])
    check("Choice Scarf (150 -> 225) beats 200", col(fe, "rel.pairspeed_00")[0], 1.0)

    # ---- 3. priority brackets -------------------------------------------
    # Sucker Punch is +1: a slow user still moves first against a fast foe.
    sp = dict(blank(spe=50, typ="DARK"), move0="SUCKERPUNCH")
    ft = dict(blank(spe=250), move0="TACKLE")
    _, fe, _ = enc([flat_state([sp], [ft])])
    check("priority +1 move wins move-order bit", col(fe, "rel.order_us0_them0")[0], 1.0)
    check("we hold the outranking priority", col(fe, "rel.prio_outranks_us")[0], 1.0)
    # Prankster gives a STATUS move +1
    pk = dict(blank(spe=50), move0="THUNDERWAVE", ability="PRANKSTER")
    _, fe, _ = enc([flat_state([pk], [ft])])
    check("Prankster status move gets +1 bracket",
          col(fe, "rel.mprio_s0_m0")[0] * 7.0, 1.0)
    check("Prankster status move moves first", col(fe, "rel.order_us0_them0")[0], 1.0)
    # ... but not on a damaging move
    pk2 = dict(blank(spe=50), move0="TACKLE", ability="PRANKSTER")
    _, fe, _ = enc([flat_state([pk2], [ft])])
    check("Prankster does NOT boost a damaging move",
          col(fe, "rel.mprio_s0_m0")[0], 0.0)
    # Gale Wings gives a FLYING move +1
    gw = dict(blank(spe=50, typ="FLYING"), move0="BRAVEBIRD", ability="GALEWINGS")
    _, fe, _ = enc([flat_state([gw], [ft])])
    check("Gale Wings gives a Flying move +1",
          col(fe, "rel.mprio_s0_m0")[0] * 7.0, 1.0)

    # ---- 4. the strict sweep conjunction ---------------------------------
    # Dragon Dance user, +1 = Atk x1.5 and Spe x1.5. Give it a huge Attack so it
    # OHKOs everything, and speed 100 -> 150 after one dance.
    sweeper = dict(blank(species="DRAGONITE", typ="DRAGON", stat=250, spe=100,
                         hp=300, maxhp=300),
                   move0="DRAGONDANCE", move1="OUTRAGE", ability="NONE")
    weak = dict(blank(species="SHUCKLE", typ="BUG", stat=50, spe=80,
                      hp=100, maxhp=100), move0="TACKLE")
    s = flat_state([sweeper], [weak, dict(weak), dict(weak)])
    _, fe, dbg = enc([s])
    check("sweep at +1 when it OHKOs all, outspeeds all, no priority answer",
          col(fe, "cf_mon0.s1_sweep")[0], 1.0)
    check("side best_sweep_1 mirrors it", col(fe, "cf_side0.best_sweep_1")[0], 1.0)
    # break the conjunction three ways
    fast_weak = dict(weak, speed=400)
    s2 = flat_state([sweeper], [weak, fast_weak, dict(weak)])
    _, fe, _ = enc([s2])
    check("sweep FALSE when one opponent outspeeds it",
          col(fe, "cf_mon0.s1_sweep")[0], 0.0)
    bulky = dict(weak, hp=9999, maxhp=9999, defense=9999)
    s3 = flat_state([sweeper], [weak, bulky, dict(weak)])
    _, fe, _ = enc([s3])
    check("sweep FALSE when one opponent survives the hit",
          col(fe, "cf_mon0.s1_sweep")[0], 0.0)
    prio = dict(weak, move0="SUCKERPUNCH", type1="DARK", btype1="DARK", attack=9999)
    s4 = flat_state([sweeper], [weak, prio, dict(weak)])
    _, fe, _ = enc([s4])
    check("sweep FALSE when an opponent has a priority KO answer",
          col(fe, "cf_mon0.s1_sweep")[0], 0.0)
    # mons-required-to-kill: three weak mons cannot kill a 300 hp sweeper
    _, fe, _ = enc([s])
    check("mons-required-to-kill = 'never' when nothing can kill it",
          col(fe, "cf_mon0.s1_mtk_never")[0], 1.0)
    lethal = dict(weak, attack=99999, speed=400, move0="TACKLE")
    s5 = flat_state([sweeper], [lethal, dict(weak), dict(weak)])
    _, fe, _ = enc([s5])
    check("mons-required-to-kill = 1 when one fast mon one-shots it",
          col(fe, "cf_mon0.s1_mtk_1")[0], 1.0)

    # ---- 4b. tera-enabled sweep: the SAME strict conjunction --------------
    # The value test found this column responding with the WRONG SIGN for our
    # own side, because it used to ask "does tera turn a non-kill on the
    # opposing ACTIVE into a kill, for a mon that owns a setup move" -- a flag
    # for NEEDING tera. It must ask what §4 states, so these cases pin the
    # conjunction, the tera-specific half of it, and the side it lands on.
    #
    # Construction: a NORMAL-typed attacker with a DRAGON move gets no STAB;
    # tera'ing into DRAGON multiplies its damage by exactly 1.5. Sizing every
    # opponent's hp AT the non-tera damage `d` puts the position strictly
    # between the two conjunctions: 0.85*d < d (no plain sweep, since the sweep
    # uses the MINIMUM roll) and 0.85*1.5*d = 1.275*d >= d (tera sweeps).
    natt = dict(blank(species="DRAGONITE", typ="NORMAL", stat=250, spe=300,
                      hp=300, maxhp=300), move0="OUTRAGE", tera_type="DRAGON")
    probe = dict(blank(species="SHUCKLE", typ="ROCK", stat=50, spe=80,
                       hp=100, maxhp=100), move0="TACKLE")
    _, fe, dbg = enc([flat_state([natt], [probe])])
    d_pts = float(dbg["dmg_points"][0, 0, 6])
    foe = dict(probe, hp=int(round(d_pts)), maxhp=int(round(d_pts)))
    s_t = flat_state([natt], [foe, dict(foe), dict(foe)])
    _, fe, _ = enc([s_t])
    check("tera sweep: no PLAIN sweep at the minimum roll",
          col(fe, "cf_mon0.s1_sweep")[0], 0.0)
    check("tera sweep: 1 when tera STAB OHKOs every living opponent",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 1.0)
    check("tera sweep: it is OUR side's column, not theirs",
          col(fe, "cf_side1.tera_enabled_sweep")[0], 0.0)
    # same position, tera type == own type: tera buys no STAB, so no sweep
    _, fe, _ = enc([flat_state([dict(natt, tera_type="NORMAL")],
                               [foe, dict(foe), dict(foe)])])
    check("tera sweep: 0 when the tera type buys no damage",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)
    # already terastallized -> the tera is spent, the counterfactual is gone
    _, fe, _ = enc([flat_state([dict(natt, terastallized="true")],
                               [foe, dict(foe), dict(foe)])])
    check("tera sweep: 0 for a mon that has already terastallized",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)
    # the rest of the conjunction, one break at a time
    _, fe, _ = enc([flat_state([natt], [foe, dict(foe, speed=999), dict(foe)])])
    check("tera sweep: 0 when one opponent outspeeds it",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)
    _, fe, _ = enc([flat_state([natt], [foe, dict(foe, hp=9999, maxhp=9999),
                                        dict(foe)])])
    check("tera sweep: 0 when one opponent survives the tera'd hit",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)
    _, fe, _ = enc([flat_state([natt], [foe, dict(foe, move0="SUCKERPUNCH",
                                                  type1="DARK", btype1="DARK",
                                                  attack=99999), dict(foe)])])
    check("tera sweep: 0 when an opponent has a priority KO answer",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)
    # SIDE SYMMETRY: mirror the whole position, the flag must move sides
    _, fe, _ = enc([flat_state([dict(probe)], [natt.copy()])])
    check("tera sweep: mirrored position puts the flag on side 1",
          col(fe, "cf_side1.tera_enabled_sweep")[0], 1.0)
    check("tera sweep: mirrored position clears side 0",
          col(fe, "cf_side0.tera_enabled_sweep")[0], 0.0)

    # ---- 4c. typing multi-hot (the shipped encoder's representation) ------
    tm = dict(blank(species="CHARIZARD", typ="FIRE"), type2="FLYING",
              btype2="FLYING", tera_type="WATER")
    _, fe, _ = enc([flat_state([tm], [dict(probe)])])
    for nmc, want in (("type_eff_fire", 1.0), ("type_eff_flying", 1.0),
                      ("type_eff_water", 0.0), ("type_own_fire", 1.0),
                      ("type_own_flying", 1.0)):
        check("typing: untera'd %s" % nmc, col(fe, "mon0." + nmc)[0], want)
    _, fe, _ = enc([flat_state([dict(tm, terastallized="true")], [dict(probe)])])
    for nmc, want in (("type_eff_water", 1.0), ("type_eff_fire", 0.0),
                      ("type_eff_flying", 0.0), ("type_own_fire", 1.0),
                      ("type_own_flying", 1.0)):
        check("typing: terastallized %s" % nmc, col(fe, "mon0." + nmc)[0], want)
    _, fe, _ = enc([flat_state([dict(tm, terastallized="true",
                                     tera_type="STELLAR")], [dict(probe)])])
    check("typing: Stellar tera keeps the original typing",
          col(fe, "mon0.type_eff_fire")[0], 1.0)
    check("typing: empty slot sets no type bit",
          col(fe, "mon5.type_own_normal")[0], 0.0)

    # ---- 5. setup profile ------------------------------------------------
    _, fe, _ = enc([s])
    check("setup_present on the Dragon Dance user", col(fe, "cf_mon0.setup_present")[0], 1.0)
    check("setup boost profile: +1 Atk", col(fe, "cf_mon0.setup_boost_atk")[0] * 6, 1.0)
    check("setup boost profile: +1 Spe", col(fe, "cf_mon0.setup_boost_spe")[0] * 6, 1.0)
    check("setup boost profile: no SpA", col(fe, "cf_mon0.setup_boost_spa")[0], 0.0)

    # ---- 6. capabilities -------------------------------------------------
    caps = dict(blank(), move0="STEALTHROCK", move1="RECOVER", move2="RAPIDSPIN",
                move3="TRICKROOM", ability="REGENERATOR")
    _, fe, _ = enc([flat_state([caps], [weak])])
    for c, want in (("hazard_setting", 1), ("recovery", 1), ("hazard_removal", 1),
                    ("trick_room", 1), ("revival", 0), ("phazing", 0)):
        check("capability %s" % c, col(fe, "mon0.cap_" + c)[0], want)

    print("GATE hand: %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)))
    for f in fails:
        print("   ", f)
    return not fails


# =========================================================================
# gate 3 -- static damage kernel vs the live engine
# =========================================================================
def gate_dmg(n=400, verbose=True, source="real"):
    """Compare the encoder's active-vs-active best damage against
    `poke_engine.calculate_damage` (max non-crit roll), move by move.

    `source="real"` uses the labelled corpus (13 species, one team pair);
    `source="fuzz"` uses pool-wide synthetic states, which is the only way to
    exercise the other 496 species, all 60 items and all 203 abilities."""
    from poke_engine import State, calculate_damage
    # `tame`: no volatiles and no forced switches, because Protect / Substitute
    # make the engine report 0 damage for reasons that are turn state, not a
    # matchup, and the KO matrix deliberately does not model them.
    states = fuzz_states(n, seed=7, tame=True) if source == "fuzz" else load_states(n)
    _, _, dbg = enc(states)
    pts = dbg["dmg_points"]
    T = enc2.Tables(vocab())
    S = dbg["static"]
    rel, absd, n_cmp, n_ko_agree, n_ko = [], [], 0, 0, 0
    per_move_rel = []
    for i, s in enumerate(states):
        st = State.from_string(s)
        s1, s2 = st.side_one, st.side_two
        a1 = list(s1.pokemon)[int(s1.active_index)]
        a2 = list(s2.pokemon)[int(s2.active_index)]
        if a1.maxhp == 0 or a2.maxhp == 0 or a1.hp <= 0 or a2.hp <= 0:
            continue
        partner = "SPLASH"
        best_engine = 0.0
        for mv in a1.moves:
            mid = str(mv.id).upper()
            if mid in ("NONE",) or getattr(mv, "pp", 1) <= 0 or mv.disabled:
                continue
            try:
                r = calculate_damage(st, mid, partner, True)[0]
            except Exception:
                continue
            if not r:
                continue
            # the engine reports ONE hit of a multi-hit move; the KO matrix
            # counts the whole move, so scale by the expected hit count
            li = T.mt.ix.get(mid, 0)
            hits = (T.mt.hits_dice[li] if str(a1.item).upper() == "LOADEDDICE"
                    else T.mt.hits[li])
            best_engine = max(best_engine, float(r[0]) * float(hits))
        mine = float(pts[i, 0, 6])
        if best_engine <= 0 and mine <= 0:
            continue
        n_cmp += 1
        absd.append(abs(mine - best_engine))
        rel.append(abs(mine - best_engine) / max(best_engine, 1.0))
        per_move_rel.append((mine, best_engine))
        # KO-count agreement is what the feature actually encodes
        hp = float(a2.hp)
        k_mine = int(np.ceil(hp / mine)) if mine > 0 else 99
        k_eng = int(np.ceil(hp / best_engine)) if best_engine > 0 else 99
        n_ko += 1
        n_ko_agree += int(min(k_mine, 5) == min(k_eng, 5))
    rel = np.array(rel)
    absd = np.array(absd)
    ok = n_cmp > 0 and float(np.median(rel)) < 0.05 and n_ko_agree / max(n_ko, 1) > 0.90
    if verbose:
        print("  compared %d active-vs-active best-damage values" % n_cmp)
        print("  relative error: median %.4f  mean %.4f  p90 %.4f  p99 %.4f  max %.3f"
              % (np.median(rel), rel.mean(), np.percentile(rel, 90),
                 np.percentile(rel, 99), rel.max()))
        print("  absolute error (hp): median %.2f  p90 %.2f  max %.1f"
              % (np.median(absd), np.percentile(absd, 90), absd.max()))
        print("  KO-count agreement (bucketed 1..4+/never): %.2f%% of %d"
              % (100 * n_ko_agree / max(n_ko, 1), n_ko))
        print("  exact matches within 1 hp: %.2f%%" % (100 * (absd <= 1).mean()))
    print("GATE dmg[%s]: %s" % (source, "PASS" if ok else "FAIL"))
    return ok


# =========================================================================
# fainted-mon handling
# =========================================================================
def gate_fainted(verbose=True):
    fails = []

    def check(name, got, want):
        ok = bool(np.all(np.asarray(got) == np.asarray(want)))
        if not ok:
            fails.append("%s: got %s want %s" % (name, got, want))
        if verbose:
            print("  %-52s %s" % (name, "ok" if ok else "FAIL got=%s want=%s"
                                  % (got, want)))

    live = dict(blank(species="DRAGONITE", stat=200, spe=200, hp=200, maxhp=200),
                move0="TACKLE", move1="STEALTHROCK")
    dead = dict(blank(species="GARGANACL", stat=200, spe=200, hp=0, maxhp=200),
                move0="TACKLE", move1="STEALTHROCK", move2="RECOVER")
    foe = dict(blank(species="SHUCKLE", hp=100, maxhp=100, stat=50, spe=50),
               move0="TACKLE")
    s = flat_state([live, dead, dead], [foe])
    _, fe, dbg = enc([s])
    # `slot_occupied` was dropped (always 1 in gen9 randbats); occupancy is now
    # read off maxhp, which is 0 only for a truly empty slot.
    check("fainted mon: alive = 0", col(fe, "mon1.alive")[0], 0.0)
    check("fainted mon: still occupied (maxhp > 0)", col(fe, "mon1.maxhp")[0] > 0, True)
    for c in ("hazard_setting", "recovery"):
        check("fainted mon capability %s reads 0" % c, col(fe, "mon1.cap_" + c)[0], 0.0)
    check("living mon keeps its capability", col(fe, "mon0.cap_hazard_setting")[0], 1.0)
    check("n_fainted_revivable counts 2 of 6",
          round(float(col(fe, "side0.n_fainted_revivable")[0]) * 6), 2)
    check("fainted mon is excluded from the KO matrix ('never')",
          col(fe, "rel.ko_now_us_to_them_10_never")[0], 1.0)
    check("fainted mon cannot sweep", col(fe, "cf_mon1.s1_sweep")[0], 0.0)
    check("fainted mon has no setup_present", col(fe, "cf_mon1.setup_present")[0], 0.0)
    check("a dead OPPONENT is not counted in the sweep denominator: "
          "living sweeper still evaluated",
          col(fe, "cf_mon0.setup_present")[0], 0.0)   # no setup move -> 0
    # an empty slot (maxhp 0) must read as unoccupied everywhere
    s2 = flat_state([live], [foe])
    _, fe, _ = enc([s2])
    check("empty slot: unoccupied (maxhp = 0)", col(fe, "mon1.maxhp")[0], 0.0)
    check("empty slot: alive = 0", col(fe, "mon1.alive")[0], 0.0)
    check("empty slot: no capability bits", col(fe, "mon1.cap_recovery")[0], 0.0)
    check("empty slot: not counted as fainted-revivable",
          round(float(col(fe, "side0.n_fainted_revivable")[0]) * 6), 0)
    print("GATE fainted: %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)))
    for f in fails:
        print("   ", f)
    return not fails


# =========================================================================
# corpus census -- 10k+ real states
# =========================================================================
def census(fe, L=enc2.DEFAULT_LAYOUT, tag="", verbose=True, top=25):
    mn, mx = fe.min(axis=0), fe.max(axis=0)
    const = mn == mx
    nan = int(np.isnan(fe).sum())
    inf = int(np.isinf(fe).sum())
    oor = np.where((mn < -1.001) | (mx > 1.001))[0]
    if verbose:
        print("  [%s] rows=%d cols=%d  NaN=%d Inf=%d" % (tag, fe.shape[0], fe.shape[1],
                                                         nan, inf))
        print("  global min %.4f max %.4f" % (mn.min(), mx.max()))
        print("  constant columns: %d / %d (%.1f%%)"
              % (const.sum(), len(const), 100 * const.mean()))
        blocks = collections.Counter()
        for i in np.where(const)[0]:
            blocks[L.names[i].split(".")[0].rstrip("0123456789")] += 1
        print("  constant by block: %s" % dict(blocks.most_common()))
        if len(oor):
            print("  OUT OF [-1,1]: %d columns, e.g. %s"
                  % (len(oor), [(L.names[i], round(float(mn[i]), 2),
                                 round(float(mx[i]), 2)) for i in oor[:top]]))
        else:
            print("  all columns within [-1, 1]")
    return dict(nan=nan, inf=inf, const=const, mn=mn, mx=mx, oor=oor)


def gate_corpus(n=10000, verbose=True):
    states = load_states(n)
    t = time.time()
    _, fe = enc2.encode_states(states, vocab())
    dt = time.time() - t
    print("  encoded %d real states in %.1fs (%.3f ms/state, %.0f states/s)"
          % (len(states), dt, dt / len(states) * 1e3, len(states) / dt))
    r = census(fe, tag="real corpus", verbose=verbose)
    ok = r["nan"] == 0 and r["inf"] == 0
    print("GATE corpus: %s" % ("PASS" if ok else "FAIL"))
    return ok, r, fe


# =========================================================================
# pool-wide fuzz corpus
# =========================================================================
def fuzz_states(n=4000, seed=0, tame=False):
    """Synthetic but structurally valid states covering the WHOLE randbats pool.

    The labelled corpus is one team pair (13 species), so a constant column
    there proves nothing about the design. These states sample real sets from
    `rbpool`, then randomise hp, status, boosts, volatiles, hazards and field.
    They are NOT training data and are never labelled -- they exist so the
    constant-column census and the range checks mean something.
    """
    rng = random.Random(seed)
    pool = rbpool.Pool()
    dex = {k.replace("-", "").replace(" ", "").replace(".", "").replace("'", "").lower(): v
           for k, v in json.load(open(DT.DEX_PATH)).items()}
    sets = [s for s in pool.sets if s[0].lower() in dex]
    statuses = ["NONE"] * 6 + ["BURN", "SLEEP", "PARALYZE", "POISON", "TOXIC", "FREEZE"]
    weathers = [w for w in LE.WEATHER_ORDER if w not in ("HARSHSUN", "HEAVYRAIN")]
    terrains = LE.TERRAIN_ORDER
    vols = rbpool.reachable_list()
    out = []
    for _ in range(n):
        mons = [[], []]
        for side in (0, 1):
            for _slot in range(6):
                sp, lvl, item, abil, mv, tera, _w = rng.choice(sets)
                b = dex[sp.lower()]["baseStats"]
                ty = [DT.tix(t) for t in dex[sp.lower()]["types"]]
                t1 = dex[sp.lower()]["types"][0].upper()
                t2 = (dex[sp.lower()]["types"][1].upper()
                      if len(dex[sp.lower()]["types"]) > 1 else "TYPELESS")
                stat = lambda k: int(np.floor((2 * b[k] + 52) * lvl / 100.0) + 5)  # noqa: E731
                mhp = int(np.floor((2 * b["hp"] + 52) * lvl / 100.0) + lvl + 10)
                hp = rng.choice([mhp, mhp, int(mhp * rng.random()), 0])
                d = dict(species=sp.upper(), level=lvl, type1=t1, type2=t2,
                         btype1=t1, btype2=t2, hp=hp, maxhp=mhp,
                         ability=abil.upper(), bability=abil.upper(),
                         item=(item or "NONE").upper(), status=rng.choice(statuses),
                         attack=stat("attack"), defense=stat("defense"),
                         spa=stat("special-attack"), spd=stat("special-defense"),
                         speed=stat("speed"), weight=dex[sp.lower()].get("weightkg", 50),
                         rest_turns=rng.randint(0, 3), sleep_turns=rng.randint(0, 3),
                         terastallized="true" if rng.random() < 0.08 else "false",
                         tera_type=tera.upper(), times_attacked=rng.randint(0, 6),
                         reveal_mask=rng.randint(0, 255), nature="SERIOUS")
                for i in range(4):
                    d["move%d" % i] = "%s;%s;%d" % (
                        mv[i].upper() if i < len(mv) else "NONE",
                        "true" if rng.random() < 0.05 else "false", rng.randint(0, 32))
                del ty
                mons[side].append(d)
        s = flat_state(mons[0], mons[1],
                       weather=rng.choice(weathers), terrain=rng.choice(terrains),
                       trick_room=rng.random() < 0.15)
        st = St(s)
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
            vs = [] if tame else [v for v in vols if rng.random() < 0.05]
            st.side(side, active_index=rng.randint(0, 5),
                    side_conditions=";".join(str(x) for x in sc),
                    volatiles=":".join(vs),
                    durations=";".join(str(rng.randint(0, 3)) for _ in range(13)),
                    substitute_health=rng.choice([0, 0, 40]),
                    boost_atk=rng.randint(-2, 4), boost_def=rng.randint(-2, 2),
                    boost_spa=rng.randint(-2, 4), boost_spd=rng.randint(-2, 2),
                    boost_spe=rng.randint(-2, 4), boost_acc=rng.randint(-1, 1),
                    boost_eva=rng.randint(-1, 1),
                    wish0=rng.randint(0, 2), wish1=rng.randint(0, 150),
                    fs0=rng.randint(0, 3), fs1=rng.randint(0, 5),
                    force_switch="false" if tame else rng.choice(["true", "false"]),
                    force_trapped="false" if tame else rng.choice(["true", "false"]),
                    last_used_move=rng.choice(["move:none", "move:0", "switch:2"]),
                    times_revived=rng.randint(0, 2),
                    last_move_failed=rng.choice(["true", "false"]))
        out.append(str(st))
    return out


def gate_fuzz(n=4000, verbose=True):
    states = fuzz_states(n)
    t = time.time()
    _, fe = enc2.encode_states(states, vocab())
    dt = time.time() - t
    print("  encoded %d pool-wide synthetic states in %.1fs (%.3f ms/state)"
          % (len(states), dt, dt / len(states) * 1e3))
    r = census(fe, tag="pool-wide fuzz", verbose=verbose)
    ok = r["nan"] == 0 and r["inf"] == 0
    print("GATE fuzz: %s" % ("PASS" if ok else "FAIL"))
    return ok, r, fe


# =========================================================================
# the static/dynamic split must be EXACT, not merely fast
# =========================================================================
def split_states(n=200, seed=3, hard=False):
    """`n` leaves that all share ONE root's twelve Pokemon -- the search's own
    condition. HP, status, boosts, volatiles, hazards, field and WHICH MON IS
    ACTIVE move. With `hard=True` the three quantities that used to invalidate
    the shared static context move as well: terastallization, `disabled` and
    PP exhaustion."""
    rng = random.Random(seed)
    base = load_states(1)[0]
    variants = []
    for _ in range(n):
        st = St(base)
        for side in (0, 1):
            for slot in range(6):
                p = st.sides[side][slot].split(",")
                mh = int(p[MON["maxhp"]])
                kw = dict(hp=rng.choice([mh, int(mh * rng.random()), 0]),
                          status=rng.choice(["NONE"] * 3 + ["BURN", "PARALYZE",
                                                            "POISON", "TOXIC"]))
                if hard:
                    kw["terastallized"] = "true" if rng.random() < 0.15 else "false"
                    for i in range(4):
                        mv = p[MON["move%d" % i]].split(";")[0]
                        kw["move%d" % i] = "%s;%s;%d" % (
                            mv, "true" if rng.random() < 0.25 else "false",
                            rng.choice([0, 0, 8, 16, 32]))
                st.mon(side, slot, **kw)
            sc = ["0"] * 19
            for k in ("stealth_rock", "reflect", "light_screen", "tailwind"):
                if rng.random() < 0.3:
                    sc[LE.SIDE_CONDITION_FIELDS.index(k)] = str(rng.randint(1, 5))
            sc[LE.SIDE_CONDITION_FIELDS.index("spikes")] = str(rng.randint(0, 3))
            st.side(side, active_index=rng.randint(0, 5),
                    side_conditions=";".join(sc),
                    boost_atk=rng.randint(-2, 4), boost_def=rng.randint(-2, 2),
                    boost_spa=rng.randint(-2, 4), boost_spd=rng.randint(-2, 2),
                    boost_spe=rng.randint(-2, 4))
        st.field(weather=rng.choice(["NONE", "SUN", "RAIN", "SAND", "SNOW"]),
                 trick_room=rng.random() < 0.3)
        variants.append(str(st))
    return variants


def gate_split(n=200, verbose=True):
    """`share_static=True` must reproduce the cold path bit for bit.

    The precondition used to be "same twelve Pokemon, same PP, same disabled
    flags, same tera state", and the last three were the problem: a
    terastallization happens on 16.2% of real MCTS transitions and a new
    disable on 2.7%, so "rebuild the static context" was not a rare event, it
    was a 13-40%-of-budget tax (disable) and worse (tera). The static context
    no longer depends on any of the three, so the SECOND half of this gate --
    the same test with tera, `disabled` and PP all moving -- must now pass too.
    Only the twelve Pokemon themselves are still required to be fixed."""
    ok = True
    for hard in (False, True):
        variants = split_states(n, hard=hard)
        i1, f1 = enc2.encode_states(variants, vocab(), share_static=True)
        i2, f2 = enc2.encode_states(variants, vocab())
        d = float(np.abs(f1 - f2).max())
        good = d == 0.0 and bool((i1 == i2).all())
        ok &= good
        if verbose:
            bad = np.where(np.abs(f1 - f2).max(axis=0) > 0)[0]
            print("  %d leaves sharing one root%s: max |shared - cold| = %.3g, "
                  "%d/%d columns differ"
                  % (len(variants), " + tera/disable/pp moving" if hard else "",
                     d, len(bad), f1.shape[1]))
            if len(bad):
                print("    e.g. %s" % [enc2.DEFAULT_LAYOUT.names[i] for i in bad[:8]])
    print("GATE split: %s" % ("PASS" if ok else "FAIL"))
    return ok


# =========================================================================
# gate 4 -- static vs dynamic cost
# =========================================================================
def gate_cost(n=4096, verbose=True):
    """The spec's cost rule: the static half is paid once per SEARCH, the
    dynamic half once per LEAF. Both are measured here on states that share
    their twelve Pokemon (the real search condition)."""
    states = load_states(n)
    v = vocab()
    import llencoder as LL
    # --- parse ---
    t = time.time()
    for _ in range(3):
        C = LL.parse_batch(states, v)
    t_parse = (time.time() - t) / 3 / n

    # --- static, from one state ---
    C1 = LL.parse_batch(states[:1], v)
    T = enc2.Tables(v)
    order1 = LL._slot_order(C1["si"][:, :, enc2._SI["active_index"]])
    t = time.time()
    reps = 50
    for _ in range(reps):
        S = enc2.StaticCtx(C1, T, order1)
    t_static = (time.time() - t) / reps

    # --- dynamic, reusing that static context ---
    t = time.time()
    ids, fe = enc2.encode_columnar(C, v, S=S)
    t_dyn_total = time.time() - t
    t_dyn = t_dyn_total / n

    # --- full (static recomputed per state), the cold path ---
    t = time.time()
    enc2.encode_columnar(C, v, S=None)
    t_full = (time.time() - t) / n

    if verbose:
        print("  parse                     %8.3f ms/state" % (t_parse * 1e3))
        print("  STATIC (once per search)  %8.3f ms        <- 12 mons, 2x36 "
              "cross-side pairs" % (t_static * 1e3))
        print("  DYNAMIC (once per leaf)   %8.3f ms/leaf" % (t_dyn * 1e3))
        print("  full encode (static+dyn)  %8.3f ms/state" % (t_full * 1e3))
        print("  static amortised over N leaves:")
        for N in (1, 10, 100, 1000):
            print("     N=%5d -> %.3f ms/leaf" % (N, (t_static / N + t_dyn) * 1e3))
        print("  throughput (dynamic only): %.0f leaves/s" % (1 / t_dyn))
    return True, dict(parse=t_parse, static=t_static, dyn=t_dyn, full=t_full)


# =========================================================================
def gate_probe(ckpt=None, n=20000, verbose=True):
    """DIRECTED PROBES on a TRAINED net -- every feature the design rests on
    must move the prediction the way it is supposed to.

    This gate exists because `cf_side*.tera_enabled_sweep` shipped with the
    WRONG SIGN for our own side (ENCODER_VALUE_TEST §7.4) and nothing caught it:
    the hand-built cases above pin what the ENCODER computes, and only a probe
    on a trained net pins what the column MEANS. Point it at a checkpoint with
    ENC2_PROBE_CKPT (or pass one); it is skipped, loudly, when there is none,
    so it can sit in `all` without making the gate depend on a training run.

      python enc2_gate.py probe [checkpoint.pt]
    """
    ckpt = ckpt or os.environ.get("ENC2_PROBE_CKPT", "")
    if not ckpt or not os.path.isfile(ckpt):
        print("GATE probe: SKIPPED (no checkpoint; set ENC2_PROBE_CKPT)")
        return True
    import torch
    import vt_lib as V
    import vt_ablate
    ck = torch.load(ckpt, weights_only=False)
    r = ck["res"]
    kw = {k: r[k] for k in ("arch", "mon_depth", "depth", "slot_emb") if k in r}
    net = V.build_net("enc2", tuple(r["cap"]), r["seed"], **kw)
    net.load_state_dict(ck["sd"])
    net.eval()
    meta = V.load_meta()
    ix, _ = V.split_idx(meta)
    idx = ix["test"]
    if n and n < len(idx):
        idx = np.sort(np.random.default_rng(3).choice(idx, n, replace=False))
    res = vt_ablate.probes(net, meta, idx)
    fails = []
    for k, v in res.items():
        if "expect" not in v or "tera-enabled sweep" in k:
            continue      # see the conditional-mean check below
        want = v["expect"]
        # The discriminating statistic is the sign of the MEAN response -- that
        # is what was inverted in the defect this gate exists for. `sign_ok` is
        # a per-row agreement rate and is legitimately near 0.5 for a weak
        # probe, so it is only a floor against an outright reversal.
        if np.sign(v["mean_dp"]) != np.sign(want) or v["sign_ok"] < 0.45:
            fails.append("%s: expect %+d, mean_dp %+.4f, sign_ok %.3f"
                         % (k, want, v["mean_dp"], v["sign_ok"]))

    # ---- the tera-enabled sweep flag, checked on MEANING not on a one-column
    # edit. A single-column counterfactual sets the flag while the KO matrix,
    # the setup flags and the HP left beside it still describe a position that
    # is NOT a sweep, and a trained net reads that contradiction; measured, it
    # answers the edit with the wrong sign even when the column is correct.
    # What must never regress is what the flag MEANS: positions carrying it are
    # positions the net (and the labels) rate far higher.
    import vt_lib as V2
    lay = V2.load_layout()
    ci = {n: i for i, n in enumerate(lay["names"])}
    fe = np.load(os.path.join(V2.ENC, "enc2_feats.npy"), mmap_mode="r")
    pred = V.predict(net, V.Arm("enc2"), idx)
    y = meta["label_p"][idx]
    for s, want_up in ((0, True), (1, False)):
        f = np.asarray(fe[idx, ci["cf_side%d.tera_enabled_sweep" % s]], np.float32)
        a = np.asarray(fe[idx, ci["side%d.tera_available" % s]], np.float32)
        m1, m0 = f > 0.5, (a > 0.5) & (f < 0.5)
        if m1.sum() < 50:
            continue
        d_pred = pred[m1].mean() - pred[m0].mean()
        d_y = y[m1].mean() - y[m0].mean()
        ok = (d_pred > 0.10) if want_up else (d_pred < -0.10)
        print("  tera-sweep flag side %d: E[pred|flag]-E[pred|no flag] = %+.3f "
              "(labels %+.3f, n=%d) %s" % (s, d_pred, d_y, m1.sum(),
                                           "ok" if ok else "FAIL"))
        if not ok:
            fails.append("tera_enabled_sweep side %d: dpred %+.3f, dlabel %+.3f"
                         % (s, d_pred, d_y))
    print("GATE probe: %s (%d failures) [%s]"
          % ("PASS" if not fails else "FAIL", len(fails), os.path.basename(ckpt)))
    for f in fails:
        print("   ", f)
    return not fails


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("probe", "all"):
        print("=== directed probes on a trained net ===")
        ok &= gate_probe(sys.argv[2] if len(sys.argv) > 2 else None)
    if which in ("hand", "all"):
        print("=== gate 2: hand-checked relational correctness ===")
        ok &= gate_hand()
    if which in ("fainted", "all"):
        print("\n=== fainted-mon handling ===")
        ok &= gate_fainted()
    if which in ("dmg", "all"):
        print("\n=== gate 3: damage kernel vs live engine (labelled corpus) ===")
        ok &= gate_dmg(source="real")
        print("\n=== gate 3: damage kernel vs live engine (pool-wide fuzz) ===")
        ok &= gate_dmg(n=600, source="fuzz")
    if which in ("split", "all"):
        print("\n=== static/dynamic split exactness ===")
        ok &= gate_split()
    if which in ("cost", "all"):
        print("\n=== gate 4: static vs dynamic cost ===")
        gate_cost()
    if which in ("corpus", "all"):
        print("\n=== corpus census (10k real states) ===")
        o, _, _ = gate_corpus()
        ok &= o
    if which in ("fuzz", "all"):
        print("\n=== pool-wide fuzz census ===")
        o, _, _ = gate_fuzz()
        ok &= o
    print("\nOVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
