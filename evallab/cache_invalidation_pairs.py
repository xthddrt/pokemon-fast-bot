"""CACHE-INVALIDATION PAIRS — the gen9 mid-battle identity changes, as
before/after state-string pairs for `leaf_prof invalidation`.

Each pair is IDENTICAL except for the fields one identity change actually
moves, as the engine itself moves them (`genx/abilities.rs` for the forme
changes, `apply_transform` for Imposter, the `illusion_broken` /
`terastallized` flags for the other two). The Rust side primes the enc2 static
cache with `before`, encodes `after` through the cached path, and demands it be
bit-identical to `after` encoded on a virgin thread with an empty cache.

Emits `label <TAB> before <TAB> after` on stdout.
"""
import copy
import os
import random
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: E402,F401
import synth_pool as SP  # noqa: E402

SEED = 20260815


def filler(rng):
    """Eleven ordinary mons, fixed across every pair, all alive so `wrap`'s
    active-slot draw is identical for before and after."""
    out = []
    for sp, ab, it, mv in [
        ("dragapult", "INFILTRATOR", "CHOICESPECS", ["SHADOWBALL", "DRACOMETEOR", "UTURN", "THUNDERBOLT"]),
        ("greattusk", "PROTOSYNTHESIS", "BOOSTERENERGY", ["HEADLONGRUSH", "EARTHQUAKE", "CLOSECOMBAT", "RAPIDSPIN"]),
        ("garganacl", "PURIFYINGSALT", "LEFTOVERS", ["SALTCURE", "RECOVER", "EARTHQUAKE", "STEALTHROCK"]),
        ("gholdengo", "GOODASGOLD", "AIRBALLOON", ["MAKEITRAIN", "SHADOWBALL", "RECOVER", "NASTYPLOT"]),
        ("kingambit", "SUPREMEOVERLORD", "BLACKGLASSES", ["KOWTOWCLEAVE", "SUCKERPUNCH", "IRONHEAD", "SWORDSDANCE"]),
        ("ironvaliant", "QUARKDRIVE", "BOOSTERENERGY", ["MOONBLAST", "CLOSECOMBAT", "KNOCKOFF", "SWORDSDANCE"]),
        ("corviknight", "PRESSURE", "ROCKYHELMET", ["BRAVEBIRD", "BODYPRESS", "ROOST", "UTURN"]),
        ("slowkinggalar", "REGENERATOR", "HEAVYDUTYBOOTS", ["FUTURESIGHT", "SLUDGEBOMB", "CHILLYRECEPTION", "THUNDERWAVE"]),
        ("hydrapple", "REGENERATOR", "LEFTOVERS", ["FICKLEBEAM", "GIGADRAIN", "RECOVER", "NASTYPLOT"]),
        ("ogerpon", "DEFIANT", "LEFTOVERS", ["IVYCUDGEL", "KNOCKOFF", "HORNLEECH", "SWORDSDANCE"]),
        ("ceruledge", "FLASHFIRE", "LIFEORB", ["BITTERBLADE", "CLOSECOMBAT", "SHADOWSNEAK", "SWORDSDANCE"]),
    ]:
        m = SP.mon(sp, 100, it, ab, mv, "FIRE", rng)
        m["hp"] = m["maxhp"]  # keep every filler alive and fixed
        out.append(m)
    return out


def base(sp, ab, it, mv, tera, rng, **over):
    m = SP.mon(sp, 100, it, ab, mv, tera, rng)
    m["hp"] = m["maxhp"]
    m["status"] = "NONE"
    m["terastallized"] = "false"
    m.update(over)
    return m


def reforme(before, sp, rng, **over):
    """`before`, refolded onto a new forme's dex row: species, base/current
    types, base stats, weight and maxhp all come from `sp`, every dynamic field
    (hp, status, pp/disabled flags, times_attacked, reveal mask, tera) is
    carried over unchanged. This is what `active.id = X; recalculate_stats()`
    leaves behind."""
    mv = [before["move%d" % i].split(";")[0] for i in range(4)]
    after = SP.mon(sp, before["level"], before["item"], before["ability"], mv,
                   before["tera_type"], rng)
    for k in ("hp", "status", "rest_turns", "sleep_turns", "terastallized",
              "times_attacked", "reveal_mask", "move0", "move1", "move2",
              "move3", "bability"):
        after[k] = before[k]
    after.update(over)
    return after


def pair(label, mon_before, mon_after, out):
    """Twelve mons -> two state strings differing ONLY in side-one slot 0."""
    for who, m0 in (("b", mon_before), ("a", mon_after)):
        rng = random.Random(SEED)
        mons = [m0] + filler(rng)[:5]
        mons += filler(random.Random(SEED + 1))[:6]
        s = SP.wrap(mons, random.Random(SEED + 2), tame_vol=True)
        out.setdefault(label, {})[who] = s


def main():
    SP._pool()  # populate the dex/set tables `SP.mon` reads
    pairs = {}
    r = random.Random(SEED)

    # 1. Palafin -> Palafin-Hero (Zero to Hero on switch-out; the engine also
    #    calls recalculate_stats, so attack 70->160 etc. move with the id).
    pal = base("palafin", "ZEROTOHERO", "CHOICEBAND", ["JETPUNCH", "WAVECRASH", "FLIPTURN", "CLOSECOMBAT"], "WATER", r)
    pair("palafin_zero_to_hero", pal, reforme(pal, "palafinhero", r), pairs)

    # 2. Terapagos -> Terapagos-Terastal (Tera Shift on switch-in: stats, maxhp
    #    AND ability Tera Shift -> Tera Shell).
    ter = base("terapagos", "TERASHIFT", "LEFTOVERS", ["TERASTARSTORM", "EARTHPOWER", "CALMMIND", "PROTECT"], "STELLAR", r)
    pair("terapagos_tera_shift", ter,
         reforme(ter, "terapagosterastal", r, ability="TERASHELL", bability="TERASHELL"), pairs)

    # 3. Mimikyu -> Mimikyu-Busted (Disguise). Identical base stats, types,
    #    weight, ability and moves; the form costs maxhp/8, which is DYNAMIC.
    mim = base("mimikyu", "DISGUISE", "LEFTOVERS", ["PLAYROUGH", "SHADOWSNEAK", "SWORDSDANCE", "SUBSTITUTE"], "GHOST", r)
    busted = reforme(mim, "mimikyubusted", r)
    busted["hp"] = mim["maxhp"] - mim["maxhp"] // 8
    pair("mimikyu_disguise_busted", mim, busted, pairs)

    # 4. Ditto / Imposter transform: apply_transform overwrites id, types,
    #    ability, all five stats, weight and all four moves. HP is never touched.
    dit = base("ditto", "IMPOSTER", "CHOICESCARF", ["TRANSFORM", "NONE", "NONE", "NONE"], "NORMAL", r)
    tra = SP.mon("dragapult", 100, "CHOICESCARF", "INFILTRATOR",
                 ["SHADOWBALL", "DRACOMETEOR", "UTURN", "THUNDERBOLT"], "NORMAL", r)
    for k in ("hp", "maxhp", "status", "rest_turns", "sleep_turns",
              "terastallized", "times_attacked", "reveal_mask", "bability"):
        tra[k] = dit[k]
    pair("ditto_imposter_transform", dit, tra, pairs)

    # 5. Zoroark / Illusion break. The engine flips `illusion_broken` and
    #    nothing else; RawState does not even carry the flag, so the correct
    #    behaviour is exact reuse with NO rebuild.
    zor = base("zoroark", "ILLUSION", "FOCUSSASH", ["NASTYPLOT", "DARKPULSE", "FLAMETHROWER", "SLUDGEBOMB"], "DARK", r,
               illusion_broken="false")
    zor_b = copy.deepcopy(zor)
    zor_b["illusion_broken"] = "true"
    pair("zoroark_illusion_break", zor, zor_b, pairs)

    # 6. Terastallization. ToggleTerastallized flips one bool; the declared tera
    #    TYPE is keyed and unchanged, and tera is an AXIS of the damage table,
    #    so again: exact reuse, no rebuild.
    tez = base("ironvaliant", "QUARKDRIVE", "BOOSTERENERGY",
               ["MOONBLAST", "CLOSECOMBAT", "KNOCKOFF", "SWORDSDANCE"], "FAIRY", r)
    tez_a = copy.deepcopy(tez)
    tez_a["terastallized"] = "true"
    pair("terastallization", tez, tez_a, pairs)

    # 7-8. Knock Off — the plain keyed change, and the one Stage 2 measured as
    #      the driver of single-entry thrash. Two flavours on purpose: Choice
    #      Scarf moves the setup block's outspeed counts, Leftovers moves the
    #      static but NOT its 14-column projection, which is what the encoder
    #      emits. Both must rebuild; only the first is OBSERVABLE at the output,
    #      and that contrast is why `keycheck` (whole-struct) and not this test
    #      is the load-bearing check for key completeness.
    ks = base("dragapult", "INFILTRATOR", "CHOICESCARF",
              ["SHADOWBALL", "DRACOMETEOR", "UTURN", "THUNDERBOLT"], "GHOST", r)
    ks_a = copy.deepcopy(ks)
    ks_a["item"] = "NONE"
    ks_a["last_item"] = "CHOICESCARF"
    pair("knock_off_choice_scarf", ks, ks_a, pairs)

    ko = base("garganacl", "PURIFYINGSALT", "LEFTOVERS",
              ["SALTCURE", "RECOVER", "EARTHQUAKE", "STEALTHROCK"], "FAIRY", r)
    ko_a = copy.deepcopy(ko)
    ko_a["item"] = "NONE"
    ko_a["last_item"] = "LEFTOVERS"
    pair("knock_off_leftovers", ko, ko_a, pairs)

    for label, d in pairs.items():
        assert d["b"] != d["a"], label
        sys.stdout.write("%s\t%s\t%s\n" % (label, d["b"], d["a"]))


if __name__ == "__main__":
    main()
