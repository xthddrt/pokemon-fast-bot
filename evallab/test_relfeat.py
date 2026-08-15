"""Hand-checkable correctness gate for relfeat.py.

Every assertion below is a fact about gen9 Pokemon that can be verified without
running anything -- which is the point. A relational feature that is silently
wrong would produce a clean null result in the experiment and we would conclude
"relational features do not help" when what we measured was a bug.

Run:  python test_relfeat.py
"""

import labenv  # noqa: F401
import numpy as np

import labteams
import relfeat
from encoder import Vocab, bench_order  # noqa: E402
from fp.battle import Battle, Move, Pokemon  # noqa: E402
from fp.helpers import normalize_name  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from poke_engine import State  # noqa: E402

FAILS = []


def check(name, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) <= tol
    print("%-4s %-46s got=%-10.4f want=%.4f" % ("ok" if ok else "FAIL", name, got, want))
    if not ok:
        FAILS.append(name)


def mk(spec):
    p = Pokemon(normalize_name(spec["species"]), spec["level"])
    p.ability = normalize_name(spec["ability"])
    p.item = normalize_name(spec["item"]) or None
    p.moves = [Move(normalize_name(m)) for m in spec["moves"]]
    p.tera_type = normalize_name(spec["teraType"])
    return p


def state_with(lead1, lead2, pair="A"):
    """Full-info state with a chosen lead on each side of `pair`."""
    t1 = [mk(s) for s in labteams.team_spec(pair, 0)]
    t2 = [mk(s) for s in labteams.team_spec(pair, 1)]
    i = [p.name for p in t1].index(lead1)
    j = [p.name for p in t2].index(lead2)
    t1 = [t1[i]] + [p for k, p in enumerate(t1) if k != i]
    t2 = [t2[j]] + [p for k, p in enumerate(t2) if k != j]
    b = Battle("g")
    b.battle_type = None
    b.user.active, b.user.reserve = t1[0], t1[1:]
    b.opponent.active, b.opponent.reserve = t2[0], t2[1:]
    return State.from_string(battle_to_poke_engine_state(b).to_string())


def main():
    G = {n: i for i, n in enumerate(relfeat.GX_NAMES)}
    S = {n: i for i, n in enumerate(relfeat.SX_NAMES)}

    # ---- type chart itself
    check("chart FIRE->GRASS", relfeat.type_mult("FIRE", ("GRASS",)), 2.0)
    check("chart GROUND->FLYING", relfeat.type_mult("GROUND", ("FLYING",)), 0.0)
    check("chart ROCK->dragonite(DRAGON,FLYING)", relfeat.type_mult("ROCK", ("DRAGON", "FLYING")), 2.0)
    check("chart ELECTRIC->GROUND/STEEL", relfeat.type_mult("ELECTRIC", ("GROUND", "STEEL")), 0.0)
    check("chart FIGHTING->corviknight(FLYING,STEEL)",
          relfeat.type_mult("FIGHTING", ("FLYING", "STEEL")), 1.0)

    # ---- dragonite (Dragon/Flying, base spd 80, L74) vs great tusk (Ground/Fighting, 87, L77)
    st = state_with("dragonite", "greattusk")
    gx = relfeat.global_extra(st)
    a1 = list(st.side_one.pokemon)[int(st.side_one.active_index)]
    a2 = list(st.side_two.pokemon)[int(st.side_two.active_index)]
    print("\n  dragonite spd=%d  greattusk spd=%d" % (a1.speed, a2.speed))
    # dragonite: earthquake(ground) 1x, ironhead(steel) 1x, outrage(dragon) 1x into Ground/Fighting
    check("A off_us dragonite->greattusk (x1)", gx[G["off_us"]] * 4, 1.0)
    # great tusk's RAPID SPIN set is bulkup/closecombat/earthquake/rapidspin:
    # closecombat(fighting) 0.5x, earthquake(ground) 0x, rapidspin(normal) 1x
    check("A off_them greattusk->dragonite (x1)", gx[G["off_them"]] * 4, 1.0)
    # great tusk STAB = ground/fighting -> best is closecombat at 0.5x (EQ is 0x)
    check("A stab_them greattusk->dragonite (x0.5)", gx[G["stab_them"]] * 4, 0.5)
    # dragonite STAB = dragon/flying -> outrage 1x
    check("A stab_us dragonite->greattusk (x1)", gx[G["stab_us"]] * 4, 1.0)
    check("A they_first (tusk 87 base > nite 80)", gx[G["they_first"]], 1.0)
    check("A we_first", gx[G["we_first"]], 0.0)
    check("A spd_margin sign", 1.0 if gx[G["spd_margin"]] < 0 else 0.0, 1.0)

    # ---- rotom-wash (Electric/Water, Levitate) vs great tusk: EQ is 0x, and
    # rotom is NOT grounded, so Spikes cost it nothing while SR costs 1/8.
    st = state_with("rotomwash", "greattusk")
    gx = relfeat.global_extra(st)
    # LEVITATE zeroes Earthquake; closecombat 1x, rapidspin 1x -> 1x
    check("B off_them greattusk->rotomwash (levitate kills EQ)", gx[G["off_them"]] * 4, 1.0)
    # rotom hydropump(water) into Ground/Fighting = 2x; discharge(electric) 0x
    check("B off_us rotomwash->greattusk (x2)", gx[G["off_us"]] * 4, 2.0)
    check("B we_first (rotom 86 base L82 vs tusk 87 base L77)", gx[G["we_first"]], 1.0)

    # ---- hazards. SideConditions is read-only from python, so hazard_cost is
    # tested as the pure function it is, against a duck-typed conditions object.
    class SC:
        def __init__(self, sr, sp):
            self.stealth_rock, self.spikes = sr, sp

    st = state_with("dragonite", "greattusk")
    by = {p.id: p for p in st.side_one.pokemon}
    by.update({p.id: p for p in st.side_two.pokemon})
    # Dragonite: Dragon/Flying -> SR 2x = 1/4; NOT grounded so Spikes free; but
    # it carries Heavy-Duty Boots -> 0 either way.
    check("C dragonite (boots)", relfeat.hazard_cost(by["DRAGONITE"], SC(1, 3)), 0.0)
    # Rotom-Wash: Electric/Water -> SR 1x = 1/8; Levitate -> Spikes free.
    check("C rotomwash SR only (levitate)", relfeat.hazard_cost(by["ROTOMWASH"], SC(1, 3)), 1 / 8, 1e-6)
    check("C rotomwash no hazards", relfeat.hazard_cost(by["ROTOMWASH"], SC(0, 0)), 0.0)
    # Corviknight: Flying/Steel -> Rock is 2x on Flying, 0.5x on Steel = 1x = 1/8;
    # Flying -> Spikes free.
    check("C corviknight SR 1x, no spikes", relfeat.hazard_cost(by["CORVIKNIGHT"], SC(1, 2)), 1 / 8, 1e-6)
    # Great Tusk: Ground/Fighting -> Rock is 0.5x on both = 0.25x = 1/32;
    # grounded -> 2 Spikes layers = 1/6.
    check("C greattusk SR 0.25x + 2spikes", relfeat.hazard_cost(by["GREATTUSK"], SC(1, 2)),
          0.25 / 8 + 1 / 6, 1e-6)
    check("C greattusk 3 spikes only", relfeat.hazard_cost(by["GREATTUSK"], SC(0, 3)), 1 / 4, 1e-6)
    # Kingambit: Dark/Steel -> SR 0.5x = 1/16, grounded
    check("C kingambit SR 0.5x", relfeat.hazard_cost(by["KINGAMBIT"], SC(1, 0)), 0.5 / 8, 1e-6)

    # sx wiring: slot k of sx is slot k of the encoder bench block
    vocab = Vocab(frozen=True)
    mons = list(st.side_one.pokemon)
    bench = bench_order(mons, int(st.side_one.active_index), vocab)
    sx = relfeat.side_extra(st, st.side_one, st.side_two, bench)
    names = [p.id for _, p in bench]
    print("\n  bench slots:", names)
    for k, (_i, p) in enumerate(bench):
        off, _ = relfeat.offense(p, list(st.side_two.pokemon)[int(st.side_two.active_index)])
        check("C sx benchoff slot%d (%s)" % (k, names[k]),
              sx[S["benchoff_b%d" % k]] * 4, min(off, 4.0), 1e-5)

    # ---- swap symmetry: gx computed on the mirrored state must equal gx_swap(gx)
    st = state_with("kingambit", "ironvaliant")
    gx = relfeat.global_extra(st)
    sw = relfeat.gx_swap(gx)
    check("D swap involution", float(np.abs(relfeat.gx_swap(sw) - gx).max()), 0.0, 1e-6)
    check("D swap moves we_first->they_first", sw[G["they_first"]], gx[G["we_first"]])
    check("D swap negates spd_margin", sw[G["spd_margin"]], -gx[G["spd_margin"]])

    print()
    if FAILS:
        raise SystemExit("FAILED: %s" % FAILS)
    print("ALL RELFEAT CHECKS PASS")


if __name__ == "__main__":
    main()
