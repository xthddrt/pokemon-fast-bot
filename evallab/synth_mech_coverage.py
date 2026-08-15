"""Did LEG B's corpus actually reach the mechanics the gate is REQUIRED to cover?

`synth_pool.coverage` counts species/items/abilities/moves/tera. That is not the
same as "Loaded Dice multi-hit was exercised": the named awkward cases are the
ones the last port got wrong, so they are counted BY NAME here, parsed back out
of the emitted state strings (never from the generator's intent).
"""
import json
import os
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import labenv  # noqa: E402,F401
import enc2_gate as G  # noqa: E402
import synth_pool as SP  # noqa: E402

M = G.MON


def mons(states):
    for si_state in states:
        t = si_state.split("/")
        for si in (0, 1):
            for slot in t[si].split("=")[:6]:
                p = slot.split(",")
                if len(p) >= 32 and p[M["maxhp"]] != "0":
                    yield p


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    states, cov = SP.build(n, seed)

    setup_all = set(SP.setup_moves())
    c = {k: 0 for k in (
        "loaded_dice", "loaded_dice_with_multihit", "multihit_move",
        "ate_ability", "protean_libero", "terapagos", "tera_stellar",
        "stellar_terastallized", "palafin", "zerotohero", "ditto", "imposter",
        "mimikyu", "disguise", "illusion", "revival_blessing_move",
        "ragefist", "times_attacked_nonzero", "setup_move_on_mon")}
    setup_seen, mh = set(), set(SP.MULTIHIT)
    for p in mons(states):
        mv = {p[M["move%d" % i]].split(";")[0] for i in range(4)}
        it, ab, sp = p[M["item"]], p[M["ability"]], p[M["species"]]
        c["loaded_dice"] += it == "LOADEDDICE"
        c["loaded_dice_with_multihit"] += it == "LOADEDDICE" and bool(mv & mh)
        c["multihit_move"] += bool(mv & mh)
        c["ate_ability"] += ab in SP.ATE
        c["protean_libero"] += ab in ("PROTEAN", "LIBERO")
        c["terapagos"] += sp.startswith("TERAPAGOS")
        c["tera_stellar"] += p[M["tera_type"]] == "STELLAR"
        c["stellar_terastallized"] += (p[M["tera_type"]] == "STELLAR"
                                       and p[M["terastallized"]] == "true")
        c["palafin"] += sp.startswith("PALAFIN")
        c["zerotohero"] += ab == "ZEROTOHERO"
        c["ditto"] += sp == "DITTO"
        c["imposter"] += ab == "IMPOSTER"
        c["mimikyu"] += sp.startswith("MIMIKYU")
        c["disguise"] += ab == "DISGUISE"
        c["illusion"] += ab == "ILLUSION"
        c["revival_blessing_move"] += "REVIVALBLESSING" in mv
        c["ragefist"] += "RAGEFIST" in mv
        c["times_attacked_nonzero"] += int(p[M["times_attacked"]]) > 0
        hit = mv & setup_all
        c["setup_move_on_mon"] += bool(hit)
        setup_seen |= hit

    out = dict(pool_coverage=cov, mechanic_mon_counts=c,
               setup_moves_total=len(setup_all), setup_moves_reached=len(setup_seen),
               setup_moves_missing=sorted(setup_all - setup_seen),
               multihit_moves_total=len(mh),
               multihit_moves_reached=len({m for p in mons(states)
                                           for m in ({p[M["move%d" % i]].split(";")[0]
                                                      for i in range(4)} & mh)}),
               ate_abilities_reached=sorted({p[M["ability"]] for p in mons(states)}
                                            & set(SP.ATE)))
    print(json.dumps(out, indent=1))
    missing = [k for k, v in c.items() if v == 0] + out["setup_moves_missing"]
    print("MECHANIC COVERAGE: %s" % ("COMPLETE" if not missing else "GAPS %s" % missing))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
