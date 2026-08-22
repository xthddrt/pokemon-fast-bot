"""The three FIXED team pairs the lab is built on.

PAIR_A  primary        -- ~10k games, the in-distribution corpus
PAIR_B  half-shared    -- 6 of its 12 species are PAIR_A species (3 per side),
                          6 are new.  The transfer probe: how much of what the
                          net learned about A survives when half the roster is
                          familiar and half is not.
PAIR_C  disjoint       -- 12 species that appear in neither A nor B.  The pure
                          out-of-distribution probe.

Sets are the MODAL randbats set for each species (highest count in
gen9randombattle.json), so a pair is fully determined by its species list plus
that file -- no hidden randomness, and the pair is reproducible from the repo.

Species were chosen to span the archetypes the audit says the net cannot
currently compare: fast frail attackers vs slow bulky walls (speed order),
Ground/Flying/Steel/Fairy/Fire/Water cores (type effectiveness), hazard setters
and removers (hazard switch-in cost).
"""

import json

import labenv  # noqa: F401  (must import first)

_SETS = None

PAIR_A = (
    # side one: offence-leaning, two of them faster than most of side two
    ["dragonite", "glimmora:stealthrock", "rotomwash", "kingambit", "volcarona", "garganacl"],
    # side two: a defensive core plus two speed threats
    ["corviknight", "gholdengo", "greattusk:rapidspin", "ironvaliant", "slowkinggalar", "ogerpon"],
)

PAIR_B = (
    # 3 shared with A1 (dragonite, rotomwash, garganacl) + 3 new
    ["dragonite", "rotomwash", "garganacl", "hydreigon", "meowscarada", "tinkaton"],
    # 3 shared with A2 (corviknight, gholdengo, ironvaliant) + 3 new
    ["corviknight", "gholdengo", "ironvaliant", "clefable", "dragapult", "quagsire"],
)

PAIR_C = (
    ["skeledirge", "annihilape", "cinderace", "toxapex", "hatterene", "barraskewda"],
    ["azumarill", "amoonguss", "bisharp", "espeon", "sandaconda:stealthrock", "electrode"],
)

PAIRS = {"A": PAIR_A, "B": PAIR_B, "C": PAIR_C}

# Extra pairs without editing this file: EVALLAB_PAIR_JSON -> {"D0": [[6
# entries],[6 entries]], ...}, same "species[:requiredmove]" entry format,
# modal sets as above. Used by the onepair experiment's sampled candidates.
import os as _os
if _os.environ.get("EVALLAB_PAIR_JSON"):
    PAIRS.update({k: (v[0], v[1]) for k, v in
                  json.load(open(_os.environ["EVALLAB_PAIR_JSON"])).items()})


def sets():
    global _SETS
    if _SETS is None:
        _SETS = json.load(open(labenv.SETS_PATH))
    return _SETS


def modal_set(entry):
    """The most common randbats set for `entry`, as a build spec.

    `entry` is "species" or "species:requiredmove". The filter exists because
    the modal set is not always the interesting one: pair A's hazard features
    are identically zero unless SOMEONE on it carries Stealth Rock, and a
    feature that is constant across the corpus cannot be measured at all.
    """
    species, _, need = entry.partition(":")
    d = sets()[species]
    if need:
        d = {k: v for k, v in d.items() if need in k.split(",")[3:-1]}
        assert d, "no %s set carries %s" % (species, need)
    # sort by (-count, key) so ties are broken deterministically by the key text
    key = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    parts = key.split(",")
    return {
        "species": species,
        "level": int(parts[0]),
        "item": parts[1],
        "ability": parts[2],
        "moves": parts[3:-1],
        "teraType": parts[-1],
        "_setkey": key,
    }


def team_spec(pair, side):
    """pair in {'A','B','C'}, side in {0,1} -> list of 6 build specs."""
    return [modal_set(e) for e in PAIRS[pair][side]]


def species_of(pair, side):
    return [e.partition(":")[0] for e in PAIRS[pair][side]]


def validate():
    s = sets()
    for name, (t1, t2) in PAIRS.items():
        for side, team in enumerate((t1, t2)):
            sp = [e.partition(":")[0] for e in team]
            assert len(sp) == 6 and len(set(sp)) == 6, (name, side, "not 6 distinct")
            for x in sp:
                assert x in s, "%s: unknown species %r" % (name, x)
            for e in team:
                modal_set(e)
    a = set(species_of("A", 0)) | set(species_of("A", 1))
    b = set(species_of("B", 0)) | set(species_of("B", 1))
    c = set(species_of("C", 0)) | set(species_of("C", 1))
    return {
        "A_species": sorted(a),
        "B_species": sorted(b),
        "C_species": sorted(c),
        "A_and_B_shared": sorted(a & b),
        "A_and_C_shared": sorted(a & c),
        "B_and_C_shared": sorted(b & c),
    }


if __name__ == "__main__":
    v = validate()
    for k in v:
        print("%-16s n=%2d  %s" % (k, len(v[k]), " ".join(v[k])))
    for p in PAIRS:
        for side in (0, 1):
            for sp in team_spec(p, side):
                print("%s%d %-14s L%-3d %-14s %-16s %s | tera %s"
                      % (p, side + 1, sp["species"], sp["level"], sp["item"],
                         sp["ability"], ",".join(sp["moves"]), sp["teraType"]))
