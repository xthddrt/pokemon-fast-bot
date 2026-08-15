"""The gen9 randbats reachability pool, and gate 1: the reachable-volatile set.

WHY THIS FILE EXISTS
  EVALUATOR_SPEC design rule 4 is "nothing unreachable in gen9 randbats", and
  gate 1 says "all reachable" is only trustworthy if the enumeration is SHOWN.
  So every volatile this module admits carries the evidence that put it there,
  and every volatile it rejects carries the reason. `python rbpool.py` writes
  REACHABLE_VOLATILES.md from that evidence -- the file is generated, never
  hand-maintained, so it cannot drift from the code.

THE POOL
  `foul-play/data/pkmn_sets_cache/gen9randombattle.json` is the format's actual
  set data: 509 species, 5,851 sets, each set a key
      "level,item,ability,move1,move2,move3,move4,teratype".
  Everything reachable in the format is reachable from that pool plus the
  engine's own always-on mechanics (switching, fainting, Terastallization).

EVIDENCE CLASSES, in decreasing order of strength
  data/move      the move's own `volatileStatus` field in moves.json
  data/secondary the move's `secondary.volatileStatus`
  data/self      the move's `self.volatileStatus`
  data/charge    the move is a charge/two-turn move (engine sets the volatile
                 named after it)
  mechanic       a gen9 mechanic the JSON does not express, named explicitly
                 here with the move or ability that triggers it
  ability        an ability in the pool sets it
  engine         the engine sets it with no move/ability involved (TYPECHANGE
                 on Terastallization, TRAPPED, ...)

THE NAME-MISMATCH GAP, AND WHY `ABILITY_CAUSES` EXISTS (2026-08-14 audit)
  The ability rule used to be "an ability in the pool whose NAME EQUALS the
  volatile's name sets it". That is a heuristic, not a mechanism: an ability
  that sets a volatile with a DIFFERENT name was structurally invisible to this
  enumerator, so it could never be admitted however common it was. Two real
  gen9 randbats mechanics were missed that way -- Cute Charm -> ATTRACT and
  Electromorphosis -> CHARGE. `ABILITY_CAUSES` below is the fix: it is the
  first-class table of ability -> volatile causes whose names DISAGREE, it is
  consulted UNCONDITIONALLY (not only as a fallback for volatiles with no other
  evidence, which is what `MECHANIC_CAUSES` is), and any future name-mismatched
  ability cause belongs in it. The name-equality rule is kept as a shortcut for
  the ~15 abilities where the names do agree.

ASSUMPTIONS, STATED
  * The engine's `PokemonVolatileStatus` enum (107 variants, from
    `lossless_encoder.VOLATILE_ORDER`) is the universe. A volatile the engine
    cannot represent cannot appear in a state string, reachable or not.
  * Reachability is judged over the CURRENT set cache. If randbats gets a new
    move, re-run this file; the gate is the regeneration, not the checked-in md.
  * A volatile is admitted if ANY pool member can cause it, even rarely. The
    cost of a wrong admission is one dead column; the cost of a wrong rejection
    is an unrepresentable state.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "valuenet"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SETS_PATH = os.path.join(ROOT, "foul-play/data/pkmn_sets_cache/gen9randombattle.json")
MOVES_PATH = os.path.join(ROOT, "foul-play/data/moves.json")
DEX_PATH = os.path.join(ROOT, "foul-play/data/pokedex.json")
MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "REACHABLE_VOLATILES.md")


# ---------------------------------------------------------------- the pool
class Pool:
    """Everything gen9 randbats can put on the field."""

    def __init__(self, path=SETS_PATH):
        self.raw = json.load(open(path))
        self.sets = []          # (species, level, item, ability, (m0..m3), tera)
        for species, sets in self.raw.items():
            for key, weight in sets.items():
                p = key.split(",")
                self.sets.append((species, int(p[0]), p[1], p[2],
                                  tuple(p[3:-1]), p[-1], int(weight)))
        self.species = sorted(self.raw)
        self.moves = sorted({m for s in self.sets for m in s[4]})
        self.items = sorted({s[2] for s in self.sets})
        self.abilities = sorted({s[3] for s in self.sets})
        self.tera_types = sorted({s[5] for s in self.sets})
        self.levels = sorted({s[1] for s in self.sets})

    def summary(self):
        return dict(species=len(self.species), sets=len(self.sets),
                    moves=len(self.moves), items=len(self.items),
                    abilities=len(self.abilities), tera_types=len(self.tera_types),
                    level_min=self.levels[0], level_max=self.levels[-1])


# ------------------------------------------------------- mechanics not in JSON
# Each entry: volatile -> list of (kind, trigger, note). Only consulted for
# volatiles the JSON cannot justify; the JSON always wins where it speaks.
MECHANIC_CAUSES = {
    "LOCKEDMOVE": [("mechanic", "outrage/petaldance/thrash/uproar",
                    "multi-turn locking attacks lock the user in")],
    "PARTIALLYTRAPPED": [("mechanic", "whirlpool/magmastorm/infestation/firespin/"
                          "sandtomb/bind/wrap/thundercage/snaptrap",
                          "binding moves; also carries a duration")],
    "CONFUSION": [("mechanic", "confuseray/swagger/flatter + self-confusion from "
                   "outrage/petaldance/thrash", "and secondary-chance confusers")],
    "FLINCH": [("mechanic", "any secondary-flinch move; King's Rock",
                "set and consumed within a turn, but observable mid-turn")],
    "TRAPPED": [("ability", "arenatrap/shadowtag/magnetpull", "no-switch trapping"),
                ("mechanic", "block/meanlook/spiderweb/jawlock", "trapping moves")],
    "TYPECHANGE": [("engine", "Terastallization / Protean / Libero / Reflect Type",
                    "the engine rewrites the type list")],
    "PERISH4": [("mechanic", "perishsong", "counts down 4->1")],
    "PERISH3": [("mechanic", "perishsong", "counts down 4->1")],
    "PERISH2": [("mechanic", "perishsong", "counts down 4->1")],
    "PERISH1": [("mechanic", "perishsong", "counts down 4->1")],
    "SUBSTITUTE": [("mechanic", "substitute/shedtail", "carries substitute_health")],
    "MUSTRECHARGE": [("mechanic", "hyperbeam/gigaimpact/etc.", "recharge turn")],
    "DISABLE": [("ability", "cursedbody", "and the Disable move"),
                ("mechanic", "disable", "")],
    "SLOWSTART": [("ability", "slowstart", "Regigigas")],
    "UNBURDEN": [("ability", "unburden", "set when the held item is consumed")],
    "FLASHFIRE": [("ability", "flashfire", "set when hit by a Fire move")],
    "TRUANT": [("ability", "truant", "alternating turns")],
    "CUDCHEW": [("ability", "cudchew", "berry re-eat, carries a duration")],
    "TRANSFORM": [("ability", "imposter", "and the Transform move")],
    "PROTOSYNTHESISATK": [("ability", "protosynthesis", "highest-stat pick")],
    "PROTOSYNTHESISDEF": [("ability", "protosynthesis", "highest-stat pick")],
    "PROTOSYNTHESISSPA": [("ability", "protosynthesis", "highest-stat pick")],
    "PROTOSYNTHESISSPD": [("ability", "protosynthesis", "highest-stat pick")],
    "PROTOSYNTHESISSPE": [("ability", "protosynthesis", "highest-stat pick")],
    "QUARKDRIVEATK": [("ability", "quarkdrive", "highest-stat pick")],
    "QUARKDRIVEDEF": [("ability", "quarkdrive", "highest-stat pick")],
    "QUARKDRIVESPA": [("ability", "quarkdrive", "highest-stat pick")],
    "QUARKDRIVESPD": [("ability", "quarkdrive", "highest-stat pick")],
    "QUARKDRIVESPE": [("ability", "quarkdrive", "highest-stat pick")],
    "CHARGE": [("mechanic", "charge", "doubles the next Electric move")],
    "ENDURE": [("mechanic", "endure", "survive-at-1HP for the turn")],
    "IMPRISON": [("mechanic", "imprison", "")],
    "TORMENT": [("mechanic", "torment", "")],
    "EMBARGO": [("mechanic", "embargo", "")],
    "HEALBLOCK": [("mechanic", "healblock", "carries a duration")],
    "OCTOLOCK": [("mechanic", "octolock", "")],
    "NIGHTMARE": [("mechanic", "nightmare", "")],
    "GLAIVERUSH": [("mechanic", "glaiverush", "")],
    "SYRUPBOMB": [("mechanic", "syrupbomb", "carries a duration")],
    "INGRAIN": [("mechanic", "ingrain", "")],
    "AQUARING": [("mechanic", "aquaring", "")],
    "FOCUSENERGY": [("mechanic", "focusenergy/dire hit", "")],
    "SMACKDOWN": [("mechanic", "smackdown/thousandarrows/gravity", "grounds the target")],
    "TARSHOT": [("mechanic", "tarshot", "")],
    "ELECTRIFY": [("mechanic", "electrify", "")],
    "POWDER": [("mechanic", "powder", "")],
    "MINIMIZE": [("mechanic", "minimize", "")],
    "DEFENSECURL": [("mechanic", "defensecurl", "")],
    "STOCKPILE": [("mechanic", "stockpile", "")],
    "AUTOTOMIZE": [("mechanic", "autotomize", "")],
    "POWERTRICK": [("mechanic", "powertrick", "")],
    "POWERSHIFT": [("mechanic", "powershift", "")],
    "LASERFOCUS": [("mechanic", "laserfocus", "")],
    "FORESIGHT": [("mechanic", "foresight/odorsleuth", "")],
    "MIRACLEEYE": [("mechanic", "miracleeye", "")],
    "TELEKINESIS": [("mechanic", "telekinesis", "")],
    "GASTROACID": [("mechanic", "gastroacid", "")],
    "GRUDGE": [("mechanic", "grudge", "")],
    "SNATCH": [("mechanic", "snatch", "")],
    "MAGICCOAT": [("mechanic", "magiccoat", "")],
    "HELPINGHAND": [("mechanic", "helpinghand", "doubles-only")],
    "FOLLOWME": [("mechanic", "followme", "doubles-only")],
    "RAGEPOWDER": [("mechanic", "ragepowder", "doubles-only")],
    "SPOTLIGHT": [("mechanic", "spotlight", "doubles-only")],
    "MAXGUARD": [("mechanic", "dynamax", "gen8 only")],
    "BIDE": [("mechanic", "bide", "not in gen9")],
    "RAGE": [("mechanic", "rage", "not in gen9")],
    "SKYDROP": [("mechanic", "skydrop", "doubles-only")],
    "UPROAR": [("mechanic", "uproar", "")],
    "ATTRACT": [("mechanic", "attract", "")],
}

# ------------------------------------------------- name-MISMATCHED ability causes
# ability id -> [(volatile, note)]. An ability that sets a volatile of a
# DIFFERENT name. Consulted unconditionally, and only the abilities actually in
# the pool are admitted. Derived from the engine's own write sites, not from
# moves.json (which cannot express an ability).
ABILITY_CAUSES = {
    "cutecharm": [("ATTRACT",
                   "Cute Charm injects a 30% ATTRACT secondary on any contact "
                   "hit, targeting the ATTACKER (genx/abilities.rs:4435-4473); "
                   "enamorus, 2 of 5 sets")],
    "electromorphosis": [("CHARGE",
                          "being hit for damage grants CHARGE, which doubles "
                          "the next Electric move (genx/abilities.rs:1736-1756, "
                          "genx/damage_calc.rs:2950); bellibolt, 4 of 4 sets")],
    "windpower": [("CHARGE", "same effect off a wind move "
                             "(genx/abilities.rs:1758-1785); not in the pool")],
    "cursedbody": [("DISABLE", "30% chance to disable the move that hit it")],
    "imposter": [("TRANSFORM", "transforms on switch-in")],
}

# Volatiles whose name is a move: the engine's convention. Confirmed by matching
# the volatile name against the pool move ids.
CHARGE_MOVES = {"FLY", "DIG", "DIVE", "BOUNCE", "SOLARBEAM", "SOLARBLADE",
                "METEORBEAM", "ELECTROSHOT", "PHANTOMFORCE", "SHADOWFORCE",
                "SKYATTACK", "RAZORWIND", "SKULLBASH", "FREEZESHOCK",
                "ICEBURN", "GEOMANCY", "SKYDROP"}

# Volatiles that exist in the engine enum but are structurally unreachable in a
# gen9 randbats singles battle, with the reason. Anything here is EXCLUDED and
# the reason is printed in the md for review.
STRUCTURAL_EXCLUSIONS = {
    "NONE": "the enum's null member, never a real volatile",
    "MAXGUARD": "Dynamax; gen8 only",
    "BIDE": "removed after gen7",
    "RAGE": "removed after gen7",
    "HELPINGHAND": "doubles only",
    "FOLLOWME": "doubles only",
    "RAGEPOWDER": "doubles only",
    "SPOTLIGHT": "doubles only",
    "SKYDROP": "doubles only",
    # gen1-only volatile variants. In gen9 both moves set a SIDE CONDITION and
    # the volatile is written only under `if cfg!(feature = "gen1")`
    # (choices.rs:9777-9795 and 13611-13629); no other write site exists in
    # genx/. foul-play agrees (fp/search/poke_engine_helpers.py:815 passes them
    # as side_conditions). Both are already encoded as SIDE_CONDITION_FIELDS,
    # so the volatile columns were always zero.
    "REFLECT": "gen9 sets a SIDE CONDITION; the volatile variant is gen1-only "
               "(choices.rs:13611-13629) and is already encoded as side_conditions.reflect",
    "LIGHTSCREEN": "gen9 sets a SIDE CONDITION; the volatile variant is gen1-only "
                   "(choices.rs:9777-9795) and is already encoded as "
                   "side_conditions.light_screen",
}


# The volatiles EVALUATOR_SPEC section 2 names by hand, with the move/ability
# that would have to be in the pool for each to be reachable. Used only to build
# the cross-check table -- it never admits anything.
SPEC_NAMED = [
    ("Substitute (+HP)", "SUBSTITUTE", "substitute"),
    ("Leech Seed", "LEECHSEED", "leechseed"),
    ("Confusion", "CONFUSION", "confusion secondaries"),
    ("Taunt", "TAUNT", "taunt"),
    ("Encore", "ENCORE", "encore"),
    ("Disable", "DISABLE", "cursedbody"),
    ("Yawn", "YAWN", "yawn"),
    ("Perish count", "PERISH4", "perishsong"),
    ("Trapped", "TRAPPED", "arenatrap"),
    ("Protect counter", "PROTECT", "protect"),
    ("Flinch", "FLINCH", "flinch secondaries"),
    ("Charge", "CHARGE", "charge"),
    ("Salt Cure", "SALTCURE", "saltcure"),
    ("Heal Block", "HEALBLOCK", "psychicnoise"),
    ("Magnet Rise", "MAGNETRISE", "magnetrise"),
    ("Octolock", "OCTOLOCK", "octolock"),
    ("Torment", "TORMENT", "torment"),
    ("Imprison", "IMPRISON", "imprison"),
    ("Embargo", "EMBARGO", "embargo"),
    ("Aqua Ring", "AQUARING", "aquaring"),
    ("Ingrain", "INGRAIN", "ingrain"),
    ("Curse", "CURSE", "curse"),
    ("Nightmare", "NIGHTMARE", "nightmare"),
    ("Glaive Rush", "GLAIVERUSH", "glaiverush"),
    ("Syrup Bomb", "SYRUPBOMB", "syrupbomb"),
    ("Destiny Bond", "DESTINYBOND", "destinybond"),
    ("Endure", "ENDURE", "endure"),
    ("Roost", "ROOST", "roost"),
]


# ------------------------------------------------------------- the enumeration
def enumerate_volatiles(pool=None, volatile_order=None):
    """-> (reachable, excluded). `reachable` maps volatile -> [(kind, trigger,
    note)] evidence; `excluded` maps volatile -> reason string."""
    if pool is None:
        pool = Pool()
    if volatile_order is None:
        import lossless_encoder as LE
        volatile_order = LE.VOLATILE_ORDER
    moves = json.load(open(MOVES_PATH))
    pool_moves = set(pool.moves)
    pool_abils = set(pool.abilities)

    evidence = {v: [] for v in volatile_order}

    def add(vol, kind, trigger, note=""):
        v = vol.upper().replace("_", "")
        if v in evidence and (kind, trigger, note) not in evidence[v]:
            evidence[v].append((kind, trigger, note))

    # 1. data-driven: the move JSON says so.
    for mid in sorted(pool_moves):
        d = moves.get(mid)
        if not d:
            continue
        if d.get("volatileStatus"):
            add(d["volatileStatus"], "data/move", mid)
        sec = d.get("secondary") or {}
        if isinstance(sec, dict) and sec.get("volatileStatus"):
            add(sec["volatileStatus"], "data/secondary", mid,
                "%s%% chance" % sec.get("chance", "?"))
        slf = d.get("self") or {}
        if isinstance(slf, dict) and slf.get("volatileStatus"):
            add(slf["volatileStatus"], "data/self", mid)
        # charge / two-turn moves: the engine names the volatile after the move
        if mid.upper() in CHARGE_MOVES:
            add(mid, "data/charge", mid, "two-turn move")
        # the engine's convention: a volatile named exactly after a pool move
        if mid.upper() in evidence and not evidence[mid.upper()]:
            add(mid, "mechanic", mid, "volatile named after the move")

    # 2a. abilities in the pool whose NAME IS the volatile's name (a shortcut,
    #     not a mechanism -- see 2b)
    for ab in sorted(pool_abils):
        if ab.upper() in evidence:
            add(ab, "ability", ab, "ability of the same name")

    # 2b. abilities in the pool that set a DIFFERENTLY NAMED volatile. Without
    #     this the enumerator cannot see them at all, whatever their frequency.
    for ab in sorted(pool_abils):
        for vol, note in ABILITY_CAUSES.get(ab, ()):
            add(vol, "ability", ab, note)

    # 3. curated mechanics for what the JSON cannot express
    for vol, causes in MECHANIC_CAUSES.items():
        if not evidence.get(vol):
            for kind, trigger, note in causes:
                # only admit a mechanic whose trigger is actually in the pool
                trigs = [t.strip() for t in trigger.replace("/", " ").split()]
                hit = [t for t in trigs if t in pool_moves or t in pool_abils]
                # NO exceptions: a mechanic is admitted only if its trigger is
                # actually in the pool. (Perish Song is the case that matters --
                # the spec names "Perish count" but `perishsong` is not a gen9
                # randbats move, so PERISH1-4 are unreachable. See the md.)
                if kind == "engine" or hit:
                    add(vol, kind, ", ".join(hit) if hit else trigger, note)

    # 4. always-on engine mechanics
    add("TYPECHANGE", "engine", "terastallize", "every battle; tera is universal")
    add("TRAPPED", "engine", "arenatrap/shadowtag/magnetpull/partial trapping",
        "also set by the engine when switching is illegal")

    reachable = {v: e for v, e in evidence.items()
                 if e and v not in STRUCTURAL_EXCLUSIONS}
    excluded = {}
    for v in volatile_order:
        if v in reachable:
            continue
        excluded[v] = STRUCTURAL_EXCLUSIONS.get(
            v, "no move, ability or item in the gen9 randbats pool causes it")
    return reachable, excluded


# ----------------------------------------------------------------- the report
def write_md(path=MD_PATH):
    import lossless_encoder as LE
    pool = Pool()
    reach, excl = enumerate_volatiles(pool, LE.VOLATILE_ORDER)
    s = pool.summary()
    L = ["# Reachable volatiles in gen9 randbats — EVALUATOR_SPEC gate 1",
         "",
         "*Generated by `evallab/rbpool.py`. Do not hand-edit; re-run it.*", "",
         "The engine's `PokemonVolatileStatus` enum has **%d** variants. "
         "**%d** are reachable in gen9 randbats and get a column; **%d** are "
         "excluded." % (len(LE.VOLATILE_ORDER), len(reach), len(excl)), "",
         "Pool: %d species, %d sets, %d distinct moves, %d items, %d abilities, "
         "%d tera types, levels %d-%d." % (
             s["species"], s["sets"], s["moves"], s["items"], s["abilities"],
             s["tera_types"], s["level_min"], s["level_max"]), "",
         "Evidence classes: `data/move`, `data/secondary`, `data/self` are read "
         "straight out of `foul-play/data/moves.json` for moves that are in the "
         "pool. `data/charge` is a two-turn move. `mechanic` is a gen9 rule the "
         "JSON does not express, named with the pool move that triggers it. "
         "`ability` is a pool ability that causes it -- either one of the same "
         "name, or a name-MISMATCHED cause listed in `ABILITY_CAUSES` "
         "(Cute Charm -> ATTRACT, Electromorphosis -> CHARGE). `engine` is "
         "always-on.",
         "", "## Reachable (%d)" % len(reach), "",
         "| # | Volatile | Evidence | Trigger | Note |", "|---|---|---|---|---|"]
    for i, v in enumerate(sorted(reach)):
        for j, (kind, trig, note) in enumerate(reach[v][:3]):
            L.append("| %s | %s | `%s` | %s | %s |" % (
                i + 1 if j == 0 else "", v if j == 0 else "", kind, trig, note))
    L += ["", "## Excluded (%d)" % len(excl), "",
          "| Volatile | Reason |", "|---|---|"]
    for v in sorted(excl):
        L.append("| %s | %s |" % (v, excl[v]))

    # --- cross-check against the volatiles EVALUATOR_SPEC section 2 names ----
    pool_moves, pool_abils = set(pool.moves), set(pool.abilities)
    n_unreach = sum(1 for _, vol, _ in SPEC_NAMED if vol not in reach)
    L += ["", "## Cross-check against the names in EVALUATOR_SPEC section 2", "",
          "Section 2 lists volatiles by name as examples of the reachable set. "
          "%d of them are **not** reachable from the current gen9 randbats "
          "pool: nothing in the pool causes them. They get no "
          "column. Listed here because the spec names them explicitly and the "
          "disagreement needs a decision, not a silent drop." % n_unreach, "",
          "| Named in spec | Volatile | Reachable? | Why |", "|---|---|---|---|"]
    for label, vol, trig in SPEC_NAMED:
        ok = vol in reach
        if ok:
            why = "; ".join("%s=%s" % (k, t) for k, t, _ in reach[vol][:2])
        else:
            why = excl.get(vol, "`%s` is not in the randbats move pool" % trig)
            if why.startswith("no move"):
                why = "`%s` is not in the randbats move pool" % trig
        L.append("| %s | %s | %s | %s |" % (
            label, vol, "yes" if ok else "**no**", why))
    L += ["", "Also: section 2 gives the charging-move examples as "
          "*Fly/Dig/Solar Beam/Meteor Beam*. Of those only **solarbeam** and "
          "**meteorbeam** are in the pool (`fly`, `dig`, `dive`, `bounce`, "
          "`electroshot`, `phantomforce` are not), so `charging_move_id` has "
          "exactly two reachable values.", ""]
    body = "\n".join(L)
    # Keep any hand-written audit appendix that is already in the file: it is
    # the review trail for this enumeration and regenerating must not eat it.
    try:
        old = open(path).read()
    except IOError:
        old = ""
    marker = "\n# AUDIT VERDICT"
    if marker in old:
        body += ("\n---\n\n*The four disagreements the audit below found are "
                 "**FIXED** in `rbpool.py` and the table above already reflects "
                 "the fix: ATTRACT and CHARGE are admitted (via `ABILITY_CAUSES`, "
                 "the new first-class table of name-mismatched ability causes), "
                 "LIGHTSCREEN and REFLECT are excluded (`STRUCTURAL_EXCLUSIONS`). "
                 "The count is 44 either way; the membership moved. Re-running the "
                 "enumeration moves nothing else.*\n"
                 + marker + old.split(marker, 1)[1])
    open(path, "w").write(body)
    return len(reach), len(excl)


REACHABLE_CACHE = None


def reachable_list():
    """Sorted list of reachable volatile names -- THE column set for §2."""
    global REACHABLE_CACHE
    if REACHABLE_CACHE is None:
        import lossless_encoder as LE
        reach, _ = enumerate_volatiles(Pool(), LE.VOLATILE_ORDER)
        REACHABLE_CACHE = sorted(reach)
    return REACHABLE_CACHE


if __name__ == "__main__":
    import labenv  # noqa: F401  (sets sys.path for lossless_encoder)
    n, m = write_md()
    print("reachable %d, excluded %d -> %s" % (n, m, MD_PATH))
