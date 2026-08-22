"""SECOND PASS: is each sampled world's FULL opponent team one PS could build?

    .venv/bin/python audit_team_legality.py <run_dir> [--ref ps_reference.json]

The first pass (audit_sampling.py) scores only the mons the opponent really
had. This one scores the whole 6-slot team in every sampled world -- including
the slots the sampler INVENTED -- against a corpus of real Showdown teams
(gen_ps_reference.js). That is where the open findings live:

    #6  Species Clause keyed on the cosmetic forme -> two Alcremie in one team
    #13 revealed mons drawn independently -> teams PS can never build
    #14 Zoroark unreachable in the last unrevealed slot
    #15 Zacian-Crowned collapsing to the base forme
    #16 hazard setters/removers under-produced among invented mons

HARD checks fire on invariants the reference shows PS NEVER violates (0 in
20,000 teams): duplicate species, two Stealth Rock users, a species outside the
pool, an ability/level/tera/item the generator never gives that species.
DISTRIBUTIONAL checks compare rates PS does vary -- two Spikes users is 1.1%
real, so a rate is the only meaningful comparison there.

Items are exempted once Knock Off or consumption removed them: the sampler is
then CORRECT to carry no item, and the signature legitimately stops matching
any generated set.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_sampling import (  # noqa: E402
    norm, parse_team, revealed_by_decision, _species_tera_possible)

F_LEVEL, F_ABILITY, F_ITEM, F_MOVE0, F_TERA = 1, 8, 10, 22, 27
HARD_SINGLETON = "stealthrock"   # 0/20000 real teams have two


_DEX = None
_REF = None



_SETSJSON = None


def _factored_legal(sp, moves, ability, tera):
    """Is the tuple buildable per pokemon-showdown's own sets.json?

    The 1.05M-team reference is finite: Dugtrio-Alola Sand Force rides ~0.5%
    of teams and its flagged tuple has expected count ~2, so absence is Poisson
    noise, not illegality (run10k a98b36b9). Factored check: the moveset must
    fit inside ONE set's movepool, and ability/tera must be legal for that set.
    """
    global _SETSJSON
    if _SETSJSON is None:
        try:
            _SETSJSON = json.load(open(os.path.join(
                os.environ.get("FP_ROOT", "/Users/sallyliu/pokemon-fast-bot"),
                "pokemon-showdown/data/random-battles/gen9/sets.json")))
        except OSError:
            _SETSJSON = {}
    rec = None
    for k, v in _SETSJSON.items():
        if norm(k) == sp:
            rec = v
            break
    if not rec:
        return False
    for st in rec.get("sets", []):
        pool = {norm(x) for x in st.get("movepool", ())}
        if not set(moves) <= pool:
            continue
        abils = {norm(x) for x in st.get("abilities", ())}
        if abils and ability not in abils:
            continue
        teras = {norm(x) for x in st.get("teraTypes", ())}
        if tera and teras and tera not in teras:
            continue
        return True
    return False


def _nickname_species_map(plog):
    """nickname -> normalized species id, from |switch|/|drag|/|replace| lines.

    Transform/Trick exemptions were keyed on the NICKNAME from the event line
    ('Giratina'), but every legality check keys on the serialized SPECIES id
    ('giratinaorigin'), so formes never matched and the exemption silently
    missed -- run1000 flagged three transformed Dittos this way (grp0/g12,
    grp1/g219, grp2/g118). Details fields carry the true species.
    """
    m = {}
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            mm = re.search(
                r"\|(?:switch|drag|replace)\|p[12][ab]: ([^|]+)\|([^|,]+)", line)
            if mm:
                m.setdefault(norm(mm.group(1)), set()).add(norm(mm.group(2)))
    return m


def _reference():
    global _REF
    if _REF is None:
        try:
            _REF = json.load(open(os.path.join(HERE, "ps_reference.json")))["species"]
        except OSError:
            _REF = {}
    return _REF


def _pokedex():
    global _DEX
    if _DEX is None:
        try:
            _DEX = json.load(open(
                os.path.join(os.environ.get("FP_ROOT", "/Users/sallyliu/pokemon-fast-bot"), "foul-play/data/pokedex.json")))
        except OSError:
            _DEX = {}
    return _DEX


def _base_species(species):
    """PS baseSpecies id. Shields Down renames Minior-Indigo to Minior-Meteor
    mid-battle -- a BATTLE forme, so it is deliberately not in the cosmetic
    alias map -- but on OUR OWN side we still know it is the same mon."""
    global _DEX
    if _DEX is None:
        path = os.path.join(os.environ.get("FP_ROOT", "/Users/sallyliu/pokemon-fast-bot"), "foul-play/data/pokedex.json")
        try:
            _DEX = json.load(open(path))
        except OSError:
            _DEX = {}
    entry = _DEX.get(species) or {}
    # `battleOnly` first: Ogerpon-Cornerstone-Tera's battleOnly is
    # Ogerpon-Cornerstone (L76, its own pool entry), while its baseSpecies is
    # plain Ogerpon (L80) -- resolving to the latter flagged a correct
    # post-tera forme as an illegal level.
    battle_only = entry.get("battleOnly")
    if isinstance(battle_only, str) and battle_only:
        return norm(battle_only)
    return norm(entry.get("baseSpecies") or species)



_DENSE_CACHE = {}


def _battle_forme_abilities(species):
    """Abilities of `species` plus every BATTLE forme related to it.

    Walked in BOTH directions: the sampled mon may be named for the forme
    (terapagosterastal -> its own Tera Shell) or still named for the base while
    already carrying the forme's ability (terapagos -> Tera Shell).
    """
    dex = _pokedex()
    entry = dex.get(species) or {}
    out = {norm(x) for x in (entry.get("abilities") or {}).values()}
    base = norm(entry.get("battleOnly") or species)
    for key, e in dex.items():
        if not isinstance(e, dict):
            continue
        bo = e.get("battleOnly")
        if isinstance(bo, str) and norm(bo) in (base, species):
            out |= {norm(x) for x in (e.get("abilities") or {}).values()}
    return out


def _reference_is_dense(species):
    """Can absence from the reference mean anything for this species?

    The joint signature is the PRODUCT moves x item x ability x tera. When the
    corpus holds fewer draws than that product has cells, unvisited-but-legal
    cells are guaranteed and "not in the reference" stops being evidence.
    """
    if species in _DENSE_CACHE:
        return _DENSE_CACHE[species]
    rec = _reference().get(species) or {}
    sets = rec.get("sets") or {}
    if not sets:
        _DENSE_CACHE[species] = False
        return False
    moves, items, abils, teras = set(), set(), set(), set()
    for sig in sets:
        parts = sig.split("|")
        moves.add(parts[0]); items.add(parts[1])
        abils.add(parts[2]); teras.add(parts[3])
    space = len(moves) * len(items) * len(abils) * len(teras)
    _DENSE_CACHE[species] = rec.get("n", 0) >= space
    return _DENSE_CACHE[species]


_TAIL_CACHE = {}


def _moveset_tail_saturated(species):
    """Has the reference drawn this species enough to have seen every moveset?

    The moveset check reads "absent from 100k teams" as "PS never builds it",
    which only holds once the tail is exhausted. Good-Turing gives the mass PS
    still puts on movesets the reference never drew: the count of movesets seen
    EXACTLY ONCE. Non-zero means more unseen cells are out there and absence is
    not evidence -- thundurus keeps 4 singletons over 645 draws, and its
    sludgewave/thunderwave/uturn set is a real set (count 3 in the bot's own
    PS-derived dataset) that the reference simply never rolled.
    Costs 3 species of 509; every other species has an exhausted tail.
    """
    if species in _TAIL_CACHE:
        return _TAIL_CACHE[species]
    sets = (_reference().get(species) or {}).get("sets") or {}
    counts = collections.Counter()
    for sig, c in sets.items():
        counts[sig.split("|")[0]] += c
    _TAIL_CACHE[species] = bool(counts) and not any(
        c == 1 for c in counts.values())
    return _TAIL_CACHE[species]


def _as_dict(f):
    return {
        "species": norm(f[0]), "level": f[F_LEVEL],
        "ability": norm(f[F_ABILITY]), "item": norm(f[F_ITEM]),
        "tera": norm(f[F_TERA]),
        "moves": sorted(norm(f[j].split(";")[0]) for j in
                        range(F_MOVE0, F_MOVE0 + 4)
                        if f[j].split(";")[0] != "NONE"),
    }


def _our_team(run_dir):
    """Our own six, from the first |request| that carries our side.

    We KNOW our team exactly -- it is never sampled -- so every world must
    serialize it identically. That makes it the one place a mismatch is
    unambiguously a bug rather than an unlucky draw.
    """
    path = os.path.join(run_dir, "protocol.log")
    if not os.path.isfile(path):
        return {}
    for line in open(path, errors="replace"):
        if not line.startswith("|request|") or not line[9:].strip():
            continue
        try:
            req = json.loads(line[9:])
        except ValueError:
            continue
        mons = req.get("side", {}).get("pokemon")
        if not mons:
            continue
        out = {}
        for m in mons:
            sp = norm(m["details"].split(",")[0])
            lvl = "100"
            # details is "<Species>, L83, F" -- skip the SPECIES field, and
            # require L<digits>. `part.startswith("L")` alone reads "Luvdisc"
            # as level "uvdisc" (208 phantom wrong_level flags in batch g12),
            # and would do the same for Lapras, Lycanroc, Lucario...
            for part in m["details"].split(",")[1:]:
                part = part.strip()
                if re.fullmatch(r"L\d+", part):
                    lvl = part[1:]
            # BOTH, not just the live one. `ability` is what is ACTIVE right
            # now, `baseAbility` the mon's own: a Gardevoir that has Traced an
            # opponent's Guts reports ability=guts/baseAbility=trace, and Trace
            # re-copies on every switch-in, so serialising the reserve copy
            # under `trace` is correct. Grading against the live value alone
            # flagged 152 slots in batch-20260820-215623 g6.
            out[sp] = {"level": lvl,
                       "abilities": {norm(m.get("ability")),
                                     norm(m.get("baseAbility"))} - {""},
                       "moves": {norm(x) for x in m["moves"]}}
        # An ability can legitimately CHANGE mid-battle, and the first |request|
        # is then stale. Ogerpon terastallizes into Ogerpon-Teal-Tera and swaps
        # Defiant for Embody Aspect (run vg_t2: 128 phantom mismatches against
        # the turn-1 snapshot). Accept anything the server itself announced.
        for line in open(path, errors="replace"):
            m = re.search(r"\|-ability\|p1a: ([^|]+)\|([^|]+)", line)
            if m:
                for rec in out.values():
                    rec["abilities"].add(norm(m.group(2)))
            # Terapagos swaps Tera Shift -> Tera Shell on its forme change and
            # announces it as a formechange, not an -ability line.
            m = re.search(r"\|(?:detailschange|-formechange)\|p1a: ([^|]+)\|([^|,]+)", line)
            if m:
                nf = norm(m.group(2))
                # A forme change swaps the ability to the NEW forme's -- e.g.
                # Terapagos-Terastal runs Tera Shell where Terapagos runs Tera
                # Shift. Battle formes are not generated directly, so the
                # reference has no sets for them; take the abilities from the
                # pokedex instead.
                # ...and the BASE forme's too: PS reverts a fainted battle
                # forme in side requests (a fainted Terapagos-Stellar reports
                # ability=terashift again), and the bot serializes that server
                # truth byte-for-byte. run1000 grp1/g223: 64 phantom flags on
                # the faint-revert value.
                _forms = [nf]
                _bs = _pokedex().get(nf) or {}
                if _bs.get("baseSpecies"):
                    _forms.append(norm(_bs["baseSpecies"]))
                for _f in _forms:
                    entry = _pokedex().get(_f) or {}
                    for ab in (entry.get("abilities") or {}).values():
                        for r2 in out.values():
                            r2["abilities"].add(norm(ab))
                rec = _reference().get(nf) or {}
                for sig in rec.get("sets", {}):
                    for r2 in out.values():
                        r2["abilities"].add(sig.split("|")[2])
        return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default=os.path.join(HERE, "ps_reference.json"))
    a = ap.parse_args()

    ref = json.load(open(a.ref))
    R = ref["species"]
    COSMETIC = ref.get("cosmetic", {})
    # Gastrodon-East and Maushold-Four are the SAME battle entity as their base;
    # PS's pool is keyed on the base, so they must collapse before any pool or
    # Species-Clause test. Zacian-Crowned is NOT collapsed -- different stats.
    base = lambda sp: COSMETIC.get(sp, sp)

    truth = json.load(open(os.path.join(a.run_dir, "truth.json")))
    # `ident` is the nickname PS shows ("p2: Ursaluna"); `details` carries the
    # real species ("Ursaluna-Bloodmoon"). The protocol speaks idents, the
    # sampler speaks species -- joining on the wrong one made every knocked-off
    # Ursaluna-Bloodmoon look like an illegal item.
    ident2sp = {}
    true_species = set()
    for m in truth["team"]:
        ident = norm(m["ident"].split(": ")[1])
        sp = norm(m["details"].split(",")[0])
        ident2sp[ident] = sp
        true_species.add(base(sp))
        # Ice Face renames Eiscue to Eiscue-Noice mid-battle; it is not a
        # COSMETIC forme (the stats swap), so base() leaves it alone and the
        # mon stopped being recognised as one of the opponent's real six.
        true_species.add(_base_species(sp))
    legal_moves, legal_ab, legal_it, legal_tera, legal_lv, legal_sig = (
        {}, {}, {}, {}, {}, {})
    for sp, rec in R.items():
        legal_lv[sp] = set(rec["levels"])
        ms, ab, it, te, sg = set(), set(), set(), set(), set()
        for sig in rec["sets"]:
            mv, item, abil, tera = sig.split("|")
            ms.add(frozenset(mv.split(",")))
            ab.add(abil); it.add(item); te.add(tera); sg.add(sig)
        legal_moves[sp], legal_ab[sp] = ms, ab
        legal_it[sp], legal_tera[sp], legal_sig[sp] = it, te, sg

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "worlds.jsonl"))]
    per_dec = revealed_by_decision(a.run_dir)
    removed = set()
    for snap in per_dec.values():
        for ident, v in snap[1].items():
            if v == "__removed__":
                sp_ = ident2sp.get(ident, ident)
                # the loop compares base-normalized species, so a cosmetic
                # forme (vivillonelegant -> vivillon) would otherwise miss its
                # own knock-off exemption
                removed.update({sp_, base(sp_), _base_species(sp_)})

    # Mons the sampler had to build with populate_pkmn_from_fallback_set: it
    # deliberately MERGES revealed moves with a drawn set, so the result is not
    # a generated set by construction and "is this a real PS set?" is the wrong
    # question for it. What IS fair to ask is whether the fallback respected
    # negative item evidence -- finding #5 -- so that is checked instead.
    fallback_mons = set()
    blog = os.path.join(a.run_dir, "battle.log")
    if os.path.isfile(blog):
        for line in open(blog, errors="replace"):
            m = re.search(r"No candidate set survived the evidence for (\S+)", line)
            if m:
                fallback_mons.add(norm(m.group(1)))
    # Trick / Switcheroo hand a mon an item its species never generates, so the
    # PS item list stops applying once we have SEEN its item change.
    swapped = set()
    plog = os.path.join(a.run_dir, "protocol.log")
    if os.path.isfile(plog):
        _pending_items = []      # -item nicknames since the slot last changed
        for line in open(plog, errors="replace"):
            if re.search(r"\|(?:switch|drag)\|p2a:", line):
                _pending_items = []
            m = re.search(r"\|-item\|p2a: ([^|]+)\|", line)
            if m:
                _pending_items.append(norm(m.group(1)))
                sp_ = ident2sp.get(norm(m.group(1)), norm(m.group(1)))
                swapped.update({sp_, base(sp_), _base_species(sp_)})
            # Illusion: a |replace| means every -item since the slot changed
            # actually happened to the mon it reveals, not to the disguise.
            # run1000 grp3/g48: Zoroark-Hisui Tricked its Specs away while
            # disguised as Koraidon; the exemption landed on koraidon and
            # zoroarkhisui was graded item-illegal 176 times.
            m = re.search(r"\|replace\|p2a: [^|]+\|([^|,]+)", line)
            if m and _pending_items:
                sp_ = norm(m.group(1))
                swapped.update({sp_, base(sp_), _base_species(sp_)})
                # different worlds legitimately hypothesize the SIBLING forme
                # until the bot pins it -- exempt both (run25k, 5 illegal_item
                # on worlds carrying the other forme of a Trick-ing zoroark)
                if sp_ in ("zoroark", "zoroarkhisui"):
                    swapped.update({"zoroark", "zoroarkhisui"})

    # Abilities the server itself announced for the opponent. Trace/Receiver
    # copy another mon's ability, so the sampled mon legitimately carries one
    # its own species never generates.
    announced_ab = set()
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            m = re.search(r"\|-ability\|p2a: [^|]+\|([^|]+)", line)
            if m:
                announced_ab.add(norm(m.group(1)))

    # A TRANSFORMED mon (Ditto/Imposter) is serialized under the species it
    # COPIED, so every species-keyed check grades it against the wrong mon:
    # level, moveset, tera, item and the reveal history all belong to the
    # copy target. Three Dittos produced 1,256 missing-move + 457
    # incomplete-move + 96 level flags in the 100-game batch.
    transformed = set()
    _nick_map = _nickname_species_map(plog)
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            m = re.search(r"\|-transform\|p2a: ([^|]+)\|p1a: ([^|]+)", line)
            if m:
                # nickname norms AND the species each nickname resolves to --
                # 'Giratina' alone never matches species key 'giratinaorigin'
                for nick in (m.group(1), m.group(2)):
                    transformed.add(norm(nick))
                    transformed.update(_nick_map.get(norm(nick), ()))

    S = collections.Counter()
    examples = collections.defaultdict(list)
    permove = collections.defaultdict(collections.Counter)
    sampled_species = collections.Counter()
    n_worlds = 0

    for r in rows:
        _ours_raw, opp_raw = parse_team(r["state"])
        opp = [_as_dict(f) for f in opp_raw]
        if len(opp) < 6:
            S["short_team_%d" % len(opp)] += 1
        n_worlds += 1
        seen = collections.Counter()
        movecount = collections.Counter()
        # exclusive moves genuinely revealed on the opponent's team by now --
        # PS culls these from every later team slot, so invented mons can
        # legally carry re-rolled movesets/items the marginal reference never
        # shows (run10k ae5419c1: webs-less Choice-Specs Galvantula beside a
        # revealed Ariados with Sticky Web -- teams.ts:512 cull + the getItem
        # all-special branch). Transform copies deliberately DO NOT count: a
        # copied Sticky Web conditioning the fill-ins is bot bug B5, and an
        # exemption keyed on it would mask exactly that.
        _dec_rev = per_dec.get(r["decision"], ({}, {}, {}, {}))[0]
        _EXCL = ("stickyweb", "stealthrock", "spikes", "toxicspikes",
                 "defog", "rapidspin", "mortalspin")
        _culled_ctx = {
            mv for spx, mvs in _dec_rev.items()
            if spx not in transformed
            for mv in mvs if mv in _EXCL
        }
        # ...or carried by another NON-TRANSFORM mon in THIS world: PS's
        # sequential fill-in build is internally consistent, so one invented
        # mon taking webs legally forces the webless re-roll on a later one
        # (run10k grp3/g767, d13). Transform copies stay excluded -- a copied
        # Sticky Web conditioning the build is bot bug B5, and counting it
        # here would mask exactly that (grp3/g490 must keep flagging).
        _culled_ctx |= {
            mv for mm in opp
            if mm["species"] not in transformed
            for mv in mm["moves"] if mv in _EXCL
        }
        for m in opp:
            sp = m["species"]
            sampled_species[sp] += 1
            seen[base(sp)] += 1
            # transform copies carry the COPY TARGET's moves; counting them
            # made two legit worlds read as two-Stealth-Rock teams (run10k
            # a05bf559/a1cc3960 -- Imposter Ditto copying our SR carrier).
            # Same `transformed` set the per-mon checks already consume.
            if sp not in transformed:
                for mv in m["moves"]:
                    movecount[mv] += 1
            if base(sp) not in R and sp not in R and _base_species(sp) not in R:
                S["species_not_in_ps_pool"] += 1
                examples["species_not_in_ps_pool"].append((r["decision"], sp))
                continue
            if sp not in R:
                sp = base(sp) if base(sp) in R else _base_species(sp)
            if sp in transformed:
                S["transformed_exempt"] += 1
                continue
            if m["level"] not in legal_lv[sp]:
                S["illegal_level"] += 1
                examples["illegal_level"].append(
                    (r["decision"], sp, "L%s not in %s" % (m["level"], sorted(legal_lv[sp]))))
            # A BATTLE FORME's ability belongs to the forme: Terapagos-Terastal
            # runs Tera Shell where Terapagos runs Tera Shift, and PS announces
            # it as `|-activate|...|ability: Tera Shell`, which `announced_ab`
            # (an `|-ability|` regex) never sees. The sampled mon can also still
            # be NAMED for the base while carrying the forme's ability, so the
            # relation is walked in both directions.
            ok_ab = (legal_ab[sp] | _battle_forme_abilities(m["species"])
                     | _battle_forme_abilities(sp))
            if m["ability"] not in ok_ab and m["ability"] not in announced_ab:
                S["illegal_ability"] += 1
                examples["illegal_ability"].append(
                    (r["decision"], sp, "%s not in %s" % (m["ability"], sorted(legal_ab[sp]))))
            if (m["tera"] and m["tera"] not in legal_tera[sp]
                    # SCOPED zoroark-pivot-exit exemption: only when the
                    # truth team actually holds an Illusion form AND that
                    # form's reference pool carries the type -- a global
                    # poison/fighting pass would mask real bugs (run25k,
                    # 168 illegal_tera on greninja/Fighting = hisui zoroark
                    # tera'd under its disguise, no |replace| that stint)
                    and not any(
                        _zf in true_species
                        and _species_tera_possible(_zf, m["tera"])
                        for _zf in ("zoroark", "zoroarkhisui"))):
                S["illegal_tera"] += 1
                examples["illegal_tera"].append((r["decision"], sp, m["tera"]))
            if (sp not in removed and sp not in swapped
                    and sp not in fallback_mons
                    and m["item"] not in legal_it[sp]):
# applies to REVEALED-species-with-unknown-set too, not just
                # invented slots: a revealed Galvantula whose teammate showed
                # webs is exactly as cull-conditioned as an invented one
                # (grp3/g767 d13). B5 protection lives in the transform
                # exclusion of _culled_ctx, not here.
                if (_culled_ctx
                        and _factored_legal(sp, m["moves"], m["ability"],
                                            m["tera"])):
                    S["team_culled_variant"] += 1
                else:
                    S["illegal_item"] += 1
                    examples["illegal_item"].append(
                        (r["decision"], sp, "%s not in %s"
                         % (m["item"], sorted(legal_it[sp]))))
            if sp in fallback_mons:
                S["fallback_populated"] += 1
            elif not _moveset_tail_saturated(sp):
                S["skip_moveset_sparse"] += 1
            elif frozenset(m["moves"]) not in legal_moves[sp]:
                # applies to REVEALED-species-with-unknown-set too, not just
                # invented slots: a revealed Galvantula whose teammate showed
                # webs is exactly as cull-conditioned as an invented one
                # (grp3/g767 d13). B5 protection lives in the transform
                # exclusion of _culled_ctx, not here.
                if (_culled_ctx
                        and _factored_legal(sp, m["moves"], m["ability"],
                                            m["tera"])):
                    S["team_culled_variant"] += 1
                else:
                    S["illegal_moveset"] += 1
                examples["illegal_moveset"].append(
                    (r["decision"], sp, ",".join(m["moves"])))
            else:
                S["ok_moveset"] += 1
            # JOINT signature. The four checks above are INDEPENDENT, so a mon
            # can pass all of them and still be a set PS never builds -- a
            # Choice-Band moveset wearing Leftovers. Only applied to INVENTED
            # slots: a revealed mon's item/tera legitimately drift from its
            # generated set as the game knocks items off and teras happen.
            # SPARSITY-GATED (Sally 2026-08-20). The joint signature is the
            # product moves x item x ability x tera, so a species whose
            # signature space dwarfs its reference draws always shows a few
            # unvisited-but-legal cells -- thundurus is 1317 signatures against
            # 645 draws, mew 971 against 175. Every flag this produced across
            # 200 games was that, never a real defect, and the noise masks
            # genuine HARD violations. Only score species the reference covers
            # densely enough for absence to mean anything.
            # measured against the SIGNATURE SPACE (movesets x items x
            # abilities x teras), not the signatures already SEEN: thundurus has
            # 130 seen but a 1317-cell space against 645 draws, so counting seen
            # signatures made a 2x under-sampled species look dense.
            _dense = _reference_is_dense(sp)
            if (_dense
                    and sp not in true_species
                    and base(m["species"]) not in true_species
                    and _base_species(m["species"]) not in true_species
                    and sp not in fallback_mons
                    and sp not in removed and sp not in swapped):
                sig = "|".join([",".join(sorted(m["moves"])),
                                m["item"] or "none", m["ability"], m["tera"]])
                if sig in legal_sig[sp]:
                    S["ok_joint_signature"] += 1
                elif _factored_legal(sp, m["moves"], m["ability"], m["tera"]):
                    # legal per PS's own sets.json -- absence from the finite
                    # reference is Poisson noise, not a violation (or a
                    # team-conditional cull variant when _culled_ctx is set)
                    S["rare_joint_unseen"] += 1
                else:
                    S["illegal_joint_signature"] += 1
                    examples["illegal_joint_signature"].append(
                        (r["decision"], sp, sig))
        for sp, c in seen.items():
            # An opponent Ditto that copied one of OUR mons serialises under the
            # species it copied, so the world legitimately holds two of them and
            # PS's Species Clause is not violated. batch-20260820-205240 g132
            # d9: "|-transform|p2a: Ditto|p1a: Baxcalibur|[from] ability:
            # Imposter", then two Baxcalibur in the sampled team. The
            # transform-exempt path already existed for the illegal-ability
            # check and for our own party; the duplicate counter never used it.
            if c > 1 and sp not in transformed:
                S["duplicate_species"] += 1
                examples["duplicate_species"].append((r["decision"], sp, "x%d" % c))
        if movecount[HARD_SINGLETON] >= 2:
            S["two_stealthrock"] += 1
            examples["two_stealthrock"].append((r["decision"], "", "x%d" % movecount[HARD_SINGLETON]))
        for m in opp:
            if (sp in true_species or base(m["species"]) in true_species
                    or _base_species(m["species"]) in true_species):
                continue        # a REVEALED mon: fixed by reality, not sampled
            S["invented_slots"] += 1
            for mv in ref["teamStats"]["perMove"]:
                if mv in m["moves"]:
                    permove[mv]["hit"] += 1

    print("# Team-legality audit — %s" % os.path.basename(a.run_dir))
    print("%d sampled worlds x 6 opponent slots = %d mon-instances"
          % (n_worlds, n_worlds * 6))
    print("reference: %d real PS teams\n" % ref["teams"])

    print("## HARD violations (reference rate in real PS teams is exactly 0)")
    hard = ["species_not_in_ps_pool", "duplicate_species", "two_stealthrock",
            "illegal_level", "illegal_ability", "illegal_tera", "illegal_item",
            "illegal_moveset", "illegal_joint_signature"]
    for k in hard:
        flag = "  <-- VIOLATION" if S[k] else ""
        print("   %-26s %6d%s" % (k, S[k], flag))
        for ex in examples[k][:4]:
            print("        e.g. d%-3s %s %s" % (ex[0], ex[1], ex[2] if len(ex) > 2 else ""))

    if S["fallback_populated"]:
        print("   %-26s %6d  (exempt: merged set, not a generated one)"
              % ("fallback_populated", S["fallback_populated"]))

    print("\n## DISTRIBUTIONAL — per-slot rate over INVENTED slots only")
    print("   (revealed mons are fixed by reality, not sampled, so including")
    print("    them would compare our constrained team to PS's unconstrained one)")
    inv = S["invented_slots"] or 1
    ps_mons = ref["teams"] * 6
    ps_rate = {}
    for mv in ref["teamStats"]["perMove"]:
        hit = 0
        for sp, rec in R.items():
            for sig, c in rec["sets"].items():
                if mv in sig.split("|")[0].split(","):
                    hit += c
        ps_rate[mv] = hit / ps_mons
    print("   %-14s %10s %10s %8s" % ("move", "ours/slot", "PS/mon", "ratio"))
    for mv, pr in ps_rate.items():
        our = permove[mv]["hit"] / inv
        ratio = our / pr if pr else float("nan")
        mark = "  <-- under-produced" if pr and ratio < 0.75 else (
               "  <-- over-produced" if pr and ratio > 1.33 else "")
        print("   %-14s %10.4f %10.4f %8.2f%s" % (mv, our, pr, ratio, mark))
    print("   invented slots scored: %d" % S["invented_slots"])

    print("\n## OUR OWN SIDE (side_one) — known perfectly, so it must be EXACT")
    print("   (nothing here is sampled; any divergence is a serialization bug.")
    print("    finding #8 lives here: a mon fainted in the FIRST request gets")
    print("    max_hp=0 and serializes as an EMPTY party slot)")
    # base-normalize: Alcremie-Lemon-Cream is a cosmetic forme and serializes
    # as its base, so an un-normalized truth key reported it "absent from our
    # own party" in all 144 slots of run g18.
    ours_truth = {base(k): v for k, v in _our_team(a.run_dir).items()}
    ours_truth = {_base_species(k): v for k, v in ours_truth.items()}
    # OUR OWN Ditto serializes under the species it copied, so it reads as
    # "absent from our own party" for every world it is transformed in.
    our_transformed = set()
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            m = re.search(r"\|-transform\|p1a: ([^|]+)\|p[12]a: ([^|]+)", line)
            if m:
                # BOTH participants, nickname AND resolved species: our
                # Ditto->Rotom-Mow serializes as ROTOMMOW, base-collapses to
                # 'rotom', and collided with our genuine Rotom-Fan -- 80
                # phantom wrong_level/wrong_moves in run10k adab1639.
                for nick in (m.group(1), m.group(2)):
                    nm = norm(nick)
                    our_transformed.update({nm, base(nm), _base_species(nm)})
                    for spx in _nick_map.get(nm, ()):
                        our_transformed.update(
                            {spx, base(spx), _base_species(spx)})
    if not ours_truth:
        print("   no |request| with our side found — skipped")
    else:
        O = collections.Counter()
        oex = []
        for r in rows:
            mine_raw, _opp_raw = parse_team(r["state"])
            mine = [_as_dict(f) for f in mine_raw]
            O["worlds"] += 1
            if len(mine) != 6:
                O["short_party"] += 1
                oex.append((r["decision"], "party=%d" % len(mine), ""))
            # A forme change on our own side (Minior-Meteor <-> Minior-Indigo
            # via Shields Down) renames the mon mid-battle; we still know it is
            # ours, so match on the PS baseSpecies rather than the exact forme.
            got = {}
            for m in mine:
                got[base(m["species"])] = m
                got.setdefault(_base_species(m["species"]), m)
                # Terastallizing a battle forme stacks TWO suffixes:
                # Ogerpon-Hearthflame-Tera. _base_species is deliberately
                # one-level (see the note at the top of this file -- the level
                # check must NOT collapse straight to plain Ogerpon), so it
                # lands on "ogerponhearthflame" while the truth side keys on
                # "ogerpon" and the mon reads as missing. Widen ONLY the
                # party-membership lookup. batch-20260820-200253 g69: 16
                # spurious missing_mon, all of them decisions 36-37 x 8 worlds,
                # i.e. exactly the post-tera decisions.
                got.setdefault(_base_species(_base_species(m["species"])), m)
            for sp, tm in ours_truth.items():
                if sp in our_transformed:
                    continue
                m = got.get(sp)
                if m is None:
                    O["missing_mon"] += 1
                    oex.append((r["decision"], sp, "absent from our own party"))
                    continue
                O["slots"] += 1
                if m["level"] != tm["level"]:
                    O["wrong_level"] += 1
                    oex.append((r["decision"], sp, "L%s != L%s" % (m["level"], tm["level"])))
                if m["ability"] not in tm["abilities"]:
                    O["wrong_ability"] += 1
                    oex.append((r["decision"], sp, "%s not in %s"
                                % (m["ability"], sorted(tm["abilities"]))))
                if set(m["moves"]) - tm["moves"]:
                    O["wrong_moves"] += 1
                    oex.append((r["decision"], sp, ",".join(sorted(set(m["moves"]) - tm["moves"]))))
        for k in ("short_party", "missing_mon", "wrong_level", "wrong_ability", "wrong_moves"):
            print("   %-26s %6d%s" % (k, O[k], "  <-- VIOLATION" if O[k] else ""))
        print("   %-26s %6d" % ("our slots scored", O["slots"]))
        for e in oex[:5]:
            print("        e.g. d%-3s %s %s" % e)

    print("\n## SPECIES COVERAGE of invented slots")
    print("   distinct species sampled: %d   PS pool: %d"
          % (len(sampled_species), len(R)))
    zo = [s for s in R if s.startswith("zoroark")]
    for s in zo:
        print("   %-16s sampled %d   (PS pool frequency %.4f/mon-slot)"
              % (s, sampled_species.get(s, 0), R[s]["n"] / (ref["teams"] * 6)))


if __name__ == "__main__":
    main()
