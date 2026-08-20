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
from audit_sampling import norm, parse_team, revealed_by_decision  # noqa: E402

F_LEVEL, F_ABILITY, F_ITEM, F_MOVE0, F_TERA = 1, 8, 10, 22, 27
HARD_SINGLETON = "stealthrock"   # 0/20000 real teams have two


_DEX = None
_REF = None


def _reference():
    global _REF
    if _REF is None:
        try:
            _REF = json.load(open(os.path.join(HERE, "ps_reference.json")))["species"]
        except OSError:
            _REF = {}
    return _REF


def _base_species(species):
    """PS baseSpecies id. Shields Down renames Minior-Indigo to Minior-Meteor
    mid-battle -- a BATTLE forme, so it is deliberately not in the cosmetic
    alias map -- but on OUR OWN side we still know it is the same mon."""
    global _DEX
    if _DEX is None:
        path = "/Users/sallyliu/pokemon-fast-bot/foul-play/data/pokedex.json"
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
            out[sp] = {"level": lvl, "abilities": {norm(m.get("ability"))},
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
                removed.add(sp_)
                removed.add(base(sp_))

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
        for line in open(plog, errors="replace"):
            m = re.search(r"\|-item\|p2a: ([^|]+)\|", line)
            if m:
                sp_ = ident2sp.get(norm(m.group(1)), norm(m.group(1)))
                swapped.add(sp_)
                swapped.add(base(sp_))

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
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            m = re.search(r"\|-transform\|p2a: ([^|]+)\|p1a: ([^|]+)", line)
            if m:
                transformed.add(norm(m.group(1)))
                transformed.add(norm(m.group(2)))

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
        for m in opp:
            sp = m["species"]
            sampled_species[sp] += 1
            seen[base(sp)] += 1
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
            if m["ability"] not in legal_ab[sp] and m["ability"] not in announced_ab:
                S["illegal_ability"] += 1
                examples["illegal_ability"].append(
                    (r["decision"], sp, "%s not in %s" % (m["ability"], sorted(legal_ab[sp]))))
            if m["tera"] and m["tera"] not in legal_tera[sp]:
                S["illegal_tera"] += 1
                examples["illegal_tera"].append((r["decision"], sp, m["tera"]))
            if (sp not in removed and sp not in swapped
                    and sp not in fallback_mons
                    and m["item"] not in legal_it[sp]):
                S["illegal_item"] += 1
                examples["illegal_item"].append(
                    (r["decision"], sp, "%s not in %s" % (m["item"], sorted(legal_it[sp]))))
            if sp in fallback_mons:
                S["fallback_populated"] += 1
            elif frozenset(m["moves"]) not in legal_moves[sp]:
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
            if base(m["species"]) not in true_species and sp not in fallback_mons:
                sig = "|".join([",".join(sorted(m["moves"])),
                                m["item"] or "none", m["ability"], m["tera"]])
                if sig in legal_sig[sp]:
                    S["ok_joint_signature"] += 1
                else:
                    S["illegal_joint_signature"] += 1
                    examples["illegal_joint_signature"].append(
                        (r["decision"], sp, sig))
        for sp, c in seen.items():
            if c > 1:
                S["duplicate_species"] += 1
                examples["duplicate_species"].append((r["decision"], sp, "x%d" % c))
        if movecount[HARD_SINGLETON] >= 2:
            S["two_stealthrock"] += 1
            examples["two_stealthrock"].append((r["decision"], "", "x%d" % movecount[HARD_SINGLETON]))
        for m in opp:
            if base(m["species"]) in true_species:
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
            m = re.search(r"\|-transform\|p1a: ([^|]+)\|", line)
            if m:
                nm = norm(m.group(1))
                our_transformed.update({nm, base(nm), _base_species(nm)})
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
