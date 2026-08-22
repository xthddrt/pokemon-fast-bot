"""Score one validation game's SAMPLING against the opponent's true team.

    .venv/bin/python audit_sampling.py <run_dir> [--verbose]

For every decision, for every sampled world, compare what the sampler invented
for the opponent against what the server actually generated (truth.json), and
against what had genuinely been REVEALED by that point (protocol.log).

Three questions, matching the three properties a sampled world must have:

  COMPLIANT   nothing in the sampled set contradicts an established fact --
              a move already used, an item/ability/tera already shown.
              A violation is an `admits-impossible` defect.
  REACHABLE   the TRUE set is not excluded. Reported as the per-decision rate
              at which the true moves/item/ability appear across the 8 worlds.
              Zero coverage of a still-possible truth is an `excludes-truth`
              defect -- the Volcarona/Quiver-Dance failure mode.
  COMPLETE    every sampled mon carries 4 moves, a real ability, a real item
              and a tera type. NONE/UNKNOWNITEM reaching the engine prices a
              live threat as harmless.
"""
import argparse
import collections
import json
import os
import re
import sys

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


# engine state field offsets, verified against a known serialization
F_NAME, F_LEVEL, F_HP, F_MAXHP = 0, 1, 6, 7
F_ABILITY, F_ITEM = 8, 10
F_MOVE0, F_TERA = 22, 27
NONE_MOVE = ("NONE",)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


MON_RE = re.compile(r"^[A-Z][A-Z0-9]*,\d+,[A-Z]")


def parse_team(state):
    """[side_one(6), side_two(6)] in serialized party order.

    Replaces a `state.find(SPECIES)` scan (Sally 2026-08-20). That scan had two
    bugs and both fire in the same game: it searched the WHOLE state, so with a
    Rotom on each side it returned OURS -- serialized first -- and graded it
    against the opponent's reveals; and it matched by PREFIX, so "ROTOM" also
    hit "ROTOMFROST". Run /tmp/vg_f2 produced 24 phantom
    `revealed move blizzard absent` violations against our own base Rotom.
    """
    mons = []
    for chunk in state.split("="):
        for piece in chunk.split("/"):
            if not MON_RE.match(piece):
                continue
            f = piece.split(",")
            if len(f) < 28:
                continue
            mons.append(f)
    return mons[:6], mons[6:12]


def opponent_mon(state, species):
    """The opponent's `species`, matched EXACTLY, or None."""
    _ours, opp = parse_team(state)
    for f in opp:
        if norm(f[F_NAME]) != species:
            continue
        return {
            "level": f[F_LEVEL], "hp": f[F_HP], "maxhp": f[F_MAXHP],
            "ability": f[F_ABILITY], "item": f[F_ITEM], "tera": f[F_TERA],
            "moves": [f[j].split(";")[0] for j in range(F_MOVE0, F_MOVE0 + 4)],
        }
    return None


def parse_mon(state, species):
    """Pull `species`' serialized chunk out of a world state string."""
    i = state.find(species.upper())
    if i < 0:
        return None
    chunk = state[i:].split("=")[0]
    f = chunk.split(",")
    if len(f) < 28:
        return None
    moves = []
    for j in range(F_MOVE0, F_MOVE0 + 4):
        nm = f[j].split(";")[0]
        moves.append(nm)
    return {
        "level": f[F_LEVEL], "hp": f[F_HP], "maxhp": f[F_MAXHP],
        "ability": f[F_ABILITY], "item": f[F_ITEM],
        "moves": moves, "tera": f[F_TERA],
    }


RE_ARMS = re.compile(r"\[d (\d+)\] INFO\s+(Opp)?WorldStats \d+: (.*)")
RE_ARM_NAME = re.compile(r"^(.*?) [\d.]+%/")


def tera_decisions(run_dir, ):
    """First decision by which each side had ALREADY terastallized.

    Tera is once per side per battle, so from that decision on the search must
    never offer that side a `-tera` action again. Read off battle.log so the
    decision index is exact (see revealed_by_decision).
    """
    out = {}
    decision = 1
    for raw in open(os.path.join(run_dir, "battle.log"), errors="replace"):
        if "Choice:" in raw:
            decision += 1
            continue
        m = re.search(r"\|-terastallize\|(p[12])a: ", raw)
        if m:
            side = "us" if m.group(1) == "p1" else "opp"
            out.setdefault(side, decision)
    return out


def tera_arms_after_use(run_dir):
    """[(decision, side, arm)] where a spent-tera side actually CHOSE (or was
    predicted to answer with) a `-tera` action.

    Keyed on battle.log `Choice:` lines and protocol Invalid-choice NAKs, NOT
    on search.log WorldStats arm listings: those are logged from the raw
    engine root enumeration, UPSTREAM of the selection-pool strip that makes
    the arms unselectable (fp/search/selection.py user_tera_used). run10kb
    flagged 752 such arms across 3 games with zero illegal choices sent --
    verified: every sent Choice at those decisions was a bare move. The arm
    enumeration is expected residue of the Revival-Blessing tera_spent corner
    and is counted separately as informational.
    """
    tera_at = tera_decisions(run_dir)
    bad = []
    enumerated = 0
    path = os.path.join(run_dir, "search.log")
    if os.path.isfile(path):
        for line in open(path, errors="replace"):
            m = RE_ARMS.match(line)
            if not m:
                continue
            d, is_opp, blob = int(m.group(1)), bool(m.group(2)), m.group(3)
            side = "opp" if is_opp else "us"
            used = tera_at.get(side)
            if used is None or d <= used:
                continue
            for arm in blob.split(" | "):
                nm = RE_ARM_NAME.match(arm.strip())
                name = nm.group(1) if nm else arm.strip()
                if name.endswith("-tera"):
                    enumerated += 1
    blog = os.path.join(run_dir, "battle.log")
    if os.path.isfile(blog):
        d = 0
        for line in open(blog, errors="replace"):
            if "Choice:" in line:
                d += 1
                choice = line.rsplit("Choice:", 1)[1].strip()
                used = tera_at.get("us")
                if used is not None and d > used and choice.endswith("-tera"):
                    bad.append((d, "us", choice))
    plog = os.path.join(run_dir, "protocol.log")
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            if "[Invalid choice]" in line and "erastallize" in line:
                bad.append(("nak", "us", line.strip()[:80]))
    tera_at["_enumerated_arms"] = enumerated
    return tera_at, bad


_MOVEPOOL = None


def _species_cannot_have(species, move):
    """True when NO set PS generates for `species` carries `move`.

    Read off the 100k-team reference, i.e. the JS generator's own output, so it
    is a statement about what Showdown builds rather than about what our bot
    believes.
    """
    global _MOVEPOOL
    if _MOVEPOOL is None:
        _MOVEPOOL = {}
        try:
            ref = json.load(open(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ps_reference.json")))
        except OSError:
            return False
        for sp, rec in ref["species"].items():
            pool = set()
            for sig in rec["sets"]:
                pool.update(sig.split("|")[0].split(","))
            _MOVEPOOL[sp] = pool
    pool = _MOVEPOOL.get(species)
    return bool(pool) and move not in pool


def _species_tera_possible(species, tera):
    """Does the reference ever give this species this tera type?"""
    _species_cannot_have(species, "")
    if not _MOVEPOOL:
        return True
    global _TERAPOOL
    if _TERAPOOL is None:
        import json as _json
        _TERAPOOL = {}
        try:
            ref = _json.load(open(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "ps_reference.json")))
            for sp, rec in ref["species"].items():
                pool = set()
                for sig in rec["sets"]:
                    pool.add(sig.split("|")[3])
                _TERAPOOL[sp] = pool
        except OSError:
            pass
    pool = _TERAPOOL.get(species)
    return not pool or tera in pool


_TERAPOOL = None


def _species_movepool_size(species):
    """Distinct moves the species ever carries in the real-PS reference.

    0 when the reference is unavailable or the species unknown -- callers must
    treat 0 as "no information", never as "empty pool"."""
    _species_cannot_have(species, "")     # forces the lazy _MOVEPOOL load
    return len(_MOVEPOOL.get(species) or ()) if _MOVEPOOL else 0


def revealed_by_decision(run_dir, opp_prefix="p2a"):
    """decision N -> what the opponent had revealed WHEN DECISION N WAS MADE.

    Walks battle.log, where the server protocol and our own `Choice:` lines are
    interleaved in true chronological order, so the join is exact. Keying on
    turns instead is WRONG and silently so: a forced switch adds a decision
    without adding a turn, so the indices drift apart (25 turns / 26 decisions
    in run 20260820-142839) and the audit ends up grading decision N against
    reveals that had not happened yet -- which is what made a correct Clawitzer
    sample look like 13 violations.

    The snapshot for decision N is the state at the moment its `Choice:` is
    emitted: everything the sampler could legitimately have known.
    """
    per_dec = {}
    moves = collections.defaultdict(set)
    items, abils, teras = {}, {}, {}
    _pending_evts = []  # (kind, nickname, value) reveals since the slot changed
    _retro = []         # replace-driven re-attributions, applied to ALL decisions
    _ability_overwritten = set() # mons whose live ability is no longer their set's
    decision = 1
    for raw in open(os.path.join(run_dir, "battle.log"), errors="replace"):
        line = raw.rstrip("\n")
        if "Choice:" in line:
            per_dec[decision] = (
                {k: set(v) for k, v in moves.items()},
                dict(items), dict(abils), dict(teras),
            )
            decision += 1
            continue
        m = re.search(r"\|move\|%s: ([^|]+)\|([^|]+)" % opp_prefix, line)
        if m:
            mv = norm(m.group(2))
            # STRUGGLE is what a mon does with no PP left -- it is never in a
            # moveset, and counting it made every sampled Gothitelle in batch
            # game g1 look non-compliant (120 flags). A move CALLED by another
            # (Sleep Talk, Copycat, Metronome, Nature Power, Mirror Move) is
            # tagged `[from]` and likewise is not evidence of the caller's own
            # moveset -- g1's Lapras runs Rest + Sleep Talk.
            if mv == "struggle" or "[from]" in line:
                continue
            # ILLUSION RESIDUE. A move PS never generates on this species was
            # used by a Zoroark wearing its name -- the bot detects exactly this
            # ("dedenne using sludgebomb means it is zoroark", battle_modifier
            # move handler), swaps the Zoroark into the active slot and purges
            # the residue from the impersonated mon. Recording it as that mon's
            # own reveal made a CORRECT sample look non-compliant: 1256 flags
            # across three batches. Judged against PS's own generated corpus,
            # so this stays independent of the bot's heuristic.
            if _species_cannot_have(norm(m.group(1)), mv):
                continue
            moves[norm(m.group(1))].add(mv)
            _pending_evts.append(("move", norm(m.group(1)), mv))
            continue
        m = re.search(r"\|-item\|%s: ([^|]+)\|([^|]+)" % opp_prefix, line)
        if m:
            items[norm(m.group(1))] = norm(m.group(2))
            _pending_evts.append(("item", norm(m.group(1)), norm(m.group(2))))
            continue
        m = re.search(r"\|-terastallize\|%s: ([^|]+)\|([^|]+)" % opp_prefix, line)
        if m:
            _tnick, _ttype = norm(m.group(1)), norm(m.group(2))
            # A (species, tera) pair the 1.05M-team reference NEVER builds
            # is a disguised Zoroark terastallizing under the disguise's
            # name -- when it pivots out no |replace| ever corrects the
            # pin (run25k: 648 flags, ampharos/Poison + greninja/Fighting).
            # Re-attribute to the Illusion form whose pool carries it.
            if not _species_tera_possible(_tnick, _ttype):
                for _zf in ("zoroark", "zoroarkhisui"):
                    if _species_tera_possible(_zf, _ttype):
                        _tnick = _zf
                        break
            teras[_tnick] = _ttype
            _pending_evts.append(("tera", _tnick, _ttype))
            continue
        # Illusion: |replace| reveals who every reveal since the last slot
        # change REALLY belonged to. Items were already re-attributed; MOVES
        # were not -- _species_cannot_have passes whenever the disguise could
        # legally roll the move, so a disguised Zoroark's Nasty Plot stayed
        # pinned on Grumpig (run10k: 4 games, 145 flags) -- and the STALE
        # disguise-keyed item entry was never deleted (a Trick under disguise
        # left Calyrex-Shadow graded against Leftovers, 288 flags).
        m = re.search(r"\|replace\|%s: ([^|]+)\|([^|,]+)" % opp_prefix, line)
        if m:
            _true_sp = norm(m.group(2))
            for _kind, _nick, _val in _pending_evts:
                if _kind == "move":
                    moves[_nick].discard(_val)
                    if not _species_cannot_have(_true_sp, _val):
                        moves[_true_sp].add(_val)
                elif _kind == "enditem":
                    if items.get(_nick) == "__removed__":
                        del items[_nick]
                    items[_true_sp] = "__removed__"
                elif _kind == "item":
                    if items.get(_nick) == _val:
                        del items[_nick]
                    items[_true_sp] = _val
                else:
                    if teras.get(_nick) == _val:
                        del teras[_nick]
                    teras[_true_sp] = _val
                # ...and RETROACTIVELY: decisions between the event and this
                # |replace| already snapshotted the disguise-keyed pin, and
                # the bot may legitimately have detected the Zoroark before
                # the server said so ("X using Y means it is zoroark") -- its
                # during-stay worlds then carry the TRUE mon's state and were
                # graded against the disguise pin (run10kb g474 d2: hariyama
                # tera steel vs "revealed poison" that was Zoroark's).
                # Safe because a side teras at most once and the disguise
                # window is one contiguous stay.
                _retro.append((_kind, norm(m.group(1)), _nick, _val, _true_sp))
            _pending_evts = []
            continue
        if re.search(r"\|(?:switch|drag)\|%s:" % opp_prefix, line):
            # NOTE: tera pins from a disguised stint that ends in a pivot
            # (no |replace|) stay on the disguise -- re-attributing them from
            # a LATER stint's |replace| mis-assigns a real mon's genuine tera
            # to the zoroark (tried; -152 net). Known residual, needs
            # species-tera-impossibility to resolve safely.
            _pending_evts = []
        # Knock Off / consumption REMOVES the item; "NONE" afterwards is the
        # correct sample, not an underspecified one.
        m = re.search(r"\|-enditem\|%s: ([^|]+)\|" % opp_prefix, line)
        if m:
            items[norm(m.group(1))] = "__removed__"
            # under Illusion the removal belongs to whoever |replace|
            # later reveals -- run25k: 80 viol_item_should_be_gone pinned
            # on the disguise after a disguised Trick gave its item away
            _pending_evts.append(("enditem", norm(m.group(1)), None))
            continue
        # Magician/Pickpocket theft announces only on the THIEF's side -- no
        # victim-side -enditem ever prints, so the victim's item stayed pinned
        # forever while the bot correctly tracked it as gone (run10k
        # a8d18cb5/abd5e8a5: 224 flags). Slot letter optional: the victim may
        # have just fainted.
        m = re.search(
            r"\|-item\|p1[ab]: [^|]+\|[^|]+\|\[from\] ability: "
            r"(?:Magician|Pickpocket)\|\[of\] %s[ab]?: ([^|]+)"
            % opp_prefix.rstrip("a"), line)
        if m:
            items[norm(m.group(1))] = "__removed__"
            continue
        m = re.search(r"\|-ability\|%s: ([^|]+)\|([^|]+)" % opp_prefix, line)
        if m:
            # A COPIED ability announces someone else's, not this mon's own:
            #   |-ability|p2a: Gardevoir|Competitive|Trace|[from] ability: Trace
            # Recording "competitive" as Gardevoir's ability made every sampled
            # Gardevoir (correctly carrying Trace) look like a violation -- 104
            # of them in batch game g20.
            if "[from] ability:" in line and "[of]" in line:
                # ...and the guard must be STICKY: after Trace fires, the
                # traced ability keeps re-announcing on BARE lines (a Ruin
                # ability's switch-in announce, "Stamina|boost"), which is what
                # slipped past the one-line guard -- run1000b flagged 3 games,
                # all Gardevoir, 504 worlds. Trace itself is positive evidence:
                # the [from] tag names the mon's real set ability.
                # Do NOT pin "trace" either: the bot rightly serializes the
                # LIVE (traced) ability -- that is what the engine must model
                # -- so after an overwrite neither the set value nor the live
                # value is a checkable "revealed ability". Exempt the mon.
                abils.pop(norm(m.group(1)), None)
                _ability_overwritten.add(norm(m.group(1)))
                continue
            if norm(m.group(1)) in _ability_overwritten:
                continue
            # As One (Glastrier) manifests as Unnerve + Chilling Neigh and the
            # server announces the COMPONENT, so the sampled `asoneglastrier`
            # is right and the announcement is not the mon's own ability name.
            _ab = norm(m.group(2))
            if _ab in ("asone", "unnerve", "chillingneigh", "grimneigh",
                       # Terapagos: Tera Shift -> Terastal forme announces
                       # Teraform Zero/Tera Shell; the server silently reverts
                       # the forme on faint and every later CORRECT world was
                       # graded against the stale pin (run10k a3cbb819, 88).
                       "teraformzero", "terashell") \
                    or _ab.startswith("embodyaspect"):
                # Ogerpon-Cornerstone runs Sturdy and announces Embody Aspect
                # (Cornerstone) only once it terastallizes -- the announcement
                # is the tera forme's ability, not the set's.
                continue
            abils[norm(m.group(1))] = norm(m.group(2))
            continue
    for _kind, _rnick, _nick, _val, _true_sp in _retro:
        for _dec, (dm, di, da, dt) in per_dec.items():
            if _kind == "move":
                if _val in dm.get(_nick, set()):
                    dm[_nick].discard(_val)
                    dm.setdefault(_true_sp, set()).add(_val)
            elif _kind == "item":
                if di.get(_nick) == _val:
                    del di[_nick]
                    di[_true_sp] = _val
            else:
                if dt.get(_nick) == _val:
                    del dt[_nick]
                    dt[_true_sp] = _val
    return per_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    truth = json.load(open(os.path.join(a.run_dir, "truth.json")))
    tmons, ident2sp = {}, {}
    for m in truth["team"]:
        # details, not ident: the nickname is "Rotom" while the species is
        # "Rotom-Frost" (same trap as Ursaluna/Ursaluna-Bloodmoon)
        sp = norm(m["details"].split(",")[0])
        ident2sp[norm(m["ident"].split(": ")[1])] = sp
        tmons[sp] = {
            "moves": {norm(x) for x in m["moves"]},
            "item": norm(m.get("item")), "ability": norm(m.get("ability")),
            "tera": norm(m.get("teraType")), "species": sp,
        }
    # A mon the opponent's Ditto COPIED is serialized twice under one species:
    # the real one and the transform. Every reveal- and completeness-check then
    # grades the Ditto against the real mon's history. Exempt the copy target.
    transformed = set()
    plog = os.path.join(a.run_dir, "protocol.log")
    _nick_map = _nickname_species_map(plog)
    if os.path.isfile(plog):
        for line in open(plog, errors="replace"):
            # BOTH sides of a transform are contaminated: the copy TARGET is
            # duplicated in the opponent's serialized party, and the
            # TRANSFORMER (Ditto) shows the target's moves in the reveal log
            # while its own sampled set is the real one -- {'transform'}.
            # Neither is evidence about the mon the checker thinks it is.
            for pat in (r"\|-transform\|p2a: ([^|]+)\|p1a: ([^|]+)",
                        r"\|-transform\|p1a: ([^|]+)\|p2a: ([^|]+)"):
                m = re.search(pat, line)
                if m:
                    # nicknames AND their resolved species ids: 'Giratina'
                    # never matches the species key 'giratinaorigin'
                    for nick in (m.group(1), m.group(2)):
                        transformed.add(norm(nick))
                        transformed.update(_nick_map.get(norm(nick), ()))
    if transformed:
        print("   (exempt, copied by a transform: %s)" % ", ".join(sorted(transformed)))

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "worlds.jsonl"))]
    by_dec = collections.defaultdict(list)
    for r in rows:
        by_dec[r["decision"]].append(r)

    # map decision -> turn by counting Choice lines per turn in search.log order
    per_dec = revealed_by_decision(a.run_dir)

    stats = collections.Counter()
    viol = []
    incomplete = []
    cov_rows = []

    for d in sorted(by_dec):
        worlds = sorted(by_dec[d], key=lambda r: r["world"])
        _rm, _ri, _ra, _rt = per_dec.get(d, ({}, {}, {}, {}))
        tr_ = lambda dd: {ident2sp.get(k, k): v for k, v in dd.items()}
        rmoves, ritems, rabils, rteras = tr_(_rm), tr_(_ri), tr_(_ra), tr_(_rt)

        for sp, tr in tmons.items():
            if sp in transformed:
                continue
            seen_true_moves = collections.Counter()
            present = 0
            exact = 0
            for w in worlds:
                mon = opponent_mon(w["state"], sp)
                if not mon:
                    continue
                present += 1
                smoves = {norm(x) for x in mon["moves"] if x not in NONE_MOVE}
                sitem, sabil = norm(mon["item"]), norm(mon["ability"])

                # ---- COMPLETE ----
                # "fewer than 4 moves" is only underspecification when the
                # species HAS 4: Ditto's whole gen9 randbats movepool is
                # {transform}, so its 1-move world is exactly right and flagged
                # 176 times in run1000 grp1/g180. Bound by the species' true
                # pool size from the reference.
                _pool = _species_movepool_size(sp)
                if len(smoves) < min(4, _pool if _pool else 4):
                    stats["incomplete_moves"] += 1
                    incomplete.append((d, sp, "moves=%d" % len(smoves), mon["moves"]))
                if sabil in ("none", ""):
                    stats["incomplete_ability"] += 1
                    incomplete.append((d, sp, "ability=NONE", ""))
                if sitem == "unknownitem":
                    stats["incomplete_item"] += 1
                    incomplete.append((d, sp, "item=UNKNOWNITEM", ""))
                if norm(mon["tera"]) in ("typeless", "none", ""):
                    stats["incomplete_tera"] += 1
                    incomplete.append((d, sp, "tera=%s" % mon["tera"], ""))

                # ---- COMPLIANT (vs what was actually revealed) ----
                for mv in rmoves.get(sp, set()):
                    if mv not in smoves:
                        stats["viol_missing_revealed_move"] += 1
                        viol.append((d, sp, "revealed move %s absent" % mv, sorted(smoves)))
                if sp in ritems:
                    want = ritems[sp]
                    if want == "__removed__":
                        # item was knocked off/consumed: every world must agree
                        if sitem not in ("none", ""):
                            stats["viol_item_should_be_gone"] += 1
                            viol.append((d, sp, "item %s but it was removed" % sitem, ""))
                        else:
                            stats["ok_item_removal_tracked"] += 1
                    elif sitem != want:
                        stats["viol_item"] += 1
                        viol.append((d, sp, "item %s != revealed %s" % (sitem, want), ""))
                if sp in rabils and sabil != rabils[sp]:
                    stats["viol_ability"] += 1
                    viol.append((d, sp, "ability %s != revealed %s" % (sabil, rabils[sp]), ""))
                # An observed |-terastallize| pins the tera TYPE for good: it
                # cannot drift like items. rteras was collected since the first
                # coverage pass and never compared -- a post-tera world
                # carrying the wrong type passed every check (coverage-gap
                # assessment, check 4).
                if sp in rteras and norm(mon["tera"]) != rteras[sp]:
                    stats["viol_tera"] += 1
                    viol.append((d, sp, "tera %s != revealed %s"
                                 % (norm(mon["tera"]), rteras[sp]), ""))

                # ---- REACHABLE (vs truth) ----
                for mv in tr["moves"]:
                    if mv in smoves:
                        seen_true_moves[mv] += 1
                if smoves == tr["moves"]:
                    exact += 1
            if present:
                cov_rows.append({
                    "d": d, "sp": sp, "worlds": present, "exact": exact,
                    "per_move": {mv: seen_true_moves[mv] for mv in sorted(tr["moves"])},
                    "revealed": sorted(rmoves.get(sp, set())),
                })

    print("# Sampling audit — %s" % os.path.basename(a.run_dir))
    print("battle %s, winner %s, %d decisions x 8 worlds\n"
          % (truth["battle_tag"], truth.get("winner"), len(by_dec)))

    print("## 1. COMPLIANCE (sampled world contradicts an established fact)")
    for k in ("viol_missing_revealed_move", "viol_item", "viol_tera",
              "viol_item_should_be_gone", "viol_ability"):
        print("   %-32s %d" % (k, stats[k]))
    print("   %-32s %d  (item removals correctly propagated to every world)"
          % ("ok_item_removal_tracked", stats["ok_item_removal_tracked"]))
    if viol[:8]:
        print("   first violations:")
        for v in viol[:8]:
            print("     d%-3d %-14s %s %s" % v)

    print("\n## 2. COMPLETENESS (mon reaches the engine underspecified)")
    for k in ("incomplete_moves", "incomplete_ability", "incomplete_item", "incomplete_tera"):
        print("   %-32s %d" % (k, stats[k]))
    if incomplete[:8]:
        print("   first cases:")
        for v in incomplete[:8]:
            print("     d%-3d %-14s %-16s %s" % v)

    tera_at, tera_bad = tera_arms_after_use(a.run_dir)
    print("\n## 3. TERA LEGALITY (one terastallization per side, per battle)")
    print("   terastallized at decision: %s" % (tera_at or "neither side"))
    print("   %-32s %d%s" % ("tera_arm_offered_after_use", len(tera_bad),
                             "  <-- VIOLATION" if tera_bad else ""))
    seen = set()
    for d, side, name in tera_bad:
        if (side, name) in seen:
            continue
        seen.add((side, name))
        print("        d%-3s %-4s %s" % (d, side, name))
        if len(seen) >= 6:
            break

    print("\n## 4. REACHABILITY of the TRUE set (per mon, over all decisions it appears in)")
    agg = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
    for r in cov_rows:
        A = agg[r["sp"]]
        A[0] += r["worlds"]; A[1] += r["exact"]
        for mv, c in r["per_move"].items():
            A[2][mv] += c; A[3][mv] += r["worlds"]
    print("   %-14s %8s %10s   per-true-move world coverage" % ("species", "worlds", "exact-set"))
    for sp, (w, ex, hit, tot) in sorted(agg.items()):
        cov = ", ".join("%s %.0f%%" % (mv, 100 * hit[mv] / max(tot[mv], 1))
                        for mv in sorted(hit))
        print("   %-14s %8d %9.1f%%   %s" % (sp, w, 100 * ex / max(w, 1), cov))

    zero = [(sp, mv) for sp, (w, ex, hit, tot) in agg.items()
            for mv in hit if hit[mv] == 0]
    print("\n   TRUE MOVES NEVER SAMPLED IN ANY WORLD: %d" % len(zero))
    for sp, mv in zero:
        print("     %-14s %s" % (sp, mv))


if __name__ == "__main__":
    main()
