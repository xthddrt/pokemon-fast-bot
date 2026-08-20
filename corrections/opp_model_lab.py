"""MO-ISMCTS CEILING LAB — asymmetric-information machinery (Sally 2026-08-17).

Module 1 of the opponent-modeling ceiling experiment: everything needed to
give an agent only its LEGITIMATE view of a game state.

  * revealed-info model: a mon is revealed when it has switched in; a move
    when it has been used; tera type when terastallized. Items/abilities are
    treated as hidden until set-conditioning implies them (the PS set tables
    couple item/ability to the sampled set, which mirrors how humans infer).
  * belief sampler ("blinding"): replace everything the observer cannot know
    with a sample from the PS randbats set distribution CONDITIONED on what
    is revealed. Unrevealed mons are full-HP no-status by definition (they
    were never on the field).
  * donor construction reuses the battle-tested pipeline end to end:
    ps_teams set sampling -> run_duels.opening_state -> parse the mon
    substring back out. No hand-built mon strings.

Self-test: python opp_model_lab.py selftest <worlds.jsonl>
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "valuenet", "sprt"))
sys.path.insert(0, os.path.join(ROOT, "foul-play"))

MON_HP, MON_MAXHP, MON_STATUS = 6, 7, 18
MON_MOVES = (22, 23, 24, 25)
MON_TERA_USED, MON_TERA_TYPE = 26, 27


def split_state(s):
    """state -> (side1_mons, side2_mons, side1_rest, side2_rest, tail).
    Each side segment is 6 '='-joined mon substrings followed by '='-joined
    side context; sides and trailing segments are '/'-joined."""
    parts = s.split("/")
    sides = []
    for seg in parts[:2]:
        toks = seg.split("=")
        sides.append((toks[:6], toks[6:]))
    return sides[0], sides[1], parts[2:]


def join_state(s1, s2, tail):
    a = "=".join(list(s1[0]) + s1[1])
    b = "=".join(list(s2[0]) + s2[1])
    return "/".join([a, b] + tail)


def mon_fields(m):
    return m.split(",")


def mon_species(m):
    return m.split(",", 1)[0]


def mon_moves(m):
    f = mon_fields(m)
    return [f[i].split(";")[0] for i in MON_MOVES]


class Revealed:
    """What ONE side has shown the opponent. keyed by species."""

    def __init__(self):
        self.mons = set()          # species that have been on the field
        self.moves = {}            # species -> set(move names used)
        self.tera = {}             # species -> tera type IF terastallized

    def switch_in(self, species):
        self.mons.add(species)
        self.moves.setdefault(species, set())

    def used_move(self, species, move):
        self.switch_in(species)
        if move and move.lower() not in ("none", "no move", "switch"):
            self.moves[species].add(move.upper())

    def terastallized(self, species, ttype):
        self.switch_in(species)
        self.tera[species] = ttype

    def to_json(self):
        return {"mons": sorted(self.mons),
                "moves": {k: sorted(v) for k, v in self.moves.items()},
                "tera": dict(self.tera)}

    @classmethod
    def from_json(cls, d):
        r = cls()
        r.mons = set(d["mons"])
        r.moves = {k: set(v) for k, v in d["moves"].items()}
        r.tera = dict(d["tera"])
        return r


# ---------------------------------------------------------------- donors ----
_DONOR_CACHE = {}


def _fresh_team_state(species_list, seed):
    """Build an opening state whose side-one team is exactly species_list
    (sets sampled by the PS generator), then return its mon substrings."""
    import tempfile
    from fp.search import ps_teams
    import run_duels as rd
    ps_teams.seed(seed)
    team = []
    for sp in species_list:
        team.append(ps_teams.random_set_for(sp) if hasattr(ps_teams, "random_set_for")
                    else None)
    if any(t is None for t in team):
        # public API path: sample whole teams until the species appear is too
        # slow; instead build via the set table directly
        team = [_set_from_tables(sp, seed + i) for i, sp in enumerate(species_list)]
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"teams": {"p1": {"team": team}, "p2": {"team": team}}}, tf)
    tf.close()
    s = rd.opening_state(tf.name, "p1", tf.name, "p2")
    os.unlink(tf.name)
    side1, _, _ = split_state(s)
    return side1[0]


def _set_from_tables(species, seed):
    """Sample one legal randbats set dict for species from the PS tables."""
    from fp.search import _ps_team_loop as L
    rng = random.Random(seed)
    key = None
    for k in L.RANDOM_SETS:
        if k.replace("-", "").lower() == species.replace("-", "").lower():
            key = k
            break
    if key is None:
        raise KeyError(f"species {species} not in RANDOM_SETS")
    entry = L.RANDOM_SETS[key]
    role = rng.choice(entry["sets"]) if "sets" in entry else entry
    moves = list(role.get("movepool", role.get("moves", [])))
    rng.shuffle(moves)
    return {"speciesId": key, "role": role, "_moves": moves[:4], "_seed": seed}


def _set_to_mon_substring(ps_set, seed):
    """One complete PS set dict -> engine mon substring, via the battle-tested
    team-file -> opening_state pipeline (never hand-built)."""
    import tempfile
    from fp.search import ps_teams
    import run_duels as rd
    ps_teams.seed(seed)
    team = [ps_set] + ps_teams.random_team()[:5]
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"teams": {"p1": {"team": team}, "p2": {"team": team}}}, tf)
    tf.close()
    st = rd.opening_state(tf.name, "p1", tf.name, "p2")
    os.unlink(tf.name)
    side1, _, _ = split_state(st)
    return side1[0][0]


def donor_mon(species, revealed_moves, seed, tries=12):
    """Set-conditioned donor via the live sampler's own random_set — works for
    every species incl. rare ones (canary catch: CALYREX broke the old
    whole-team rejection path)."""
    from fp.search import ps_teams
    from fp.search import _ps_team_loop as L
    want = {m.upper().replace(" ", "") for m in revealed_moves}
    sid = species.lower().replace("-", "").replace(" ", "")
    if sid not in L.RANDOM_SETS:
        # formes appear in states under their forme id (MINIORMETEOR) while
        # the sets table keys the base (minior): longest prefix match wins
        cands = [k for k in L.RANDOM_SETS
                 if sid.startswith(k.replace("-", "").lower())
                 or k.replace("-", "").lower().startswith(sid)]
        if cands:
            sid = max(cands, key=len)
    best, best_hit = None, -1
    for t in range(tries):
        ps_teams.seed(seed + t * 7919)
        try:
            st = ps_teams._GEN.random_set(sid)
        except Exception:
            break
        have = {m.upper().replace(" ", "") for m in st["moves"]}
        hit = len(want & have)
        if hit > best_hit:
            best, best_hit = st, hit
        if hit == len(want):
            break
    if best is not None:
        m = _set_to_mon_substring(best, seed)
        if best_hit < len(want):
            f = mon_fields(m)
            have = {x.upper().replace(" ", "") for x in mon_moves(m)}
            missing = sorted(want - have)
            slots = [i for i in MON_MOVES
                     if f[i].split(";")[0].upper().replace(" ", "") not in want]
            for i, mv in zip(slots, missing):
                f[i] = f"{mv};false;{f[i].split(';')[2]}"
            m = ",".join(f)
        return m
    return _donor_mon_legacy(species, revealed_moves, seed, tries)


def _donor_mon_legacy(species, revealed_moves, seed, tries=12):
    """A mon substring for `species` sampled from its set distribution,
    preferring sets that contain every revealed move; falls back to
    overwriting move slots with the revealed ones."""
    from fp.search import ps_teams
    want = {m.upper() for m in revealed_moves}
    best, best_hit = None, -1
    for t in range(tries):
        ck = (species, seed + t * 7919)
        if ck not in _DONOR_CACHE:
            try:
                ps_teams.seed(seed + t * 7919)
                team = ps_teams.random_team_with(species) if hasattr(
                    ps_teams, "random_team_with") else None
            except Exception:
                team = None
            if team is None:
                _DONOR_CACHE[ck] = _donor_via_full_team(species, seed + t * 7919)
            else:
                _DONOR_CACHE[ck] = _donor_from_team(team, species, seed)
        m = _DONOR_CACHE[ck]
        if m is None:
            continue
        have = {x.upper() for x in mon_moves(m)}
        hit = len(want & have)
        if hit > best_hit:
            best, best_hit = m, hit
        if hit == len(want):
            break
    if best is None:
        raise RuntimeError(f"no donor for {species}")
    if best_hit < len(want):
        f = mon_fields(best)
        missing = sorted(want - {x.upper() for x in mon_moves(best)})
        slots = [i for i in MON_MOVES
                 if f[i].split(";")[0].upper() not in want]
        for i, mv in zip(slots, missing):
            f[i] = f"{mv};false;{f[i].split(';')[2]}"
        best = ",".join(f)
    return best


def _donor_via_full_team(species, seed):
    """Sample whole teams until one contains `species`; extract its mon."""
    import tempfile
    from fp.search import ps_teams
    import run_duels as rd
    for k in range(40):
        ps_teams.seed(seed + k * 104729)
        team = ps_teams.random_team()
        ids = [m["speciesId"].upper().replace("-", "") for m in team]
        tgt = species.upper().replace("-", "")
        if tgt in ids:
            tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump({"teams": {"p1": {"team": team}, "p2": {"team": team}}}, tf)
            tf.close()
            s = rd.opening_state(tf.name, "p1", tf.name, "p2")
            os.unlink(tf.name)
            side1, _, _ = split_state(s)
            for m in side1[0]:
                if mon_species(m).upper().replace("-", "") == tgt:
                    return m
    return None


_FILLER_POOL = []


def fill_pool(n_teams=20, seed=1):
    """Pre-generate a pool of mon substrings so per-turn blinding never pays
    team-generation cost (duel harness calls this once per game)."""
    import tempfile
    from fp.search import ps_teams
    import run_duels as rd
    _FILLER_POOL.clear()
    for k in range(n_teams):
        ps_teams.seed(seed + k * 7919)
        team = ps_teams.random_team()
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"teams": {"p1": {"team": team}, "p2": {"team": team}}}, tf)
        tf.close()
        st = rd.opening_state(tf.name, "p1", tf.name, "p2")
        os.unlink(tf.name)
        side1, _, _ = split_state(st)
        _FILLER_POOL.extend(side1[0])


def random_filler_mon(exclude, seed):
    """A random mon whose species is not in `exclude` (unrevealed slot)."""
    import tempfile
    from fp.search import ps_teams
    import run_duels as rd
    ex = {e.upper().replace("-", "") for e in exclude}
    if _FILLER_POOL:
        rng = random.Random(seed)
        for _ in range(60):
            m = rng.choice(_FILLER_POOL)
            if mon_species(m).upper().replace("-", "") not in ex:
                return m
    for k in range(20):
        ps_teams.seed(seed + k * 15485863)
        team = ps_teams.random_team()
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"teams": {"p1": {"team": team}, "p2": {"team": team}}}, tf)
        tf.close()
        s = rd.opening_state(tf.name, "p1", tf.name, "p2")
        os.unlink(tf.name)
        side1, _, _ = split_state(s)
        random.Random(seed + k).shuffle(side1[0])
        for m in side1[0]:
            if mon_species(m).upper().replace("-", "") not in ex:
                return m
    raise RuntimeError("no filler mon found")


# ---------------------------------------------------------------- blind -----
def blind_side_masked(mons, revealed, seed):
    """blind_side + the phantom mask: party slots whose content the observer
    invented (never-revealed mons). Returns (mons, sorted slot list)."""
    out = blind_side(mons, revealed, seed)
    mask = [i for i, m in enumerate(mons) if mon_species(m) not in revealed.mons]
    return out, mask


def blind_side(mons, revealed, seed):
    """Return the observer's-belief version of one side's 6 mon substrings.
    Revealed mons keep true public context (hp, status, revealed move slots,
    tera if shown); their hidden move slots / item / ability / stats come from
    a set-conditioned donor. Unrevealed mons are replaced entirely."""
    rng = random.Random(seed)
    out = []
    known = {mon_species(m) for m in mons if mon_species(m) in revealed.mons}
    for m in mons:
        sp = mon_species(m)
        if sp not in revealed.mons:
            out.append(random_filler_mon(known | {mon_species(x) for x in out},
                                         rng.randrange(1 << 30)))
            continue
        shown = revealed.moves.get(sp, set())
        d = donor_mon(sp, shown, rng.randrange(1 << 30))
        df, tf_ = mon_fields(d), mon_fields(m)
        # public context stays true
        df[MON_HP], df[MON_MAXHP] = tf_[MON_HP], tf_[MON_MAXHP]
        df[MON_STATUS] = tf_[MON_STATUS]
        # revealed move slots keep the TRUE token (name + pp); donor fills rest
        donor_slots = [i for i in MON_MOVES
                       if df[i].split(";")[0].upper() not in shown]
        true_shown = [tf_[i] for i in MON_MOVES
                      if tf_[i].split(";")[0].upper() in shown]
        for i, tok in zip([i for i in MON_MOVES
                           if df[i].split(";")[0].upper() in shown] + donor_slots,
                          true_shown + [df[i] for i in donor_slots]):
            df[i] = tok
        if sp in revealed.tera:
            df[MON_TERA_USED] = tf_[MON_TERA_USED]
            df[MON_TERA_TYPE] = tf_[MON_TERA_TYPE]
        out.append(",".join(df))
    return out


def blinded_state(true_state, my_side, opp_revealed_of_me, seed):
    """The state as the OPPONENT of `my_side` may see it: my side's mons are
    blinded through their revealed-info; everything else (their own side,
    field, hazards) stays true. my_side in (1, 2)."""
    s1, s2, tail = split_state(true_state)
    if my_side == 1:
        s1 = (blind_side(s1[0], opp_revealed_of_me, seed), s1[1])
    else:
        s2 = (blind_side(s2[0], opp_revealed_of_me, seed), s2[1])
    return join_state(s1, s2, tail)


# ---------------------------------------------------------------- test ------
def selftest(worlds_path):
    import labcheck  # noqa: F401  (not required; keep import cost visible)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        wp = sys.argv[2]
        rec = json.loads(open(wp).readline())
        st = rec["state"]
        s1, s2, tail = split_state(st)
        print(f"parsed: side1 {len(s1[0])} mons + {len(s1[1])} ctx, "
              f"side2 {len(s2[0])} mons, tail {len(tail)}")
        assert join_state(s1, s2, tail) == st, "round-trip failed"
        print("round-trip: OK")
        rev = Revealed()
        rev.switch_in(mon_species(s1[0][0]))
        rev.used_move(mon_species(s1[0][0]), mon_moves(s1[0][0])[0])
        b = blinded_state(st, 1, rev, seed=7)
        from poke_engine import State
        State.from_string(b)
        print("engine parses blinded state: OK")
        b1, _, _ = split_state(b)
        assert mon_species(b1[0][0]) == mon_species(s1[0][0])
        assert mon_fields(b1[0][0])[MON_HP] == mon_fields(s1[0][0])[MON_HP]
        kept = mon_moves(s1[0][0])[0].upper()
        assert kept in [x.upper() for x in mon_moves(b1[0][0])], "revealed move lost"
        changed = sum(mon_species(a) != mon_species(bm)
                      for a, bm in zip(s1[0], b1[0]))
        print(f"unrevealed mons replaced: {changed}/6 (expect 5)")
        print("SELFTEST PASS")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
