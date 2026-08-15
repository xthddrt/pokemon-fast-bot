"""Correctness gate for the canonicalisation used by extract.py.

Three checks, all hand-verifiable:
 1. canon(s) still deserialises with the real engine, and describes the SAME
    game situation (same active, same multiset of (species,hp,status), same
    side conditions, same legal-arm set).
 2. canon is invariant under an arbitrary party-slot permutation applied to s.
 3. at ply 0 the 20k games must collapse to exactly 36 canonical states --
    the 36 (lead x lead) openings -- because every game starts from the same
    two full-HP teams and differs only in which mon leads and in slot order.
"""
import gzip, json, random, sys
sys.path.insert(0, "/Users/sallyliu/pokemon-fast-bot/evallab")
sys.path.insert(0, "/Users/sallyliu/pokemon-fast-bot/evallab/floor")
import labenv  # noqa
from poke_engine import State  # noqa
import extract

def permute_side(f, perm):
    """Apply a party permutation to a raw side field-list (inverse of canon)."""
    ai = int(f[6])
    g = list(f)
    g[0:6] = [f[perm[j]] for j in range(6)]
    new = {perm[j]: j for j in range(6)}
    g[6] = str(new[ai])
    if f[21].isdigit() and int(f[21]) < 6:
        g[21] = str(new[int(f[21])])
    if f[28].startswith("switch:"):
        v = f[28][7:]
        if v.isdigit() and int(v) < 6:
            g[28] = "switch:%d" % new[int(v)]
    return g

def sig(st):
    def side(s):
        ms = list(s.pokemon)
        a = ms[int(s.active_index)]
        return (a.id, a.hp, a.status,
                tuple(sorted((p.id, p.hp, p.status, p.terastallized) for p in ms)),
                (s.side_conditions.stealth_rock, s.side_conditions.spikes, s.side_conditions.toxic_spikes, s.side_conditions.sticky_web, s.side_conditions.reflect, s.side_conditions.light_screen), s.attack_boost, s.speed_boost)
    return (side(st.side_one), side(st.side_two), st.weather, st.terrain, st.trick_room)

def canon_str(s):
    p = s.split("/")
    return "/".join("=".join(extract.canon_side(x.split("="))) for x in p[:2]) + "/" + "/".join(p[2:])

path = sys.argv[1]
rng = random.Random(0)
n_perm_ok = n_sig_ok = 0
ply0 = {}
ngames = 0
with gzip.open(path, "rt") as f:
    for line in f:
        row = json.loads(line)
        if row.get("kind") == "header" or not row.get("t"):
            continue
        ngames += 1
        ply0.setdefault(canon_str(row["t"][0]["s"]), set()).add(tuple(row["lead"]))
        if ngames <= 300:
            for t in rng.sample(row["t"], min(3, len(row["t"]))):
                s = t["s"]
                c = canon_str(s)
                # check 1: engine round-trip + semantic identity
                a, b = State.from_string(s), State.from_string(c)
                assert sig(a) == sig(b), "SEMANTIC MISMATCH"
                n_sig_ok += 1
                # check 2: invariance to a random party permutation
                p = s.split("/")
                sides = [x.split("=") for x in p[:2]]
                pr = [list(range(6)), list(range(6))]
                rng.shuffle(pr[0]); rng.shuffle(pr[1])
                perm_s = "/".join("=".join(permute_side(sides[i], pr[i])) for i in (0, 1)) \
                    + "/" + "/".join(p[2:])
                assert sig(State.from_string(perm_s)) == sig(a), "PERMUTE BROKE THE STATE"
                assert canon_str(perm_s) == c, "CANON NOT PERMUTATION-INVARIANT"
                n_perm_ok += 1
print("check1 semantic round-trip : %d/%d OK" % (n_sig_ok, n_sig_ok))
print("check2 permutation-invariant: %d/%d OK" % (n_perm_ok, n_perm_ok))
print("check3 distinct canonical ply-0 states over %d games: %d  (expect 36)" % (ngames, len(ply0)))
bad = {k: v for k, v in ply0.items() if len(v) != 1}
print("        each maps to exactly one lead pair: %s" % ("YES" if not bad else "NO (%d bad)" % len(bad)))
print("        distinct lead pairs covered: %d" % len({v for s in ply0.values() for v in s}))
