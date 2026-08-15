"""Stream a corpus shard -> per-decision arrays for the FLOOR measurement.

MODEL-FREE AND ENCODER-FREE. Nothing here touches a net, an encoder, or a
search value. The only things read out of the corpus are: the serialized state,
the game outcome, the per-decision exploration tag, and the number of legal
arms. Everything else is derived by string manipulation on the state.

EXACT-STATE IDENTITY (level `E`)
  The engine's serialized state, canonicalised for the ONE semantically
  irrelevant degree of freedom it carries: the party-slot order. generate.py
  shuffles each side's party per game, so two games standing in the identical
  game situation serialise to different strings purely because the bench sits in
  different slots. Canonicalisation puts the active mon in slot 0 and sorts the
  remaining five by their own serialisation, then remaps every slot-indexed
  field: `active_index` (f6), `future_sight.1` (f21) and `last_used_move`
  (f28, "switch:<idx>"). Nothing else in the string is slot-indexed
  (verified against poke-engine src/state.rs Side::serialize).
  After that, `E` is bit-exact: same HP, same PP, same volatile durations.

NEAR-DUPLICATE LADDER (L1..L7), each a strict coarsening of the state:
  L1  exact minus move PP and pure bookkeeping (revealed / known /
      illusion_broken / times_attacked / last_consumed_item /
      once_per_battle_ability_used / active_move_actions / stellar_boosted /
      reveal_mask / level / nature / EVs / stats / weight / base types).
      Everything that can change the legal move set or the dynamics is kept.
  L2  L1 with each mon's HP bucketed to 1/32 of its max.
  L3  L2 at 1/16 HP, volatile-status DURATIONS dropped.
  L4  "the position as a player would restate it": both actives' species +
      tera, each side's roster as a sorted (species, 1/8 HP bucket, status,
      tera) multiset, both sides' side-condition strings, both actives' boosts,
      weather/terrain/trick-room.
  L5  the `stats.py` coarse key: (active pair, alive counts, both actives' HP
      quartile, both sides' side conditions, tera flags).
  L6  (alive counts, each side's team HP fraction in 1/8 buckets).
  L7  (alive counts).
  L8  everything in one group  ->  the base-rate constant predictor.

USAGE  python extract.py <shard_glob> <out.npz>
"""

import glob
import gzip
import hashlib
import json
import sys
import time

import numpy as np

# ---- serialized-field layout (poke-engine src/state.rs) --------------------
# Side split on '=' : f0..f5 mons, f6 active_index, f7 side_conditions,
# f8 volatile_statuses, f9 volatile_status_durations, f10 substitute_health,
# f11..f17 boosts, f18..f19 wish, f20..f21 future_sight, f22 force_switch,
# f23 switch_out_move_second_saved_move, f24 baton_passing, f25 shed_tailing,
# f26 revival_blessing, f27 force_trapped, f28 last_used_move,
# f29 slow_uturn_move, f30 times_revived, f31 last_move_failed
MON_ID, MON_T1, MON_T2 = 0, 2, 3
MON_HP, MON_MAXHP = 6, 7
MON_ABIL, MON_ITEM = 8, 10
MON_STATUS, MON_REST, MON_SLEEP = 18, 19, 20
MON_MOVES = (22, 23, 24, 25)
MON_TERA, MON_TERATYPE = 26, 27

SIDE_KEEP_L1 = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def canon_side(f):
    """Party-slot canonicalisation: active first, rest sorted; indices remapped."""
    ai = int(f[6])
    order = [ai] + sorted((j for j in range(6) if j != ai), key=lambda j: f[j])
    new = {o: n for n, o in enumerate(order)}
    g = list(f)
    g[0:6] = [f[j] for j in order]
    g[6] = "0"
    if f[21].isdigit() and int(f[21]) < 6:
        g[21] = str(new[int(f[21])])
    if f[28].startswith("switch:"):
        v = f[28][7:]
        if v.isdigit() and int(v) < 6:
            g[28] = "switch:%d" % new[int(v)]
    return g


def mon_l1(m, hpbins=0):
    """Mon fields that can affect the game; PP and bookkeeping dropped."""
    hp, mx = int(m[MON_HP]), max(int(m[MON_MAXHP]), 1)
    hpv = str(hp) if hpbins == 0 else ("0" if hp == 0 else
                                       str(1 + min(hpbins - 1, (hp * hpbins - 1) // mx)))
    mv = ";".join(m[i].rsplit(";", 1)[0] for i in MON_MOVES)   # name;disabled  (PP dropped)
    return ",".join((m[MON_ID], m[MON_T1], m[MON_T2], hpv, m[MON_ABIL], m[MON_ITEM],
                     m[MON_STATUS], m[MON_REST], m[MON_SLEEP], mv,
                     m[MON_TERA], m[MON_TERATYPE]))


def hpbin(hp, mx, nb):
    if hp <= 0:
        return 0
    return 1 + min(nb - 1, (hp * nb - 1) // max(mx, 1))


def keys_for(s):
    """All grouping keys for one serialized state. Returns list of hashes."""
    p = s.split("/")
    glob_ = "/".join(p[2:])
    sides = [x.split("=") for x in p[:2]]
    csides = [canon_side(f) for f in sides]
    cmons = [[c[j].split(",") for j in range(6)] for c in csides]

    # E : exact canonical
    kE = h64("/".join("=".join(c) for c in csides) + "/" + glob_)

    def lvl(hpbins, keep_durations=True, keep_last_move=True):
        outs = []
        for c, mons in zip(csides, cmons):
            body = "|".join(mon_l1(m, hpbins) for m in mons)
            side_fields = [c[i] for i in SIDE_KEEP_L1
                           if (keep_durations or i != 9) and (keep_last_move or i != 28)]
            outs.append(body + "#" + "=".join(side_fields))
        return h64(outs[0] + "/" + outs[1] + "/" + glob_)

    k1 = lvl(0)
    k2 = lvl(32)
    k3 = lvl(16, keep_durations=False)

    # L4 : the position as a player would restate it
    l4 = []
    for c, mons in zip(csides, cmons):
        a = mons[0]
        roster = sorted(",".join((m[MON_ID], str(hpbin(int(m[MON_HP]), int(m[MON_MAXHP]), 8)),
                                  m[MON_STATUS], m[MON_TERA])) for m in mons)
        l4.append("|".join([a[MON_ID], a[MON_TERA], a[MON_TERATYPE]] + roster
                           + [c[7]] + [c[i] for i in range(11, 18)]))
    k4 = h64(l4[0] + "/" + l4[1] + "/" + glob_)

    # L5 : stats.py's coarse key
    l5 = []
    for c, mons in zip(csides, cmons):
        a = mons[0]
        alive = sum(int(m[MON_HP]) > 0 for m in mons)
        l5.append("%s,%s,%d,%d,%s" % (a[MON_ID], a[MON_TERA], alive,
                                      hpbin(int(a[MON_HP]), int(a[MON_MAXHP]), 4), c[7]))
    k5 = h64(l5[0] + "/" + l5[1])

    # L6 : alive counts + team HP fraction in 1/8
    l6 = []
    for mons in cmons:
        alive = sum(int(m[MON_HP]) > 0 for m in mons)
        frac = sum(int(m[MON_HP]) / max(int(m[MON_MAXHP]), 1) for m in mons) / 6.0
        l6.append("%d,%d" % (alive, min(7, int(frac * 8))))
    k6 = h64(l6[0] + "/" + l6[1])

    # L7 : alive counts only
    k7 = h64("%d/%d" % tuple(sum(int(m[MON_HP]) > 0 for m in mons) for mons in cmons))
    return [kE, k1, k2, k3, k4, k5, k6, k7]


LEVELS = ["E", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
TAGS = {"": 0, "rand": 1, "temp": 2, "eps": 3}


def main():
    paths = sorted(glob.glob(sys.argv[1]))
    assert paths, "no shards matched %r" % sys.argv[1]
    out = sys.argv[2]
    gid, ply, y, tag, nrand, glen, nleg = [], [], [], [], [], [], []
    kk = [[] for _ in LEVELS]
    t0 = time.time()
    ng = 0
    for pth in paths:
        with gzip.open(pth, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header" or "error" in row or not row.get("t"):
                    continue
                ts = row["t"]
                o = float(row["outcome"])
                nr = 0
                for t in ts:
                    if t["e"] == "rand":
                        nr += 1
                    else:
                        break
                T = len(ts)
                for pi, t in enumerate(ts):
                    for j, k in enumerate(keys_for(t["s"])):
                        kk[j].append(k)
                    gid.append(ng); ply.append(pi); y.append(o)
                    tag.append(TAGS[t["e"]]); nrand.append(nr); glen.append(T)
                    nleg.append(len(t["n"]) * 100 + min(len(t["n2"]), 99))
                ng += 1
                if ng % 2000 == 0:
                    print("  %d games  %d rows  %.0fs" % (ng, len(gid), time.time() - t0), flush=True)
    d = {"gid": np.asarray(gid, np.int32), "ply": np.asarray(ply, np.int32),
         "y": np.asarray(y, np.float32), "tag": np.asarray(tag, np.int8),
         "nrand": np.asarray(nrand, np.int8), "glen": np.asarray(glen, np.int32),
         "nleg": np.asarray(nleg, np.int32)}
    for j, L in enumerate(LEVELS):
        d["k_" + L] = np.asarray(kk[j], np.uint64)
    np.savez(out, **d)
    print("wrote %s: %d games, %d rows, %.0fs" % (out, ng, len(gid), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
