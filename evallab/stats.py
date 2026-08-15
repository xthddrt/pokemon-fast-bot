"""Corpus diversity report -- the honest answer to "did self-play on ONE fixed
pair actually explore anything, or did it play the same game 10,000 times?"

Exact-state distinctness is a useless metric here (HP and PP make almost every
state unique), so the report leads with a COARSE state key -- the things a
player would call "the position" -- plus arm coverage, which is what the
sibling metric depends on.

USAGE  python stats.py <shard_glob...>
"""

import glob
import gzip
import json
import sys
from collections import Counter, defaultdict

import labenv  # noqa: F401
from poke_engine import State  # noqa: E402


def coarse(st):
    """(active pair, alive counts, HP quartile of each active, hazard state)."""
    s1, s2 = st.side_one, st.side_two
    a1 = list(s1.pokemon)[int(s1.active_index)]
    a2 = list(s2.pokemon)[int(s2.active_index)]
    def hpq(p):
        return int(min(3, max(0, (p.hp * 4) // max(p.maxhp, 1))))
    def hz(s):
        c = s.side_conditions
        return (min(c.stealth_rock, 1), min(c.spikes, 3), min(c.toxic_spikes, 2), min(c.sticky_web, 1))
    return (a1.id, a2.id,
            sum(p.hp > 0 for p in s1.pokemon), sum(p.hp > 0 for p in s2.pokemon),
            hpq(a1), hpq(a2), hz(s1), hz(s2),
            int(a1.terastallized), int(a2.terastallized))


def report(paths):
    games = 0
    turns = 0
    outcomes = Counter()
    tags = Counter()
    leads = Counter()
    exact = set()
    crs = set()
    ctx_arms = defaultdict(set)     # (active pair) -> arms ever PLAYED
    ctx_legal = defaultdict(set)    # (active pair) -> arms ever LEGAL
    nlegal = []
    hdr = None
    for p in sorted(paths):
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "header":
                    hdr = hdr or row
                    continue
                if "error" in row or not row.get("t"):
                    continue
                games += 1
                outcomes[row["outcome"]] += 1
                leads[tuple(row["lead"])] += 1
                for t in row["t"]:
                    turns += 1
                    tags[t["e"] or "argmax"] += 1
                    exact.add(t["s"])
                    st = State.from_string(t["s"])
                    c = coarse(st)
                    crs.add(c)
                    key = (c[0], c[1])
                    ctx_arms[key].add(t["m"])
                    ctx_legal[key] |= set(t["n"])
                    nlegal.append(len(t["n"]))
    cov = [len(ctx_arms[k]) / max(len(ctx_legal[k]), 1) for k in ctx_legal]
    print("pair=%s iters=%s  games=%d decisions=%d  mean turns/game=%.1f"
          % (hdr.get("pair"), hdr.get("iters"), games, turns, turns / max(games, 1)))
    print("outcomes (s1 win / loss / draw): %d / %d / %d   win rate %.3f"
          % (outcomes[1.0], outcomes[0.0], outcomes[0.5],
             (outcomes[1.0] + 0.5 * outcomes[0.5]) / max(games, 1)))
    print("move selection: %s" % dict(tags))
    print("distinct EXACT states     : %d / %d  (%.1f%%)"
          % (len(exact), turns, 100.0 * len(exact) / max(turns, 1)))
    print("distinct COARSE positions : %d  (active pair x alive x hp-quartile x hazards x tera)"
          % len(crs))
    print("active-pair contexts seen : %d / 36 possible lead pairs; %d distinct (mon vs mon) contexts"
          % (len(leads), len(ctx_legal)))
    print("ARM COVERAGE per context  : mean %.3f of legal arms ever played (median %.3f, min %.3f)"
          % (sum(cov) / max(len(cov), 1), sorted(cov)[len(cov) // 2] if cov else 0, min(cov) if cov else 0))
    print("mean legal arms/decision  : %.2f" % (sum(nlegal) / max(len(nlegal), 1)))
    return {"games": games, "decisions": turns, "distinct_exact": len(exact),
            "distinct_coarse": len(crs), "contexts": len(ctx_legal),
            "lead_pairs": len(leads), "arm_coverage": sum(cov) / max(len(cov), 1),
            "win_rate": (outcomes[1.0] + 0.5 * outcomes[0.5]) / max(games, 1),
            "mean_turns": turns / max(games, 1), "tags": dict(tags)}


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(a)
    assert paths, "no shards matched"
    json.dump(report(paths), open("/dev/null", "w"))
