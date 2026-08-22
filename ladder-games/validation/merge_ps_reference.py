"""Merge gen_ps_reference.js shards into one reference corpus.

gen_ps_reference.js is single-threaded (~770 teams/s), so a large corpus is
generated as parallel shards and merged here:

    for i in 1 2 3; do node gen_ps_reference.js 350000 > shard$i.json & done; wait
    python3 merge_ps_reference.py shard*.json -o ps_reference.json

WHY BIG: the auditor reads "absent from the reference" as "PS cannot build
this", which only holds once the corpus is deep enough to have drawn the tail.
At 100k teams (~1.2k draws/species) that was false often enough to dominate the
findings -- pincurchin discharge/recover/scald/thunderbolt is a REAL set that
appears 3 times in 1.05M teams and zero times in 100k, and it was flagged as
illegal. At 1.05M teams (~12.4k draws/species) no species has a singleton
moveset left, i.e. the moveset tail is exhausted everywhere.

Shards must come from SEPARATE processes: Teams.generate seeds its PRNG per
process, so separate runs are independent draws (verified -- shard counts
differ). Do NOT reuse one shard for both this reference and the sampler's
weights in data/ps/gen9randombattle_set_dist.json when comparing the two
distributions to each other; for pure LEGALITY (is this set in PS's support)
more teams is strictly better and sharing is fine, because the answer is the
real generator's output either way.
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    shards = []
    for p in a.shards:
        with open(p) as f:
            shards.append(json.load(f))

    sp = collections.defaultdict(
        lambda: {"n": 0, "levels": collections.Counter(),
                 "sets": collections.Counter()})
    for s in shards:
        for k, rec in s["species"].items():
            sp[k]["n"] += rec["n"]
            sp[k]["levels"].update(rec["levels"])
            sp[k]["sets"].update(rec["sets"])

    stats = {"n": sum(s["teamStats"]["n"] for s in shards),
             "dupSpecies": sum(s["teamStats"]["dupSpecies"] for s in shards),
             "perMove": {}}
    for mv in shards[0]["teamStats"]["perMove"]:
        c = collections.Counter()
        for s in shards:
            c.update(s["teamStats"]["perMove"][mv])
        stats["perMove"][mv] = dict(c)

    out = {
        "teams": sum(s["teams"] for s in shards),
        "species": {k: {"n": v["n"], "levels": dict(v["levels"]),
                        "sets": dict(v["sets"])} for k, v in sp.items()},
        "teamStats": stats,
        # top-level alias map, NOT per-species -- dropping it silently breaks
        # cosmetic-forme resolution and every Florges-White/Gastrodon-East slot
        # reads as species_not_in_ps_pool.
        "cosmetic": shards[0]["cosmetic"],
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    tot = sum(v["n"] for v in out["species"].values())
    print("%d teams -> %d species, %d mon draws (%.0f/species), %d cosmetic aliases"
          % (out["teams"], len(out["species"]), tot,
             tot / len(out["species"]), len(out["cosmetic"])))


if __name__ == "__main__":
    main()
