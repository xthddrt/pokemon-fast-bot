"""THIRD PASS: do the 8 worlds' SET DISTRIBUTION match the true posterior?

    .venv/bin/python audit_posterior.py <run_dir> [--ref ps_reference.json]

Passes 1 and 2 ask whether each world is individually legal. This one asks
whether the WORLDS TOGETHER are drawn at the right rates -- the property the
seat allocator controls, and the one the old largest-remainder planner broke.

Method: for each decision and each revealed opponent mon, take the set of
movesets PS actually generates for that species, keep those consistent with
the moves revealed BY THAT DECISION, and weight them by PS's own counts. That
is an independent estimate of the posterior the sampler should be drawing
from -- independent because it comes from the JS generator, not from our
candidate-filtering code. Then compare the 8 sampled worlds against it.

Three statistics:
  TV        total-variation distance between sampled and reference posterior.
            Noisy at 8 worlds; only the MEAN over many decisions is meaningful.
  TOP-RATIO sampled share of the highest-probability moveset divided by its
            reference share. The old allocator renormalized over SEATED sets
            only, handing the unseated tail's mass to whoever was already
            seated, so a systematic value > 1 is that bias's signature.
  ORPHANS   movesets with reference probability >= 1/n_worlds that got ZERO
            worlds. Those are truths the search could not price at all.

Comparison is on MOVESETS, not full signatures: items and tera drift as the
game knocks things off, movesets do not.
"""
import argparse
import collections
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_sampling import norm, parse_mon, revealed_by_decision  # noqa: E402



def item_exclusions(run_dir, ident2sp, opp_prefix="p2a"):
    """species -> set of items the RAW PROTOCOL rules out, by decision.

    Conditioning the reference posterior on revealed MOVES alone over-reports:
    the sampler legitimately narrows further on item evidence, and a moves-only
    reference then calls a correct 8/8 concentration an "orphaned" 80% of the
    posterior. That is exactly what happened to Snorlax in run 20260820-142927,
    where the bot had correctly ruled out Leftovers.

    Derived HERE from the protocol, not read out of foul-play's own ledger --
    reusing the bot's conclusions to grade the bot would be circular. Two rules,
    both objective:
      * a mon that sat through a residual phase BELOW max hp without a
        Leftovers/Black Sludge heal cannot be holding one
      * a mon that used two different moves cannot be Choice-locked
    """
    CHOICE = {"choiceband", "choicescarf", "choicespecs"}
    LEFTY = {"leftovers", "blacksludge"}
    out = collections.defaultdict(set)
    per_dec_excl = {}
    decision = 1
    active = None
    below_max = set()
    healed_this_turn = set()
    moves_used = collections.defaultdict(set)
    for raw in open(os.path.join(run_dir, "battle.log"), errors="replace"):
        if "Choice:" in raw:
            per_dec_excl[decision] = {k: set(v) for k, v in out.items()}
            decision += 1
            continue
        if re.search(r"\|turn\|\d+", raw):
            # residual phase of the turn just ended
            if active and active in below_max and active not in healed_this_turn:
                out[ident2sp.get(active, active)] |= LEFTY
            healed_this_turn = set()
            continue
        m = re.search(r"\|switch\|%s: ([^|]+)\|" % opp_prefix, raw)
        if m:
            active = norm(m.group(1))
            below_max.discard(active)
            continue
        m = re.search(r"\|-damage\|%s: ([^|]+)\|(\d+)\\?/(\d+)" % opp_prefix, raw)
        if m and int(m.group(2)) < int(m.group(3)):
            below_max.add(norm(m.group(1)))
            continue
        m = re.search(r"\|-heal\|%s: ([^|]+)\|.*item:", raw)
        if m:
            healed_this_turn.add(norm(m.group(1)))
            continue
        m = re.search(r"\|move\|%s: ([^|]+)\|([^|]+)\|" % opp_prefix, raw)
        if m:
            who = norm(m.group(1))
            moves_used[who].add(norm(m.group(2)))
            if len(moves_used[who]) > 1:
                out[ident2sp.get(who, who)] |= CHOICE
    return per_dec_excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default=os.path.join(HERE, "ps_reference.json"))
    a = ap.parse_args()

    ref = json.load(open(a.ref))
    R = ref["species"]
    truth = json.load(open(os.path.join(a.run_dir, "truth.json")))
    ident2sp, species = {}, []
    for m in truth["team"]:
        ident = norm(m["ident"].split(": ")[1])
        sp = norm(m["details"].split(",")[0])
        ident2sp[ident] = sp
        species.append(sp)

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "worlds.jsonl"))]
    by_dec = collections.defaultdict(list)
    for r in rows:
        by_dec[r["decision"]].append(r)
    per_dec = revealed_by_decision(a.run_dir)
    excl_by_dec = item_exclusions(a.run_dir, ident2sp)

    tvs, ratios, orphan_rows = [], [], []
    scored = 0
    for d in sorted(by_dec):
        worlds = by_dec[d]
        n = len(worlds)
        rmoves = per_dec.get(d, ({}, {}, {}, {}))[0]
        for sp in species:
            if sp not in R:
                continue
            known = set()
            for ident, mvs in rmoves.items():
                if ident2sp.get(ident) == sp:
                    known |= mvs
            # reference posterior over movesets, filtered by what was revealed
            banned = excl_by_dec.get(d, {}).get(sp, set())
            exp = collections.Counter()
            for sig, c in R[sp]["sets"].items():
                parts = sig.split("|")
                ms = frozenset(parts[0].split(","))
                if known <= ms and parts[1] not in banned:
                    exp[ms] += c
            tot = sum(exp.values())
            if tot == 0 or len(exp) < 2:
                continue
            expp = {k: v / tot for k, v in exp.items()}

            obs = collections.Counter()
            seen = 0
            for w in worlds:
                mon = parse_mon(w["state"], sp)
                if not mon:
                    continue
                ms = frozenset(x for x in map(norm, mon["moves"]) if x != "none")
                obs[ms] += 1
                seen += 1
            if seen < n:          # mon not in every world -> not comparable
                continue
            scored += 1
            obsp = {k: v / seen for k, v in obs.items()}
            keys = set(expp) | set(obsp)
            tvs.append(0.5 * sum(abs(obsp.get(k, 0) - expp.get(k, 0)) for k in keys))
            top = max(expp, key=expp.get)
            if expp[top] > 0:
                ratios.append(obsp.get(top, 0) / expp[top])
            orph = [k for k, p in expp.items() if p >= 1.0 / seen and obsp.get(k, 0) == 0]
            if orph:
                orphan_rows.append((d, sp, len(orph),
                                    round(sum(expp[k] for k in orph), 3)))

    print("# Posterior audit — %s" % os.path.basename(os.path.normpath(a.run_dir)))
    print("reference: %d real PS teams | comparable (decision, mon) pairs: %d\n"
          % (ref["teams"], scored))
    if not scored:
        print("nothing comparable — the mons were never in every world at once")
        return

    def stat(v):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
        return m, 1.96 * sd / math.sqrt(len(v))

    tv_m, tv_ci = stat(tvs)
    r_m, r_ci = stat(ratios)
    print("   mean TV distance          %.3f +/- %.3f" % (tv_m, tv_ci))
    print("   mean TOP-SET ratio        %.3f +/- %.3f   (1.0 = unbiased; the old"
          % (r_m, r_ci))
    print("                                              allocator inflated this)")
    verdict = ("no top-set inflation detected" if abs(r_m - 1.0) <= r_ci
               else ("TOP SET OVER-SAMPLED" if r_m > 1 else "TOP SET UNDER-SAMPLED"))
    print("   verdict: %s" % verdict)
    print("\n   orphaned movesets (reference p >= 1/worlds, ZERO worlds): %d of %d pairs"
          % (len(orphan_rows), scored))
    for row in sorted(orphan_rows, key=lambda x: -x[3])[:8]:
        print("      d%-3d %-16s %d moveset(s), %.1f%% of posterior unreachable"
              % (row[0], row[1], row[2], 100 * row[3]))


if __name__ == "__main__":
    main()
