"""FOURTH PASS: how often do sampled worlds carry the revealed mon's TRUE set,
and how close is that to the best any sampler could do?

    python3 audit_setaccuracy.py <run_dir> [run_dir...]

For every (decision, revealed opponent mon) with the mon present in sampled
worlds, two numbers:

  HIT    fraction of worlds whose sampled JOINT signature (4 moves + item +
         ability) equals the mon's true one. Tera is excluded: it drifts
         legitimately once used, and truth.json holds the pre-battle value.
  CEIL   the Bayes ceiling for a sampler that draws from the exact posterior:
         restrict the real-PS reference (1.05M teams) to this species' sets
         consistent with the evidence *visible in the world itself* (revealed
         moves are a subset, level matches), weight by generator counts, and
         read off the posterior mass of the true signature. A perfect sampler's
         expected HIT is exactly CEIL; HIT persistently below CEIL means our
         posterior is mis-shaped, HIT above CEIL means the bot is using
         evidence this ceiling does not condition on (damage, speed, item
         reveals) -- which is fine and expected, so CEIL here is a FLOOR-ish
         reference, not a hard bound. To keep the comparison honest the
         ceiling also conditions on item/ability when the game revealed them
         (they are in the world's own reveal history).

Evidence tiers report both, bucketed by how many of the true set's moves were
revealed at decision time, so "we should be nailing fully-revealed mons" is
visible separately from "1 move seen" guessing.
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_sampling import (  # noqa: E402
    norm, revealed_by_decision, _nickname_species_map)

_REF = None
_COSMETIC = None


def ref():
    global _REF, _COSMETIC
    if _REF is None:
        d = json.load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ps_reference.json")))
        _REF, _COSMETIC = d["species"], d.get("cosmetic", {})
    return _REF


def ref_key(sp):
    """Reference key for a truth species: itself, or its cosmetic base."""
    ref()
    return sp if sp in _REF else _COSMETIC.get(sp)


def opponent_chunks(state):
    out = []
    for chunk in state.split("=")[31:37]:
        f = chunk.split(",")
        if len(f) < 30 or not f[0]:
            continue
        sp = norm(f[0].split("/")[-1])
        if not sp or sp in ("0", "false", "none"):
            continue
        out.append({
            "species": sp, "level": f[1],
            "ability": norm(f[8]), "item": norm(f[10]),
            "moves": sorted(norm(x.split(";")[0]) for x in f[22:26]
                            if norm(x.split(";")[0]) not in ("", "none")),
        })
    return out


def main():
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])  # tier -> [hit, ceil, n]
    per_species_gap = collections.Counter()
    # CHECK 1 (share calibration): for every DISTINCT sampled joint signature,
    # (share k of n worlds, was it the truth). If a set carries k/8 of the
    # worlds it should BE the truth about k/8 of the time -- the property the
    # search actually consumes, which the tier MEANS above average away. A
    # mode-seeking sampler (8/8 agreement, 60% truth) passes every other pass.
    # Note the worlds are a systematic-PPS seating, not iid draws, so the
    # bucket target k/n holds to within one seat's rounding.
    calib = collections.defaultdict(lambda: [0, 0])  # k -> [n_sigs, n_true]
    # CHECK 2 (truth orphans): the true set had posterior mass >= 1/n_worlds
    # yet got ZERO worlds -- the catastrophic-miss list tier means hide.
    orphans = []
    n_pts = 0
    for run_dir in sys.argv[1:]:
        tpath = os.path.join(run_dir, "truth.json")
        wpath = os.path.join(run_dir, "worlds.jsonl")
        if not (os.path.isfile(tpath) and os.path.isfile(wpath)):
            continue
        truth = {}
        ident2sp = {}
        for m in json.load(open(tpath))["team"]:
            sp = norm(m["details"].split(",")[0])
            ident2sp[norm(m["ident"].split(":", 1)[1])] = sp
            truth[sp] = {
                "moves": sorted(norm(x) for x in m["moves"]),
                "item": norm(m.get("item") or ""),
                "ability": norm(m.get("ability") or ""),
                "level": (re.search(r"L(\d+)", m["details"]) or [None, "100"])[1],
            }
        plog = os.path.join(run_dir, "protocol.log")
        # Transform exemption (as pass 1) + the NAME of each lost item: the
        # -enditem line names it, and the ceiling must keep conditioning on
        # evidence the sampler genuinely had after a Knock Off.
        transformed = set()
        removed_item = {}
        nickmap = _nickname_species_map(plog)
        if os.path.isfile(plog):
            for line in open(plog, errors="replace"):
                mm = re.search(
                    r"\|-transform\|p[12]a: ([^|]+)\|p[12]a: ([^|]+)", line)
                if mm:
                    for nick in (mm.group(1), mm.group(2)):
                        transformed.add(norm(nick))
                        transformed.update(nickmap.get(norm(nick), ()))
                mm = re.search(r"\|-enditem\|p2a: ([^|]+)\|([^|]+)", line)
                if mm:
                    spx = ident2sp.get(norm(mm.group(1)), norm(mm.group(1)))
                    removed_item.setdefault(spx, norm(mm.group(2)))
        # Reveal maps key on protocol NICKNAMES, truth on details species --
        # unmapped, every forme mon (Sawsbuck-Winter, Rotom-*, regional formes)
        # silently vanished from the metric (verified on g1: 18 datapoints
        # dropped). Remap through truth.json's ident field.
        def remap(d):
            return {ident2sp.get(k, k): v for k, v in d.items()}
        revealed = {dec: tuple(remap(x) for x in r)
                    for dec, r in revealed_by_decision(run_dir).items()}
        # presence-by-decision: which species had SWITCHED IN by each decision
        # (battle.log interleaves protocol lines with per-decision "Choice:"
        # markers, same framing revealed_by_decision uses)
        present_by_dec = {}
        _seen = set()
        _dec = 1
        blog = os.path.join(run_dir, "battle.log")
        if os.path.isfile(blog):
            for line in open(blog, errors="replace"):
                if "Choice:" in line:
                    present_by_dec[_dec] = set(_seen)
                    _dec += 1
                    continue
                mm = re.search(r"\|(?:switch|drag)\|p2a: [^|]+\|([^|,]+)", line)
                if mm:
                    _seen.add(norm(mm.group(1)))
        worlds_by_dec = collections.defaultdict(list)
        for line in open(wpath):
            r = json.loads(line)
            worlds_by_dec[r["decision"]].append(r["state"])
        for dec, states in worlds_by_dec.items():
            rev = revealed.get(dec)
            if not rev:
                continue
            rmoves, ritems, rabils, _ = rev
            # every species the opponent has SHOWN, not only ones that used a
            # move -- pinning from species+level alone (the 0/4 tier) is where
            # early-game sampling matters most and was previously unmeasured.
            audited = set(rmoves) | (
                present_by_dec.get(dec, set()) & set(truth))
            for sp in audited:
                seen_moves = rmoves.get(sp, set())
                t = truth.get(sp)
                rk = ref_key(sp)
                if t is None or rk is None or sp in transformed:
                    continue
                # worlds where the mon appears
                # A knocked-off/consumed item makes the world's "none" CORRECT
                # while truth still holds the original -- 97% of raw 4/4-tier
                # item misses were exactly this (13,310-instance sample), and
                # itemless sets normalize "" vs "none" inconsistently. Compare
                # items only while the item is still comparable.
                item_removed = ritems.get(sp) == "__removed__"
                _n = lambda x: "" if x in ("", "none") else x
                hits = tot = 0
                for st in states:
                    # FIRST matching chunk only: a transform copy or forme
                    # collision can put the species in a world twice, and
                    # double-counting produced impossible >8/8 shares
                    for c in opponent_chunks(st):
                        if c["species"] != sp:
                            continue
                        tot += 1
                        if (c["moves"] == t["moves"]
                                and (item_removed
                                     or _n(c["item"]) == _n(t["item"]))
                                and c["ability"] == t["ability"]):
                            hits += 1
                        break
                if tot == 0:
                    continue
                # Bayes ceiling from the reference
                seen = {norm(x) for x in seen_moves}
                item_known = ritems.get(sp) not in (None, "__removed__")
                abil_known = sp in rabils
                num = den = 0
                for sig, cnt in ref()[rk]["sets"].items():
                    mv, it, ab, _te = sig.split("|")
                    mvset = set(mv.split(","))
                    if not seen <= mvset:
                        continue
                    if item_known and it != ritems[sp]:
                        continue
                    if item_removed and sp in removed_item \
                            and _n(it) != _n(removed_item[sp]):
                        continue
                    if abil_known and ab != rabils[sp]:
                        continue
                    den += cnt
                    if (sorted(mvset) == t["moves"] and ab == t["ability"]
                            and (item_removed or _n(it) == _n(t["item"]))):
                        num += cnt
                ceil = (num / den) if den else 0.0
                tier = "%d/4 moves%s%s" % (
                    len(seen & set(t["moves"])),
                    "+item" if item_known else "",
                    "+abil" if abil_known else "")
                a = agg[tier]
                a[0] += hits / tot; a[1] += ceil; a[2] += 1
                n_pts += 1
                if ceil - hits / tot > 0.25:
                    per_species_gap[sp] += 1
                if hits == 0 and ceil >= 1.0 / max(1, tot):
                    orphans.append((run_dir, dec, sp, ceil, tot))
                sig_counts = collections.Counter()
                truth_sig = None
                for st in states:
                    for c in opponent_chunks(st):
                        if c["species"] != sp:
                            continue
                        k = (tuple(c["moves"]),
                             "" if item_removed else _n(c["item"]),
                             c["ability"])
                        sig_counts[k] += 1
                        if (c["moves"] == t["moves"] and c["ability"] == t["ability"]
                                and (item_removed
                                     or _n(c["item"]) == _n(t["item"]))):
                            truth_sig = k
                        break
                for k, cnt in sig_counts.items():
                    # probe-phase decisions log 16 worlds, so bucket by the
                    # NORMALIZED 8-world share, not the raw count -- raw
                    # counts produced impossible 9/8..16/8 rows
                    b = max(1, min(8, round(8.0 * cnt / max(1, tot))))
                    calib[b][0] += 1
                    if k == truth_sig:
                        calib[b][1] += 1

    print("# TRUE-SET SAMPLE RATE vs BAYES CEILING  (%d datapoints)" % n_pts)
    print("%-28s %10s %10s %8s %6s" % ("evidence tier", "HIT", "CEIL", "gap", "n"))
    for tier in sorted(agg, key=lambda t: -agg[t][2]):
        h, c, n = agg[tier]
        print("%-28s %9.1f%% %9.1f%% %+7.1f%% %6d"
              % (tier, 100 * h / n, 100 * c / n, 100 * (h - c) / n, n))
    if per_species_gap:
        print("\nspecies most often >25pt below ceiling:")
        for sp, c in per_species_gap.most_common(10):
            print("   %-22s %d" % (sp, c))

    print("\n## SHARE CALIBRATION (a set at k/8 worlds should be truth ~k/8)")
    print("%6s %10s %12s %10s %8s" % ("share", "n_sigs", "truth-rate", "target", "delta"))
    ece_num = ece_den = 0.0
    for k in sorted(calib):
        n_sigs, n_true = calib[k]
        if not n_sigs:
            continue
        rate, tgt = n_true / n_sigs, k / 8.0
        ece_num += n_sigs * abs(rate - tgt)
        ece_den += n_sigs
        print("%5d/8 %10d %11.1f%% %9.1f%% %+7.1f%%"
              % (k, n_sigs, 100 * rate, 100 * tgt, 100 * (rate - tgt)))
    if ece_den:
        print("ECE (share-weighted) = %.4f" % (ece_num / ece_den))

    if orphans:
        print("\n## TRUTH ORPHANS (true set had >=1/8 posterior mass, got 0 worlds)")
        sev = sorted(orphans, key=lambda o: -o[3])
        for run_dir, dec, sp, ceil, tot in sev[:15]:
            print("   %-28s d%-4d %-18s ceil=%.2f worlds=%d"
                  % (os.path.basename(run_dir), dec, sp, ceil, tot))
        print("   total: %d (%d with ceil>=0.5)"
              % (len(orphans), sum(1 for o in orphans if o[3] >= 0.5)))
    else:
        print("\n## TRUTH ORPHANS: none")


if __name__ == "__main__":
    main()
