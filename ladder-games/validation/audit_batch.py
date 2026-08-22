"""Run all three audit passes over a batch of validation games and tally.

    .venv/bin/python audit_batch.py <batch_dir>

Prints one row per game plus a pooled total, so a defect that shows up in 2 of
20 games is visible as a RATE rather than as a one-off. Anything non-zero in
the HARD columns is a team Showdown could not have generated.
"""
import collections
import io
import json
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_sampling          # noqa: E402
import audit_team_legality     # noqa: E402
import audit_posterior         # noqa: E402

PASS1 = ["viol_missing_revealed_move", "viol_item", "viol_item_should_be_gone",
         "viol_ability", "viol_tera", "incomplete_moves", "incomplete_ability",
         "incomplete_item", "incomplete_tera", "tera_arm_offered_after_use"]
PASS2 = ["species_not_in_ps_pool", "duplicate_species", "two_stealthrock",
         "illegal_level", "illegal_ability", "illegal_tera", "illegal_item",
         "illegal_moveset", "illegal_joint_signature",
         "short_party", "missing_mon", "wrong_level", "wrong_ability",
         "wrong_moves"]


def run(mod, d):
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["x", d]
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    except SystemExit:
        pass
    except Exception as e:                      # keep the batch alive
        return "", "%s: %s" % (type(e).__name__, e)
    finally:
        sys.argv = argv
    return buf.getvalue(), None


def scrape(text, keys):
    out = {}
    for k in keys:
        m = re.search(r"^\s+%s\s+(\d+)" % re.escape(k), text, re.M)
        if m:
            out[k] = int(m.group(1))
    return out


def main():
    batch = sys.argv[1]
    games = sorted(
        (os.path.join(batch, g) for g in os.listdir(batch)
         if os.path.isdir(os.path.join(batch, g))
         and os.path.isfile(os.path.join(batch, g, "worlds.jsonl"))),
        key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0),
    )
    # A game whose log captured TWO battles cannot be graded: the auditor would
    # match one battle's truth against the other's worlds. run_validation_batch
    # marks those with an EXCLUDED file. Counted and reported rather than
    # silently dropped -- a silent skip reads as "covered everything".
    excluded = [g for g in games if os.path.isfile(os.path.join(g, "EXCLUDED"))]
    games = [g for g in games if g not in set(excluded)]
    tot = collections.Counter()
    rows, errs = [], []
    ndec = nworld = 0
    for d in games:
        name = os.path.basename(d)
        t1, e1 = run(audit_sampling, d)
        t2, e2 = run(audit_team_legality, d)
        t3, e3 = run(audit_posterior, d)
        for e in (e1, e2, e3):
            if e:
                errs.append((name, e))
        s = scrape(t1, PASS1)
        s.update(scrape(t2, PASS2))
        dec = sum(1 for _ in open(os.path.join(d, "search.log"))
                  if "Choice:" in _) if os.path.isfile(os.path.join(d, "search.log")) else 0
        w = sum(1 for _ in open(os.path.join(d, "worlds.jsonl")))
        ndec += dec
        nworld += w
        for k, v in s.items():
            tot[k] += v
        m = re.search(r"mean TOP-SET ratio\s+([\d.]+)", t3)
        rows.append((name, dec, w, sum(s.values()), m.group(1) if m else "-",
                     {k: v for k, v in s.items() if v}))

    print("\n# BATCH AUDIT — %d games, %d decisions, %d sampled worlds, %d mon-instances"
          % (len(games), ndec, nworld, nworld * 6))
    print("\n%-6s %5s %7s %7s %9s  %s"
          % ("game", "dec", "worlds", "flags", "top-ratio", "non-zero checks"))
    for name, dec, w, f, tr, nz in rows:
        print("%-6s %5d %7d %7d %9s  %s"
              % (name, dec, w, f, tr, nz or "-"))
    if excluded:
        print("\n## EXCLUDED (cross-talk: >1 battle in the log, ungradeable)")
        for g in excluded:
            print("   %s" % os.path.basename(g))
    # Pass 4/5: true-set rate vs Bayes ceiling, share calibration, truth
    # orphans -- batch-pooled (per-game numbers are 8-world noise).
    import subprocess
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(HERE, "audit_setaccuracy.py")]
            + games, capture_output=True, text=True, timeout=3600).stdout
        print("\n" + out.strip())
    except Exception as e:
        errs.append(("setaccuracy", repr(e)))

    print("\n## POOLED TOTALS")
    bad = {k: v for k, v in tot.items() if v}
    if not bad:
        print("   every check zero across the whole batch")
    for k in PASS1 + PASS2:
        if tot[k]:
            games_hit = sum(1 for r in rows if r[5].get(k))
            print("   %-30s %6d   (in %d/%d games)"
                  % (k, tot[k], games_hit, len(games)))
    if errs:
        print("\n## AUDIT ERRORS")
        for n, e in errs[:10]:
            print("   %-6s %s" % (n, e))


if __name__ == "__main__":
    main()
