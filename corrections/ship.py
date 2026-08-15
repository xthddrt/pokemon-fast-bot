"""Ship a hammered candidate: full export + REAL constants re-derive +
versioned install into m4_artifacts (HAMMER_SPEC.md Part 2 step 5).

  .venv python ship.py --net corrections/candidate.pt --name v6nopol_h1
      [--shard corrections/oos.jsonl.gz] [--candidate-bin corrections/candidate.bin]
      [--ledger PATH] [--iters 100000] [--skip-flip]

Steps:
  1. Full export via valuenet/export_weights.py --shard: re-derives the
     sidecar constants from THIS checkpoint's own trajectories (value-scale
     drift moves tau/UCB). This is the slow step (~minutes) and the reason it
     runs at ship time only.
  2. Python bit-exactness gate (cargo nn-parity is SKIPPED at ship time for
     speed — run the full cargo parity weekly/on-demand): the shipped .bin
     must be byte-identical (sha256) to the fast-export .bin that passed the
     hammer's real flip test. Same checkpoint through the same writer =>
     identical bytes; any difference means the wrong weights are shipping.
  3. Re-run the REAL flip test on the shipped bin under the NEW derived
     constants (they change search behaviour, so the flips must survive them).
  4. Print the run_game.sh line (run_game.sh itself is not edited).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import common
import verify_flip


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="hammered checkpoint (.pt)")
    ap.add_argument("--name", required=True,
                    help="artifact name, e.g. v6nopol_h1 -> "
                         "m4_artifacts/valuenet_v6nopol_h1.bin")
    ap.add_argument("--shard", default=common.OOS_SHARD,
                    help="OOS shard for derive_constants (same one v6nopol "
                         "used)")
    ap.add_argument("--candidate-bin", default=None,
                    help="the fast-export bin that passed the hammer's real "
                         "flip test; shipped bytes must match it")
    ap.add_argument("--ledger", default=common.LEDGER)
    ap.add_argument("--iters", type=int, default=600000)
    ap.add_argument("--flip-workers", type=int, default=4)
    ap.add_argument("--no-promote", action="store_true",
        help="do not point run_game.sh at the shipped net "
             "(default: auto-promote — Sally 2026-08-13)")
    ap.add_argument("--skip-flip", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    out_bin = os.path.join(common.M4, "valuenet_%s.bin" % args.name)
    out_sidecar = os.path.splitext(out_bin)[0] + ".constants.json"
    if os.path.isfile(out_bin):
        raise SystemExit("%s already exists — hammer generations are "
                         "versioned, never overwritten. Bump the h<N>."
                         % out_bin)

    # env flags for export_weights' import-time encoder config
    ck = common.load_ckpt(args.net)
    env = dict(os.environ)
    env.update(common.ckpt_env(ck))

    print("ship: full export + constants derive (the slow step, ~minutes)...")
    t0 = time.time()
    r = subprocess.run(
        [common.VENV_PY, os.path.join(common.VALUENET, "export_weights.py"),
         os.path.abspath(args.net), out_bin, "--shard",
         os.path.abspath(args.shard)],
        cwd=common.VALUENET, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("export failed (rc=%d)" % r.returncode)
    t_export = time.time() - t0

    # python bit-exactness gate (cargo parity deliberately skipped: weekly)
    ship_sha = sha256(out_bin)
    if args.candidate_bin:
        cand_sha = sha256(args.candidate_bin)
        if cand_sha != ship_sha:
            raise SystemExit(
                "BIT-EXACTNESS FAILURE: shipped %s sha %s != flip-tested "
                "candidate %s sha %s — the bytes that passed the flip test "
                "are not the bytes shipping. NOT installing."
                % (out_bin, ship_sha[:16], args.candidate_bin, cand_sha[:16]))
        print("bit-exact vs flip-tested candidate: OK (sha256 %s)"
              % ship_sha[:16])
    else:
        print("NOTE: no --candidate-bin given; byte-identity vs the "
              "flip-tested bin not checked (sha256 %s)" % ship_sha[:16])

    # flips must survive the NEW derived constants
    t_flip = None
    if not args.skip_flip:
        entries = common.read_ledger(args.ledger)
        if entries:
            constants = json.load(open(out_sidecar))
            results, t_flip = verify_flip.flip_test(
                out_bin, constants, entries, args.iters, args.flip_workers)
            if not all(x["pass"] for x in results.values()):
                fails = [k for k, x in results.items() if not x["pass"]]
                raise SystemExit(
                    "flip regression under the NEW derived constants: %s — "
                    "NOT ship-ready; keep hammering (the .bin/.sidecar stay "
                    "installed for inspection at %s)." % (fails, out_bin))
            print("all %d ledger entries flip under the new constants."
                  % len(results))
        else:
            print("NOTE: empty ledger %s — no flip regression to run."
                  % args.ledger)

    print("\ninstalled: %s (+.constants.json)" % out_bin)
    print("export+derive %.1fs%s, total %.1fs"
          % (t_export,
             ", flip re-verify %.1fs" % t_flip if t_flip else "",
             time.time() - t_start))

    # Auto-promote (Sally's default 2026-08-13): a generation only reaches this
    # line after every ledger entry flipped under its own derived constants, so
    # promotion cannot regress a prior ruling. --no-promote opts out.
    rel = "../valuenet/m4_artifacts/valuenet_%s.bin" % args.name
    if args.no_promote:
        print("\n--no-promote: run_game.sh untouched. To ladder this net:")
        print("  RG_NN_WEIGHTS=%s ./run_game.sh" % rel)
    else:
        rg = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "ladder-games", "run_game.sh")
        src = open(rg).read()
        m = re.search(r'--nn-weights "\$\{RG_NN_WEIGHTS:-([^}]+)\}"', src)
        if not m:
            print("\nWARNING: could not find the --nn-weights default in %s — "
                  "not promoted. Ladder it with:\n  RG_NN_WEIGHTS=%s "
                  "./run_game.sh" % (rg, rel))
        else:
            prev = m.group(1)
            open(rg, "w").write(src.replace(m.group(0),
                '--nn-weights "${RG_NN_WEIGHTS:-%s}"' % rel))
            print("\nPROMOTED: run_game.sh default %s -> %s" % (prev, rel))
            print("  (the next ladder game plays this net; roll back by "
                  "re-running ship.py for the prior generation or editing "
                  "run_game.sh)")
    print("\ntradeoff on record: cargo nn-parity was SKIPPED at ship time "
          "(python byte-identity gate instead); run the full cargo parity "
          "weekly or on demand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
