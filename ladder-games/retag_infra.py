"""Retro-tag infra losses (inactivity/illegal-choice timeouts) in archived games.

Idempotent: re-run any time (e.g. at each reconciliation). Scans every LOSS's
battle log, and when the loss was caused by our own machinery rather than by
play, writes infra="inactivity_timeout" into meta.json — which excludes it from
pi_human training data (critically, from the beat-us x2 weight) and from the
loss register.

Usage:
  retag_infra.py /path/to/archive              # local archive dir
  retag_infra.py --s3 <bucket> <prefix>        # fleet boxes on S3
"""
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile

AWS = "/Users/sallyliu/.awscli-venv/bin/aws"
PATTERN = re.compile(r"lost due to inactivity|Invalid choice.*(?:nomove|No Move)")


def is_infra_loss(log_bytes_path):
    with gzip.open(log_bytes_path, "rt", errors="replace") as f:
        for line in f:
            if PATTERN.search(line):
                return True
    return False


def retag_local(archive_dir):
    changed = 0
    for meta_path in glob.glob(os.path.join(archive_dir, "games", "*", "meta.json")):
        meta = json.load(open(meta_path))
        if meta.get("result") != "L" or meta.get("infra"):
            continue
        log = os.path.join(os.path.dirname(meta_path), "battle.log.gz")
        if os.path.exists(log) and is_infra_loss(log):
            meta["infra"] = "inactivity_timeout"
            json.dump(meta, open(meta_path, "w"), indent=2)
            print("tagged:", meta["battle_tag"], meta.get("opponent"))
            changed += 1
    # rewrite index rows to match
    idx = os.path.join(archive_dir, "index.jsonl")
    if os.path.exists(idx):
        rows = [json.loads(l) for l in open(idx)]
        by_tag = {}
        for meta_path in glob.glob(os.path.join(archive_dir, "games", "*", "meta.json")):
            m = json.load(open(meta_path))
            by_tag[m["battle_tag"]] = m.get("infra")
        for r in rows:
            t = by_tag.get(r.get("battle_tag"))
            if t and not r.get("infra"):
                r["infra"] = t
        open(idx, "w").writelines(json.dumps(r) + "\n" for r in rows)
    return changed


def retag_s3(bucket, prefix):
    changed = 0
    boxes = subprocess.run(
        [AWS, "s3", "ls", f"s3://{bucket}/{prefix}/"],
        capture_output=True, text=True).stdout.split()
    boxes = [b.strip("/") for b in boxes if b.startswith("box-")]
    for box in boxes:
        base = f"s3://{bucket}/{prefix}/{box}/archive/games"
        listing = subprocess.run([AWS, "s3", "ls", base + "/"],
                                 capture_output=True, text=True).stdout
        games = [l.split()[-1].strip("/") for l in listing.splitlines() if "PRE" in l]
        for g in games:
            if not g.endswith("_L"):
                continue
            with tempfile.TemporaryDirectory() as td:
                mp, lp = os.path.join(td, "meta.json"), os.path.join(td, "b.gz")
                if subprocess.run([AWS, "s3", "cp", f"{base}/{g}/meta.json", mp,
                                   "--quiet"]).returncode:
                    continue
                meta = json.load(open(mp))
                if meta.get("infra"):
                    continue
                if subprocess.run([AWS, "s3", "cp", f"{base}/{g}/battle.log.gz", lp,
                                   "--quiet"]).returncode:
                    continue
                if is_infra_loss(lp):
                    meta["infra"] = "inactivity_timeout"
                    json.dump(meta, open(mp, "w"), indent=2)
                    subprocess.run([AWS, "s3", "cp", mp, f"{base}/{g}/meta.json",
                                    "--quiet"])
                    print("tagged:", box, meta["battle_tag"], meta.get("opponent"))
                    changed += 1
    return changed


if __name__ == "__main__":
    if sys.argv[1] == "--s3":
        n = retag_s3(sys.argv[2], sys.argv[3])
    else:
        n = retag_local(sys.argv[1])
    print(f"infra-tagged {n} game(s)")
