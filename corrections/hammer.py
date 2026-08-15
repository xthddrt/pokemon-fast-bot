"""The trainer: fine-tune the current target net until every ledger ruling
flips under the REAL production search (HAMMER_SPEC.md Part 2,
corrections/hammer.py).

  .venv python hammer.py --net <ckpt.pt> [--ledger PATH] [--workdir DIR]
      [--constants sidecar.json] [--lr 1e-4] [--steps 200] [--cap 20000]
      [--iters 100000] [--flip-workers 4] [--seed 7] [--tag candidate]

Loop: batch = ALL ledger entries' ruled-move successors (cumulative — every
past ruling rides along, nothing un-bakes), BCE toward value 1.0, LR 1e-4,
25-step rounds (default; --steps to change). After each round the FAST PRECHECK (net-level ranking, ~ms)
runs; when it flips for every entry, the REAL flip test runs (fast .bin
export + production MCTS at a fixed budget + the real selection pipeline).
Ship-ready when the real test passes for ALL entries. 20k-step cap:
report, don't ship.

Device: MPS if torch.backends.mps.is_available() else CPU with 4 torch
threads. Combined with the flip pool (4 workers x 1 thread, never concurrent
with training) the tool stays within the <=4-6 core budget.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import common
import verify_flip


def find_constants(net_path, override):
    if override:
        return override
    cands = [os.path.splitext(net_path)[0] + ".constants.json"]
    stem = os.path.basename(net_path)
    if stem.startswith("net_"):
        cands.append(os.path.join(
            common.M4, "valuenet_%s.constants.json"
            % os.path.splitext(stem[len("net_"):])[0]))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit("no constants sidecar found for %s (tried %s); pass "
                     "--constants — flip tests must run this net's search "
                     "constants" % (net_path, cands))


def fast_export(ckpt_path, out_bin, constants_path):
    t0 = time.time()
    r = subprocess.run(
        [common.VENV_PY, os.path.join(common.CORR, "_fast_export.py"),
         ckpt_path, out_bin, constants_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("fast export failed:\n%s\n%s" % (r.stdout, r.stderr))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="target checkpoint (.pt)")
    ap.add_argument("--ledger", default=common.LEDGER)
    ap.add_argument("--workdir", default=common.CORR)
    ap.add_argument("--constants", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=25,
                    help="steps per round; the precheck between rounds costs "
                         "~18ms, so a small round stops within --steps of the "
                         "minimum needed and limits overfit (Sally 2026-08-13)")
    ap.add_argument("--cap", type=int, default=20000)
    # 600k ~= the production 4500ms budget at the measured ~130-160k iters/s.
    # At 100k the UNhammered v6nopol already picks the yraf-T16 ruled move, so
    # a 100k flip test can pass vacuously; 600k reproduces the archived choice.
    ap.add_argument("--iters", type=int, default=600000,
                    help="real flip test MCTS iterations per world")
    ap.add_argument("--flip-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="candidate")
    args = ap.parse_args()

    t_start = time.time()
    entries = common.read_ledger(args.ledger)
    if not entries:
        raise SystemExit("empty ledger: %s" % args.ledger)
    constants_path = find_constants(args.net, args.constants)
    constants = json.load(open(constants_path))
    os.makedirs(args.workdir, exist_ok=True)

    # --- net + device (env flags BEFORE valuenet imports) ------------------
    ck = common.load_ckpt(args.net)
    common.apply_env(common.ckpt_env(ck))
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        torch.set_num_threads(4)
    net = common.build_net(ck).to(device)
    vocab = common.frozen_vocab()
    print("hammer: net=%s device=%s entries=%d constants=%s"
          % (os.path.basename(args.net), device, len(entries),
             os.path.basename(constants_path)))

    # --- one-time successor fan + encode (precheck cache + train batch) ----
    t0 = time.time()
    caches = verify_flip.build_precheck_cache(entries, vocab, device)
    train_parts = [caches[e["id"]]["options"][e["ruled_move"]][0]
                   for e in entries]
    batch = {k: torch.cat([p[k] for p in train_parts]) for k in train_parts[0]}
    n_train = next(iter(batch.values())).shape[0]
    t_prepare = time.time() - t0
    print("prepared %d training successors (+%d option batches) in %.1fs"
          % (n_train,
             sum(len(c["options"]) for c in caches.values()), t_prepare))

    # --- baseline values (the from->to report) ------------------------------
    pc0 = verify_flip.precheck(net, caches)
    base_vals = {eid: dict(r["values"]) for eid, r in pc0.items()}
    for eid, r in pc0.items():
        top = sorted(r["values"].items(), key=lambda kv: -kv[1])[:4]
        print("baseline %s: ruled %r=%.4f | top: %s"
              % (eid, caches[eid]["ruled"], r["values"][caches[eid]["ruled"]],
                 [(o, round(v, 4)) for o, v in top]))

    # --- fine-tune loop ------------------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ones = torch.ones(n_train, device=device)
    lossf = torch.nn.BCEWithLogitsLoss()
    steps_done = 0
    steps_to_precheck_flip = None
    round_times, precheck_times = [], []
    flip_attempts = []
    shipped = False

    while steps_done < args.cap:
        t0 = time.time()
        net.train()
        for _ in range(args.steps):
            opt.zero_grad()
            out = net(batch)
            logits = out[0] if isinstance(out, tuple) else out
            loss = lossf(logits, ones)
            loss.backward()
            opt.step()
        steps_done += args.steps
        net.eval()
        round_times.append(time.time() - t0)

        t0 = time.time()
        pc = verify_flip.precheck(net, caches)
        precheck_times.append(time.time() - t0)
        all_flip = all(r["flip"] for r in pc.values())
        print("round %d (steps %d): loss=%.4f precheck_flips=%d/%d [%.2fs+"
              "%.3fs]" % (len(round_times), steps_done, loss.item(),
                          sum(r["flip"] for r in pc.values()), len(pc),
                          round_times[-1], precheck_times[-1]))
        if not all_flip:
            continue
        if steps_to_precheck_flip is None:
            steps_to_precheck_flip = steps_done

        # --- REAL flip test on a fast-exported bin of this exact candidate --
        cand_pt = os.path.join(args.workdir, "%s.pt" % args.tag)
        cand_bin = os.path.join(args.workdir, "%s.bin" % args.tag)
        cand_ck = dict(ck)
        cand_ck["model"] = {k: v.detach().cpu()
                            for k, v in net.state_dict().items()}
        cand_ck["hammer"] = {"ledger": args.ledger, "steps": steps_done,
                             "lr": args.lr, "base": os.path.abspath(args.net),
                             "ts": common.now_ts()}
        torch.save(cand_ck, cand_pt)
        t_export = fast_export(cand_pt, cand_bin, constants_path)
        results, t_flip = verify_flip.flip_test(
            cand_bin, constants, entries, args.iters, args.flip_workers,
            args.seed)
        flip_attempts.append({"steps": steps_done, "export_s": t_export,
                              "flip_s": t_flip,
                              "results": {k: {kk: r[kk] for kk in
                                              ("pass", "choice")}
                                          for k, r in results.items()}})
        if all(r["pass"] for r in results.values()):
            shipped = True
            break
        print("real flip test: %d/%d — continuing hammering"
              % (sum(r["pass"] for r in results.values()), len(results)))

    # --- report --------------------------------------------------------------
    pc_final = verify_flip.precheck(net, caches)
    report = {
        "ship_ready": shipped,
        "steps": steps_done,
        "steps_to_precheck_flip": steps_to_precheck_flip,
        "cap": args.cap,
        "entries": {e["id"]: {
            "ruled": e["ruled_move"],
            "value_from": base_vals[e["id"]][e["ruled_move"]],
            "value_to": pc_final[e["id"]]["values"][e["ruled_move"]],
            "values_final": pc_final[e["id"]]["values"],
        } for e in entries},
        "timings_s": {
            "prepare": round(t_prepare, 2),
            "per_round_mean": round(sum(round_times) / len(round_times), 2)
            if round_times else None,
            "precheck_mean": round(sum(precheck_times) /
                                   len(precheck_times), 3)
            if precheck_times else None,
            "flip_attempts": flip_attempts,
            "total": round(time.time() - t_start, 1),
        },
        "candidate_pt": os.path.join(args.workdir, "%s.pt" % args.tag),
        "candidate_bin": os.path.join(args.workdir, "%s.bin" % args.tag),
        "device": device,
    }
    rpt_path = os.path.join(args.workdir, "hammer_report.json")
    json.dump(report, open(rpt_path, "w"), indent=1)
    if shipped:
        print("SHIP-READY after %d steps (%.1fs total). Candidate: %s"
              % (steps_done, time.time() - t_start, report["candidate_bin"]))
        print("next: .venv python ship.py --net %s --name <base>_h<N> "
              "--candidate-bin %s"
              % (report["candidate_pt"], report["candidate_bin"]))
    else:
        print("CAP HIT (%d steps) without a full real-test flip — NOT "
              "shipping. Report: %s" % (steps_done, rpt_path))
    print("report: %s" % rpt_path)
    return 0 if shipped else 2


if __name__ == "__main__":
    sys.exit(main())
