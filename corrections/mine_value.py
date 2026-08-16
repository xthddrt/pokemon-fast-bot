"""ACTIVE MINING — backward endgame-first scan (ACTIVE_MINING.md, Sally's
2026-08-15 test protocol).

Per round:
  1. G full-info self-play games: champion (--audit net) vs itself, one
     decoupled search per turn (--ms), argmax visits both sides, every
     decision state recorded, outcome from alive counts.
  2. The audited net's RAW eval of every recorded state (leaf_prof logits).
  3. Per game, scan decisions BACKWARD from the last turn. At each state run
     --screen-n playouts with the LABEL PLAYER (v8b_s1 @ 2000 iters — pinned:
     the player that defined ruling v1's target; s1 is frozen so targets stay
     comparable across ledger generations). STOP at the first state where
     |eval − p̂| >= 0.15 and z >= 2 (Agresti-Coull SE — plain SE degenerates
     at p̂ ∈ {0,1}).
  4. Confirm that one state with --confirm-n FRESH-seeded playouts (winner's
     curse: the screening sample that flagged a state overstates its gap).
     Confirmed if |eval − p̂_c| >= 0.10 and z_c >= 3.
  5. Print the assessment table + write candidates.json with ready ledger
     rows. NO HAMMERING here — Sally assesses, then the rows are appended to
     value_ledger.jsonl and hammer_value.py runs on her command.

Usage (orchestrator; spawns its own game/scan subprocesses for env isolation):
    foul-play/.venv/bin/python corrections/mine_value.py \
        [--games 5] [--ms 4500] [--tag mine1]
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "foul-play", ".venv", "bin", "python")
LEAF_PROF = os.path.join(ROOT, "poke-engine", "target", "release", "leaf_prof")
CORPUS = "/Users/sallyliu/pokemon-ai/synthetic-corpus-holdout10"
AUDIT_BIN = os.path.join(ROOT, "valuenet/nets_v8b/v8b_h1.bin")
LABEL_BIN = os.path.join(ROOT, "valuenet/nets_v8b/v8b_s1.bin")
MAX_STEPS = 300
LABEL_ITERS = 2000
MAX_CONCURRENT = 4  # half the cores, always

def net_env(bin_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PE_")}
    env["PE_NN_WEIGHTS"] = bin_path
    side = os.path.splitext(bin_path)[0] + ".constants.json"
    for k, v in json.load(open(side)).items():
        if k.startswith("PE_"):
            env[k] = str(v)
    return env

def arm_to_move(a):
    a = a.lower()
    if a in ("no move", "nomove"):
        return "none"
    return a[7:] if a.startswith("switch ") else a

def ac_stats(outs):
    """mean, Agresti-Coull SE (well-defined at p̂ ∈ {0,1}; ties count 0.5)."""
    n = len(outs)
    x = sum(outs)
    pt = (x + 2.0) / (n + 4.0)
    return x / n, math.sqrt(pt * (1.0 - pt) / (n + 4.0))

def one_playout(state_str, base_seed):
    from poke_engine import State, generate_instructions, monte_carlo_tree_search
    rng = random.Random(base_seed)
    state = State.from_string(state_str)
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        res = monte_carlo_tree_search(state, 0, LABEL_ITERS, 1,
                                      (base_seed * 7919 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        p1 = max(s1, key=lambda m: m.visits).move_choice
        p2 = max(s2, key=lambda m: m.visits).move_choice
        try:
            branches = [b for b in generate_instructions(
                state, arm_to_move(p1), arm_to_move(p2)) if b.percentage > 0]
        except Exception:
            break
        if not branches:
            break
        state = state.apply_instructions(
            rng.choices(branches, weights=[b.percentage for b in branches])[0])
    a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
    return 1.0 if (a1 > 0 and a2 == 0) else 0.0 if (a2 > 0 and a1 == 0) else 0.5

def describe(state_str):
    """Light context for Sally's assessment; best-effort."""
    try:
        from poke_engine import State
        st = State.from_string(state_str)
        out = []
        for side in (st.side_one, st.side_two):
            alive = sum(p.hp > 0 for p in side.pokemon)
            act = getattr(side, "active_index", 0)
            try:
                mon = side.pokemon[int(act)]
                out.append(f"{getattr(mon, 'id', '?')} {mon.hp}/{getattr(mon, 'maxhp', '?')}hp ({alive} alive)")
            except Exception:
                out.append(f"({alive} alive)")
        return " vs ".join(str(x) for x in out)
    except Exception:
        return ""

# ---------------------------------------------------------------- subcommands

def cmd_game(a):
    """One self-play game. Env (audited net) set by the orchestrator."""
    sys.path.insert(0, os.path.join(ROOT, "valuenet", "sprt"))
    import run_duels as rd
    from poke_engine import State, generate_instructions, monte_carlo_tree_search
    fa, fb = a.teams.split(",")
    s = rd.opening_state(fa, "p1", fb, "p2")
    rng = random.Random(a.seed)
    state = State.from_string(s)
    recs = []
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        ss = state.to_string()
        res = monte_carlo_tree_search(state, a.ms, 0, 1,
                                      (a.seed * 104729 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        recs.append({"t": step, "s": ss})
        p1 = max(s1, key=lambda m: m.visits).move_choice
        p2 = max(s2, key=lambda m: m.visits).move_choice
        try:
            branches = [b for b in generate_instructions(
                state, arm_to_move(p1), arm_to_move(p2)) if b.percentage > 0]
        except Exception:
            break
        if not branches:
            break
        state = state.apply_instructions(
            rng.choices(branches, weights=[b.percentage for b in branches])[0])
    a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
    a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
    outcome = 1.0 if (a1 > 0 and a2 == 0) else 0.0 if (a2 > 0 and a1 == 0) else 0.5
    json.dump({"seed": a.seed, "outcome": outcome, "teams": a.teams,
               "states": recs}, open(a.out, "w"))
    print(f"game {a.seed}: {len(recs)} decisions, outcome {outcome}", flush=True)

def cmd_scan(a):
    """Backward scan one game. Env (label player = s1) set by orchestrator."""
    g = json.load(open(a.game))
    res = {"seed": g["seed"], "outcome": g["outcome"], "scanned": [],
           "candidate": None, "near_misses": []}
    for rec in reversed(g["states"][-a.max_scan:]):
        base = g["seed"] * 1_000_003 + rec["t"] * 8191
        outs = [one_playout(rec["s"], (base + j * 7919) & 0x7FFFFFFF)
                for j in range(a.screen_n)]
        p, se = ac_stats(outs)
        gap = abs(rec["e"] - p)
        z = gap / se
        res["scanned"].append({"t": rec["t"], "e": round(rec["e"], 3),
                               "p10": round(p, 3), "z": round(z, 2)})
        print(f"game {g['seed']} t{rec['t']}: e={rec['e']:.3f} "
              f"p̂{a.screen_n}={p:.3f} z={z:.1f}", flush=True)
        if gap >= 0.15 and z >= 2.0:
            outs_c = [one_playout(rec["s"], (base + 500_000 + j * 104729) & 0x7FFFFFFF)
                      for j in range(a.confirm_n)]
            pc, sec = ac_stats(outs_c)
            gc = abs(rec["e"] - pc)
            zc = gc / sec
            k = {
                "t": rec["t"], "e": round(rec["e"], 4),
                "p_screen": round(p, 4), "p_confirm": round(pc, 4),
                "se_confirm": round(sec, 4), "z_confirm": round(zc, 2),
                "confirmed": bool(gc >= 0.10 and zc >= a.confirm_z),
                "wins": sum(1 for o in outs_c if o == 1.0),
                "losses": sum(1 for o in outs_c if o == 0.0),
                "ties": sum(1 for o in outs_c if o == 0.5),
                "context": describe(rec["s"]), "s": rec["s"],
            }
            if k["confirmed"]:
                res["candidate"] = k
                break
            # screen hit that failed confirm = probably noise: record it for
            # Sally's soften-to-2.5σ/2σ judgment call, keep scanning backward
            res["near_misses"].append(k)
    json.dump(res, open(a.out, "w"))

def wait_slots(procs, limit):
    while sum(1 for p in procs if p.poll() is None) >= limit:
        time.sleep(2)

def cmd_run(a):
    work = os.path.join(HERE, "_mine_work", a.tag)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(CORPUS, "*.teams.json")))
    assert len(files) >= 2, f"need teams.json files in {CORPUS}"
    # ring pairing: game i = file_i's p1 vs file_{i+1}'s p2 — no team reused
    genv = net_env(a.audit)
    procs = []
    for i in range(a.games):
        fa, fb = files[i % len(files)], files[(i + 1) % len(files)]
        wait_slots(procs, MAX_CONCURRENT)
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "game",
             "--out", os.path.join(work, f"g{i}.json"),
             "--seed", str(101 + i), "--ms", str(a.ms),
             "--teams", f"{fa},{fb}"],
            env=genv, stdout=sys.stdout, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"[{time.time()-t0:.0f}s] games done", flush=True)

    # audited net's raw eval of every state, one leaf_prof pass
    states_file = os.path.join(work, "states.txt")
    keys = []
    with open(states_file, "w") as f:
        for i in range(a.games):
            g = json.load(open(os.path.join(work, f"g{i}.json")))
            for rec in g["states"]:
                keys.append((i, rec["t"]))
                f.write(rec["s"] + "\n")
    out = subprocess.run([LEAF_PROF, "logits", states_file],
                         env=net_env(a.audit), capture_output=True, text=True,
                         check=True).stdout
    logits = [float(l.split("\t")[1]) for l in out.splitlines() if "\t" in l]
    assert len(logits) == len(keys)
    evals = {k: 1 / (1 + math.exp(-x)) for k, x in zip(keys, logits)}
    for i in range(a.games):
        path = os.path.join(work, f"g{i}.json")
        g = json.load(open(path))
        for rec in g["states"]:
            rec["e"] = evals[(i, rec["t"])]
        json.dump(g, open(path, "w"))
    print(f"[{time.time()-t0:.0f}s] evals injected ({len(keys)} states)", flush=True)

    senv = net_env(a.label)  # label player pinned to s1
    procs = []
    for i in range(a.games):
        wait_slots(procs, MAX_CONCURRENT)
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "scan",
             "--game", os.path.join(work, f"g{i}.json"),
             "--out", os.path.join(work, f"cand{i}.json"),
             "--screen-n", str(a.screen_n), "--confirm-n", str(a.confirm_n),
             "--max-scan", str(a.max_scan), "--confirm-z", str(a.confirm_z)],
            env=senv, stdout=sys.stdout, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()

    # assessment table + ready ledger rows
    cands = [json.load(open(os.path.join(work, f"cand{i}.json")))
             for i in range(a.games)]
    rows = []
    print("\n=== MINING CANDIDATES (%s) ===" % a.tag, flush=True)
    for c in cands:
        for k in c.get("near_misses", []):
            print(f"game {c['seed']} NEAR-MISS turn {k['t']}: eval {k['e']:.3f} "
                  f"vs truth {k['p_confirm']:.3f}±{k['se_confirm']:.3f} "
                  f"(z={k['z_confirm']}) — screen hit, confirm below threshold",
                  flush=True)
        if not c["candidate"]:
            print(f"game {c['seed']} (outcome {c['outcome']}): no confirmed "
                  f"eval error in last {a.max_scan} decisions", flush=True)
            continue
        k = c["candidate"]
        tgt = k["p_confirm"]
        print(f"game {c['seed']} (outcome {c['outcome']}) turn {k['t']}: "
              f"eval {k['e']:.3f} vs truth {tgt:.3f}±{k['se_confirm']:.3f} "
              f"(z={k['z_confirm']}, {k['wins']}W-{k['ties']}T-{k['losses']}L) "
              f"{'CONFIRMED' if k['confirmed'] else 'not confirmed'} | {k['context']}",
              flush=True)
        if k["confirmed"]:
            rows.append({
                "id": f"{a.tag}-g{c['seed']}-t{k['t']}",
                "game": f"selfplay-{a.tag}-{c['seed']}", "decision": k["t"],
                "target": tgt,
                "band": [round(max(0.0, tgt - k["se_confirm"]), 4),
                         round(min(1.0, tgt + k["se_confirm"]), 4)],
                "states": [k["s"]], "n_playouts": a.confirm_n,
                "note": f"mined: eval {k['e']:.3f} vs playout {tgt:.3f}, "
                        f"z={k['z_confirm']} | {k['context']}",
                "ts": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
            })
    json.dump(rows, open(os.path.join(work, "ledger_rows.json"), "w"), indent=1)
    print(f"\n[{time.time()-t0:.0f}s] {len(rows)} confirmed ruling(s) ready in "
          f"{os.path.join(work, 'ledger_rows.json')} — NOT hammered; awaiting "
          f"assessment.", flush=True)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("game", "scan", "run"):
        sys.argv.insert(1, "run")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("game")
    g.add_argument("--out"); g.add_argument("--seed", type=int)
    g.add_argument("--ms", type=int); g.add_argument("--teams")
    s = sub.add_parser("scan")
    s.add_argument("--game"); s.add_argument("--out")
    s.add_argument("--screen-n", type=int, default=10)
    s.add_argument("--confirm-n", type=int, default=30)
    s.add_argument("--max-scan", type=int, default=1000)
    s.add_argument("--confirm-z", type=float, default=3.0)
    r = sub.add_parser("run")
    r.add_argument("--games", type=int, default=5)
    r.add_argument("--ms", type=int, default=4500)
    r.add_argument("--tag", default="mine1")
    r.add_argument("--screen-n", type=int, default=10)
    r.add_argument("--confirm-n", type=int, default=30)
    r.add_argument("--max-scan", type=int, default=1000)
    r.add_argument("--confirm-z", type=float, default=3.0)
    r.add_argument("--audit", default=AUDIT_BIN)
    r.add_argument("--label", default=LABEL_BIN)
    a = ap.parse_args()
    {"game": cmd_game, "scan": cmd_scan, "run": cmd_run}[a.cmd or "run"](a)

if __name__ == "__main__":
    main()
