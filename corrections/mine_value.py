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
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "foul-play", ".venv", "bin", "python")
LEAF_PROF = os.path.join(ROOT, "poke-engine", "target", "release", "leaf_prof")
CORPUS = "/Users/sallyliu/pokemon-ai/synthetic-corpus-holdout10"
AUDIT_BIN = os.path.join(ROOT, "valuenet/nets_v8b/v8b_h2.bin")
LABEL_BIN = os.path.join(ROOT, "valuenet/nets_v8b/v8b_s1.bin")
# Mining plays stalls out to PS-like resolution (PP drain -> Struggle);
# 1000 mirrors Showdown's own turn-limit backstop. The corpus labeler keeps
# its 200-step cap for farm-scale economics — mining's 30 playouts/spot can
# afford the tail. (Sally, 2026-08-15.) MINE_MAX_STEPS overrides for exact
# replication of runs made under the old 300 cap.
MAX_STEPS = int(os.environ.get("MINE_MAX_STEPS", "1000"))
LABEL_ITERS = 2000
MAX_CONCURRENT = int(os.environ.get("MINE_CONCURRENT", "4"))  # half the Mac's cores; cloud boxes set MINE_CONCURRENT to their vCPUs

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

INS_RE = re.compile(r"^(Damage|Heal) Side(One|Two)\S*: (-?\d+)")

def double_ko_winner(pre_state, branch):
    """PS gen9 rule: the side whose mon faints LAST wins a double KO."""
    hp = {s: pre_state.side_one.pokemon[int(getattr(pre_state.side_one, "active_index", 0))].hp
          if s == "One" else
          pre_state.side_two.pokemon[int(getattr(pre_state.side_two, "active_index", 0))].hp
          for s in ("One", "Two")}
    seq = {}
    for i, ins in enumerate(branch.instruction_list):
        m = INS_RE.match(str(ins))
        if not m:
            continue
        kind, side, amt = m.groups()
        hp[side] += int(amt) if kind == "Heal" else -int(amt)
        if hp[side] <= 0 and side not in seq:
            seq[side] = i
    if len(seq) < 2:
        return None
    return 1.0 if seq["One"] > seq["Two"] else 0.0

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
        pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        nxt = state.apply_instructions(pick)
        if (not any(p.hp > 0 for p in nxt.side_one.pokemon)
                and not any(p.hp > 0 for p in nxt.side_two.pokemon)):
            w = double_ko_winner(state, pick)
            return w if w is not None else 0.5
        state = nxt
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

def _playout_worker(args):
    state_str, seed = args
    return one_playout(state_str, seed)

def cmd_scan(a):
    """Backward scan one game. Env (label player = s1) set by orchestrator.

    Playouts fan out over a small PROCESS pool (--pool; default sized so
    orchestrator-level scan concurrency x pool = MINE_CONCURRENT). Separate
    processes double as the playout-repro-anomaly mitigation: screen and
    confirm draws span independent process instances, like block confirms.
    Screens are STAGED: 4 playouts first, the remaining 6 only when the
    first 4 disagree with the eval (gap >= 0.10) — most states pass at 4.
    Flag decisions always use the full sample.
    """
    import concurrent.futures as cf
    pool = cf.ProcessPoolExecutor(max_workers=a.pool) if a.pool > 1 else None

    def run_playouts(state_str, seeds):
        if pool is None:
            return [one_playout(state_str, s) for s in seeds]
        return list(pool.map(_playout_worker, [(state_str, s) for s in seeds]))

    g = json.load(open(a.game))
    res = {"seed": g["seed"], "outcome": g["outcome"], "scanned": [],
           "candidate": None, "near_misses": []}
    for rec in reversed(g["states"][-a.max_scan:]):
        base = g["seed"] * 1_000_003 + rec["t"] * 8191
        outs = run_playouts(rec["s"], [(base + j * 7919) & 0x7FFFFFFF
                                       for j in range(a.screen_n)])
        p_stage1 = sum(outs) / len(outs)
        if abs(rec["e"] - p_stage1) >= 0.15 and a.confirm_n > a.screen_n:
            outs += run_playouts(rec["s"], [(base + j * 7919) & 0x7FFFFFFF
                                            for j in range(a.screen_n, a.confirm_n)])
        p, se = ac_stats(outs)
        gap = abs(rec["e"] - p)
        z = gap / se
        res["scanned"].append({"t": rec["t"], "e": round(rec["e"], 3),
                               "p10": round(p, 3), "z": round(z, 2)})
        print(f"game {g['seed']} t{rec['t']}: e={rec['e']:.3f} "
              f"p̂{len(outs)}={p:.3f} z={z:.1f}", flush=True)
        if len(outs) > a.screen_n and gap >= 0.10 and z >= 2.0:
            k = {
                "t": rec["t"], "e": round(rec["e"], 4),
                "p_screen": round(p_stage1, 4), "p_confirm": round(p, 4),
                "se_confirm": round(se, 4), "z_confirm": round(z, 2),
                "confirmed": bool(z >= a.confirm_z),
                "wins": sum(1 for o in outs if o == 1.0),
                "losses": sum(1 for o in outs if o == 0.0),
                "ties": sum(1 for o in outs if o == 0.5),
                "context": describe(rec["s"]), "s": rec["s"],
            }
            if k["confirmed"]:
                res["candidate"] = k
                break
            # screen hit that failed confirm = probably noise: record it for
            # Sally's soften-to-2.5σ/2σ judgment call, keep scanning backward
            res["near_misses"].append(k)
    json.dump(res, open(a.out, "w"))

def cmd_block(a):
    """One independent seed-block of playouts for a candidate (env = label)."""
    g = json.load(open(a.game))
    rec = next(r for r in g["states"] if r["t"] == a.t)
    base = g["seed"] * 1_000_003 + a.t * 8191 + 900_000 + a.block * 1013
    outs = [one_playout(rec["s"], (base + j * 104729) & 0x7FFFFFFF)
            for j in range(a.n)]
    json.dump({"t": a.t, "block": a.block, "e": rec["e"], "outs": outs},
              open(a.out, "w"))

def wait_slots(procs, limit):
    while sum(1 for p in procs if p.poll() is None) >= limit:
        time.sleep(2)

def fresh_teams_file(work, i, seed):
    """Two brand-new teams from the PS-exact generator port, one file per
    game (Sally 2026-08-15: every mining round plays never-seen teams)."""
    if os.path.join(ROOT, "foul-play") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "foul-play"))
    from fp.search import ps_teams
    path = os.path.join(work, f"teams_g{i}.json")
    teams = {}
    for k, salt in (("p1", 11), ("p2", 12)):
        ps_teams.seed(seed * 977 + salt)
        teams[k] = {"team": ps_teams.random_team()}
    json.dump({"teams": teams}, open(path, "w"))
    return path

def cmd_run(a):
    work = os.path.join(HERE, "_mine_work", a.tag)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    genv = net_env(a.audit)
    procs = []
    for i in range(a.games):
        if a.teams_source == "ps":
            fa = fb = fresh_teams_file(work, i, a.seed_base + i)
        else:
            files = sorted(glob.glob(os.path.join(CORPUS, "*.teams.json")))
            n = len(files)
            off = 1 + (i // n) % (n - 1)
            fa, fb = files[i % n], files[(i + off) % n]
        wait_slots(procs, MAX_CONCURRENT)
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "game",
             "--out", os.path.join(work, f"g{i}.json"),
             "--seed", str(a.seed_base + i), "--ms", str(a.ms),
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
    # scans each own a playout pool; concurrency x pool = MINE_CONCURRENT
    scan_pool = max(1, MAX_CONCURRENT // 4)
    scan_conc = max(1, MAX_CONCURRENT // scan_pool)
    procs = []
    for i in range(a.games):
        wait_slots(procs, scan_conc)
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "scan",
             "--game", os.path.join(work, f"g{i}.json"),
             "--out", os.path.join(work, f"cand{i}.json"),
             "--screen-n", str(a.screen_n), "--confirm-n", str(a.confirm_n),
             "--max-scan", str(a.max_scan), "--confirm-z", str(a.confirm_z),
             "--pool", str(scan_pool)],
            env=senv, stdout=sys.stdout, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()

    # block re-measure every scan candidate: 6 fresh processes x 5 playouts,
    # pooled; the scan's own single-process confirm is provisional only
    # (playout-repro-anomaly). Verdict needs gap >= 0.10 and BOTH z >= 2.5.
    cands = [json.load(open(os.path.join(work, f"cand{i}.json")))
             for i in range(a.games)]
    procs = []
    for i, c in enumerate(cands):
        if not c["candidate"]:
            continue
        for b in range(6):
            wait_slots(procs, MAX_CONCURRENT)
            procs.append(subprocess.Popen(
                [PY, os.path.abspath(__file__), "block",
                 "--game", os.path.join(work, f"g{i}.json"),
                 "--t", str(c["candidate"]["t"]), "--block", str(b),
                 "--n", "5", "--out", os.path.join(work, f"blk{i}_{b}.json")],
                env=senv, stdout=sys.stdout, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"[{time.time()-t0:.0f}s] block re-measures done", flush=True)

    rows = []
    print("\n=== MINING CANDIDATES (%s) — block-remeasured ===" % a.tag, flush=True)
    for i, c in enumerate(cands):
        for k in c.get("near_misses", []):
            print(f"game {c['seed']} NEAR-MISS t{k['t']}: eval {k['e']:.3f} vs "
                  f"scan-confirm {k['p_confirm']:.3f} (z={k['z_confirm']})", flush=True)
        if not c["candidate"]:
            print(f"game {c['seed']} (outcome {c['outcome']}): no confirmed "
                  f"eval error", flush=True)
            continue
        k = c["candidate"]
        blocks = [json.load(open(os.path.join(work, f"blk{i}_{b}.json")))
                  for b in range(6)]
        outs = [o for b in blocks for o in b["outs"]]
        p, se = ac_stats(outs)
        e = k["e"]
        zp = abs(e - p) / se
        bm = [sum(b["outs"]) / len(b["outs"]) for b in blocks]
        bmean = sum(bm) / len(bm)
        bsd = (sum((m - bmean) ** 2 for m in bm) / (len(bm) - 1)) ** 0.5
        zb = abs(e - bmean) / (bsd / len(bm) ** 0.5) if bsd > 0 else float("inf")
        ok = abs(e - p) >= 0.10 and min(zp, zb) >= 2.5
        w = sum(1 for o in outs if o == 1.0)
        t_ = sum(1 for o in outs if o == 0.5)
        print(f"game {c['seed']} (outcome {c['outcome']}) turn {k['t']}: "
              f"eval {e:.3f} vs truth {p:.3f}±{se:.3f} "
              f"(z_pooled={zp:.2f} z_block={zb:.2f}, {w}W-{t_}T-{len(outs)-w-t_}L) "
              f"{'CONFIRMED' if ok else 'NOT confirmed'} | blocks={[round(m,2) for m in bm]} "
              f"| {k['context']}", flush=True)
        if ok:
            rows.append({
                "id": f"{a.tag}-g{c['seed']}-t{k['t']}",
                "game": f"selfplay-{a.tag}-{c['seed']}", "decision": k["t"],
                "target": round(p, 4),
                "band": [round(max(0.0, p - se), 4), round(min(1.0, p + se), 4)],
                "states": [k["s"]], "n_playouts": len(outs),
                "note": f"block-remeasured 6x5 PS-scored: eval {e:.3f} vs "
                        f"truth {p:.3f} zp={zp:.1f} zb={zb:.1f} | {k['context']}",
                "ts": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
            })
    json.dump(rows, open(os.path.join(work, "ledger_rows.json"), "w"), indent=1)
    print(f"\n[{time.time()-t0:.0f}s] {len(rows)} confirmed ruling(s) staged in "
          f"{os.path.join(work, 'ledger_rows.json')} — NOT hammered; awaiting "
          f"assessment.", flush=True)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("game", "scan", "run", "block"):
        sys.argv.insert(1, "run")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("game")
    g.add_argument("--out"); g.add_argument("--seed", type=int)
    g.add_argument("--ms", type=int); g.add_argument("--teams")
    s = sub.add_parser("scan")
    s.add_argument("--game"); s.add_argument("--out")
    s.add_argument("--screen-n", type=int, default=8)
    s.add_argument("--confirm-n", type=int, default=30)
    s.add_argument("--max-scan", type=int, default=1000)
    s.add_argument("--confirm-z", type=float, default=3.0)
    s.add_argument("--pool", type=int, default=max(1, MAX_CONCURRENT // 4))
    b = sub.add_parser("block")
    b.add_argument("--game"); b.add_argument("--t", type=int)
    b.add_argument("--block", type=int); b.add_argument("--n", type=int, default=5)
    b.add_argument("--out")
    r = sub.add_parser("run")
    r.add_argument("--games", type=int, default=5)
    r.add_argument("--ms", type=int, default=1000)
    r.add_argument("--tag", default="mine1")
    r.add_argument("--screen-n", type=int, default=8)
    r.add_argument("--confirm-n", type=int, default=30)
    r.add_argument("--max-scan", type=int, default=1000)
    r.add_argument("--confirm-z", type=float, default=3.0)
    r.add_argument("--seed-base", type=int, default=101)
    r.add_argument("--teams-source", choices=["ps", "corpus"], default="ps")
    r.add_argument("--audit", default=AUDIT_BIN)
    r.add_argument("--label", default=LABEL_BIN)
    a = ap.parse_args()
    {"game": cmd_game, "scan": cmd_scan, "run": cmd_run, "block": cmd_block}[a.cmd or "run"](a)

if __name__ == "__main__":
    main()
