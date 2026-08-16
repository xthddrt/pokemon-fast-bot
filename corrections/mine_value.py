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
AUDIT_BIN = os.path.join(ROOT, "valuenet/nets_v8c/v8c_h1g.bin")
LABEL_BIN = os.path.join(ROOT, "valuenet/nets_v8c/v8c_h1g.bin")  # label player = current champion (Sally 2026-08-16)
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

def mirror_state(s):
    """Side-swap: the game is side-symmetric, so a mirrored position is a
    legitimate new state. Its truth is MEASURED, not assumed 1-t: the label
    player is measurably side-asymmetric (mean 0.027 antisymmetry deviation),
    so naive 1-t labels are only safe at the extremes."""
    p = s.split("/")
    assert len(p) == 6, f"unexpected state format ({len(p)} segments)"
    p[0], p[1] = p[1], p[0]
    return "/".join(p)

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

    def block_confirm(state_str, gseed, t):
        """6 seed-blocks x 5 playouts over FRESH processes (the authority;
        playout-repro-anomaly demands cross-process draws)."""
        base = gseed * 1_000_003 + t * 8191 + 900_000
        args = [(state_str, (base + b * 1013 + j * 104729) & 0x7FFFFFFF)
                for b in range(6) for j in range(5)]
        with cf.ProcessPoolExecutor(max_workers=min(6, max(2, a.pool))) as bp:
            outs = list(bp.map(_playout_worker, args))
        bm = [sum(outs[b * 5:(b + 1) * 5]) / 5 for b in range(6)]
        return outs, bm

    g = json.load(open(a.game))
    res = {"seed": g["seed"], "outcome": g["outcome"], "scanned": [],
           "candidate": None, "near_misses": []}
    seq = [r for r in g["states"] if a.start_t is None or r["t"] < a.start_t]
    for rec in reversed(seq[-a.max_scan:]):
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
                "context": describe(rec["s"]), "s": rec["s"],
            }
            if z >= a.confirm_z:
                # scan-level hit: the block re-measure is the authority. A
                # rejection records a near-miss and the scan RESUMES — every
                # game ends block-CONFIRMED or clean-to-turn-1 (Sally).
                bouts, bm = block_confirm(rec["s"], g["seed"], rec["t"])
                bp_, bse = ac_stats(bouts)
                zp = abs(rec["e"] - bp_) / bse
                bmean = sum(bm) / 6
                bsd = (sum((m - bmean) ** 2 for m in bm) / 5) ** 0.5
                zb = abs(rec["e"] - bmean) / (bsd / 6 ** 0.5) if bsd > 0 else float("inf")
                k.update({"p_block": round(bp_, 4), "se_block": round(bse, 4),
                          "z_pooled": round(zp, 2),
                          "z_block": (round(zb, 2) if zb != float("inf") else "inf"),
                          "blocks": [round(m, 2) for m in bm],
                          "wins": sum(1 for o in bouts if o == 1.0),
                          "losses": sum(1 for o in bouts if o == 0.0),
                          "ties": sum(1 for o in bouts if o == 0.5),
                          "confirmed": bool(abs(rec["e"] - bp_) >= 0.10
                                            and min(zp, zb) >= 2.5)})
                if k["confirmed"]:
                    # mirror ruling (Sally: permanent up/down balance): the
                    # side-swapped state, block-confirmed on its own.
                    ms = mirror_state(rec["s"])
                    mouts, mbm = block_confirm(ms, g["seed"] + 500, rec["t"])
                    mp, mse = ac_stats(mouts)
                    k["mirror"] = {"s": ms, "p_block": round(mp, 4),
                                   "se_block": round(mse, 4),
                                   "blocks": [round(m, 2) for m in mbm],
                                   "naive_1mt": round(1 - k["p_block"], 4)}
                    res["candidate"] = k
                    break
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

def cmd_confirm_pairs(a):
    """20+20 pair confirm for corpus mining (Sally 2026-08-16): each input row
    {"id","s","e","s_mir","e_mir"} gets 20 playouts per seating (4 blocks x 5,
    distinct seed streams). Qualification is judged on the PAIR: mean gap
    >= 0.10 and combined z >= 2.5. Playouts are flattened across all rows so
    the process pool never idles between candidates."""
    import concurrent.futures as cf
    # SELF-SUFFICIENT LABEL ENV (canary catch 2026-08-16: the cloud bootstrap
    # ran this netless — 2,000 "playouts" in 10s measuring nothing). If the
    # label player isn't configured, configure it from the packed champion;
    # then PROVE the net actually loads before spending a single playout.
    if "PE_NN_WEIGHTS" not in os.environ:
        for k, v in net_env(LABEL_BIN).items():
            if k.startswith("PE_"):
                os.environ[k] = v
        print(f"label env self-configured from {LABEL_BIN}", flush=True)
    import subprocess as _sp
    chk = _sp.run([LEAF_PROF, "logits", "/dev/null"], env=dict(os.environ),
                  capture_output=True, text=True)
    if "valuenet: loaded" not in (chk.stderr + chk.stdout):
        raise SystemExit(f"FATAL: label net failed to load "
                         f"(PE_NN_WEIGHTS={os.environ.get('PE_NN_WEIGHTS')})")
    print("label net verified", flush=True)
    rows = [json.loads(l) for l in open(a.states) if l.strip()]
    lo, hi = a.start, (a.start + a.count if a.count else len(rows))
    rows = rows[lo:hi]
    n_pl = a.pair_n  # per seating
    print(f"confirm-pairs: {len(rows)} candidates [{lo}:{hi}], {n_pl}+{n_pl} playouts", flush=True)
    tasks = []
    for r in rows:
        base = (abs(hash(r["id"])) % 0xFFFFF) * 1013 + 300_000
        for j in range(n_pl):
            tasks.append((r["s"], (base + j * 104729) & 0x7FFFFFFF))
        for j in range(n_pl):
            tasks.append((r["s_mir"], (base + 777_777 + j * 104729) & 0x7FFFFFFF))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        outs = list(ex.map(_playout_worker, tasks, chunksize=4))
    el = time.time() - t0
    print(f"{len(tasks)} playouts in {el:.0f}s ({el/max(1,len(tasks)):.2f}s each)", flush=True)
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            o = outs[i*2*n_pl : i*2*n_pl + n_pl]
            m = outs[i*2*n_pl + n_pl : (i+1)*2*n_pl]
            po, seo = ac_stats(o)
            pm, sem = ac_stats(m)
            ga = abs(r["e"] - po); gb = abs(r["e_mir"] - pm)
            g = (ga + gb) / 2
            se = math.sqrt(seo**2 + sem**2) / 2
            z = g / se if se > 0 else float("inf")
            f.write(json.dumps({
                "id": r["id"], "e": r["e"], "e_mir": r["e_mir"],
                "target": round(po, 4), "se": round(seo, 4),
                "band": [round(max(0.0, po-seo), 4), round(min(1.0, po+seo), 4)],
                "target_mir": round(pm, 4), "se_mir": round(sem, 4),
                "band_mir": [round(max(0.0, pm-sem), 4), round(min(1.0, pm+sem), 4)],
                "pair_gap": round(g, 4), "pair_z": round(z, 2) if z != float("inf") else "inf",
                "confirmed": bool(g >= 0.10 and z >= 2.5),
                "s": r["s"], "s_mir": r["s_mir"],
            }) + "\n")
    print(f"confirm-pairs done -> {a.out}", flush=True)


def cmd_confirm_batch(a):
    """Block-confirm a LIST of candidate states (corpus mining path).

    Input JSONL: {"id": str, "s": state string, "e": net eval}
    Output JSONL: id, target (30-playout mean), se, band, blocks, W/T/L, z.
    30 playouts as 6 independent seed-blocks; if a candidate CONFIRMS and its
    SE > --tighten-se, a second 30 is added (cap 60) so murky positions get a
    narrower band (Sally 2026-08-15). Env = label player (s1), set by caller.
    """
    import concurrent.futures as cf
    rows = [json.loads(l) for l in open(a.states) if l.strip()]
    lo, hi = a.start, (a.start + a.count if a.count else len(rows))
    rows = rows[lo:hi]
    print(f"confirm-batch: {len(rows)} candidates [{lo}:{hi}]", flush=True)

    def blocks_for(s, key, n_blocks, off):
        args = [(s, (abs(hash(key)) % 0xFFFFF * 1013 + (b + off) * 7919
                     + j * 104729 + 600_000) & 0x7FFFFFFF)
                for b in range(n_blocks) for j in range(5)]
        with cf.ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
            outs = list(ex.map(_playout_worker, args))
        return outs

    t0 = time.time()
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            outs = blocks_for(r["s"], r["id"], 6, 0)
            p, se = ac_stats(outs)
            e = float(r.get("e", 0.0))
            zp = abs(e - p) / se if se > 0 else float("inf")
            bm = [sum(outs[b * 5:(b + 1) * 5]) / 5 for b in range(len(outs) // 5)]
            bmean = sum(bm) / len(bm)
            bsd = (sum((m - bmean) ** 2 for m in bm) / (len(bm) - 1)) ** 0.5
            zb = abs(e - bmean) / (bsd / len(bm) ** 0.5) if bsd > 0 else float("inf")
            ok = abs(e - p) >= 0.10 and min(zp, zb) >= 2.5
            if ok and se > a.tighten_se:
                outs += blocks_for(r["s"], r["id"], 6, 100)
                p, se = ac_stats(outs)
                zp = abs(e - p) / se if se > 0 else float("inf")
                bm = [sum(outs[b * 5:(b + 1) * 5]) / 5 for b in range(len(outs) // 5)]
                bmean = sum(bm) / len(bm)
                bsd = (sum((m - bmean) ** 2 for m in bm) / (len(bm) - 1)) ** 0.5
                zb = abs(e - bmean) / (bsd / len(bm) ** 0.5) if bsd > 0 else float("inf")
                ok = abs(e - p) >= 0.10 and min(zp, zb) >= 2.5
            w = sum(1 for o in outs if o == 1.0)
            t_ = sum(1 for o in outs if o == 0.5)
            f.write(json.dumps({
                "id": r["id"], "e": round(e, 4), "target": round(p, 4),
                "se": round(se, 4), "n": len(outs),
                "band": [round(max(0.0, p - se), 4), round(min(1.0, p + se), 4)],
                "z_pooled": round(zp, 2) if zp != float("inf") else "inf",
                "z_block": round(zb, 2) if zb != float("inf") else "inf",
                "confirmed": bool(ok), "wins": w, "ties": t_,
                "losses": len(outs) - w - t_,
                "blocks": [round(m, 2) for m in bm], "s": r["s"],
            }) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {el/(i+1)*(len(rows)-i-1):.0f}s",
                      flush=True)
    print(f"confirm-batch done: {len(rows)} in {time.time()-t0:.0f}s -> {a.out}",
          flush=True)


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

    if a.bench_per_game:
        # BENCH MODE (Sally 2026-08-15): no scans; sample states across ply
        # bands (40/40/20 early/mid/late, evallab's split) and block-label
        # each with 6 fresh processes x 5 playouts. Output bench.jsonl.
        bands = (("early", 0, 10), ("mid", 10, 25), ("late", 25, 10 ** 9))
        alloc = {"early": 0.4, "mid": 0.4, "late": 0.2}
        picks = []
        for i in range(a.games):
            g = json.load(open(os.path.join(work, f"g{i}.json")))
            rng = random.Random(a.seed_base + i * 31 + 7)
            for name, lo, hi in bands:
                ts = [r["t"] for r in g["states"] if lo <= r["t"] < hi]
                k = min(len(ts), max(1, round(alloc[name] * a.bench_per_game)))
                picks += [(i, t, name) for t in rng.sample(ts, k)]
        procs = []
        for (i, t, _) in picks:
            for b in range(6):
                wait_slots(procs, MAX_CONCURRENT)
                procs.append(subprocess.Popen(
                    [PY, os.path.abspath(__file__), "block",
                     "--game", os.path.join(work, f"g{i}.json"),
                     "--t", str(t), "--block", str(b), "--n", "5",
                     "--out", os.path.join(work, f"bb{i}_{t}_{b}.json")],
                    env=senv, stdout=sys.stdout, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        out_path = os.path.join(work, "bench.jsonl")
        with open(out_path, "w") as f:
            for (i, t, band) in picks:
                g = json.load(open(os.path.join(work, f"g{i}.json")))
                rec = next(r for r in g["states"] if r["t"] == t)
                outs = []
                for b in range(6):
                    outs += json.load(open(os.path.join(work, f"bb{i}_{t}_{b}.json")))["outs"]
                p_, se = ac_stats(outs)
                f.write(json.dumps({"key": f"{a.tag}-g{g['seed']}-t{t}",
                                    "s": rec["s"], "truth": round(p_, 4),
                                    "se": round(se, 4), "n": len(outs),
                                    "band": band, "e": round(rec["e"], 4)}) + "\n")
        print(f"[{time.time()-t0:.0f}s] BENCH: {len(picks)} block-labeled states "
              f"-> {out_path}", flush=True)
        return

    # scans each own a playout pool; concurrency x pool = MINE_CONCURRENT
    scan_pool = max(1, MAX_CONCURRENT // 4)
    scan_conc = max(1, MAX_CONCURRENT // max(scan_pool, 2))
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

    cands = [json.load(open(os.path.join(work, f"cand{i}.json")))
             for i in range(a.games)]
    rows = []
    print("\n=== MINING CANDIDATES (%s) — block-verified in-scan ===" % a.tag, flush=True)
    for i, c in enumerate(cands):
        for k in c.get("near_misses", []):
            extra = (f" blocks {k['p_block']} zp={k['z_pooled']} zb={k['z_block']}"
                     if "p_block" in k else "")
            print(f"game {c['seed']} NEAR-MISS t{k['t']}: eval {k['e']:.3f} vs "
                  f"scan {k['p_confirm']:.3f} (z={k['z_confirm']}){extra}", flush=True)
        if not c["candidate"]:
            print(f"game {c['seed']} (outcome {c['outcome']}): CLEAN to turn 1",
                  flush=True)
            continue
        k = c["candidate"]
        print(f"game {c['seed']} (outcome {c['outcome']}) turn {k['t']}: "
              f"eval {k['e']:.3f} vs truth {k['p_block']:.3f}±{k['se_block']:.3f} "
              f"(z_pooled={k['z_pooled']} z_block={k['z_block']}, "
              f"{k['wins']}W-{k['ties']}T-{k['losses']}L) CONFIRMED | "
              f"blocks={k['blocks']} | {k['context']}", flush=True)
        rows.append({
            "id": f"{a.tag}-g{c['seed']}-t{k['t']}",
            "game": f"selfplay-{a.tag}-{c['seed']}", "decision": k["t"],
            "target": k["p_block"],
            "band": [round(max(0.0, k["p_block"] - k["se_block"]), 4),
                     round(min(1.0, k["p_block"] + k["se_block"]), 4)],
            "states": [k["s"]], "n_playouts": 30,
            "note": f"in-scan block-verified: eval {k['e']:.3f} vs truth "
                    f"{k['p_block']:.3f} zp={k['z_pooled']} zb={k['z_block']} | {k['context']}",
            "ts": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        })
        if "mirror" in k:
            m = k["mirror"]
            rows.append({
                "id": f"{a.tag}-g{c['seed']}-t{k['t']}-mir",
                "game": f"selfplay-{a.tag}-{c['seed']}", "decision": k["t"],
                "target": m["p_block"],
                "band": [round(max(0.0, m["p_block"] - m["se_block"]), 4),
                         round(min(1.0, m["p_block"] + m["se_block"]), 4)],
                "states": [m["s"]], "n_playouts": 30,
                "note": f"MIRROR of t{k['t']}: measured {m['p_block']:.3f} "
                        f"(naive 1-t {m['naive_1mt']:.3f}) — up/down balance",
                "ts": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
            })
    json.dump(rows, open(os.path.join(work, "ledger_rows.json"), "w"), indent=1)
    print(f"\n[{time.time()-t0:.0f}s] {len(rows)} confirmed ruling(s) staged in "
          f"{os.path.join(work, 'ledger_rows.json')} — NOT hammered; awaiting "
          f"assessment.", flush=True)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("game", "scan", "run", "block", "confirm-batch", "confirm-pairs"):
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
    s.add_argument("--start-t", type=int, default=None)
    cp = sub.add_parser("confirm-pairs")
    cp.add_argument("--states", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--start", type=int, default=0)
    cp.add_argument("--count", type=int, default=0)
    cp.add_argument("--pair-n", type=int, default=20)
    cb = sub.add_parser("confirm-batch")
    cb.add_argument("--states", required=True)
    cb.add_argument("--out", required=True)
    cb.add_argument("--start", type=int, default=0)
    cb.add_argument("--count", type=int, default=0)
    cb.add_argument("--tighten-se", type=float, default=0.07)
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
    r.add_argument("--bench-per-game", type=int, default=0)
    r.add_argument("--audit", default=AUDIT_BIN)
    r.add_argument("--label", default=LABEL_BIN)
    a = ap.parse_args()
    {"game": cmd_game, "scan": cmd_scan, "run": cmd_run, "block": cmd_block,
     "confirm-batch": cmd_confirm_batch,
     "confirm-pairs": cmd_confirm_pairs}[a.cmd or "run"](a)

if __name__ == "__main__":
    main()
