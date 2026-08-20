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
# The audit/label player is ALWAYS the production ladder net (Sally
# 2026-08-19): valuenet/PRODUCTION_NET is the single pointer the ladder
# launchers (run_game.sh/run_parallel.sh) and generation pipelines all read,
# so shipping a new champion updates every consumer at once. net_env() below
# applies the net's own constants sidecar, so calibrated constants follow
# automatically. MINE_BIN overrides for deliberate off-champion experiments.
def _production_bin():
    with open(os.path.join(ROOT, "valuenet/PRODUCTION_NET")) as f:
        return os.path.join(ROOT, f.read().strip())


AUDIT_BIN = os.environ.get("MINE_BIN") or _production_bin()
LABEL_BIN = AUDIT_BIN
# GAME-LENGTH CAP (Sally 2026-08-19: 1000 -> 250). Distinct from the per-
# PLAYOUT cap inside labeling, which is also 250 (--cap-steps).
#
# 1000 mirrored Showdown's own turn-limit backstop, letting stalls play out to
# PS-like resolution (PP drain -> Struggle). Measured over 6,000 games that
# tail is not worth its cost: median game ends at step ~36, p99 at 97, but the
# longest ran 961 -- ~27x the median in generation compute for one game. Only
# 0.17% of games reach a harvested position past step 250, so the cap bounds
# the worst generation outlier at ~7x median while touching almost nothing.
#
# Interacts with the unresolved-game discard below: a game that HITS this cap
# has no winner, so it is dropped entirely rather than mined. That is the
# intent -- a 250-turn stall war has no trustworthy outcome to label with.
# MINE_MAX_STEPS overrides for exact replication of older runs (1000 for the
# v10 corpus, 300 for the pre-v10 mining bootstrap).
MAX_STEPS = int(os.environ.get("MINE_MAX_STEPS", "250"))
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

def double_ko_winner(pre_state, instrs):
    r"""A DOUBLE KO IS NOT A DRAW -- who faints FIRST loses (Sally 2026-08-19).

    Ground truth is pokemon-showdown/sim/battle.ts checkWin():

        if (this.sides.every(side => !side.pokemonLeft)) {
            this.win(faintData && this.gen > 4 ? faintData.target.side : null);

    `faintData` after the faintQueue drain is the LAST faint processed and
    win() takes the WINNER, so for gen > 4 (gen9 qualifies) the side that
    faints LAST wins. Only gen <= 4 is a genuine tie.

    The engine emits a branch's instructions in application order, so replaying
    their HP deltas reproduces that faint ordering. Damage and Heal are the only
    variants that move a live pokemon's HP (DamageSubstitute and
    ChangeSubstituteHealth hit the substitute; ChangeMaxHP does not occur in
    gen9 randbats), and Heal amounts can be NEGATIVE (Steel Beam's self-cost) --
    hence -?\d+ in INS_RE.

    Tracks SIDE TOTALS rather than the active slot: a branch can switch the
    active mid-stream (U-turn, post-faint replacement), which makes the
    pre-state's active_index stale, and totals are immune to that. Both agree
    in the common case because a wiped side's last faint is necessarily its
    active.

    `instrs` is an instruction LIST (pass branch.instruction_list). Returns 1.0
    if side one wins, 0.0 if side two wins, or None when the order cannot be
    established -- callers must treat None as "no outcome", never as 0.5.
    """
    if pre_state is None or not instrs:
        return None
    tot = {"One": sum(p.hp for p in pre_state.side_one.pokemon if p.hp > 0),
           "Two": sum(p.hp for p in pre_state.side_two.pokemon if p.hp > 0)}
    fell = {}
    for i, ins in enumerate(instrs):
        m = INS_RE.match(str(ins))
        if not m:
            continue
        kind, side, amt = m.groups()
        tot[side] += int(amt) if kind == "Heal" else -int(amt)
        if tot[side] <= 0 and side not in fell:
            fell[side] = i
    if len(fell) < 2:
        return None
    return 1.0 if fell["One"] > fell["Two"] else 0.0

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
            w = double_ko_winner(state, pick.instruction_list)
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
    try:
        return one_playout(state_str, seed)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # engine panics (pyo3 PanicException) are BaseException and cannot
        # even be pickled back -- swallow to a None row, never kill the pool
        # (fleet incident 2026-08-17: one panic killed a whole box mid-chunk)
        return None

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



def phase_of(state_str):
    """hp-mass phase p = 1 - sum(hp_frac)/12 over both sides (v9 recipe)."""
    tot = 0.0
    for side in state_str.split("/")[:2]:
        for m in side.split("=")[:6]:
            f = m.split(",")
            mx = float(f[7])
            tot += (float(f[6]) / mx) if mx else 0.0
    return 1.0 - tot / 12.0


def _soft_pick(arms, rng):
    """Sally 2026-08-19 gen diversity: contenders = every arm with visits >=
    70% of the argmax's; sample one with probability proportional to visit
    share squared. Reduces to pure argmax when nothing is close."""
    best = max(m.visits for m in arms)
    cands = [m for m in arms if m.visits >= 0.7 * best]
    if len(cands) == 1:
        return cands[0].move_choice
    return rng.choices(cands, weights=[float(m.visits) ** 2 for m in cands])[0].move_choice


_GEN_FAILS = []


def _gen_game_worker(args):
    """Play one fresh-team self-play game; return phase-harvested positions
    (each carrying the game's final outcome as the free posterior seed).

    FAIL LOUD (Sally 2026-08-19): this used to swallow BaseException and
    return [], so a box whose every game threw still reported "DONE" with 0
    positions, uploaded ten empty chunks and self-terminated -- a whole fleet
    burned producing nothing. The first few failures now print a full
    traceback to stderr (captured in the box's run.log)."""
    seed, ms, gen_iters, keep, work, rand_ply = args
    try:
        return _gen_game_inner(seed, ms, gen_iters, keep, work, rand_ply)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        if len(_GEN_FAILS) < 3:
            _GEN_FAILS.append(1)
            import traceback
            print("GEN-GAME-FAILED seed=%s" % (seed,), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        return []


def _gen_game_inner(seed, ms, gen_iters, keep, work, rand_ply):
    sys.path.insert(0, os.path.join(ROOT, "valuenet", "sprt"))
    import run_duels as rd
    from poke_engine import State, generate_instructions, monte_carlo_tree_search
    tf = fresh_teams_file(work, seed, seed)
    rng = random.Random(seed ^ 0x5EED)
    state = State.from_string(rd.opening_state(tf, "p1", tf, "p2"))
    cand = [[], [], []]   # per-phase candidate turns; ONE random pick each
    prev_state, last_instrs = None, None
    for step in range(MAX_STEPS):
        if not any(p.hp > 0 for p in state.side_one.pokemon):
            break
        if not any(p.hp > 0 for p in state.side_two.pokemon):
            break
        ss = state.to_string()
        # A2: fixed-iteration game policy (== the label policy); gen_iters=0
        # restores the old timed search.
        res = monte_carlo_tree_search(state, 0 if gen_iters else ms, gen_iters,
                                      1, (seed * 104729 + step) & 0x7FFFFFFF)
        s1 = [m for m in res.side_one if m.visits > 0]
        s2 = [m for m in res.side_two if m.visits > 0]
        if not s1 or not s2:
            break
        ph = phase_of(ss)
        band = 0 if ph < 1 / 3 else 1 if ph < 2 / 3 else 2
        cand[band].append({"s": ss, "ph": round(ph, 4), "g": seed, "t": step})
        # DIVERSITY (Sally 2026-08-19, replaces the 15% random-ply): each side
        # samples among near-argmax contenders -- any arm with a visit count
        # >= 70% of the argmax's -- with probability proportional to visit
        # share SQUARED. E.g. shares 50%/40%: P(argmax) = .25/(.25+.16) = .61.
        # Squaring keeps the argmax favored while real near-ties genuinely
        # branch; arms the search dismissed can never be played (no garbage).
        p1 = _soft_pick(s1, rng)
        p2 = _soft_pick(s2, rng)
        try:
            branches = [b for b in generate_instructions(
                state, arm_to_move(p1), arm_to_move(p2)) if b.percentage > 0]
        except Exception:
            break
        if not branches:
            break
        chosen = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        prev_state, last_instrs = state, chosen.instruction_list
        state = state.apply_instructions(chosen)
    try:
        os.unlink(tf)
    except OSError:
        pass
    # A3 (Sally 2026-08-19): the game outcome is one free on-policy playout
    # sample for every position mined from it; 0.5 when the loop ended on the
    # step cap or a dead search rather than a winner.
    a1 = any(p.hp > 0 for p in state.side_one.pokemon)
    a2 = any(p.hp > 0 for p in state.side_two.pokemon)
    if a1 and not a2:
        yg = 1.0
    elif a2 and not a1:
        yg = 0.0
    elif not a1 and not a2:
        # BOTH sides wiped -- resolve by faint order, NEVER a draw.
        yg = double_ko_winner(prev_state, last_instrs)
    else:
        # both sides still alive -> the loop broke abnormally (MAX_STEPS, a
        # dead search, or failed instruction generation). No outcome exists.
        yg = None
    # UNRESOLVED-GAME DISCARD (Sally 2026-08-19): a game that ended with no
    # winner -- the MAX_STEPS cap, a dead search, or a failed instruction
    # generation -- has no real outcome, so yg=0.5 is a FABRICATED posterior
    # seed. Measured over 6,000 games: unresolved games are 0.92% of games and
    # 1.33% of the labeling budget, but average 5.24 step-capped playouts per
    # position vs 0.07 for resolved games -- the least trustworthy labels in
    # the corpus. Drop the whole game rather than mine it.
    # NOTE: this is a label-QUALITY fix, not a straggler fix. Stragglers are
    # CLOSE games: 81% of the playout budget sits in n>=20 positions and 98.4%
    # of those come from games that finished normally. LPT ordering is what
    # addresses the tail.
    if yg is None:
        return []
    # AT MOST 3 ROWS PER GAME (Sally 2026-08-17): one draw per phase bucket
    # kills within-game correlation. TARGET-THEN-NEAREST (Sally 2026-08-19):
    # a uniform draw over the band's STATES samples by dwell time, starving
    # the band edges (games spend ~1 turn at ph~0 and rush through ph>0.9).
    # Drawing a target ph uniformly over the band's INTERVAL and keeping the
    # state closest to it makes the sample ph-uniform within each band, so
    # turn-1 full-roster and deep-late states get real probability mass.
    out = []
    for b, lo in ((0, 0.0), (1, 1.0 / 3.0), (2, 2.0 / 3.0)):
        if not cand[b]:
            continue
        target = lo + rng.random() / 3.0
        out.append(min(cand[b], key=lambda r: abs(r["ph"] - target)))
    for r in out:
        r["yg"] = yg
    return out


def _label_worker(args):
    """One position -> one adaptive-n label via the native label_position
    (mercy + cap-bootstrap + step-0 sharing + Beta stopping, all in Rust).
    Returns (label, n, v0, n_mercy, n_capped, steps) or None on failure."""
    (state_str, seed, yg, iters, cap, nmin, nmax, h, m, mlo, mhi, mcons) = args
    try:
        from poke_engine import State, label_position
        return label_position(
            State.from_string(state_str), iters=iters, cap_steps=cap,
            n_min=nmin, n_max=nmax, ci_halfwidth=h, prior_m=m,
            mercy_lo=mlo, mercy_hi=mhi, mercy_consec=mcons,
            seed=seed, game_outcome=yg,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


def cmd_gen(a):
    """V10 CORPUS GENERATION (Sally 2026-08-19): fresh-team self-play games at
    --gen-iters (the label policy), <=3 phase-drawn positions per game, then
    ONE native label_position call per position: adaptive-n Beta labeling
    (prior = net value, +1 free sample from the source game outcome), mercy
    rule, 250-step cap with net bootstrap. One shard jsonl.gz out."""
    import concurrent.futures as cf
    import gzip
    if "PE_NN_WEIGHTS" not in os.environ:
        for k, v in net_env(a.label_bin).items():
            if k.startswith("PE_"):
                os.environ[k] = v
        print(f"label env self-configured from {a.label_bin}", flush=True)
    # Net verification via the WHEEL itself (Sally 2026-08-19): the old
    # leaf_prof subprocess check needed a locally COMPILED Rust binary, which
    # wheel-first fleet boxes deliberately do not have. engine_config() is
    # authoritative for the exact library the labeler will use. Env is set
    # above, before this first engine import (Rust LazyLock reads it once).
    import poke_engine as _pe
    _cfg = _pe.engine_config()
    if "nn_active=true" not in _cfg:
        raise SystemExit(f"FATAL: label net failed to load "
                         f"(PE_NN_WEIGHTS={os.environ.get('PE_NN_WEIGHTS')}) "
                         f"engine_config: {_cfg}")
    print(f"label net verified: {_cfg}", flush=True)
    work = os.path.join(HERE, "_mine_work", a.tag)
    os.makedirs(work, exist_ok=True)
    keep = (1.0, 1.0, 1.0)  # per-game bucket sampling replaced keep-weights
    npb = (a.n_early, a.n_mid, a.n_late)
    t0 = time.time()
    gtasks = [(a.seed_base + i, a.ms, a.gen_iters, keep, work, a.rand_ply)
              for i in range(a.games)]
    kept = []
    with cf.ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        for out in pool.map(_gen_game_worker, gtasks, chunksize=1):
            kept.extend(out)
    t1 = time.time()
    bands = [sum(1 for r in kept if (0 if r["ph"] < 1/3 else 1 if r["ph"] < 2/3 else 2) == b)
             for b in range(3)]
    print(f"games: {a.games} in {t1-t0:.0f}s -> {len(kept)} positions "
          f"(early {bands[0]} mid {bands[1]} late {bands[2]})", flush=True)
    # A chunk that harvested nothing means every game threw; uploading an
    # empty shard and exiting 0 hides a broken box (Sally 2026-08-19).
    if not kept:
        raise SystemExit("FATAL: 0 positions harvested from %d games -- see "
                         "GEN-GAME-FAILED tracebacks above" % a.games)
    tasks = [(r["s"],
              (r["g"] * 1_000_003 + r["t"] * 8191) & 0x7FFFFFFF,
              r.get("yg"), a.label_iters, a.cap_steps, a.label_nmin,
              a.label_nmax, a.label_h, a.label_m, a.mercy_lo, a.mercy_hi,
              a.mercy_consec)
             for r in kept]
    print(f"labeling: {len(tasks)} positions (adaptive n {a.label_nmin}"
          f"-{a.label_nmax}, h={a.label_h})", flush=True)
    outs = [None] * len(tasks)
    with cf.ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        for k, o in enumerate(pool.map(_label_worker, tasks, chunksize=1)):
            outs[k] = o
    t2 = time.time()
    n_total = mercy_total = cap_total = steps_total = 0
    shard = os.path.join(work, "shard.jsonl.gz")
    with gzip.open(shard, "wt") as f:
        for r, o in zip(kept, outs):
            if o is None:
                continue
            label, n, v0, n_mercy, n_capped, steps = o
            n_total += n
            mercy_total += n_mercy
            cap_total += n_capped
            steps_total += steps
            f.write(json.dumps({"s": r["s"], "y": round(label, 4), "n": n,
                                "v0": round(v0, 4), "yg": r.get("yg"),
                                "ph": r["ph"], "g": r["g"], "t": r["t"],
                                "mercy": n_mercy, "capped": n_capped}) + "\n")
    stats = {"games": a.games, "positions": len(kept),
             "bands": bands, "playouts": n_total,
             "mercy_exits": mercy_total, "cap_exits": cap_total,
             "playout_steps": steps_total,
             "game_s": round(t1 - t0, 1), "label_s": round(t2 - t1, 1),
             "s_per_playout": round((t2 - t1) * MAX_CONCURRENT / max(n_total, 1), 2)}
    json.dump(stats, open(os.path.join(work, "gen_stats.json"), "w"), indent=1)
    print(f"DONE {stats}", flush=True)



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
    if len(sys.argv) < 2 or sys.argv[1] not in ("game", "scan", "run", "block", "confirm-batch", "confirm-pairs", "gen"):
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

    ge = sub.add_parser("gen")
    ge.add_argument("--games", type=int, default=50)
    ge.add_argument("--ms", type=int, default=1000)
    # A2 (Sally 2026-08-19): game generation at fixed ITERATIONS -- the same
    # policy the labels use. 0 falls back to the timed --ms search.
    ge.add_argument("--gen-iters", type=int, default=2000)
    ge.add_argument("--tag", default="gen1")
    ge.add_argument("--seed-base", type=int, default=1)
    ge.add_argument("--keep-early", type=float, default=0.125)
    ge.add_argument("--keep-mid", type=float, default=0.5)
    ge.add_argument("--keep-late", type=float, default=1.0)
    ge.add_argument("--n-early", type=int, default=8)
    ge.add_argument("--n-mid", type=int, default=8)
    ge.add_argument("--n-late", type=int, default=10)
    # OBSOLETE (Sally 2026-08-19): random-ply replaced by _soft_pick squared-
    # share sampling among near-argmax contenders. Parsed for compat, unused.
    ge.add_argument("--rand-ply", type=float, default=0.0)
    # label player = production net (valuenet/PRODUCTION_NET), like AUDIT_BIN
    ge.add_argument("--label-bin", default=LABEL_BIN)
    # adaptive-n labeling knobs (Sally 2026-08-19; see label_position in
    # poke-engine-py): Beta posterior, net-value prior m, stop at 95% CI
    # half-width h, floor/cap n. n-early/mid/late above are OBSOLETE under
    # adaptive n and ignored unless --fixed-n 1.
    ge.add_argument("--label-iters", type=int, default=2000)
    ge.add_argument("--cap-steps", type=int, default=250)
    ge.add_argument("--label-nmin", type=int, default=4)
    ge.add_argument("--label-nmax", type=int, default=40)
    ge.add_argument("--label-h", type=float, default=0.15)
    ge.add_argument("--label-m", type=float, default=2.0)
    ge.add_argument("--mercy-lo", type=float, default=0.025)
    ge.add_argument("--mercy-hi", type=float, default=0.975)
    ge.add_argument("--mercy-consec", type=int, default=3)
    ge.set_defaults(fn=cmd_gen)
    r.add_argument("--bench-per-game", type=int, default=0)
    r.add_argument("--audit", default=AUDIT_BIN)
    r.add_argument("--label", default=LABEL_BIN)
    a = ap.parse_args()
    {"game": cmd_game, "scan": cmd_scan, "run": cmd_run, "block": cmd_block,
     "confirm-batch": cmd_confirm_batch,
     "confirm-pairs": cmd_confirm_pairs, "gen": cmd_gen}[a.cmd or "run"](a)

if __name__ == "__main__":
    main()
