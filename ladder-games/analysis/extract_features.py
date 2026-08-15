"""Per-game feature extraction for the loss/win commonality study.

Ground truth = replay.json (omniscient log). Bot internals = search.log
(per-decision, per-world visit/score/opp-policy) + worlds.jsonl (sampled
opponent completions with chances). One JSON row per non-infra game.
"""
import json
import math
import os
import re
import sys
from multiprocessing import Pool

GAMES = "/Users/sallyliu/pokemon-fast-bot/ladder-games/games"
OUT = "/Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/features.jsonl"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parse_replay(log, our_name):
    lines = log.split("\n")
    p_us = None
    for l in lines:
        m = re.match(r"\|player\|(p[12])\|([^|]*)\|", l)
        if m and norm(m.group(2)) == norm(our_name):
            p_us = m.group(1)
    if not p_us:
        return None
    p_opp = "p2" if p_us == "p1" else "p1"

    turn = 0
    species = {"p1": set(), "p2": set()}
    levels = {"p1": [], "p2": []}
    boosts = {}  # (side, species) -> net positive stage sum
    kills = {}  # (side, species) -> count
    boosted_kills = {}
    last_move = {"p1": None, "p2": None}  # (species, move, turn)
    faints = {"p1": 0, "p2": 0}
    first_faint = None
    crits_recv = {"p1": 0, "p2": 0}
    miss_by = {"p1": 0, "p2": 0}
    cant = {"p1": 0, "p2": 0}
    status_recv = {"p1": 0, "p2": 0}
    hazards = {"p1": 0, "p2": 0}
    tera_turn = {"p1": None, "p2": None}
    opp_actions = []  # (turn, action) voluntary actions by opponent
    faint_this_turn = {"p1": False, "p2": False}
    active = {"p1": None, "p2": None}
    win = None

    def sp(field):
        return norm(field.split(",")[0])

    for l in lines:
        if l.startswith("|turn|"):
            turn = int(l.split("|")[2])
            faint_this_turn = {"p1": False, "p2": False}
        elif l.startswith(("|switch|", "|drag|", "|replace|")):
            parts = l.split("|")
            side = parts[2][:2]
            s = sp(parts[3])
            species[side].add(s)
            m = re.search(r", L(\d+)", parts[3])
            if m:
                levels[side].append(int(m.group(1)))
            boosts[(side, s)] = 0
            if active[side]:
                boosts[(side, active[side])] = 0
            active[side] = s
            if (
                l.startswith("|switch|")
                and side == p_opp
                and turn > 0
                and not faint_this_turn[side]
            ):
                opp_actions.append((turn, "switch " + s))
        elif l.startswith("|move|"):
            parts = l.split("|")
            side = parts[2][:2]
            mover = sp(parts[2].split(":")[1] if ":" in parts[2] else parts[2])
            last_move[side] = (mover, norm(parts[3]), turn)
            if side == p_opp:
                opp_actions.append((turn, norm(parts[3])))
        elif l.startswith("|faint|"):
            parts = l.split("|")
            side = parts[2][:2]
            faints[side] += 1
            faint_this_turn[side] = True
            if first_faint is None:
                first_faint = (side, turn)
            other = "p2" if side == "p1" else "p1"
            lm = last_move[other]
            if lm and lm[2] >= turn - 1:
                k = (other, lm[0])
                kills[k] = kills.get(k, 0) + 1
                if boosts.get((other, lm[0]), 0) >= 2:
                    boosted_kills[k] = boosted_kills.get(k, 0) + 1
        elif l.startswith("|-boost|"):
            parts = l.split("|")
            side = parts[2][:2]
            s = sp(parts[2].split(":")[1] if ":" in parts[2] else parts[2])
            boosts[(side, s)] = boosts.get((side, s), 0) + int(parts[4])
        elif l.startswith("|-unboost|"):
            parts = l.split("|")
            side = parts[2][:2]
            s = sp(parts[2].split(":")[1] if ":" in parts[2] else parts[2])
            boosts[(side, s)] = max(0, boosts.get((side, s), 0) - int(parts[4]))
        elif l.startswith("|-crit|"):
            side = l.split("|")[2][:2]
            crits_recv[side] += 1
        elif l.startswith("|-miss|"):
            side = l.split("|")[2][:2]
            miss_by[side] += 1
        elif l.startswith("|cant|"):
            side = l.split("|")[2][:2]
            cant[side] += 1
        elif l.startswith("|-status|"):
            side = l.split("|")[2][:2]
            status_recv[side] += 1
        elif l.startswith("|-sidestart|"):
            side = l.split("|")[2][:2]
            hazards[side] += 1
        elif l.startswith("|-terastallize|"):
            side = l.split("|")[2][:2]
            if tera_turn[side] is None:
                tera_turn[side] = turn
        elif l.startswith("|win|"):
            win = l.split("|")[2].strip()

    opp_kill_list = {k[1]: v for k, v in kills.items() if k[0] == p_opp}
    our_kill_list = {k[1]: v for k, v in kills.items() if k[0] == p_us}
    opp_top_killer = max(opp_kill_list.items(), key=lambda x: x[1], default=(None, 0))
    our_top = max(our_kill_list.items(), key=lambda x: x[1], default=(None, 0))
    opp_boosted_k = {k[1]: v for k, v in boosted_kills.items() if k[0] == p_opp}

    return {
        "turns": turn,
        "our_species": sorted(species[p_us]),
        "opp_species": sorted(species[p_opp]),
        "our_lvl": round(sum(levels[p_us]) / max(len(levels[p_us]), 1), 1),
        "opp_lvl": round(sum(levels[p_opp]) / max(len(levels[p_opp]), 1), 1),
        "opp_top_killer": opp_top_killer[0],
        "opp_top_kills": opp_top_killer[1],
        "opp_top_killer_boosted_kills": opp_boosted_k.get(opp_top_killer[0], 0),
        "our_top_kills": our_top[1],
        "opp_setup_max": max(
            [v for (sd, _), v in boosts.items() if sd == p_opp] + [0]
        ),
        "crits_against": crits_recv[p_us],
        "crits_for": crits_recv[p_opp],
        "we_missed": miss_by[p_us],
        "they_missed": miss_by[p_opp],
        "we_cant": cant[p_us],
        "they_cant": cant[p_opp],
        "status_on_us": status_recv[p_us],
        "status_on_them": status_recv[p_opp],
        "hazards_on_us": hazards[p_us],
        "hazards_on_them": hazards[p_opp],
        "our_tera_turn": tera_turn[p_us],
        "opp_tera_turn": tera_turn[p_opp],
        "our_left": 6 - faints[p_us],
        "opp_left": 6 - faints[p_opp],
        "first_faint_ours": first_faint[0] == p_us if first_faint else None,
        "first_faint_turn": first_faint[1] if first_faint else None,
        "opp_actions": opp_actions,
        "won": norm(win) == norm(our_name) if win else None,
    }


def parse_search(path):
    """Per decision: mean top-move score across worlds, pooled opp-move dist."""
    evals = {}
    opp_pred = {}
    weights = {}
    try:
        with open(path, errors="replace") as f:
            for l in f:
                m = re.match(r"\[d (\d+)\] INFO\s+Policy (\d+): .*avg_score=([\d.]+) sample_chance_multiplier=([\d.eE+-]+)", l)
                if m:
                    d, w = int(m.group(1)), int(m.group(2))
                    evals.setdefault(d, []).append(float(m.group(3)))
                    weights.setdefault(d, {})[w] = float(m.group(4))
                    continue
                m = re.match(r"\[d (\d+)\] INFO\s+OppWorldStats (\d+): (.*)", l)
                if m:
                    d, w = int(m.group(1)), int(m.group(2))
                    dist = {}
                    for part in m.group(3).split(" | "):
                        mm = re.match(r"(.+?) ([\d.]+)%/", part.strip())
                        if mm:
                            dist[norm(mm.group(1))] = float(mm.group(2)) / 100.0
                    opp_pred.setdefault(d, []).append((w, dist))
    except OSError:
        return None
    pooled = {}
    for d, wl in opp_pred.items():
        ws = weights.get(d, {})
        tot = sum(ws.get(w, 1.0) for w, _ in wl) or 1.0
        agg = {}
        for w, dist in wl:
            wt = ws.get(w, 1.0) / tot
            for mv, p in dist.items():
                agg[mv] = agg.get(mv, 0.0) + wt * p
        pooled[d] = agg
    return {
        "evals": {d: sum(v) / len(v) for d, v in evals.items()},
        "opp_pred": pooled,
    }


def parse_worlds(path, our_species, opp_revealed):
    """Species coverage of the sampled opponent side vs finally-revealed team,
    at an early, mid and late decision."""
    ours = set(our_species)
    revealed = set(opp_revealed)
    if not revealed:
        return {}
    by_dec = {}
    try:
        with open(path, errors="replace") as f:
            for l in f:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                by_dec.setdefault(r["decision"], []).append(r)
    except OSError:
        return {}
    if not by_dec:
        return {}
    decs = sorted(by_dec)
    picks = {"early": decs[0], "mid": decs[len(decs) // 2], "late": decs[-1]}
    out = {}
    for label, d in picks.items():
        covs, tot_w = [], 0.0
        for r in by_dec[d]:
            sides = r["state"].split("/")
            best = None
            for s in sides[:2]:
                mons = {norm(chunk.split(",")[0]) for chunk in s.split("=") if "," in chunk}
                if len(mons & ours) < 3:  # not our side -> opponent sample
                    best = mons
            if best is None:
                continue
            w = r.get("chance", 1.0) or 1e-9
            covs.append(w * len(best & revealed) / len(revealed))
            tot_w += w
        if covs and tot_w:
            out["cov_" + label] = round(sum(covs) / tot_w, 3)
    return out


def one_game(d):
    try:
        meta = json.load(open(os.path.join(GAMES, d, "meta.json")))
        if meta.get("infra") or meta.get("result") not in ("W", "L"):
            return None
        replay = json.load(open(os.path.join(GAMES, d, "replay.json")))
        rp = parse_replay(replay.get("log", ""), meta["account"])
        if not rp or rp["turns"] < 3:
            return None
        srch = parse_search(os.path.join(GAMES, d, "search.log")) or {}
        evals = srch.get("evals", {})
        pred = srch.get("opp_pred", {})

        row = {
            "game": d,
            "result": meta["result"],
            "fmt": "blitz" if "blitz" in meta["battle_tag"] else "regular",
            "opp_elo": meta.get("opp_elo"),
            "our_elo": meta.get("our_elo"),
            "account": meta.get("account"),
        }
        opp_actions = rp.pop("opp_actions")
        row.update(rp)

        if evals:
            ds = sorted(evals)
            ev = [evals[i] for i in ds]
            row["eval_t1"] = round(ev[0], 3)
            row["eval_last"] = round(ev[-1], 3)
            row["eval_min"] = round(min(ev), 3)
            row["eval_max"] = round(max(ev), 3)
            below = [i for i, v in zip(ds, ev) if v < 0.35]
            row["first_dec_below_35"] = below[0] if below else None
            drops = [(ev[i] - ev[i + 1], ds[i + 1]) for i in range(len(ev) - 1)]
            if drops:
                bd = max(drops)
                row["biggest_drop"] = round(bd[0], 3)
                row["biggest_drop_dec"] = bd[1]
            row["n_decisions"] = len(ds)

        # prediction accuracy: sequential zip of decisions vs opponent actions
        if pred and opp_actions:
            ds = sorted(pred)
            n = min(len(ds), len(opp_actions))
            row["pred_aligned"] = abs(len(ds) - len(opp_actions)) <= 2
            hits, mass, worst = 0, [], (1.0, None)
            for i in range(n):
                dist = pred[ds[i]]
                actual = opp_actions[i][1]
                if actual.startswith("switch"):
                    key = "switch " + actual.split(" ", 1)[1]
                    key = norm(key)
                else:
                    key = actual
                # tera-suffixed prediction entries count toward the base move
                p_actual = sum(v for k, v in dist.items() if k == key or k == key + "tera")
                top = max(dist.items(), key=lambda x: x[1], default=(None, 0))
                if top[0] and (top[0] == key or top[0] == key + "tera"):
                    hits += 1
                mass.append(p_actual)
                if p_actual < worst[0]:
                    worst = (p_actual, opp_actions[i][0])
            if n:
                row["pred_top1"] = round(hits / n, 3)
                row["pred_mass"] = round(sum(mass) / n, 3)
                row["pred_worst"] = round(worst[0], 3)
                row["pred_worst_turn"] = worst[1]
                row["pred_n"] = n

        row.update(
            parse_worlds(
                os.path.join(GAMES, d, "worlds.jsonl"),
                row["our_species"],
                row["opp_species"],
            )
        )
        return row
    except Exception as e:
        return {"game": d, "error": repr(e)[:200]}


def main():
    dirs = sorted(os.listdir(GAMES))
    with Pool(6) as p:
        rows = p.map(one_game, dirs)
    ok = [r for r in rows if r and "error" not in r]
    err = [r for r in rows if r and "error" in r]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in ok)
    print(f"extracted {len(ok)} games, {len(err)} errors")
    for r in err[:5]:
        print("ERR", r["game"], r["error"])


if __name__ == "__main__":
    main()
