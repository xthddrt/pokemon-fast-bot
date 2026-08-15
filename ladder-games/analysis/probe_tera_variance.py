"""Probe A: tera-spend context (eval + decision index when we tera'd).
Probe B: variance-seeking when behind — when pooled eval < 0.40, did the
chosen move forgo a similar-mean alternative with materially higher upside
(upside = best avg_score for that move across sampled worlds)?
"""
import json
import os
import re
from multiprocessing import Pool

GAMES = "/Users/sallyliu/pokemon-fast-bot/ladder-games/games"
FEATURES = "/Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/features.jsonl"
OUT = "/Users/sallyliu/pokemon-fast-bot/ladder-games/analysis/tera_variance.jsonl"

WS_RE = re.compile(r"\[d (\d+)\] INFO\s+WorldStats (\d+): (.*)")
POL_RE = re.compile(r"\[d (\d+)\] INFO\s+Policy \d+: .*avg_score=([\d.]+)")
ARG_RE = re.compile(r"\[d (\d+)\] INFO\s+ARGMAX-ONLY selection[^:]*: (.+?) with ([\d.]+)% pooled")
MOVE_RE = re.compile(r"(.+?) ([\d.]+)%/([\d.]+)/±([\d.]+)")


def one(row):
    d = row["game"]
    path = os.path.join(GAMES, d, "search.log")
    per_dec_moves = {}   # dec -> move -> list of (avg, std)
    per_dec_eval = {}    # dec -> list of top avg_scores (position eval)
    chosen = {}          # dec -> (move, visit)
    try:
        with open(path, errors="replace") as f:
            for l in f:
                m = POL_RE.match(l)
                if m:
                    per_dec_eval.setdefault(int(m.group(1)), []).append(float(m.group(2)))
                    continue
                m = WS_RE.match(l)
                if m:
                    dec = int(m.group(1))
                    for part in m.group(3).split(" | "):
                        mm = MOVE_RE.match(part.strip())
                        if mm:
                            per_dec_moves.setdefault(dec, {}).setdefault(
                                mm.group(1), []
                            ).append((float(mm.group(3)), float(mm.group(4))))
                    continue
                m = ARG_RE.match(l)
                if m:
                    chosen[int(m.group(1))] = (m.group(2).strip(), float(m.group(3)))
    except OSError:
        return None

    out = {"game": d, "result": row["result"], "fmt": row["fmt"]}

    # A: the tera decision = ARGMAX choice with -tera suffix
    tera_decs = [dec for dec, (mv, _) in chosen.items() if mv.endswith("-tera")]
    if tera_decs:
        td = min(tera_decs)
        ev = per_dec_eval.get(td)
        out["tera_dec"] = td
        out["tera_frac_of_game"] = round(td / max(max(per_dec_eval, default=td), 1), 2)
        out["eval_at_tera"] = round(sum(ev) / len(ev), 3) if ev else None
        out["tera_visit"] = chosen[td][1]
    out["n_dec"] = len(per_dec_eval)

    # B: behind decisions
    behind = ignored = chose_max_upside = 0
    examples = []
    for dec, mvs in per_dec_moves.items():
        ev = per_dec_eval.get(dec)
        if not ev or not chosen.get(dec):
            continue
        pos = sum(ev) / len(ev)
        if pos >= 0.40:
            continue
        behind += 1
        pooled = {
            mv: (sum(a for a, _ in lst) / len(lst), max(a for a, _ in lst))
            for mv, lst in mvs.items()
        }
        ch = chosen[dec][0].replace("-tera", "")
        cm = pooled.get(ch) or pooled.get(chosen[dec][0])
        if not cm:
            continue
        best_up = max(pooled.values(), key=lambda x: x[1])
        if cm[1] >= best_up[1] - 1e-9:
            chose_max_upside += 1
        alt = [
            (mv, p) for mv, p in pooled.items()
            if p[1] >= cm[1] + 0.10 and p[0] >= cm[0] - 0.05 and mv != ch
        ]
        if alt:
            ignored += 1
            if len(examples) < 2:
                a = max(alt, key=lambda x: x[1][1])
                examples.append(
                    f"d{dec} pos={pos:.2f} chose {ch}(mean {cm[0]:.2f} up {cm[1]:.2f}) over {a[0]}(mean {a[1][0]:.2f} up {a[1][1]:.2f})"
                )
    out["behind_decs"] = behind
    out["behind_chose_max_upside"] = chose_max_upside
    out["behind_upside_ignored"] = ignored
    out["examples"] = examples
    return out


def main():
    rows = [json.loads(l) for l in open(FEATURES)]
    with Pool(6) as p:
        res = [r for r in p.map(one, rows) if r]
    with open(OUT, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in res)

    W = [r for r in res if r["result"] == "W"]
    L = [r for r in res if r["result"] == "L"]

    def agg(rs, name):
        t = [r for r in rs if r.get("tera_dec")]
        early = sum(1 for r in t if r["tera_dec"] <= 5)
        behind_tera = sum(1 for r in t if (r.get("eval_at_tera") or 1) < 0.45)
        bd = sum(r["behind_decs"] for r in rs)
        ig = sum(r["behind_upside_ignored"] for r in rs)
        mx = sum(r["behind_chose_max_upside"] for r in rs)
        print(f"{name}: tera'd={len(t)}/{len(rs)}  tera<=dec5={early}/{len(t) if t else 1}"
              f"  tera-while-behind(<.45)={behind_tera}/{len(t) if t else 1}"
              f"  med tera_frac={sorted(r['tera_frac_of_game'] for r in t)[len(t)//2] if t else '-'}")
        print(f"   behind-decisions={bd}  chose-max-upside={mx} ({100*mx/max(bd,1):.0f}%)"
              f"  upside-ignored={ig} ({100*ig/max(bd,1):.0f}%)")

    agg(W, "WIN ")
    agg(L, "LOSS")
    ex = [e for r in L for e in r.get("examples", [])][:8]
    print("\nexample ignored-upside decisions (losses):")
    for e in ex:
        print("  ", e)


if __name__ == "__main__":
    main()
