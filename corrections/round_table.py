"""Standard per-round assessment table (Sally's format), merged across fleet
boxes. Run after every mining round:

    python3 corrections/round_table.py mine3        # matches _mine_work/mine3*
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ac(outs):
    n, x = len(outs), sum(outs)
    pt = (x + 2) / (n + 4)
    return x / n, math.sqrt(pt * (1 - pt) / (n + 4))


def main(prefix):
    rows = []
    for work in sorted(glob.glob(os.path.join(HERE, "_mine_work", prefix + "*"))):
        i = 0
        while os.path.isfile(os.path.join(work, f"g{i}.json")):
            g = json.load(open(os.path.join(work, f"g{i}.json")))
            c = json.load(open(os.path.join(work, f"cand{i}.json")))
            res = {1.0: "W", 0.0: "L", 0.5: "tie"}[g["outcome"]]
            total = len(g["states"])
            nm = [f"t{m['t']} z={m['z_confirm']}" for m in c.get("near_misses", [])]
            if not c["candidate"]:
                note = "clean game" + (f" (near-miss {', '.join(nm)})" if nm else "")
                rows.append((g["seed"], res, "—", f"{total} turns", note, "", "", ""))
            else:
                k = c["candidate"]
                outs, bm = [], []
                for b in range(6):
                    o = json.load(open(os.path.join(work, f"blk{i}_{b}.json")))["outs"]
                    outs += o
                    bm.append(sum(o) / len(o))
                p, se = ac(outs)
                e = k["e"]
                zp = abs(e - p) / se
                bmean = sum(bm) / 6
                bsd = (sum((m - bmean) ** 2 for m in bm) / 5) ** 0.5
                zb = abs(e - bmean) / (bsd / 6 ** 0.5) if bsd > 0 else float("inf")
                ok = abs(e - p) >= 0.10 and min(zp, zb) >= 2.5
                w = sum(1 for o in outs if o == 1.0)
                t_ = sum(1 for o in outs if o == 0.5)
                wl = f"{w}-{len(outs)-w-t_}" if not t_ else f"{w}-{t_}T-{len(outs)-w-t_}"
                zbs = "∞" if zb == float("inf") else f"{zb:.1f}"
                rows.append((g["seed"], res, k["t"],
                             f"of {total} (end−{total - k['t']})",
                             f"{e:.3f} → {p:.3f} ± {se:.2f}",
                             f"{zp:.1f} / {zbs}", wl,
                             f"{'HAMMER' if ok else 'NOT confirmed'} | {k['context']}"))
            i += 1
    print("| game | result | turn | game len | eval → truth (n=30) | z pooled/block | W-L | verdict, position |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows):
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mine")
