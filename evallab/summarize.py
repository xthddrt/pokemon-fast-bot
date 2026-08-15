"""Turn the four cells' result JSONs into the tables that go in EVALLAB_RESULTS.md.

USAGE  python summarize.py <dir>
"""

import glob
import json
import os
import sys


def cell_name(r):
    return "%s / %s" % ("relational" if r["rel"] else "baseline",
                        "rank" if r["rank_w"] else "outcome-only")


def main():
    d = sys.argv[1]
    res = []
    for p in sorted(glob.glob(os.path.join(d, "res_*.json"))):
        res.append(json.load(open(p)))
    if not res:
        raise SystemExit("no res_*.json in %s" % d)
    order = sorted(res, key=lambda r: (r["rel"], r["rank_w"]))

    print("\n### (a) value cross-entropy on held-out games (lower is better)\n")
    pairs = sorted({k for r in order for k in r["value_ce"]})
    print("| cell | " + " | ".join("%s CE" % p for p in pairs) + " |")
    print("|---|" + "---|" * len(pairs))
    for r in order:
        cells = []
        for p in pairs:
            v = r["value_ce"].get(p)
            cells.append("%.4f" % v["ce"] if v else "-")
        print("| %s | %s |" % (cell_name(r), " | ".join(cells)))
    b = order[0]["value_ce"]
    print("\nconstant-predictor CE per pair: " +
          ", ".join("%s %.4f (base %.3f, n=%d)" % (p, b[p]["const_ce"], b[p]["base_rate"], b[p]["n"])
                    for p in pairs if p in b))

    print("\n### (b)+(c) sibling discrimination vs the 5M-iteration oracle\n")
    print("| cell | pair | n | spread | top1 | top3 | spearman | regret | random regret | oracle gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in order:
        for p in sorted(r["sibling"]):
            s = r["sibling"][p]
            print("| %s | %s | %d | %.4f | %.3f±%.3f | %.3f | %.3f | %.4f±%.4f | %.4f | %.4f |"
                  % (cell_name(r), p, s["n"], s["spread"], s["top1"], s["top1_se"],
                     s["top3"], s["spearman"], s["regret"], s["regret_se"],
                     s["random_regret"], s["oracle_gap"]))

    print("\n### transfer gap (pair A = trained, B = half-shared roster, C = disjoint)\n")
    print("| cell | metric | A | B | C | A->C gap |")
    print("|---|---|---|---|---|---|")
    for r in order:
        for m in ("top1", "spearman", "regret", "spread"):
            g = r["sibling"]
            if not all(k in g for k in "ABC"):
                continue
            print("| %s | %s | %.4f | %.4f | %.4f | %+.4f |"
                  % (cell_name(r), m, g["A"][m], g["B"][m], g["C"][m], g["C"][m] - g["A"][m]))


if __name__ == "__main__":
    main()
