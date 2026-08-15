"""Corpus summary + coverage check for a playout-label set (plc1 / plc2).

Validated by reproducing plc1's published SUMMARY.json field-for-field, which is
the only reason to trust the plc2 numbers it prints.

    python plc_summary.py <labels_dir> <n_expected> [out.json]
"""
import glob
import json
import sys
from collections import Counter

BANDS = ("early", "mid", "late")


def load(d):
    rows = {}
    dup = 0
    for p in sorted(glob.glob(d + "/labels_*.jsonl")):
        with open(p, errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "i" not in r or "label_p" not in r:
                    continue
                if r["i"] in rows:
                    dup += 1
                    # deterministic seeds -> a re-labelled row must be identical
                    assert rows[r["i"]]["label_p"] == r["label_p"], r["i"]
                    continue
                rows[r["i"]] = r
    return rows, dup


def summarize(rows):
    n = len(rows)
    rs = list(rows.values())
    bands = Counter(r["band"] for r in rs)
    dec = sum(1 for r in rs if abs(r["q_search"] - 0.5) > 0.4)
    opn = sum(1 for r in rs if r["ply"] < 2)
    mp = sum(r["label_p"] for r in rs) / n
    dec_counts = Counter(min(9, int(r["label_p"] * 10)) for r in rs)
    tot_pl = sum(r["n_playouts"] for r in rs)

    def brier(f):
        return round(sum((f(r) - r["label_p"]) ** 2 for r in rs) / n, 4)

    def by_band(f):
        out = {}
        for b in BANDS:
            sel = [r for r in rs if r["band"] == b]
            out[b] = round(sum(f(r) for r in sel) / len(sel), 4)
        return out

    return {
        "n": n,
        "n_playouts": {str(k): v for k, v in Counter(r["n_playouts"] for r in rs).items()},
        "bands": {b: bands[b] for b in BANDS},
        "band_frac": {b: round(bands[b] / n, 4) for b in BANDS},
        "decided_frac": round(dec / n, 4),
        "opening_frac": round(opn / n, 4),
        "mean_label_p": round(mp, 4),
        "mean_q_search": round(sum(r["q_search"] for r in rs) / n, 4),
        "mean_y_single": round(sum(r["y_single"] for r in rs) / n, 4),
        "mean_se": round(sum(r["se"] for r in rs) / n, 4),
        "mean_ply": round(sum(r["ply"] for r in rs) / n, 2),
        "trunc_frac": round(sum(r["trunc"] for r in rs) / tot_pl, 5),
        "core_hours": round(sum(r["cs"] for r in rs) / 3600, 1),
        "mean_cs_per_position": round(sum(r["cs"] for r in rs) / n, 3),
        "label_p_decile_counts": {str(k): dec_counts[k] for k in range(10)},
        "brier_q_search_vs_label": brier(lambda r: r["q_search"]),
        "brier_constant_vs_label": brier(lambda r: 0.5),
        "brier_y_single_vs_label": brier(lambda r: r["y_single"]),
        "brier_q_by_band": by_band(lambda r: (r["q_search"] - r["label_p"]) ** 2),
        "brier_const_by_band": by_band(lambda r: (0.5 - r["label_p"]) ** 2),
        # Two conventions. plc1's published SUMMARY used the second (it is
        # ~0.9x the first at N=10); both are emitted so any comparison against
        # plc1 can be made under ONE definition rather than mixing them.
        "label_noise_floor_by_band": by_band(lambda r: r["se"] ** 2),
        "label_noise_floor_all": round(sum(r["se"] ** 2 for r in rs) / n, 6),
        "noise_floor_bern_by_band": by_band(
            lambda r: r["label_p"] * (1 - r["label_p"]) / r["n_playouts"]),
        "noise_floor_bern_all": round(
            sum(r["label_p"] * (1 - r["label_p"]) / r["n_playouts"] for r in rs) / n, 6),
    }


if __name__ == "__main__":
    d, n_exp = sys.argv[1], int(sys.argv[2])
    rows, dup = load(d)
    missing = [i for i in range(n_exp) if i not in rows]
    extra = [i for i in rows if i >= n_exp or i < 0]
    errs = sum(1 for r in rows.values() if "error" in r)
    s = summarize(rows)
    s["coverage"] = {"expected": n_exp, "present": len(rows), "missing": len(missing),
                     "missing_first20": missing[:20], "out_of_range": len(extra),
                     "duplicate_rows_deduped": dup, "error_rows": errs}
    print(json.dumps(s, indent=2))
    if len(sys.argv) > 3:
        json.dump(s, open(sys.argv[3], "w"), indent=2)
