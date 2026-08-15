"""Report on a playout-label file: cost, SE distribution, and how far the
playout-averaged label is from the single game outcome it replaces.

  python playout_report.py <labels.jsonl> [labels2.jsonl ...]
"""
import json
import statistics
import sys

rows = []
for p in sys.argv[1:]:
    for line in open(p):
        r = json.loads(line)
        if "label_p" in r:
            rows.append(r)
rows = list({r["i"]: r for r in rows}.values())
n = len(rows)
N = rows[0]["n_playouts"]
cs = sum(r["cs"] for r in rows)
steps = statistics.mean(r["steps"] for r in rows)
pl = n * N


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


print("positions %d   playouts %d (N=%d)   mean playout length %.1f decisions" % (n, pl, N, steps))
print("core-seconds %.0f  =>  %.4f core-s/playout, %.5f core-s/decision, %.2f core-s/position"
      % (cs, cs / pl, cs / pl / steps, cs / n))
print("truncated playouts (hit MAX_STEPS): %d / %d = %.4f"
      % (sum(r["trunc"] for r in rows), pl, sum(r["trunc"] for r in rows) / pl))

se = [r["se"] for r in rows]
print("\nSE of the label (n=%d)" % n)
print("  mean %.4f   median %.4f   p10 %.4f p90 %.4f   max %.4f"
      % (statistics.mean(se), statistics.median(se), q(se, .10), q(se, .90), max(se)))
print("  SE=0 (all %d playouts agreed): %d positions = %.1f%%"
      % (N, sum(x == 0 for x in se), 100.0 * sum(x == 0 for x in se) / n))
print("  single-outcome label SE for comparison: sd of one Bernoulli draw,")
print("  mean over these positions = %.4f"
      % statistics.mean((p_ * (1 - p_)) ** .5 for p_ in (r["label_p"] for r in rows)))

d = [r["label_p"] - r["y_single"] for r in rows]
ad = [abs(x) for x in d]
print("\nSINGLE RECORDED OUTCOME  vs  PLAYOUT-AVERAGED LABEL   (the noise being removed)")
print("  mean |diff| %.4f    RMSE %.4f    mean signed diff %+.4f" % (statistics.mean(ad), (sum(x * x for x in d) / n) ** .5, statistics.mean(d)))
print("  |diff| > 0.5 : %.1f%% of positions   > 0.25 : %.1f%%   > 0.10 : %.1f%%"
      % (100.0 * sum(x > .5 for x in ad) / n, 100.0 * sum(x > .25 for x in ad) / n,
         100.0 * sum(x > .10 for x in ad) / n))
print("  mean label_p %.4f   mean y_single %.4f" % (
    statistics.mean(r["label_p"] for r in rows), statistics.mean(r["y_single"] for r in rows)))
var_single = statistics.mean(x * x for x in d)
print("  implied label variance removed: E[(y_single - label_p)^2] = %.4f" % var_single)
print("  (a fair coin is 0.25; the residual label variance is now mean se^2 = %.5f, %.0fx smaller)"
      % (statistics.mean(x * x for x in se), var_single / max(statistics.mean(x * x for x in se), 1e-9)))

qs = [r for r in rows if r.get("q_search") is not None]
if qs:
    dq = [r["q_search"] - r["label_p"] for r in qs]
    print("\nGENERATING SEARCH's own root value vs the playout label (2,000 iterations)")
    print("  mean |diff| %.4f   RMSE %.4f   mean signed %+.4f (search %s)"
          % (statistics.mean(abs(x) for x in dq), (sum(x * x for x in dq) / len(dq)) ** .5,
             statistics.mean(dq), "optimistic for side one" if statistics.mean(dq) > 0 else "pessimistic for side one"))

print("\nPER-PHASE")
print("  %-6s %6s %8s %8s %8s %9s %9s %9s" % ("band", "n", "mean p", "mean se", "steps", "core-s/pos", "|p-y1|", "frac p in (.05,.95)"))
for b in ("early", "mid", "late"):
    g = [r for r in rows if r["band"] == b]
    if not g:
        continue
    print("  %-6s %6d %8.4f %8.4f %8.1f %9.2f %9.4f %9.3f"
          % (b, len(g), statistics.mean(r["label_p"] for r in g), statistics.mean(r["se"] for r in g),
             statistics.mean(r["steps"] for r in g), statistics.mean(r["cs"] for r in g),
             statistics.mean(abs(r["label_p"] - r["y_single"]) for r in g),
             sum(.05 < r["label_p"] < .95 for r in g) / len(g)))

print("\nLABEL DISTRIBUTION (how much of the corpus is genuinely uncertain)")
for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
    k = sum(lo <= r["label_p"] < hi for r in rows)
    print("  [%.2f,%.2f) %5d  %5.1f%%  %s" % (lo, hi, k, 100.0 * k / n, "#" * int(60.0 * k / n)))
