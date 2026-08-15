"""FLOOR CE from recurring positions -- model-free, encoder-free, search-free.

WHAT IS COMPUTED
  For a grouping of positions into equivalence classes, the empirical predictor
  is the win rate of the OTHER games in the same class:
        p_i = (S_g - y_(game i)  +  a * r) / (N_g - 1 + a)
  where S_g, N_g are sums/counts over the DISTINCT GAMES in class g, r is the
  corpus base rate and a is a shrinkage weight. Leaving out the whole GAME (not
  just the position) is essential: every position inside one game carries the
  same outcome label, so leave-one-POSITION-out would score a group with a
  perfect copy of its own answer.

  CE = -(1/n) sum [ y log p + (1-y) log(1-p) ] in nats, over positions.

  Draws (y = 0.5) are kept and contribute at least log 2 each; they are 0.18%
  of games, worth ~0.001 nats, and are reported separately.

BRACKET
  loo   leave-one-game-out CE            -- in expectation an UPPER bound on
                                            H(Y | class): it pays H plus the
                                            KL of a finite-sample predictor.
  mm    Miller-Madow entropy of the class win rate, H(p_hat) + 1/(2N)
                                         -- a first-order bias-corrected POINT
                                            estimate of H(Y | class).
  plug  H(p_hat) with no correction      -- a LOWER bound (plug-in entropy is
                                            biased down by concavity).
  and H(Y | class) itself is >= H(Y | exact state) for any coarsening, so every
  number here is an upper bound on the true irreducible floor.

UNCERTAINTY
  Nonparametric bootstrap over GAMES (the independent unit), 2000 resamples,
  with the class predictors held fixed -- so it is the sampling variability of
  the scored position mix, not of the predictor.
"""
import sys
import numpy as np

LEVELS = ["E", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
EPS = 1e-12


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def pairs(gr, gid, y, ngroups, ngames):
    """Collapse to (class, game) pairs. Returns per-position pair index,
    per-pair outcome, per-pair class, and class-level game sums/counts."""
    pk = gr.astype(np.int64) * np.int64(ngames) + gid.astype(np.int64)
    up, pidx = np.unique(pk, return_inverse=True)
    yp = np.zeros(len(up), np.float64)
    yp[pidx] = y
    gp = (up // np.int64(ngames)).astype(np.int64)
    S = np.bincount(gp, weights=yp, minlength=ngroups)
    N = np.bincount(gp, minlength=ngroups).astype(np.float64)
    return pidx, yp, gp, S, N


def ce_of(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def floor_level(k, gid, y, alphas=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64), base=None):
    """All floor statistics for ONE grouping."""
    _, gr = np.unique(k, return_inverse=True)
    ngroups = gr.max() + 1
    ngames = int(gid.max()) + 1
    pidx, yp, gp, S, N = pairs(gr, gid, y, ngroups, ngames)
    if base is None:
        base = float(y.mean())
    Si = S[gr] - yp[pidx]
    Ni = N[gr] - 1.0
    est = Ni >= 1.0
    res = {"ngroups": int(ngroups), "ngroups_ge2": int((N >= 2).sum()),
           "n_pos": int(len(y)), "n_est": int(est.sum()),
           "cov": float(est.mean()), "base": base,
           "games_per_group": N, "pos_per_group": np.bincount(gr, minlength=ngroups)}
    best = None
    curve = []
    for a in alphas:
        p = (Si + a * base) / (Ni + a)
        ce = ce_of(y, p)
        m_est = float(ce[est].mean()) if est.any() else float("nan")
        pf = np.where(est, p, base)                      # base-rate fallback
        m_all = float(ce_of(y, pf).mean())
        curve.append((a, m_est, m_all))
        if best is None or m_all < best[2]:
            best = (a, m_est, m_all, ce, p, est)
    res["curve"] = curve
    res["alpha"] = best[0]
    res["loo_est"] = best[1]
    res["loo_all"] = best[2]
    res["ce_pos"] = np.where(best[5], best[3], ce_of(y, np.full(len(y), base)))
    res["ce_est"] = best[3]
    res["est"] = best[5]
    # plug-in and Miller-Madow on classes with >= 2 games
    ph = np.where(N >= 2, S / np.maximum(N, 1), base)
    Hp = -(np.clip(ph, EPS, 1) * np.log(np.clip(ph, EPS, 1))
           + np.clip(1 - ph, EPS, 1) * np.log(np.clip(1 - ph, EPS, 1)))
    Hmm_ = np.minimum(Hp + 1.0 / (2.0 * np.maximum(N, 1)), np.log(2.0))
    w = np.bincount(gr, minlength=ngroups).astype(np.float64) * (N >= 2)
    res["plug_est"] = float((Hp * w).sum() / max(w.sum(), 1))
    res["mm_est"] = float((Hmm_ * w).sum() / max(w.sum(), 1))
    return res


def boot(ce, gid, ngames, reps=2000, seed=0):
    """Cluster bootstrap over games on a per-position score."""
    s = np.bincount(gid, weights=ce, minlength=ngames)
    n = np.bincount(gid, minlength=ngames).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        j = rng.integers(0, ngames, ngames)
        out[i] = s[j].sum() / max(n[j].sum(), 1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def q(a, qs=(0.5, 0.9, 0.99, 1.0)):
    return [float(np.quantile(a, x)) for x in qs]


def main():
    d = load(sys.argv[1])
    name = sys.argv[2]
    gid, y, ply, tag, glen = d["gid"], d["y"], d["ply"], d["tag"], d["glen"]
    ngames = int(gid.max()) + 1
    base = float(y.mean())
    base_game = float(np.bincount(gid, weights=y, minlength=ngames)[
        np.bincount(gid, minlength=ngames) > 0].mean() /
        1.0) if True else base
    # per-game outcome
    yg = np.zeros(ngames); yg[gid] = y
    print("== %s ==" % name)
    print("games=%d positions=%d  base rate (position-weighted)=%.4f  (game-weighted)=%.4f"
          % (ngames, len(y), base, float(yg.mean())))
    print("draws: %d games (%.3f%%)" % (int((yg == 0.5).sum()), 100.0 * (yg == 0.5).mean()))
    b = floor_level(np.zeros(len(y), np.uint64), gid, y, base=base)
    print("BASE-RATE constant predictor CE = %.4f nats  (LOO over games, a=%g)"
          % (b["loo_all"], b["alpha"]))
    lo, hi = boot(b["ce_pos"], gid, ngames)
    print("   95%% CI [%.4f, %.4f]" % (lo, hi))
    print()
    print("| level | classes | classes>=2 games | %pos estimable | median games/class | p90 | max | "
          "LOO CE (estimable) | LOO CE (all, base fallback) | 95% CI | plug-in H | Miller-Madow H | a* |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    store = {}
    for L in LEVELS:
        r = floor_level(d["k_" + L], gid, y, base=base)
        store[L] = r
        gg = r["games_per_group"]
        gg2 = gg[gg >= 2]
        lo, hi = boot(r["ce_pos"], gid, ngames)
        print("| %s | %d | %d | %.1f%% | %.0f | %.0f | %.0f | %.4f | %.4f | [%.4f, %.4f] | %.4f | %.4f | %g |"
              % (L, r["ngroups"], r["ngroups_ge2"], 100 * r["cov"],
                 np.median(gg2) if len(gg2) else 0,
                 np.quantile(gg2, .9) if len(gg2) else 0, gg.max(),
                 r["loo_est"], r["loo_all"], lo, hi, r["plug_est"], r["mm_est"], r["alpha"]))
    np.save("/private/tmp/claude-501/-Users-sallyliu-pokemon-fast-bot/410e0c58-8931-45a0-8e25-e3a8ec37baef/scratchpad/fl/%s_ce.npy" % name,
            np.vstack([store[L]["ce_pos"] for L in LEVELS]))
    print()
    # ---- alpha curves for the two headline levels
    for L in ("E", "L4"):
        print("alpha curve %s: %s" % (L, ", ".join("a=%g:%.4f" % (a, m) for a, _, m in store[L]["curve"])))
    print()
    # ---- exact-state recurrence by ply
    print("| ply band | positions | %in exact classes >=2 games | exact LOO CE | L4 LOO CE | L4 %estimable |")
    print("|---|---|---|---|---|---|")
    bands = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 14), (15, 19),
             (20, 29), (30, 49), (50, 999)]
    E, L4 = store["E"], store["L4"]
    for a_, b_ in bands:
        m = (ply >= a_) & (ply <= b_)
        if not m.any():
            continue
        me = m & E["est"]
        ml = m & L4["est"]
        print("| %d-%d | %d | %.1f%% | %s | %s | %.1f%% |"
              % (a_, b_, int(m.sum()), 100 * me.sum() / m.sum(),
                 ("%.4f" % E["ce_est"][me].mean()) if me.sum() > 30 else "-",
                 ("%.4f" % L4["ce_est"][ml].mean()) if ml.sum() > 30 else "-",
                 100 * ml.sum() / m.sum()))
    print()
    # ---- ply-0 exact floor, and the randomised n_rand contrast
    m0 = ply == 0
    print("PLY 0 (exact, 36 lead-pair classes, %d games):" % int(m0.sum()))
    r0 = floor_level(d["k_E"][m0], gid[m0], y[m0], base=base)
    lo, hi = boot(r0["ce_pos"], gid[m0], ngames)
    print("   LOO CE = %.4f nats [%.4f, %.4f]  plug-in H = %.4f  MM H = %.4f  (base %.4f)"
          % (r0["loo_all"], lo, hi, r0["plug_est"], r0["mm_est"], b["loo_all"]))
    print("   randomised forced-random-opening contrast (n_rand is drawn per game, "
          "independently of everything):")
    print("   | n_rand | games | ply-0 LOO CE | plug-in H | MM H | win rate |")
    print("   |---|---|---|---|---|---|")
    for nr in (0, 1, 2, 3):
        mm = m0 & (d["nrand"] == nr)
        if mm.sum() < 100:
            continue
        rr = floor_level(d["k_E"][mm], gid[mm], y[mm], base=base)
        print("   | %d | %d | %.4f | %.4f | %.4f | %.4f |"
              % (nr, int(mm.sum()), rr["loo_all"], rr["plug_est"], rr["mm_est"],
                 float(y[mm].mean())))
    print()
    # ---- exploration decomposition
    # suffix count of exploratory decisions inside each game
    order = np.lexsort((ply, gid))
    ex = (tag != 0).astype(np.int64)[order]
    g_o = gid[order]
    csum = np.cumsum(ex)
    starts = np.searchsorted(g_o, np.arange(ngames), "left")
    ends = np.searchsorted(g_o, np.arange(ngames), "right")
    tot = np.zeros(len(ex))
    tot[order] = csum[ends[g_o] - 1] - csum + ex          # exploratory at ply>=this
    clean = tot == 0
    Lrem = glen - ply
    print("EXPLORATION DECOMPOSITION")
    print("tag mix: argmax %.1f%%  rand %.1f%%  temp %.1f%%  eps %.1f%%"
          % tuple(100 * (tag == t).mean() for t in (0, 1, 2, 3)))
    print("positions with a fully greedy (argmax-only) remainder: %d / %d (%.2f%%); "
          "%d games are greedy end-to-end" % (int(clean.sum()), len(clean), 100 * clean.mean(),
                                              int(np.bincount(gid[clean & (ply == 0)],
                                                              minlength=ngames).sum())))
    print()
    print("(a) ONE-STEP contrast, unbiased: inside each L4 class, split positions by whether")
    print("    THIS decision was argmax or eps. The eps coin is independent of the state,")
    print("    so this is a randomised comparison. Restricted to ply >= 9 (past rand+temp).")
    late = ply >= 9
    print("   | subset | positions | LOO CE | win rate |")
    print("   |---|---|---|---|")
    for lab, msk in (("this decision argmax", late & (tag == 0)),
                     ("this decision eps", late & (tag == 3))):
        rr = floor_level(d["k_L4"][msk], gid[msk], y[msk], base=base)
        lo, hi = boot(rr["ce_pos"], gid[msk], ngames)
        print("   | %s | %d | %.4f [%.4f, %.4f] | %.4f |"
              % (lab, int(msk.sum()), rr["loo_all"], lo, hi, float(y[msk].mean())))
    print()
    print("(b) WHOLE-TAIL contrast at matched remaining length L (selection-biased: see report)")
    print("   | L band | all: n / LOO CE | greedy-tail: n / LOO CE |")
    print("   |---|---|---|")
    for a_, b_ in ((1, 3), (4, 7), (8, 12), (13, 20), (21, 30), (31, 999)):
        mb = (Lrem >= a_) & (Lrem <= b_)
        mc = mb & clean
        if mc.sum() < 500:
            continue
        ra = floor_level(d["k_L4"][mb], gid[mb], y[mb], base=base)
        rc = floor_level(d["k_L4"][mc], gid[mc], y[mc], base=base)
        print("   | %d-%d | %d / %.4f | %d / %.4f |"
              % (a_, b_, int(mb.sum()), ra["loo_all"], int(mc.sum()), rc["loo_all"]))
    print()
    mc = clean
    rc = floor_level(d["k_L4"][mc], gid[mc], y[mc], base=base)
    lo, hi = boot(rc["ce_pos"], gid[mc], ngames)
    print("   pooled greedy-tail L4 floor: n=%d  LOO CE=%.4f [%.4f, %.4f]  (mean L=%.1f vs %.1f overall)"
          % (int(mc.sum()), rc["loo_all"], lo, hi, float(Lrem[mc].mean()), float(Lrem.mean())))


if __name__ == "__main__":
    main()
