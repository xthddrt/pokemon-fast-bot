"""ENCODER VALUE TEST -- the trivial baselines, on the SAME split.

  1. constant           : the train-split mean of `label_p`
  2. hp-diff logistic   : sigmoid(a * d + b), d = sum(our hp_frac) - sum(their
                          hp_frac), a and b fitted by BCE on train. One line of
                          arithmetic.
  3. hp-diff binned     : the nonparametric ceiling of the SAME single number --
                          40 quantile bins of d, predicting the train-bin mean.
                          Anything the nets earn above this is something other
                          than counting HP.
  4. search / recal     : the generating search's own root value `q_search`,
                          raw and after a 40-bin monotone recalibration fitted
                          on train (reproduces the two published bars on THIS
                          split).
"""
import json

import numpy as np

import vt_lib as V

BANDS = V.BANDS


def brier(pred, y):
    return float(np.mean((pred - y) ** 2))


def by_band(pred, y, band):
    return {"all": brier(pred, y), **{b: brier(pred[band == b], y[band == b]) for b in BANDS}}


def fit_logistic1(x, y, iters=400, lr=0.5):
    a, b = 0.0, float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        r = p - y
        a -= lr * float((r * x).mean())
        b -= lr * float(r.mean())
    return a, b


def binned(xt, yt, xe, nb=40):
    q = np.quantile(xt, np.linspace(0, 1, nb + 1))
    q[0], q[-1] = -np.inf, np.inf
    q = np.unique(q)
    it = np.clip(np.digitize(xt, q[1:-1]), 0, len(q) - 2)
    ie = np.clip(np.digitize(xe, q[1:-1]), 0, len(q) - 2)
    m = np.array([yt[it == k].mean() if (it == k).any() else yt.mean() for k in range(len(q) - 1)])
    return m[ie]


def main():
    meta = V.load_meta()
    ix, _ = V.split_idx(meta)
    y, band = meta["label_p"], meta["band"]
    feats = np.load(V.os.path.join(V.ENC, "enc2_feats.npy"), mmap_mode="r")

    import enc2
    names = enc2.DEFAULT_LAYOUT.names
    hp = [names.index("mon%d.hp_frac" % i) for i in range(12)]
    d = (np.asarray(feats[:, hp[:6]], np.float64).sum(1)
         - np.asarray(feats[:, hp[6:]], np.float64).sum(1))

    tr, te = ix["train"], ix["test"]
    out = {}
    c = float(y[tr].mean())
    out["constant"] = by_band(np.full(len(te), c), y[te], band[te])
    a, b = fit_logistic1(d[tr], y[tr])
    out["hp_diff_logistic"] = by_band(1 / (1 + np.exp(-(a * d[te] + b))), y[te], band[te])
    out["hp_diff_logistic_coef"] = {"a": a, "b": b}
    out["hp_diff_binned40"] = by_band(binned(d[tr], y[tr], d[te]), y[te], band[te])

    q = meta["q"]
    ok = ~np.isnan(q)
    tq, eq = tr[ok[tr]], te[ok[te]]
    out["search_raw"] = by_band(q[eq], y[eq], band[eq])
    out["search_recal40"] = by_band(binned(q[tq], y[tq], q[eq]), y[eq], band[eq])
    out["label_noise_floor_se2"] = {"all": float(np.mean(meta["se"][te] ** 2)),
                                    **{b: float(np.mean(meta["se"][te][band[te] == b] ** 2))
                                       for b in BANDS}}
    n = meta["npl"][te].astype(float)
    ph = y[te]
    out["label_noise_floor_binomial"] = {
        "all": float(np.mean(ph * (1 - ph) / n)),
        **{b: float(np.mean((ph * (1 - ph) / n)[band[te] == b])) for b in BANDS}}
    out["n_test"] = int(len(te))
    print(json.dumps(out, indent=1))
    json.dump(out, open("/tmp/vt/trivial.json", "w"))


if __name__ == "__main__":
    main()
