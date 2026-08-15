"""LAMBDA-RETURN PRETRAINING EXTRACTOR over the v6 self-play shards (r4).

Emits (state_string, target, weight, metadata) rows for value-net pretraining.
One output file per input shard, so a spot reclaim costs exactly one shard.

--------------------------------------------------------------------------
WHY A LAMBDA-RETURN AND NOT RAW `v`
--------------------------------------------------------------------------
`v` (the MCTS root value at 6000-7000 iterations) is well calibrated in
aggregate -- mean |calibration error| 0.0278 over 89,868 turns -- but it has
almost NO early-game discrimination: with >20 turns left its AUC against the
eventual result is 0.573, and mean v is 0.534 on games that were eventually
won vs 0.498 on games eventually lost.  Late (<=5 turns left) it is nearly
perfect: AUC 0.988-0.999.

So training on raw `v` would teach the net to reproduce a near-coin-flip in
exactly the region where a value net has to earn its keep.  Training on the
raw outcome (lambda=1) keeps all the signal but carries full Bernoulli
variance everywhere, including late positions where `v` already knows the
answer.  The lambda-return interpolates, and the right lambda is
PHASE-DEPENDENT.

MEASURED signal-to-noise (separation between eventually-won and
eventually-lost, divided by the target's own std) of the lambda-return, by
phase -- this table is the sole justification for LAMBDA_BY_PHASE below:

    turns_left  | lam=0.0  0.8   0.9   0.95   1.0
    ------------+---------------------------------
    >20         |   0.26  0.56  1.12  1.70   2.11
    11-20       |   0.84  1.55  1.86   -     2.02
    6-10        |   1.50  1.91   -     -     2.02
    <=5         |   1.93  2.01   -     -     2.02

Rule applied: pick the LOWEST lambda that still retains >=95% of the lam=1.0
SNR in that band (linear interpolation between measured points).  Lower
lambda is strictly preferable when it is free, because it bootstraps on a
6000-iteration search instead of a single Bernoulli sample and therefore
lowers target variance.

    >20    : 0.95 keeps 1.70/2.11 = 81%  -> too expensive. lambda = 1.00
    11-20  : 0.90 keeps 1.86/2.02 = 92%; ~0.95 interpolates to ~96%
                                          -> lambda = 0.95
    6-10   : 0.80 keeps 1.91/2.02 = 95%; 0.85 interpolates to ~96-97%
                                          -> lambda = 0.85
    <=5    : 0.80 keeps 2.01/2.02 = 99.5%; 0.70 interpolates to ~99%
                                          -> lambda = 0.70

Recursion (terminal-reward-only episodes):
    G_t = (1 - lam) * v_{t+1} + lam * G_{t+1},     G_{T-1} = outcome

WHICH lam GOES IN THE RECURSION -- a real design decision, MEASURED
The obvious reading, "use lam_t = lambda_for(turns_left(t)) at each step of a
single backward pass", is WRONG and was measured to be wrong here.  An early
position's return flows through every later step, so a schedule that is 1.00
early but 0.70 late still contaminates the early target with the late
lambdas: the outcome's coefficient at an early position came out at
    0.95^10 * 0.85^5 * 0.70^5 = 0.0446
i.e. the "lambda = 1.0" early target was only 4.5% outcome, and its measured
separation was 1.751 -- 83% of the 2.11 the table promises, exactly the loss
we rejected when we declined lambda=0.95 early.

So each position gets its OWN return computed with a CONSTANT lambda over the
whole tail: lam = lambda_for(turns_left(t)), applied at every step from t to
the end.  That is what the SNR table was measured under, so it delivers the
numbers the table promises.  Cost is one backward pass per distinct lambda
(four), i.e. O(4T) per game -- negligible at T~46.

--------------------------------------------------------------------------
POSITION SELECTION
--------------------------------------------------------------------------
Games average ~46 decisions.  Taking all of them would blow the storage
budget (7,828 bytes/state encoded) and the rows inside one game are heavily
correlated anyway.  We take POS_PER_GAME per game:

  * STRATIFIED across the game.  The decision indices are split into
    POS_PER_GAME contiguous strata of equal length and exactly one position
    is drawn from each.  This makes "all four from the endgame" structurally
    impossible, which a purely crux-weighted sample would otherwise produce
    (crux spikes late).
  * WITHIN a stratum, drawn with probability proportional to
        score_t = CRUX_FLOOR + (1 - CRUX_FLOOR) * min(1, crux_t / CRUX_SCALE)
    where crux_t = |v_{t+1} - v_t| is the measured swing the search saw.
    CRUX_FLOOR is the explicit floor that guarantees quiet positions keep a
    share of the mass: at CRUX_FLOOR=0.35 a completely quiet turn still has
    35% of the sampling weight of a maximal-crux turn.  Without it the set
    becomes a diet of pure crises and the net never learns what an ordinary
    position looks like -- and ordinary positions are most MCTS leaves.

--------------------------------------------------------------------------
WEIGHT  =  w_var  x  w_region   (two separate, separately tunable terms)
--------------------------------------------------------------------------
w_var -- INVERSE VARIANCE.  Expand G_t: the eventual outcome enters with
coefficient c_t = prod_{k=t..T-2} lam_k (c_{T-1} = 1); the remaining mass
sits on `v` bootstraps, which are 6000-iteration searches and far less noisy
than one Bernoulli draw.  So Var(G_t) ~ c_t^2 * p(1-p), giving an effective
sample count N_eff = 1/c_t^2, and

    w_var  =  N_eff / (p * (1-p))          p = clip(target, PMIN, 1-PMIN)

Both factors are landmines unclipped:
  * 1/c_t^2 reaches ~500 on a long game (0.95^10 * 0.85^5 * 0.7^5 = 0.045).
    It is only that small because `v` is treated as noiseless, which it is
    not.  Its own residual error std is roughly SIGMA_V ~ 0.15 (REASONED,
    not measured -- calibration error is 0.0278 but that is a bias measure,
    not a per-position error).  That bounds the true gain at
    p(1-p)/SIGMA_V^2 = 0.25/0.0225 = 11.  We cap conservatively at
    NEFF_CAP = 4.0.
  * 1/(p(1-p)) diverges at decided positions; PMIN = 0.05 bounds it at
    1/0.0475 = 21x.

w_region -- IMPORTANCE, deliberately fighting w_var.  Inverse-variance
weighting is correct for estimating a conditional mean and wrong for a game
evaluator: it pours weight onto already-decided positions, which are both
easy and strategically worthless.  Two factors:
  * w_contest = CONTEST_FLOOR + (1-CONTEST_FLOOR) * 4p(1-p)
    -> 1.0 at p=0.5, CONTEST_FLOOR at p in {0,1}.  Partially cancels the
    inverse-variance blowup on decided positions without erasing them.
  * w_phase = EARLY_BOOST for turns_left > 20, else 1.0.  That is exactly
    the band where `v` is uninformative (AUC 0.573) and therefore the band
    where the net has something to add.

w_var is normalised at its MODAL operating point (N_eff at the cap, p=0.5) so
a typical row weighs 1.0, then w_var * w_region is hard-clipped to
[WMIN, WMAX] = [0.25, 4.0] -- a bounded 16x dynamic range.
Normalising at N_eff=1 instead was MEASURED to pin 97.5% of rows to WMAX,
i.e. a constant weight; at the modal point the achieved spread is
p10 0.94 / p50 0.98 / p90 2.13, ESS/n = 0.866, nothing at either clip.

--------------------------------------------------------------------------
EXCLUSIONS
--------------------------------------------------------------------------
  * r4x is the EVAL family and is refused outright (ALLOWED_PREFIXES).
  * r5 / r5x / r7 shard objects are truncated; excluded for now.
  * Any game whose team-pair hash appears in the fine-tune set
    (s3://<bucket>/evallab/plc1/positions.pairs.gz) is dropped, so pretrain
    and fine-tune share no team pairs.  The hash is `corpus_select.pair_hash`
    -- the exact function that produced that file -- imported, not
    reimplemented.  MEASURED on shard_0000000: 989/1953 games = 50.6% of r4
    is excluded by this, so the eligible pool is ~988k of r4's 1,999,872
    games.  At POS_PER_GAME=4 that is ~3.95M positions; reaching 8-10M
    requires POS_PER_GAME 8-10 (still <25% of the ~46 decisions per game).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python pretrain_select.py run <outdir> [seed] [shard_limit]
    python pretrain_select.py stats <outdir_or_file.jsonl.gz>   # verification

ENV
    BUCKET        default pokebot-valuenet-389825051723
    PREFIX        default r4          (r4x is refused)
    PT_WORKERS    process pool size; default max(1, cpu_count//2)
    POS_PER_GAME  default 4
    EXCLUDE_PAIRS local path to the pairs file (.gz or plain).  If unset it
                  is downloaded from S3 once into the outdir.
    NO_EXCLUDE=1  skip the pair exclusion entirely (diagnostics only)
    OUT_S3        optional s3://bucket/prefix to upload each finished shard
"""

import concurrent.futures as cf
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import boto3
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_select import read_shard, pair_hash   # noqa: E402  (exact same hash)

BUCKET = os.environ.get("BUCKET", "pokebot-valuenet-389825051723")
PREFIX = os.environ.get("PREFIX", "r4")
ALLOWED_PREFIXES = ("r4",)          # r4x is the eval family. r5/r5x/r7 truncated.
PAIRS_KEY = "evallab/plc1/positions.pairs.gz"

# ---------------------------------------------------------------- tunables
POS_PER_GAME = int(os.environ.get("POS_PER_GAME", "4"))

CRUX_FLOOR = 0.35        # quiet positions keep this share of the sampling mass
CRUX_SCALE = 0.20        # |dv| at which a position is "maximally interesting"

PMIN = 0.05              # p clip for the inverse-variance term
NEFF_CAP = 4.0           # cap on 1/c^2 (v is not noiseless; see header)
CONTEST_FLOOR = 0.35     # region weight floor at p in {0,1}
EARLY_BOOST = 1.5        # region weight multiplier for turns_left > 20
WMIN, WMAX = 0.25, 4.0   # final hard clip on the weight

# turns_left bands, high to low, matching the measured SNR table exactly.
LAMBDA_BY_PHASE = ((20, 1.00), (10, 0.95), (5, 0.85), (-1, 0.70))


def lambda_for(turns_left):
    """THE tunable lambda schedule.  turns_left = (T-1) - t.

    Piecewise-constant over exactly the bands the SNR table was measured on;
    no extrapolation beyond measured support.  See module header for the
    >=95%-of-lam=1.0-SNR rule that picked each value.
    """
    for lo, lam in LAMBDA_BY_PHASE:
        if turns_left > lo:
            return lam
    return LAMBDA_BY_PHASE[-1][1]


PHASE_BANDS = (("gt20", 21, 10 ** 9), ("11-20", 11, 21),
               ("6-10", 6, 11), ("le5", 0, 6))


def phase_of(turns_left):
    for name, lo, hi in PHASE_BANDS:
        if lo <= turns_left < hi:
            return name
    return "le5"


# ---------------------------------------------------------------- targets
def _pass(v, T, outcome, L):
    """One backward lambda-return pass at CONSTANT lambda L."""
    G = [0.0] * T
    c = [0.0] * T
    G[T - 1] = float(outcome)
    c[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        Lt, vn = L, v[t + 1]
        if vn is None:
            Lt, vn = 1.0, G[t + 1]   # no bootstrap -> that step is pure MC
        G[t] = (1.0 - Lt) * vn + Lt * G[t + 1]
        c[t] = Lt * c[t + 1]
    return G, c


def lambda_returns(turns, outcome):
    """-> (G, lam, c, crux, v) lists, one entry per decision index.

    Each position uses a return computed with a CONSTANT lambda =
    lambda_for(turns_left(t)) over its whole tail -- see the module header for
    why a single mixed-lambda backward pass is wrong.
    c_t = coefficient of `outcome` in the expansion of G_t.
    crux_t = |v_{t+1} - v_t|, with the eventual outcome standing in for v_T
    at the final decision.
    """
    T = len(turns)
    v = [None] * T
    for i, t in enumerate(turns):
        x = t.get("v")
        if x is not None:
            v[i] = float(x)

    passes = {L: _pass(v, T, outcome, L) for L in {l for _, l in LAMBDA_BY_PHASE}}

    G = [0.0] * T
    lam = [0.0] * T
    c = [0.0] * T
    for t in range(T):
        L = lambda_for((T - 1) - t)
        lam[t] = L
        G[t], c[t] = passes[L][0][t], passes[L][1][t]

    crux = [0.0] * T
    for t in range(T):
        if v[t] is None:
            crux[t] = 0.0
            continue
        nxt = v[t + 1] if (t + 1 < T and v[t + 1] is not None) else float(outcome)
        crux[t] = abs(nxt - v[t])
    return G, lam, c, crux, v


# ---------------------------------------------------------------- weights
# Normalise w_var at its MODAL operating point (N_eff at the cap, p=0.5) so a
# typical row weighs 1.0.  Normalising at N_eff=1 instead put ~everything above
# WMAX -- MEASURED: 97.5% of rows pinned to the clip, i.e. a constant weight.
_W_REF = NEFF_CAP / 0.25


def weight_of(target, c, turns_left):
    """-> (w_final, w_var_norm, w_region).  Both terms kept separate."""
    p = min(1.0 - PMIN, max(PMIN, float(target)))
    n_eff = min(NEFF_CAP, 1.0 / max(1e-9, c * c))
    w_var = (n_eff / (p * (1.0 - p))) / _W_REF

    w_contest = CONTEST_FLOOR + (1.0 - CONTEST_FLOOR) * 4.0 * p * (1.0 - p)
    w_phase = EARLY_BOOST if turns_left > 20 else 1.0
    w_region = w_contest * w_phase

    return min(WMAX, max(WMIN, w_var * w_region)), w_var, w_region


# ---------------------------------------------------------------- selection
def pick_indices(scores, k, rng):
    """Exactly one index per contiguous stratum, crux-weighted inside it."""
    T = len(scores)
    if T <= k:
        return list(range(T))
    out = []
    for j in range(k):
        lo = (j * T) // k
        hi = ((j + 1) * T) // k
        if hi <= lo:
            hi = lo + 1
        seg = list(range(lo, min(hi, T)))
        tot = sum(scores[i] for i in seg)
        if tot <= 0:
            out.append(rng.choice(seg))
            continue
        r = rng.random() * tot
        acc = 0.0
        for i in seg:
            acc += scores[i]
            if r <= acc:
                out.append(i)
                break
        else:
            out.append(seg[-1])
    return out


# ---------------------------------------------------------------- per shard
def _s3():
    return boto3.client("s3", config=Config(
        max_pool_connections=32, retries={"max_attempts": 10, "mode": "adaptive"}))


def process_shard(args):
    key, outdir, seed, exclude_path, pos_per_game = args
    name = os.path.basename(key).replace(".jsonl.gz", "")
    out_path = os.path.join(outdir, name + ".pt.jsonl.gz")
    done_path = out_path + ".done"
    if os.path.exists(done_path):                     # resume: shard already done
        try:
            return json.load(open(done_path)), True
        except Exception:
            pass

    exclude = set()
    if exclude_path:
        op = gzip.open if exclude_path.endswith(".gz") else open
        with op(exclude_path, "rt") as f:
            exclude = {l.strip() for l in f if l.strip()}

    if key.startswith("/"):
        body = open(key, "rb").read()
    else:
        body = _s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()

    # Deterministic per shard, independent of listing order and worker order.
    # (PYTHONHASHSEED-independent: derived from the shard name bytes, not hash().)
    rng = random.Random((int.from_bytes(name.encode(), "little") % (1 << 61))
                        ^ (seed * 1_000_003))

    st = Counter()
    rows = []
    for line in read_shard(body).splitlines():
        try:
            g = json.loads(line)
        except Exception:
            st["bad_lines"] += 1
            continue
        if g.get("kind") == "header":
            continue
        if "error" in g or not g.get("turns"):
            st["skip_empty"] += 1
            continue
        st["games"] += 1
        turns = g["turns"]
        if len(turns) < 2:
            st["skip_short"] += 1
            continue
        ph = pair_hash(turns[0]["s"])
        if ph in exclude:
            st["skip_pair_excluded"] += 1
            continue
        st["games_used"] += 1

        outcome = float(g["outcome"])
        G, lam, c, crux, v = lambda_returns(turns, outcome)
        scores = [CRUX_FLOOR + (1.0 - CRUX_FLOOR) * min(1.0, x / CRUX_SCALE)
                  for x in crux]
        T = len(turns)
        for t in pick_indices(scores, pos_per_game, rng):
            tl = (T - 1) - t
            w, wv, wr = weight_of(G[t], c[t], tl)
            rows.append({
                "s": turns[t]["s"],
                "y": round(G[t], 6),
                "w": round(w, 5),
                "g": "%s/%s/%s" % (PREFIX, name, g.get("file")),
                "t": t, "T": T, "tl": tl,
                "lam": round(lam[t], 4), "c": round(c[t], 6),
                "v": None if v[t] is None else round(v[t], 5),
                "crux": round(crux[t], 5),
                "out": outcome,
                "wv": round(wv, 5), "wr": round(wr, 5),
                "pair": ph, "src": PREFIX,
            })
    st["rows"] = len(rows)

    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, out_path)

    stats = dict(st)
    stats["shard"] = name
    stats["bytes_gz"] = os.path.getsize(out_path)
    stats["rows_per_game_used"] = round(len(rows) / max(1, st["games_used"]), 3)
    with open(done_path, "w") as f:
        json.dump(stats, f)

    out_s3 = os.environ.get("OUT_S3")
    if out_s3:
        b, _, p = out_s3[5:].partition("/")
        cli = _s3()
        cli.upload_file(out_path, b, "%s/%s" % (p.rstrip("/"), os.path.basename(out_path)))
        cli.put_object(Bucket=b, Key="%s/%s" % (p.rstrip("/"), os.path.basename(done_path)),
                       Body=json.dumps(stats).encode())
    return stats, False


# ---------------------------------------------------------------- driver
def list_shards():
    assert PREFIX in ALLOWED_PREFIXES, (
        "prefix %r refused. r4x is the EVAL family and must never be used for "
        "pretraining; r5/r5x/r7 objects are truncated." % PREFIX)
    cli = _s3()
    keys, tok = [], None
    pfx = "%s/shards_%s/" % (PREFIX, PREFIX)
    while True:
        kw = {"Bucket": BUCKET, "Prefix": pfx}
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])
                 if o["Key"].endswith(".jsonl.gz")]
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    keys.sort()
    assert keys, "no shards under s3://%s/%s" % (BUCKET, pfx)
    return keys


def ensure_pairs(outdir):
    if os.environ.get("NO_EXCLUDE") == "1":
        return None
    p = os.environ.get("EXCLUDE_PAIRS")
    if p and os.path.exists(p):
        return p
    p = os.path.join(outdir, "plc1.positions.pairs.gz")
    if not os.path.exists(p):
        _s3().download_file(BUCKET, PAIRS_KEY, p)
    return p


def cmd_run(argv):
    outdir = argv[0]
    seed = int(argv[1]) if len(argv) > 1 else 0
    limit = int(argv[2]) if len(argv) > 2 else 0
    os.makedirs(outdir, exist_ok=True)
    pairs = ensure_pairs(outdir)

    keys = list_shards()
    if limit:
        keys = keys[:limit]
    workers = int(os.environ.get("PT_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
    print("prefix=%s shards=%d workers=%d pos/game=%d seed=%d exclude=%s"
          % (PREFIX, len(keys), workers, POS_PER_GAME, seed, pairs), flush=True)

    tasks = [(k, outdir, seed, pairs, POS_PER_GAME) for k in keys]
    agg, t0, cached = Counter(), time.time(), 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (st, was_cached) in enumerate(ex.map(process_shard, tasks, chunksize=1)):
            cached += was_cached
            for k, v in st.items():
                if isinstance(v, (int, float)) and k != "rows_per_game_used":
                    agg[k] += v
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print("  %d/%d rows=%d cached=%d %.0fs"
                      % (i + 1, len(tasks), agg["rows"], cached, time.time() - t0),
                      flush=True)
    agg = dict(agg)
    agg["shards"] = len(keys)
    agg["cached"] = cached
    agg["rows_per_game_used"] = round(agg["rows"] / max(1, agg["games_used"]), 3)
    agg["excluded_frac"] = round(agg["skip_pair_excluded"] / max(1, agg["games"]), 4)
    agg["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2), flush=True)
    with open(os.path.join(outdir, "RUN_STATS.json"), "w") as f:
        json.dump(agg, f, indent=2)


# ---------------------------------------------------------------- stats
def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def cmd_stats(argv):
    src = argv[0]
    files = ([src] if src.endswith(".jsonl.gz")
             else sorted(os.path.join(src, f) for f in os.listdir(src)
                         if f.endswith(".pt.jsonl.gz")))
    rows = []
    for fp in files:
        with gzip.open(fp, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    n = len(rows)
    print("files=%d rows=%d" % (len(files), n))
    games = len({r["g"] for r in rows})
    print("distinct games=%d  rows/game=%.3f  distinct pairs=%d"
          % (games, n / max(1, games), len({r["pair"] for r in rows})))
    print("mean state bytes=%.0f  (encoded 7828 B/state -> %.2f GB per 1M rows)"
          % (sum(len(r["s"]) for r in rows) / n, 7828 / 1e9 * 1e6))

    print("\n-- lambda actually applied, by phase --")
    print("%-8s %8s %8s %8s %8s %8s" % ("phase", "rows", "lam_mean", "c_mean",
                                        "tl_mean", "frac"))
    byp = defaultdict(list)
    for r in rows:
        byp[phase_of(r["tl"])].append(r)
    for name, _, _ in PHASE_BANDS:
        g = byp.get(name, [])
        if not g:
            print("%-8s %8d" % (name, 0)); continue
        print("%-8s %8d %8.4f %8.4f %8.2f %8.3f"
              % (name, len(g), sum(x["lam"] for x in g) / len(g),
                 sum(x["c"] for x in g) / len(g),
                 sum(x["tl"] for x in g) / len(g), len(g) / n))

    ys = [r["y"] for r in rows]
    mu = sum(ys) / n
    sd = math.sqrt(sum((y - mu) ** 2 for y in ys) / n)
    print("\n-- target y --")
    print("mean=%.4f sd=%.4f min=%.4f max=%.4f" % (mu, sd, min(ys), max(ys)))
    print("p01=%.4f p10=%.4f p50=%.4f p90=%.4f p99=%.4f"
          % tuple(_pct(ys, q) for q in (0.01, 0.10, 0.50, 0.90, 0.99)))
    hb = Counter(min(9, int(y * 10)) for y in ys)
    print("histogram(0.1 bins): " + " ".join(
        "%.1f:%.3f" % (b / 10, hb.get(b, 0) / n) for b in range(10)))
    print("exactly 0 or 1: %.4f" % (sum(1 for y in ys if y <= 0 or y >= 1) / n))

    ws = [r["w"] for r in rows]
    wm = sum(ws) / n
    print("\n-- weight w (after clip to [%.2f,%.2f]) --" % (WMIN, WMAX))
    print("mean=%.4f sd=%.4f min=%.4f max=%.4f  ess/n=%.4f"
          % (wm, math.sqrt(sum((w - wm) ** 2 for w in ws) / n), min(ws), max(ws),
             sum(ws) ** 2 / (n * sum(w * w for w in ws))))
    print("p01=%.3f p10=%.3f p50=%.3f p90=%.3f p99=%.3f"
          % tuple(_pct(ws, q) for q in (0.01, 0.10, 0.50, 0.90, 0.99)))
    print("at WMAX: %.4f   at WMIN: %.4f"
          % (sum(1 for w in ws if w >= WMAX - 1e-9) / n,
             sum(1 for w in ws if w <= WMIN + 1e-9) / n))
    wv = [r["wv"] for r in rows]; wr = [r["wr"] for r in rows]
    print("w_var  mean=%.3f p99=%.3f max=%.3f" % (sum(wv) / n, _pct(wv, .99), max(wv)))
    print("w_reg  mean=%.3f p01=%.3f min=%.3f" % (sum(wr) / n, _pct(wr, .01), min(wr)))

    print("\n-- SEPARATION of target between eventually-won and eventually-lost --")
    print("   (mean_won - mean_lost) / sd_within_band ; ties (out=0.5) dropped")
    print("%-8s %8s %8s %8s %8s %10s %10s"
          % ("phase", "n", "won", "lost", "sd", "SEP", "ref_lam1"))
    ref = {"gt20": 2.11, "11-20": 2.02, "6-10": 2.02, "le5": 2.02}
    for name, _, _ in PHASE_BANDS:
        g = [r for r in byp.get(name, []) if r["out"] != 0.5]
        if len(g) < 20:
            print("%-8s %8d  (too few)" % (name, len(g))); continue
        a = [r["y"] for r in g if r["out"] == 1.0]
        b = [r["y"] for r in g if r["out"] == 0.0]
        allv = [r["y"] for r in g]
        m = sum(allv) / len(allv)
        s = math.sqrt(sum((y - m) ** 2 for y in allv) / len(allv))
        ma, mb = sum(a) / max(1, len(a)), sum(b) / max(1, len(b))
        print("%-8s %8d %8.4f %8.4f %8.4f %10.3f %10.2f"
              % (name, len(g), ma, mb, s, (ma - mb) / max(1e-9, s), ref[name]))

    print("\n-- crux of selected positions --")
    cx = [r["crux"] for r in rows]
    print("selected crux mean=%.4f p50=%.4f p90=%.4f"
          % (sum(cx) / n, _pct(cx, .5), _pct(cx, .9)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": cmd_run, "stats": cmd_stats}[cmd](sys.argv[2:])
