# evallab — a controlled bench for value-net changes

**What it is for.** Proving or killing an encoder/loss idea for a few dollars and
a couple of hours, instead of $30 and a day on the 150M-row pipeline. It answers
one question the full pipeline cannot: *does this change make the net choose
better moves?* — measured directly, against a deep-search oracle, on a corpus
small enough to re-generate at will.

**The controlled variables.** One fixed team pair per corpus, full information
on both sides, a fixed iteration budget per search. No determinizer, no hidden
information sampler, no team distribution. Anything that moves is the value
function.

---

## Pipeline

```
labteams.py      the three fixed pairs (A primary, B half-shared, C disjoint)
generate.py      fixed-pair full-info self-play -> shards (state, visits, avg_scores, outcome)
stats.py         corpus diversity report (coarse positions, arm coverage, lead pairs)
oracle.py        5M-iteration searches on held-out positions -> the evaluation standard
dataset.py       shards -> pos.npz (roots + outcomes) and sib.npz (sibling successors + q)
relfeat.py       the relational feature blocks (gx: 16 global, sx: 16 per side)
test_relfeat.py  hand-checkable correctness gate for relfeat  <- run this after ANY edit
labmodel.py      the net; REL=0 is bit-exact vs valuenet/train.py:ValueNet (asserted)
train_lab.py     4-cell trainer: REL={0,1} x RANK_W={0,>0}
evaluate.py      value CE + SIBLING DISCRIMINATION + transfer gap
run_all.sh       local end-to-end driver (canary or full)
cloud/           one self-terminating spot box that does generation + oracle
```

### The metric that matters

`evaluate.py`'s **sibling discrimination**. For each oracle decision, build one
successor state per legal arm (modal branch, against the opponent's oracle-best
reply), score all of them with the net, and compare that ranking to the oracle's
per-arm values:

| metric | reads as |
|---|---|
| `spread` | sd of the net's win probability across siblings. **If this is ~0 the net cannot choose a move at all**, whatever its CE says. |
| `top1` / `top3` | net's argmax is the oracle's best / in the oracle's top 3 |
| `spearman` | rank agreement with the oracle's per-arm values |
| `regret` | win probability lost per decision by picking with this net instead of the oracle. **The bottom line.** |
| `random_regret` | the same for a uniformly random arm — the floor a net must beat |
| `oracle_gap` | oracle best minus oracle worst; how much was on the table |

Value CE is reported too, but the whole point of the lab is that CE and regret
are different questions and the pipeline has only ever measured the first.

### The relational features (`relfeat.py`)

They go in the **side and global blocks, never the per-mon block** — per-mon
features are pushed through a shared MLP and then `sum`-pooled, which is exactly
the operation that destroys a relation.

* `gx` (16, global, between the two actives): signed effective-speed margin,
  who-moves-first / speed-tie bits, best offensive type multiplier each way,
  best STAB multiplier each way, best damage as a fraction of current HP each
  way, KO bits each way, KO-first bits, KO-race bits.
* `sx` (16, per side): hazard switch-in cost per bench slot, best offensive type
  multiplier of each bench mon into the opposing active, an outspeed bit per
  bench mon, and a scalar bench-answer summary.

Both are laid out so a side swap is a column permutation plus one sign flip
(`relfeat.gx_swap`).

---

## Running it

Local canary (about 4 minutes on 4 cores):

```bash
./run_all.sh canary /tmp/elcanary
```

Full run — generation and the oracle are cloud work, everything else is local:

```bash
# 1. bundle + launch (one c7a.16xlarge spot, self-terminating, ~$1)
tar czf /tmp/code.tar.gz --exclude=.venv --exclude=.git --exclude=logs \
    --exclude=__pycache__ --exclude=target --exclude=wheelout --exclude=dist \
    foul-play/fp foul-play/data foul-play/config.py foul-play/constants.py \
    foul-play/requirements.txt foul-play/teams poke-engine/src poke-engine/data \
    poke-engine/Cargo.toml poke-engine/Cargo.lock poke-engine/poke-engine-py \
    valuenet/encoder.py valuenet/train.py valuenet/vocab.json valuenet/max_pp.json \
    valuenet/m4_artifacts/valuenet_v6nopol.bin \
    valuenet/m4_artifacts/valuenet_v6nopol.constants.json evallab
BUNDLE=/tmp/code.tar.gz evallab/cloud/launch.sh <run> c7a.16xlarge

# 2. the box uploads a 6-game CANARY within ~10 minutes, before the bulk run.
#    Verify it parses with the LOCAL engine before trusting the rest:
aws s3 sync s3://.../evallab/<run>/out/ evallab/data/<run>/
python evallab/dataset.py 'evallab/data/<run>/canary/shard_A_*.jsonl.gz' /tmp/probe 100

# 3. when FINISHED appears, sync and run the local half
./run_all.sh full evallab/data/<run>
```

## Assumptions and limitations, stated

* **Root Dirichlet noise is not available** — it lives in the Rust search. The
  python-side substitute is forced-random opening plies + visit-temperature +
  epsilon-greedy + a randomised lead. `stats.py` reports what that achieved.
* **Successor states use the modal branch** against the opponent's single best
  reply. The oracle's `q` is a root Q marginalised over the opponent's root
  strategy, so the two are not the same object; using ONE construction at train
  and eval time is what makes the cells comparable.
* **No side-swap augmentation** in any cell (train.py uses it). Turning it on
  for some cells and not others would confound the comparison.
* **KO threat uses the max non-crit roll** and holds the opponent's move fixed
  at its first move; see the header of `relfeat.py` for the full approximation
  list.
* **Pair A is not balanced** — side one wins about 30% of games. Base rates are
  reported alongside every CE so absolute CE is never read as if it were.
