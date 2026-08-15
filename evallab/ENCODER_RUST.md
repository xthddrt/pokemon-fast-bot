# Encoder 2 in Rust — parity proof, per-block cost, and the answer on the budget

*Branch `feat/enc2-rust-port` off `main`, commit `4aef447`. `main` is untouched
and no shipped ladder-path file is modified: the change is three new files plus
four lines in `src/genx/mod.rs`, all behind the existing `terastallization`
feature.*

---

## 0. Headline

| | |
|---|---|
| Columns | **1,413**, all bit-identical to `enc2.py` |
| Parity corpus | **7,524 states** — 5,000 real labelled, 1,500 pool-wide synthetic, 1,024 shared-static leaves |
| Parity result | **1,413 / 1,413 columns exactly equal, max abs diff 0.0**, embedding ids identical, on every corpus |
| **Dynamic, n = 1, serial (the search regime)** | **3.43 µs/leaf** (291,000 leaves/s) |
| **Per leaf including `State` → flat view** | **3.71 µs/leaf** (269,000 leaves/s) |
| Static, once per search | **75.9 µs** — and it never rebuilds |
| Budget | 8.40 µs/leaf → the encoder uses **44 %** of it |
| **Projected search throughput** | **~93,000 iters/s** (was 133,400) |
| **Slowdown multiple** | **1.39× – 1.46×** — Sally's ceiling is 2× |
| numpy → Rust | 1,146 µs → 3.43 µs = **334×** |

**It fits, with room.** `ENCODER_PERF.md` §5 called a straight port "a coin
flip" at 4.9 – 14.6 µs, and §11.7 narrowed it to 3.0 – 9.0 µs after the n = 1
pass. The measured answer is **3.43 µs — the optimistic end of that range** —
and neither of the two further steps that report recommended (the structural
wins of §3, the incremental design of §2.3) was needed or built.

---

## 1. What was built

Three files, all new, on `feat/enc2-rust-port`:

| File | What |
|---|---|
| `poke-engine/src/genx/enc2_tables.rs` | 1,697 lines of generated constants: 885 move rows, 320 ability rows, 239 item rows, the type chart, the boost table, every layout offset. |
| `poke-engine/src/genx/enc2.rs` | `RawState::from_state(&State)`, `Enc2Static::build` (the static half), `encode` (the dynamic half), `Enc2Scratch` (per-thread buffers, no per-leaf allocation). |
| `poke-engine/src/bin/enc2_bench.rs` | Parity dumper, n = 1 serial benchmark, and the measured strip-list ablation. |

plus, on the lab side:

| File | What |
|---|---|
| `evallab/enc2_rustgen.py` | Emits `enc2_tables.rs` **from the python encoder's own objects** (`enc2.Tables(vocab)`, `dmgtab`, `enc2.DEFAULT_LAYOUT`). |
| `evallab/enc2_rust_parity.py` | The bit-identity gate. |

**The tables are generated, not transcribed, and that is a deliberate design
decision.** Every per-move, per-ability and per-item quantity is read out of
`enc2.Tables(vocab)` and written as a Rust array indexed by the *poke-engine
enum discriminant*, so a table lookup in Rust is one array index where python
does a vocab lookup plus a gather. The consequence that matters: **table parity
is true by construction**, so a parity failure can only ever be an arithmetic
bug. A hand-transcribed 885-row move table would have made every failure
ambiguous between "wrong table" and "wrong maths", and there were failures
worth diagnosing (§2.3).

The enum *orders* are read out of the poke-engine source
(`define_enum_with_from_str!` blocks) and cross-checked against the python
side at generation time: `PokemonType`, `PokemonStatus`, `Weather`, `Terrain`
and all 107 `PokemonVolatileStatus` variants must match
`lossless_encoder`'s orders exactly, or the generator refuses to emit.

---

## 2. Parity — the acceptance gate

### 2.1 Result

`python evallab/enc2_rust_parity.py --real 5000 --fuzz 1500 --shared 512`

```
=== Python/Rust bit-identity, all 1413 columns, exact float equality ===
  [real labelled corpus          ]  5000 states  1413/1413 columns bit-identical, max|d| = 0, ids ok
  [pool-wide synthetic           ]  1500 states  1413/1413 columns bit-identical, max|d| = 0, ids ok
  [one root, shared static       ]   512 states  1413/1413 columns bit-identical, max|d| = 0, ids ok
  [+ tera/disable/PP moving      ]   512 states  1413/1413 columns bit-identical, max|d| = 0, ids ok
  [rust shared vs rust cold      ]   512 states  1413/1413 columns identical,     max|d| = 0
RUST PARITY: PASS
```

The comparison is `f_py == f_rs` — **exact float equality, not a tolerance**.
There is no column with a documented divergence, no `1e-6` allowance, and no
column excluded. The 112 embedding ids are compared with `np.array_equal`.

The corpora are the same ones `enc2_equiv.py` uses:

* **real labelled corpus** — `data/pl2/out/labels_*.jsonl`, one team pair, 13
  species. This is where the encoder will actually run.
* **pool-wide synthetic** — `enc2_gate.fuzz_states`, real randbats sets across
  all 509 species / 60 items / 203 abilities, randomised HP, status, boosts,
  volatiles, hazards and field. This is the corpus that finds things (§2.3).
* **one root, shared static** — 512 leaves built from one root varying HP,
  status, boosts, volatiles, hazards, field and which mon is active.
* **+ tera / disable / PP moving** — the same, plus the three quantities that
  used to force a static rebuild.

### 2.2 The static/dynamic split, proved on the Rust side

`ENCODER2_BUILD.md` §11.3's property is that the shared-static path reproduces
the cold path bit for bit even when tera, `disabled` and PP all move. That
property is reproduced here and proved *twice*:

* against python — the shared-static Rust run matches python's **cold** path on
  all 1,413 columns on both the ordinary and the hard leaf set;
* against itself — `rust shared` vs `rust cold` on the hard set,
  **0 of 1,413 columns differ, max |d| = 0**.

The mechanism carries over unchanged. `Enc2Static` reads no `disabled`, no
`pp` and no `terastallized`. Tera is an axis of the damage table (the attacker
and defender axes are 6 party slots × {not terastallized, terastallized}), and
a leaf indexes `2 * party_slot + tera_flag`, so the slot permutation and the
tera selection are **the same gather**. There is no rebuild trigger inside a
search at all — not on tera (16.2 % of real transitions), not on a new disable
(2.7 %), not on PP.

### 2.3 What it took: numpy's dtype promotion is load-bearing

Exact equality against numpy is a much stronger constraint than a rewrite
normally faces, because `enc2.py`'s dtypes are not uniform and not accidental.
Four of them had to be reproduced deliberately, and each is commented at its
site in `enc2.rs`:

1. **`spe` and `base_speed` are f64, not f32.** `np.where(cond, 1.5, 1.0)` on
   two python floats produces a *float64* array — there is no array operand to
   anchor the weak scalars to f32 — so the paralysis factor promotes the whole
   effective-speed layer, and the speed key with it.
2. **`combine()` is f32 for three steps and f64 after.** `dm * numm`, then the
   defender boost ratios on the single active column, run in f32; `br4` and
   `msso` are float64 for the same weak-scalar reason, so everything downstream
   of them is f64. Getting this wrong changes the KO matrix.
3. **`base` inside the damage kernel is always f64** (the sand/snow `np.where`
   promotes the defensive stat), **but its numerator `lvl_term × bp × Aeff` is
   f32 unless some move in the state promotes `bp`.** Six base-power kinds do —
   Low Kick, Heavy Slam, Rage Fist, Acrobatics, Facade, Knock Off — and the
   product reaches ~1e7, where f32 spacing is 1.0. This is reproduced by a
   per-state `bp_is_f64` flag.
4. **The `-ate` ability's 1.2× base-power boost rounds at `bp`'s own dtype**,
   because `1.2` is a weak scalar. Multiplying in f64 when numpy multiplied in
   f32 is a real difference.

Two genuine bugs were caught by the pool-wide corpus and nothing else — the
5,000-state real corpus passed while they were live, because neither mechanism
occurs in its 13 species:

* **Loaded Dice was not selecting the multi-hit count.** `M["hits"]` is
  `np.where(attacker holds Loaded Dice, hits_dice, hits)`; the port used
  `hits` unconditionally.
* **The `-ate` boost was being applied in f64** (item 4 above).

Both showed up as ~64 differing columns on 1–2 of 400 synthetic states,
almost all of them KO-matrix one-hots — a one-bucket flip in `ceil(hp / dmg)`.
That is the shape a damage bug takes, and it is why the acceptance corpus has
to be pool-wide: **the real labelled corpus cannot see either bug.**

### 2.4 One flagged property, measured

`dmgtab.raw_damage`'s `bp` promotion (item 3) is evaluated over the whole numpy
**batch**, not per state, so `enc2.py`'s own output is in principle
batch-size-dependent: a state with no promoting move gets an f32 numerator
alone and an f64 numerator inside a batch that contains a Knock Off.

Measured, not assumed: of 600 pool-wide states, **588 contain a promoting move
anyway**, and of the 12 that do not, **0 change any of the 1,413 columns** when
encoded inside a batch with a promoting state rather than alone. So the effect
exists structurally and does not materialise. The Rust port replicates the
per-state semantics, which is what the search does (`build_static` sees one
root), and the parity harness therefore runs the reference at `chunk=1`.

---

## 3. Cost, measured

### 3.1 Machine and method

* **Apple M2, 8 cores** (`hw.model = Mac14,2`). *Note: `ENCODER_PERF.md`
  attributes its numbers to an M4; this machine reports M2. The python
  baselines quoted below are that report's, on whatever machine it used.*
* `cargo build --release --features terastallization`, `CARGO_BUILD_JOBS=4`,
  LTO fat / 1 codegen unit (the shipped release profile).
* **Serial, single process, one leaf at a time.** No batch dimension anywhere —
  `eval_scratch(&State)` is scalar at every signature and the batched path was
  measured at +3 % and deleted.
* 60 trials × 200 encodes each; medians reported, p10/p90 shown.
* **Machine state, stated:** not idle. Load average ~4 on 8 cores, Chrome at
  ~40 % of one core, an interactive session running. The benchmark is
  single-threaded on a machine with free cores, and the run-to-run spread is
  ±2 % across five independent runs (3.397 – 3.528 µs), so the contention is
  visible but not material. No cloud, no parallel harness.

### 3.2 Totals

| | median | p10 – p90 | 5-run spread |
|---|---:|---:|---:|
| **Dynamic, shared static (the search path)** | **3.431 µs/leaf** | 3.40 – 3.56 | 3.397 – 3.528 |
| Dynamic, rotating static (cache upper bound) | 3.498 µs/leaf | — | 3.304 – 3.646 |
| **Including `State` → flat view** | **3.711 µs/leaf** | 3.65 – 3.94 | 3.680 – 3.766 |
| `State` → flat view alone | 0.360 µs/leaf | — | 0.360 – 0.361 |
| **Static build (once per search)** | **75.9 µs** | 75.6 – 77.7 | 75.7 – 76.0 |

Two totals are quoted on purpose. **3.43 µs** is the encoder proper. **3.71 µs**
is what an `eval_scratch(&State)` actually pays, because it must first read the
engine's `State` into the flat, party-ordered view. The extraction figure is a
mild *over*-estimate: it is measured cycling 512 distinct `State` objects, where
a real search re-reads the state it has just mutated. **Use 3.71 µs; it is the
conservative one.**

The shared-static and rotating-static numbers agree within noise, which is the
useful result: the 23 KB damage table is not a cache problem.

### 3.3 Per feature block

Medians of 5 runs. Measured by instrumenting each block with `Instant::now()`
under a const-generic flag (zero cost when off), then correcting for the timer
(28.2 ns/call, 13 calls) and rescaling onto the uninstrumented total — the
instrumented build is ~20 % slower, so **the shares are the measurement and the
absolute nanoseconds are derived from them.**

| # | block | µs/leaf | % | columns produced |
|---|---|---:|---:|---:|
| 9 | **§4 counterfactual, per mon (+1 setup)** | **0.967** | **28.2 %** | 168 |
| 11 | §4 side aggregates (sweep / free turn / answers / tera cols) | 0.420 | 12.2 % | 10 (+2 on the side block) |
| 6 | **§2 per-side block** | **0.420** | **12.2 %** | 190 |
| 4 | damage combine → now + KO counts | 0.393 | 11.4 % | shared |
| 3 | damage gather + multipliers | 0.351 | 10.2 % | shared |
| 10 | §4 tera prep (tera damage table + `eff_te`) | 0.222 | 6.5 % | shared by the 4 tera cols |
| 7 | **§3 KO matrix one-hot** | **0.160** | **4.7 %** | 360 |
| 5 | §1 per-mon block + embedding ids | 0.154 | 4.5 % | 564 |
| 1 | slot reorder + gather | 0.123 | 3.6 % | shared |
| 0 | output zero-fill | 0.084 | 2.4 % | shared |
| 8 | §3 relational, speed / priority | 0.070 | 2.0 % | 106 |
| 2 | effective speed | 0.056 | 1.6 % | shared |
| 12 | §5 global | 0.011 | 0.3 % | 15 |
| | **TOTAL** | **3.430** | 100 % | 1,413 |

Rolled up onto Sally's requested breakdown:

| feature block | µs/leaf | columns | ns/column |
|---|---:|---:|---:|
| Setup counterfactual (§4 per-mon, +1) | 0.967 | 168 | 5.8 |
| Tera features (block 10 measured; +est. share of 11) | 0.222 – ~0.32 | 4 | 55 – 80 |
| Per-side block (§2) | 0.420 | 190 | 2.2 |
| KO matrix (§3, both directions) | 0.160 | 360 | 0.4 |
| Per-mon block (§1) + ids | 0.154 | 564 | 0.3 |
| Speed / priority (§3 + effective speed¹) | 0.126 | 106 | 1.2 |
| Global (§5) | 0.011 | 15 | 0.7 |
| **Shared damage infrastructure** (blocks 3 + 4) | **0.743** | — | feeds everything |
| Shared plumbing (blocks 0 + 1) | 0.207 | — | — |

¹ block 2 (effective speed, 0.056 µs) is shared — the counterfactual and the
pair-speed bits both read `skey` — so cutting §3's speed columns saves block 8
(0.070 µs) and not the whole 0.126.

**The ranking has moved relative to numpy, exactly as `ENCODER_PERF.md` §1
predicted it would.** That report's warning was "a Rust port is not a uniform
speedup; it collapses the ranking onto the elem-op column", and it does. Against
the post-§11 numpy shares:

| block | numpy share | Rust share |
|---|---:|---:|
| §3 KO matrix one-hot | 0.7 % | **4.7 %** |
| §4 cf per-mon (+1 setup) | ~14.9 % | **28.2 %** |
| §4 cf side + tera | ~19.8 % | 18.7 % |
| §2 per-side | ~11.4 % | 12.2 % |

The KO matrix takes **seven times** the share of the budget it took in numpy —
it was nearly free there because it is 660 element-ops issued as 10 numpy calls,
and dispatch was 95 % of the cost. §2 per-side barely moved despite being the
worst work-per-microsecond block in numpy (412 element-ops in 189 µs, a python
loop with dict lookups), because its Rust cost is real work rather than
dispatch: 44 volatile bits, 12 durations and 19 side conditions, twice.

### 3.4 Static cost

**75.9 µs, once per search, and it never rebuilds.** The kernel runs
5 weather × 2 blocks × 6 attackers × 2 attacker-tera × 4 moves × 6 defenders ×
2 defender-tera = **5,760 damage evaluations**, and lands inside
`ENCODER_PERF.md` §4's 48 – 146 µs estimate.

Amortised at the projected 93,000 leaves/s, a 100 ms search is ~9,300 leaves and
the static half is **0.008 µs/leaf — 0.1 % of the budget.** The invalidation
tax that §3.1 priced at 1.1 – 3.4 µs/leaf for `disabled` alone is **zero**.

---

## 4. Against the budget

The budget, from `ENCODER_PERF.md` §0: production search is **133.4k MCTS
iterations/s single-threaded = 7.50 µs/iteration**, of which the current Rust
encoder is **~0.25 – 0.75 µs**. A 2× ceiling means ≥65,000 evaluations/s, so
the new encoder's budget is **≈8.4 µs/leaf**.

| | value |
|---|---:|
| New encoder, per leaf (conservative, includes `State` read) | **3.71 µs** |
| Fraction of the 8.4 µs budget | **44 %** |
| Headroom | **2.26×** |

Projected search throughput, holding everything else fixed:

| old encoder assumed at | new µs/iteration | new iters/s | **slowdown** |
|---|---:|---:|---:|
| 0.25 µs | 10.96 | 91,200 | **1.46×** |
| 0.50 µs | 10.71 | 93,400 | **1.43×** |
| 0.75 µs | 10.46 | 95,600 | **1.39×** |

**Answer: 1.39× – 1.46×, against a ceiling of 2×.** On the encoder-only figure
(3.43 µs) it is 1.36× – 1.42×. Call it **~1.43×, and ~93,000 iters/s.**

### What that number does *not* include

Stated, not buried — these are the same three caveats `ENCODER_PERF.md` §5
raised, and none of them is resolved here:

1. **This is the encoder only.** The network forward pass over 1,413 inputs is
   a separate charge on the same 15.4 µs/iteration ceiling, and it is not
   measured here because no net of this shape exists yet.
2. **Leaf-cache hit rate is not modelled.** The engine's 20.5 % leaf-cache and
   96.9 % mon-column memo hit rates are keyed on exact encoder output bytes; a
   wider, more state-sensitive encoder will hit less often. That is a
   second-order cost on the same budget and it is unmeasured.
3. **The 7.50 µs/iteration baseline is quoted, not re-measured.** If search
   speed has moved since `SEARCH_SPEED_REPORT.md:38`, the multiple moves with it
   — but the encoder's absolute 3.71 µs does not.

---

## 5. The ranked strip list

**Not needed — the encoder clears the ceiling with 2.26× of headroom.** It is
produced anyway, because it is the table to reach for if the forward pass or
the cache-hit-rate cost turns out to eat the margin.

Ranked by **value per column lost** (highest = cut this first). Rows 1–3 are
nested, not independent: row 2 is rows 1 and 3 taken together.

| rank | candidate cut | columns lost | µs/leaf saved | **ns per column lost** | % of encoder |
|---|---|---:|---:|---:|---:|
| **1** | **the tera counterfactual** (block 10, plus its share of 11) | **4** (0.3 %) | **≥0.222 measured, ~0.32 est.** | **55 – 80** | **6 – 9 %** |
| **2** | **the whole §4 counterfactual block** (both halves — supersedes 1 and 3) | **178** (12.6 %) | **1.81** (measured by ablation) | **10.2** | **53 %** |
| 3 | the +1 setup counterfactual (§4 per-mon) | 168 (11.9 %) | 0.967 | 5.8 | 28 % |
| 4 | the §2 per-side block | 190 (13.4 %) | 0.420 | 2.2 | 12 % |
| 5 | §3 speed / priority (pair bits, margin, order-16, brackets) | 106 (7.5 %) | 0.070 – 0.126 | 0.7 – 1.2 | 2 – 4 % |
| 6 | §5 global | 15 (1.1 %) | 0.011 | 0.7 | 0.3 % |
| 7 | the KO matrix, one direction (`ko_directions=1`) | 180 (12.7 %) | ~0.08 | 0.4 | 2 % |
| 8 | **the KO matrix entirely** | **360** (25.5 %) | **0.160** | **0.4** | **5 %** |
| 9 | the §1 per-mon block | 564 (39.9 %) | 0.154 | 0.3 | 4 % |

**The two findings `ENCODER_PERF.md` §5 reached still hold, and one is now much
sharper.**

* **The KO matrix is still the worst cut available** and by a wider margin than
  before. §5 measured it at 24 % of the columns for 8 % of the work; in Rust it
  is **25.5 % of the columns for 4.7 % of the work**. Cutting it costs the net a
  quarter of its input and buys 0.16 µs. Do not cut it.
* **The counterfactual block is where the time is.** §5 found the +2 setup level
  was the best value per column and it was removed; the +1 level that remains is
  still **28 % of the encoder for 12 % of the columns**, and the block as a whole
  is **53 % of the encoder for 12.6 % of the columns**. If the budget ever
  tightens, this is the only place with real money in it.
* **New: the tera counterfactual is the single best-value cut, by ~10×.** Four
  columns — `tera_best_value` and `tera_enabled_sweep`, per side — cost **at
  least 0.222 µs/leaf**, which is block 10 measured on its own: pricing them
  needs a *second* damage-table gather and combine over every attacker and move,
  plus the defensive best-move pick over `eff_te`. Their share of block 11 (the
  `gain` / `tera_sweep` loop) is not separately instrumented; ~0.1 µs is an
  estimate and is labelled as one, so the honest range is **0.22 – 0.32 µs for
  4 columns = 55 – 80 ns per column, against 5.8 for the next candidate.** §5
  priced this at "−4 columns, ~1,660 elem-ops, 6 %" and it is 6 – 9 % in Rust,
  so that estimate was good. If one thing has to go, this is it — and it is four
  columns, so the information cost is the smallest on the list.

### How the strip list was measured, and its error bar

Two independent methods, reported together because they disagree in a way worth
knowing about.

* **Per-block instrumentation** (the µs column above) is stable to ±3 % across
  five runs and measures each block's own instructions.
* **Ablation** — rebuild with the block's columns not produced and difference
  the wall time — captures work that dies *with* the block but carries roughly
  **±0.25 µs of code-layout noise**. The tell is the §5 global row, which
  provably does ~6 stores and whose ablation ranged from −0.10 to +0.27 µs
  across three runs.

For the large cuts, where the noise is small relative to the signal, the two
agree: the whole §4 counterfactual ablates at **1.77 / 1.81 / 1.92 µs** across
three runs against a per-block sum of 1.61 µs, and the ~0.2 µs excess is the
shared work that genuinely dies with it. The §4 ablation is the one number in
the table taken from ablation rather than instrumentation, for exactly that
reason. Everything else uses the per-block figure, which is the more stable
estimator at that scale.

**Blocks 3 and 4 (0.743 µs, 22 % of the encoder) are not on the list**, because
they are shared damage infrastructure: the KO matrix, the counterfactual, the
tera valuation, the revenge-priority flags and the answers-to-best-threat count
all read `dmg_full`. They only disappear if *every* consumer does, which is
another way of saying the encoder becomes a different encoder.

---

## 6. Assumptions and scope

Stated, not hidden.

1. **The port is straight.** No structural rewrite, no incremental/dirty
   propagation, no algorithmic change. Every operation happens in the same order
   and at the same precision as numpy, because that is what bit-identity
   demands. `ENCODER_PERF.md` §5's steps B (structural wins, ×1.2–1.4) and C
   (incremental, ×2–3) are **not built** and remain available if more headroom
   is ever wanted.
2. **`Enc2Static` must be rebuilt when the twelve Pokémon change** — and only
   then. Tera, PP and `disabled` are handled dynamically and are proved not to
   invalidate it (§2.2).
3. **`ENCODER2_BUILD.md` §8 / §10.6 / §11.8's assumption lists carry over
   unchanged**, because the port is bit-identical: accuracy is not modelled,
   KO counts use the maximum roll and the sweep conjunction the minimum, Protect
   and Substitute are not in the KO matrix, and the residual damage error
   against the live engine is what it was (99.1 % KO-bucket agreement on the
   labelled corpus, 92.6 % pool-wide).
4. **The `bp_is_f64` flag is per state in Rust and per batch in numpy.** They
   agree on the search path (one state) and were measured not to differ
   anywhere else (§2.4).
5. **`State` → flat view is charged at 0.360 µs**, measured over 512 distinct
   cold `State` objects. In a search the state is hot, so the true figure is
   lower and 3.71 µs/leaf is conservative.
6. **The encoder is not wired into `eval_scratch`.** It is a library plus a
   benchmark binary. Wiring it in means a net trained on these 1,413 columns,
   which does not exist; the number this report exists to produce is whether
   that is worth doing, and the answer is yes.
7. **Single-threaded measurement on a machine with other work on it**, ±2 %
   run to run. No cloud, no training, ≤4 cores for builds.

## 7. Reproducing it

```bash
# 1. regenerate the tables from the python encoder (only needed if enc2.py moves)
cd evallab && ../foul-play/.venv/bin/python enc2_rustgen.py

# 2. build
cd ../poke-engine
CARGO_BUILD_JOBS=4 cargo build --release --features terastallization --bin enc2_bench

# 3. the acceptance gate
cd ../evallab
../foul-play/.venv/bin/python enc2_rust_parity.py --real 5000 --fuzz 1500 --shared 512

# 4. the numbers
cd ../poke-engine
./target/release/enc2_bench bench ../evallab/data/enc2/bench_leaves.txt 60
./target/release/enc2_bench strip ../evallab/data/enc2/bench_leaves.txt 60
```
