# Encoder 2 — the path to ≤2× per-evaluation cost

> **This report measures `enc2.py` as it stood on 2026-08-14 morning
> (sha256 `4c45968029d6ad18`, kept as `enc2_ref.py`). Five of its findings have
> since been acted on — the +2 setup level is gone, the two invalidation bugs of
> §3.1 are fixed outright rather than mitigated, and the structural wins of §3
> are taken. The current encoder is 1,146 µs/leaf at n = 1 with 17,962
> element-ops, and the Rust multiple is 136×, not 197×. See
> `ENCODER2_BUILD.md` §11. Everything below still describes the *before* state
> and is the reason each change was made.**

*Everything here is measured on this Mac (M4, 8 cores, single process, NumPy 2.5,
`nice -n 10`, ≤4 cores) unless labelled ESTIMATE. `enc2.py` is unmodified
(sha256 `4c45968029d6ad18…`); every prototype is a generated or separate file and
every one was proved bit-identical to `enc2.encode_columnar` before its numbers
were believed.*

Scripts: `perf_bench.py` (batch scaling) · `make_prof.py` → `enc2_prof.py` /
`enc2_count.py` (generated instrumented copies) · `npcount.py` (op counter) ·
`perf_profile.py` (block timing) · `perf_ops.py` (block work) · `perf_delta.py`
(column change fraction) · `perf_incr.py` (incremental ceiling) ·
`perf_proto.py` (static cost + incremental prototype).

---

## 0. Headline — the gap is 197×, not 11×

**The published 0.083 ms/leaf is a *batched* number.** `enc2_gate.gate_cost`
measures the dynamic half over n = 4,096 states at once and divides. The
production search has **no batch dimension** — `mcts.rs:635 → rollout →
mcts_common.rs:342 → evaluate_nn.rs:269 `eval_scratch(&State)` → `encode_state_into`
is scalar at every signature, and the batched path that exists
(`Network::forward_many`) was measured at +3 % and **permanently killed**
(commit `609b86e`). A leaf is encoded alone.

At n = 1, the regime the search actually runs in:

| | measured |
|---|---|
| dynamic half, **n = 1** | **1.654 ms/leaf** → **605 leaves/s** |
| dynamic half, n = 512 (batched) | 0.0826 ms/state → 12,101/s ← the published number |
| static half, n = 1 | 5.20 ms (median; p10 5.27, p90 9.28) |
| overhead share at n = 1 | **95.0 %** |

Per-state cost falls 21× from n = 1 to n = 512 (`perf_bench.py`), which is the
signature of a dispatch-bound path, not a FLOP-bound one.

**The budget.** Production is **133.4k MCTS iterations/s single-thread**
(`SEARCH_SPEED_REPORT.md:38`) = **7.50 µs/iteration**. The current Rust encoder
is **~0.25–0.75 µs** of that — two independent measurements: 5.4 % inclusive at
the old 13.68 µs baseline (`TRACK_P_PERF_REPORT.md:28`) and "forward is ~8.84 µs
of a ~9.08 µs eval" from the `forward_many` microbench. The encoder runs on
100 % of iterations (it produces the leaf-cache key).

> A 2× ceiling means ≥65,000 evaluations/s = ≤15.4 µs/iteration. Holding
> everything else fixed, the **new encoder's budget is ≈ 8.4 µs/leaf.**

1,654 µs vs 8.4 µs is **197×**. The 11× in `ENCODER2_BUILD.md` §10.4 compares a
batched throughput against an unbatched one and understates the gap by ~18×.

---

## 1. Profile of the dynamic half

`perf_profile.py` — generated instrumented copy, verified **1511/1511 columns
bit-identical to `enc2` on 1,024 real states**. `perf_ops.py` — same copy with
numpy replaced by a counting proxy, verified bit-identical at n = 1.

| block | n=1 µs | % | n=512 µs/st | **overhead** | elem-ops | array ops |
|---|---:|---:|---:|---:|---:|---:|
| `12` cf setup +1/+2 | **395.8** | 24.4 % | 17.10 | 378.7 | 9,936 | 202 |
| `13` cf side + tera | 258.6 | 16.0 % | 5.70 | 252.9 | 3,013 | 147 |
| `9` per-side (§2) | 189.5 | 11.7 % | 1.29 | 188.2 | 412 | 75 |
| `5` dmg gather + multipliers | 167.0 | 10.3 % | 30.49 | 136.5 | 2,911 | 44 |
| `1` static reorder | 129.9 | 8.0 % | 6.29 | 123.6 | 656 | 23 |
| `10` relational (§3) | 105.2 | 6.5 % | 2.51 | 102.6 | 1,438 | 52 |
| `6` dmg combine → now | 101.3 | 6.3 % | 7.74 | 93.5 | 3,384 | 41 |
| `8` ids + per-mon (§1) | 97.4 | 6.0 % | 3.82 | 93.6 | 3,467 | 38 |
| `4` effective speed | 56.5 | 3.5 % | 0.73 | 55.7 | 421 | 36 |
| `2` gather + parse | 44.7 | 2.8 % | 4.89 | 39.8 | 708 | 12 |
| `11` cf prep | 29.3 | 1.8 % | 0.45 | 28.8 | 482 | 7 |
| `14` global (§5) | 25.9 | 1.6 % | 0.11 | 25.8 | 21 | 9 |
| `7` KO matrix one-hot | 12.1 | 0.7 % | 0.37 | 11.7 | 660 | 10 |
| `3` spread boosts/volatiles | 7.4 | 0.5 % | 0.34 | 7.0 | 1,608 | 4 |
| **TOTAL** | **1,654** | | **82.6** | **1,539** | **29,117** | **700** |

### Overhead vs arithmetic — the number that decides the Rust estimate

**95.0 % of the n = 1 cost is dispatch, not arithmetic.** 1,654 µs = ~1,571 µs
overhead + ~83 µs of actual numpy arithmetic. The previous agent's finding for
the static half holds here too, and harder.

The overhead is *two* things, both of which Rust deletes:

* **numpy dispatch** — 700 array operations averaging 42 elements each. The
  tensors are tiny: the whole damage tensor is 360 floats.
* **Python interpretation** — block `9` is the tell: **189 µs for 412 element-ops**
  (0.46 µs per element). It is a Python `for` loop over ten side conditions with
  dict lookups. Blocks `13`, `14` and `8` are the same shape. Also: the
  column-name→index dicts `M`/`SS`/`CM`/`CS`/`G` (~185 entries) are **rebuilt on
  every single leaf**.

Ranking by *time* and by *work* disagree sharply, and that disagreement is the
point — `3_spread_boosts` is 1,608 elem-ops in 7 µs, while `9_per_side` is 412
elem-ops in 189 µs. **A Rust port is not a uniform speedup; it collapses the
ranking onto the elem-op column.**

---

## 2. Incremental update — measured, and it is *not* the 14× it looks like

### 2.1 How many columns change (`perf_delta.py`)

Real MCTS parent→child transitions, produced by the engine's own
`generate_instructions` + `apply_instructions`, sampling the (my arm, their arm)
pairs the search expands. Two independent runs:

| | 1,121 transitions | 895 transitions |
|---|---|---|
| columns changed / child | mean **106.4** (7.0 %) | mean **119.0** (7.9 %) |
| median / p90 / max | 98 / 218 / 379 | 113 / 220 / 394 |
| columns that never changed | 584 (38.6 %) | — |

Per block (1,121-transition run):

| block | cols | changed/child | % of block |
|---|---:|---:|---:|
| KO matrix | 360 | 37.2 | 10.3 % |
| §4 counterfactual per-mon | 264 | 26.7 | 10.1 % |
| §1 per-mon | 564 | 26.5 | 4.7 % |
| pair speed bits | 36 | 5.7 | 15.9 % |
| move-order 16 | 16 | 4.0 | 25.3 % |
| §4 counterfactual per-side | 12 | 2.9 | 23.8 % |
| §2 per-side | 190 | 2.1 | 1.1 % |
| priority brackets | 63 | 0.4 | 0.6 % |
| **TOTAL** | **1,511** | **106.4** | **7.0 %** |

### 2.2 What that is actually worth (`perf_incr.py`) — the important result

7 % of columns changing does **not** mean a 14× saving. What matters is the
fraction of the *dataflow graph* that must be recomputed. So I measured it
directly: run the counting encoder on a parent and on each real child, record
the output of every array operation (the op sequence is data-independent, so
the traces align call-for-call), and diff them. **Re-measured on the post-§11
encoder, 153 transitions** (the pre-§11 column is the original 180-transition
run, on an encoder that did twice the arithmetic):

| granularity | elem-ops/leaf | **speedup** | pre-§11 |
|---|---:|---:|---:|
| full recompute (today) | 14,000 | 1.00× | 29,117 |
| whole-array (cache an op or redo it) | 6,365 | **2.20×** | 1.98× |
| active/bench tiling | 5,491 | **2.55×** | 2.36× |
| **per-slot rows & columns — the design to build** | **3,154** | **4.44×** | 4.31× |
| per-element (unreachable floor) | 1,793 | 7.81× | 8.56× |

Array ops per leaf: 329, of which **128 (38.9 %) change** (was 693 / 295).
**The ceiling barely moved** — 4.31× → 4.44× — because §11 removed *whole-array*
work (the +2 setup level, the five-channel product, `spread(vol)`), not
row∪column propagation, which is what actually caps the incremental win.

**Why 4.3× and not 14×.** The expensive features are *pair* features. The KO
matrix and the counterfactual block are 6×6 tensors, so one mon losing HP
dirties an entire row **and** an entire column — 11 of 36 entries — no matter how
small the change was. 624 of the 1,511 columns are pair-structured, and they
carry 62 % of the arithmetic. That row∪column propagation is a property of the
*design*, not of the implementation, and it is what caps the incremental win.

### 2.3 Prototype (`perf_proto.py`)

An incremental implementation of the dominant chain —
`dmg_full → ko_now →` the 360 KO-matrix columns — with real dirty propagation
(which slots' HP / status / boosts / screens / alive / **tera** moved and which
moves got **disabled**; a switch permutes two slots and re-gathers rather than
recomputing; Supreme Overlord gated on the ability actually being present).

Ported to §11's **per-move** table: `S.dmg` is
(n, 5 weather, 2, 12 attackers, 4 moves, 12 defenders) and `S.pair(v, aix, dix)`
takes an already weather-selected table with `aix = 2 × party_slot + tera_flag`,
so the slot permutation and the tera selection are one gather. Two things change
for incremental update, both of them improvements:

* `disabled` / `pp` and `terastallized` are **leaf** inputs now, so they join the
  dirty set instead of forcing a static rebuild — and a rebuild is precisely what
  no incremental scheme can absorb (§3.1 priced tera's 16.2 % rate at more than
  the entire per-leaf budget).
* there is a **move axis under the pair axis**. The pre-maximum damage
  arithmetic is dirty per (pair, move); only after the max over moves does it
  collapse to per-pair KO bucketing. Both are now reported.

Re-measured on the new dataflow, **982 real transitions** (68.5 % of them
containing a switch, 0 weather fallbacks):

* **982 / 982 bit-identical to `enc2`.** 0 wrong.
* Dirty pairs: **49.0 of 72 per leaf (68.1 %) → 1.47×** on the KO chain.
* Dirty (pair, move): **196.1 of 288 (68.1 %) → 1.47×** on the damage table.

The two granularities agree to three digits: the only input finer than a whole
attacker row is `disabled`/`pp`, and at a 2.7 % rate it never fires without
something coarser firing alongside it. **The move axis buys correctness (no
rebuild), not extra sparsity.**

The prototype's 1.47× is *below* the 4.44× dataflow ceiling because its dirty
propagation is deliberately conservative (a side condition dirties a whole
block; `alive` changes propagate widely). **Read the two together:** 4.44× is
what the dataflow permits, 1.47× is what a first, safe implementation of one
chain gets. A realistic whole-encoder implementation lands at **2–3×**.

**Timing the prototype in numpy would be meaningless and is not reported.** At
n = 1 the incremental path issues *more, smaller* calls than the full recompute,
so in numpy it is slower — the design only pays off once dispatch costs nothing.
The transferable quantity is the op count, which is what is measured above.

---

## 3. Cheap structural wins, priced individually

**Weather: already fixed, no win available.** The dynamic path selects one
variant via `wsel` (`S.pair(S.dmg, order, wsel)`) and reorders only that one —
§10.2 records this. It does *not* compute five.

The **static** half does build 5 weather × 2 tera = 10 kernel variants:
**5.20 ms → 1.56 ms with one weather variant (3.30×)**. That is not waste unless
weather is known not to change inside the search, which it is not (Sun/Rain moves
are in the pool). Keep it.

| # | Win | Measured | Saving |
|---|---|---|---|
| 1 | **The bench×bench damage quadrant is fully static.** `boost12` is nonzero *only* on slots 0 and 6 — **verified: 0 of 400 real states carry a bench boost**. So for the 2×5×5 = 50 of 72 pairs (69 %) that are bench-vs-bench, every boost ratio in `_channel_damage` is exactly 1.0, and `phys`/`spec` reduce to a static 3-way and 2-way max. **Verified bit-exact on 400 states, max abs diff 0.** Yet the full 5-channel product + collapse runs on all 72 pairs, in each of the **four** `combine()` calls (base, +1, +2, tera). | verified | **~3,580 elem-ops (12 %)** |
| 2 | **`spread(vol)`** — `np.repeat(vol, 6, axis=1)` inflates (n,2,107)→(n,12,107) = **1,284 elem-ops/leaf**, to read **4 volatiles on 2 active slots** (Slow Start, Unburden, Protosynthesis/Quark Drive). Round-tripped straight back through `_sides_of`. | verified | **1,284 elem-ops (4.4 %)** |
| 3 | **Setup counterfactual attacker boosts are static for 10 of 12 slots.** `st = clip(stage + setup_boost·lv)`; `setup_boost` is static and `stage` is zero on the bench, so `st` on the bench **is** `setup_boost·lv` — a static quantity recomputed twice per leaf, then fed through the full kernel. | ESTIMATE, extends #1 | ~10–15 % |
| 4 | **Per-leaf dict rebuilds** — `M`/`SS`/`CM`/`CS`/`G`, ~185 entries, rebuilt every leaf. Free to fix, irrelevant in Rust. | — | ~1 % of numpy time |

Confirmed removable: **~16 %** of dynamic elem-ops. With #3: **~25–30 % (ESTIMATE)**.

### 3.1 Two invalidation bugs that would break the budget on their own

`ENCODER2_BUILD.md` §6 states the shared static context must be rebuilt on a
terastallization, a PP exhaustion or a new disable, and calls all three "rare".
Measured over 895 real transitions:

| cause | rate |
|---|---:|
| **terastallization** | **16.2 %** |
| **new disable** | **2.7 %** |
| PP exhaustion | **0.0 %** |

* **Disable at 2.7 %** costs 0.027 × (42–127 µs Rust static, §4) = **1.1–3.4 µs
  per leaf amortised — 13–40 % of the entire 8.4 µs budget** — for a flag that
  only feeds the `mv_present` / `mv_ok` masks. Those masks are cheap and must
  move into the dynamic half. (Choice-lock and Encore set `disabled`, which is
  why it is common.)
* **Tera at 16.2 %** must not trigger rebuilds either: cache one static context
  per tera configuration (a small set, and tera is once per side per game), not
  one per search.

Caveat: children were sampled uniformly over legal arms, so switches and tera
are over-represented relative to an MCTS visit distribution. The *direction* is
robust; the exact rates are upper bounds.

---

## 4. The Rust factor, and the assumption behind it

The n = 1 profile decomposes cleanly:

| term | numpy @ n=1 | what Rust does to it |
|---|---:|---|
| dispatch + Python interpretation | **1,571 µs (95 %)** | **→ ~0.** Becomes inlined loop bodies. Not a speedup — a deletion. |
| arithmetic (29,117 elem-ops) | **83 µs** | fused, in registers, no temporaries |

**Assumption, stated:** Rust executes enc2's elementwise work at **2–6
element-ops per nanosecond**.

*Basis, and why this is not a folk number.* The calibration is the production
Rust encoder on this same machine and problem: it writes 1,092 f32 + 108 i32
with roughly 2–5 arithmetic operations each — call it 2,400–6,000 ops — in
0.25–0.75 µs. That is **3–24 ops/ns** measured. I assume 2–6 for enc2, i.e.
1.5–8× more conservative than the calibration, because enc2's work is harder
than a flat sequence of scalar stores: f32 **divides** (1,922 elems), **gathers**
(`take_along_axis`, 3,618 elems), `argmax` / `sort` / `cumsum` reductions, and
mon axes of length 5 and 6 that vectorise poorly into f32x4 lanes.

| | elem-ops | Rust @ 2–6/ns |
|---|---:|---|
| dynamic half, full recompute | 29,117 | **4.9 – 14.6 µs/leaf** |
| static half, once per search | 253,059 | **42 – 127 µs** |

Overall numpy→Rust factor for the dynamic half: **113–338×**. Note how it
splits: the *arithmetic* speeds up only **5.7–17×** (the "5–20×" folk range is
right for that term), and everything else in the 197× comes from deleting
dispatch. **A Rust port is worth it because of the 95 %, not the 5 %.**

---

## 5. The path to ≤2×

Budget: **≈8.4 µs/leaf**.

| step | dynamic cost/leaf | factor | confidence |
|---|---:|---:|---|
| today, numpy n=1 | 1,654 µs | — | measured |
| **A.** Rust port, straight, full recompute | **4.9 – 14.6 µs** | 113–338× | ESTIMATE, calibrated (§4) |
| **B.** + structural wins §3 (−16 % verified, −25–30 % with #3) | **3.4 – 11.0 µs** | ×1.2–1.4 | #1,#2 verified; #3 estimated |
| **C.** + incremental, per-slot (ceiling 4.31×, realistic 2–3×) | **1.1 – 5.5 µs** | ×2–3 | ceiling measured; realisation estimated |
| **D.** + fix the static-invalidation bugs §3.1 | removes 1.1–3.4 µs/leaf of amortised rebuild | — | rates measured |

**Answer: ≤2× is reachable, but not by a straight port.**

* **A alone is a coin flip.** 4.9 µs clears the budget; 14.6 µs misses it by
  1.7×. The spread is the honest uncertainty in the ops/ns assumption, and
  nothing cheaper than writing the Rust will narrow it.
* **A + B is still a coin flip** at the pessimistic end (11.0 µs vs 8.4 µs).
* **A + B + C clears the budget across the whole range** (1.1–5.5 µs vs 8.4 µs),
  with 1.5–7× of margin. This is the only combination I would promise.
* **D is not optional.** Without it, a 2.7 % disable-triggered static rebuild
  alone eats 13–40 % of the budget.

### What is uncertain

1. **The 2–6 ops/ns assumption** is the dominant uncertainty — it is the entire
   width of every row above. It is calibrated against a *simpler* encoder;
   enc2's gathers and reductions may sit below the range.
2. **The incremental realisation factor.** The 4.31× ceiling is measured on the
   real dataflow and is solid. Whether an implementation captures 2×, 3× or 4×
   of it depends on how tight the dirty propagation is — the prototype's 1.43×
   on one chain is the pessimistic anchor.
3. **Incremental adds state to the search.** Every MCTS node must carry its
   parent's intermediates (~30 KB of f32 at per-slot granularity) or recompute
   on cache miss. That memory and its cache behaviour is not modelled here and
   could eat part of the win. NNUE gets away with it because the accumulator is
   one vector; here it is a set of 6×6 tensors.
4. **This is the encoder only.** The forward pass over 1,511 inputs is a separate
   charge on the same 15.4 µs. One flag, no investigation behind it: the current
   engine's 20.5 % leaf-cache and 96.9 % mon-column memo hit rates are keyed on
   *exact encoder output bytes*, and a wider, more state-sensitive encoder will
   hit less often — that is a second-order cost on the same budget.

### If it misses: what would have to go

Cost per column is wildly uneven, so a cut is cheap if aimed correctly.

| candidate cut | columns lost | dynamic work removed |
|---|---:|---:|
| **the +2 setup counterfactual** (keep +1) | −132 (8.7 %) | **~4,970 elem-ops, 17 %** |
| the whole §4 counterfactual block | −276 (18.3 %) | **~12,950 elem-ops, 44 %** |
| the KO matrix, one direction (`ko_directions=1`) | −180 (11.9 %) | ~2,400 elem-ops, 8 % |
| the tera counterfactual (`combine` on `dmg_tera`) | −4 (0.3 %) | ~1,660 elem-ops, 6 % |

**Best value per column lost: drop the +2 setup level** — 17 % of the work for
8.7 % of the columns, and +1 already carries the sweep flag the spec says should
dominate. **Worst value: the KO matrix** — it is 24 % of the columns for 8 % of
the work, so cutting it hurts the net and barely helps the clock.

---

## 6. Assumptions

1. The **8.4 µs budget** assumes everything else in the 7.50 µs iteration stays
   fixed and that "2× the per-evaluation budget" means the whole evaluation, per
   Sally's ">= 65,000/s".
2. **133.4k is MCTS *iterations*/s, not NN forwards/s.** The encoder does run on
   100 % of iterations, so it is the right denominator for the encoder. The
   forward pass runs less often (leaf cache).
3. All parent→child transitions come from the **labelled corpus, which is one
   team pair** (13 species). The change *fractions* are structural and should
   travel; the absolute rates may not.
4. Children were sampled **uniformly over legal arms**, not by MCTS visit
   counts. This over-weights switches (~56 % of arms) and tera.
5. Element-ops are counted as **output elements of each array operation**, so a
   multiply and a divide on the same array count twice, and an allocation counts
   once. That is the right unit for a fused Rust loop and an over-count for
   numpy (which fuses nothing).
6. `enc2.py` was **not modified**. `enc2_prof.py` / `enc2_count.py` are generated
   by `make_prof.py` and re-verified bit-identical on every run.
7. Single process, `nice -n 10`. Variance is reported as p10–p90 where it
   matters; the n = 1 dynamic figure moved 1.65–1.78 ms across runs, and one run
   overlapped a concurrent agent (that run is not quoted).
