# evallab: lossless encoder + flat architecture — build report

**Date:** 2026-08-13 · **Scope:** the evallab fixed-pair setting only (**pair A, full
information**). Nothing in `valuenet/` or `foul-play/` was modified; the shipped training
pipeline, the Rust encoder and the ladder path are untouched. No cloud spend, no game
generation, all compute local under `nice -n 10`, ≤3 worker threads.

**What this delivers, in one line:** an encoder that round-trips **101/101 field groups at
100.000000% over 50,000 real pair-A states** (and 80,110 mixed states as a supplementary
check), at **6.8k states/s single-core**, in two selectable widths (1,863 and **1,206 —
narrower than today's 1,260 and still provably lossless**); plus a flat, pooling-free
architecture with a capacity knob and a switchable old-encoder/old-architecture baseline in
the same harness; plus a smoke train showing the whole pipeline runs and learns.

---

## 0. Files

| file | what it is |
|---|---|
| `evallab/llencoder.py` | the encoder. Columnar batch parser + fully vectorised numpy encoder. Shares every constant **and the decoder** with `valuenet/lossless_encoder.py`, so there is one layout definition and one decoder. Two variants, `full` / `lean`. |
| `evallab/ll_gate.py` | the acceptance gate: `corpus`, `gate`, `switchsib`, `bench`, `fp16`, `all`. |
| `evallab/lldataset.py` | shards → `.npz` cache (`ids`, `feats`, `y`, `gid`, `ply`, `nlegal`). |
| `evallab/flatnet.py` | the flat net + `build("old")` switch to the incumbent pooled net. `python flatnet.py` prints the layout. |
| `evallab/train_flat.py` | one trainer, two arms; the arm is inferred from which `.npz` you hand it. |

Reproduce everything:

```bash
cd evallab
../foul-play/.venv/bin/python ll_gate.py all -n 50000        # gate + switchsib + bench + fp16
POS_STRIDE=4 ../foul-play/.venv/bin/python lldataset.py 'data/el1/A2k/shard_*.jsonl.gz' \
    data/el1/A2k --variant full
EPOCHS=15 LR=3e-4 WIDTH=512 ../foul-play/.venv/bin/python train_flat.py \
    data/el1/A2k_llfull_pos.npz /tmp/net.pt data/el1/A2k_llfull_hold_pos.npz
```

---

## 1. ROUND-TRIP GATE — the acceptance criterion

### 1.1 What is actually being proven

The gate does **not** compare the encoder to itself. For every state:

```
s  --LE.parse_state-->        src     (the reference parser, itself validated field-by-field
                                       against the real poke_engine bindings on 10,000 states
                                       by valuenet/roundtrip_gate.py --validate-parser)
s  --llencoder.parse_batch--> columnar --encode--> (ids, feats) --decode--> dst
compare flatten(src) vs flatten(dst)      [roundtrip_gate.flatten, verbatim]
```

`src` comes from the reference parser and `dst` comes from **this module's** parser, so one
pass proves the fast columnar parser *and* the vectorised encoder *and* the decoder. Equality
is exact-integer per **value**, not per state — 101 comparison groups covering all 113
encodable leaves.

### 1.2 Result — 50,000 real pair-A states, both variants

```
#### VARIANT FULL  (1863 numeric columns, 184 ids) ####
states: 50000
fields compared: 101    fields at 100.000%: 101    fields failing: 0
value comparisons: 36150000   pass: 36150000   rate: 100.000000%
ALL FIELDS ROUND-TRIP EXACTLY.
vocab: 0 misses (every categorical id is injective on this corpus)
GATE full: PASS

#### VARIANT LEAN  (1206 numeric columns, 184 ids) ####
states: 50000
fields compared: 101    fields at 100.000%: 101    fields failing: 0
value comparisons: 36150000   pass: 36150000   rate: 100.000000%
ALL FIELDS ROUND-TRIP EXACTLY.
vocab: 0 misses
GATE lean: PASS
```

**Zero field failures. Zero vocabulary misses.** The lean variant round-trips because the
decoder reads EXACT columns only — DERIVED columns carry no information by construction.

### 1.3 Field-by-field (all 101 groups, `full` variant; `lean` is identical)

Every group below is `100.000000%`. Counts are value comparisons over the 50,000 states.

| group | pass/total | group | pass/total | group | pass/total |
|---|---|---|---|---|---|
| `mon.id` | 600000/600000 | `mon.level` | 600000/600000 | `mon.hp` | 600000/600000 |
| `mon.maxhp` | 600000/600000 | `mon.attack` | 600000/600000 | `mon.defense` | 600000/600000 |
| `mon.special_attack` | 600000/600000 | `mon.special_defense` | 600000/600000 | `mon.speed` | 600000/600000 |
| `mon.ability` | 600000/600000 | `mon.base_ability` | 600000/600000 | `mon.item` | 600000/600000 |
| `mon.last_consumed_item` | 600000/600000 | `mon.nature` | 600000/600000 | `mon.evs` | 3600000/3600000 |
| `mon.type1` | 600000/600000 | `mon.type2` | 600000/600000 | `mon.base_type1` | 600000/600000 |
| `mon.base_type2` | 600000/600000 | `mon.tera_type` | 600000/600000 | `mon.terastallized` | 600000/600000 |
| `mon.status` | 600000/600000 | `mon.rest_turns` | 600000/600000 | `mon.sleep_turns` | 600000/600000 |
| `mon.weight_kg` | 600000/600000 | `mon.revealed` | 600000/600000 | `mon.known` | 600000/600000 |
| `mon.illusion_broken` | 600000/600000 | `mon.times_attacked` | 600000/600000 | `mon.active_move_actions` | 600000/600000 |
| `mon.once_per_battle_ability_used` | 600000/600000 | `mon.stellar_boosted_types` | 600000/600000 | `mon.reveal_mask` | 600000/600000 |
| `mon.move.id` | 2400000/2400000 | `mon.move.pp` | 2400000/2400000 | `mon.move.disabled` | 2400000/2400000 |
| `side.active_index` | 100000/100000 | `side.volatile_statuses` | 100000/100000 | `side.substitute_health` | 100000/100000 |
| `side.attack_boost` | 100000/100000 | `side.defense_boost` | 100000/100000 | `side.special_attack_boost` | 100000/100000 |
| `side.special_defense_boost` | 100000/100000 | `side.speed_boost` | 100000/100000 | `side.accuracy_boost` | 100000/100000 |
| `side.evasion_boost` | 100000/100000 | `side.wish.0` | 100000/100000 | `side.wish.1` | 100000/100000 |
| `side.future_sight.0` | 100000/100000 | `side.future_sight.1` | 100000/100000 | `side.force_switch` | 100000/100000 |
| `side.force_trapped` | 100000/100000 | `side.baton_passing` | 100000/100000 | `side.shed_tailing` | 100000/100000 |
| `side.revival_blessing` | 100000/100000 | `side.slow_uturn_move` | 100000/100000 | `side.times_revived` | 100000/100000 |
| `side.last_move_failed` | 100000/100000 | `side.lum.kind` | 100000/100000 | `side.lum.slot` | 100000/100000 |
| `side.lum.choice` | 100000/100000 | `side.switch_out_move_second_saved_move` | 100000/100000 | | |
| `side.sc.*` (19 groups) | 100000/100000 each | `side.dur.*` (13 groups) | 100000/100000 each | | |
| `global.weather` | 50000/50000 | `global.weather_turns` | 50000/50000 | `global.terrain` | 50000/50000 |
| `global.terrain_turns` | 50000/50000 | `global.trick_room` | 50000/50000 | `global.trick_room_turns` | 50000/50000 |
| `global.team_preview` | 50000/50000 | | | | |

The 19 `side.sc.*` groups are `aurora_veil, crafty_shield, healing_wish, light_screen,
lucky_chant, lunar_dance, mat_block, mist, protect, quick_guard, reflect, safeguard, spikes,
stealth_rock, sticky_web, tailwind, toxic_count, toxic_spikes, wide_guard`; the 13
`side.dur.*` groups are `confusion, encore, lockedmove, slowstart, taunt, yawn, throatchop,
cudchew, disable, syrupbomb, healblock, partiallytrapped, magnetrise`.

### 1.4 Supplementary — the same encoder on the 80,110-state MIXED corpus

**The pair-A corpus is narrow.** Measured on it, these fields never leave 0:
`active_move_actions`, `times_attacked`, `substitute_health`, `wish.*`, `future_sight.*`,
`times_revived`, `reveal_mask`, `stellar_boosted_types`, `rest_turns`, `sleep_turns`,
`terrain_turns`, `trick_room_turns`, and `evs` is constant 85. `weather_turns` is always −1.
So a pair-A-only gate exercises the *encode path* of those columns at value 0 and nothing
more. That is a real limitation of the acceptance corpus, not of the encoder, and it is why
§2(B) below exists.

Because it costs 3 minutes and removes the doubt, the identical gate was also run on the
pre-existing 80,110-state mixed corpus (`/tmp/states.tsv`: 40,097 ladder serve-time states +
40,013 engine self-play states — read from files already in the repo; **no generation**).
That corpus does exercise the wide ranges (`times_attacked` to 28, `active_move_actions` to
39, negative pp, banked moves on 7.5% of side vectors, switch targets on 33%):

```
VARIANT FULL : 101/101 groups, 57,919,530 comparisons, 100.000000%, 0 vocab misses — PASS
VARIANT LEAN : 101/101 groups, 57,919,530 comparisons, 100.000000%, 0 vocab misses — PASS
```

This reproduces the spec's headline number with the production encoder rather than the
reference one.

### 1.5 Range assertions

No `RANGE_ASSERTS` violation on either corpus. Observed extremes on pair A stay well inside
the declared ranges; the mixed corpus reproduces the spec's §6.3 table (`pp` ∈ [−1, 64],
`times_attacked` to 28, `active_move_actions` to 39 — the two "caps" that the engine comments
claim and the data refutes, and which the new columns carry unclipped).

---

## 2. SWITCH-SIBLING DISTINGUISHABILITY

### 2.1 (A) The required test — real successors of one root

For each root with ≥2 legal switch arms, build the modal successor of each switch arm against
the **same** opponent reply (`dataset.successor`, the lab's own construction), then compare.

```
roots used                                        200
switch-sibling PAIRS compared                    1238
pairs with DIFFERENT encodings                   1238   (100.0000%)
successor states encoded                          777
  active_index decodes correctly (both sides)   777/777
  last_used_move decodes correctly (both sides) 777/777
  ENTIRE state round-trips (all 101 groups)     777/777
RESULT (A): PASS
```

Asserted in code, not merely printed: `assert n_diff == n_pairs` and
`assert n_ai_ok == n_lum_ok == n_full_ok == n_succ`.

### 2.2 (B) The sharper test — one-field mutations, new encoder vs today's

(A) is necessary but weak: two switch successors also differ in hp, in which species is
active, and so on, so *any* encoder would separate most of them. (B) isolates each critical
fix by mutating **exactly one field** of a real state string and asking whether the encoding
moves at all. Today's shipped `valuenet/encoder.py` (v6 lab recipe: `BENCH_SORT=1`,
`PP_TRUE_MAX=1`) is measured on the identical pair.

| mutated field | new encoder | today's encoder | what today's loses |
|---|---|---|---|
| `last_used_move` `switch:1` → `switch:3` | **DIFFERS** | COLLIDES | `encoder.py:497` discards the `Switch` payload — 33.34% of side vectors |
| banked move `UTURN` → `VOLTSWITCH` | **DIFFERS** | COLLIDES | `encoder.py:539` collapses the move identity to a bool |
| `active_move_actions` 5 → 9 | **DIFFERS** | COLLIDES | `encoder.py:376` clips at 2 — 14.96% of serve-time slots exceed it |
| `times_attacked` 7 → 12 | **DIFFERS** | COLLIDES | `encoder.py:375` clips at 6 — 1.41% at serve |
| volatile `ROOST` → `OCTOLOCK` | **DIFFERS** | COLLIDES | 73 of 107 variants have no column, dropped **silently** |
| bench party permutation (party slots 2↔3) | **DIFFERS** | COLLIDES | `BENCH_SORT` re-sorts by species vocab id; party order unrecoverable |
| `wish` timer 1 → 2 | **DIFFERS** | COLLIDES | `encoder.py:466` collapses the timer to a bool |
| `substitute_health` 0 → 40 with no `SUBSTITUTE` volatile | **DIFFERS** | COLLIDES | `encoder.py:491` gates the value on the volatile |

```
RESULT (B): PASS — 8/8 mutations separated by the new encoder, 0/8 by today's.
```

Eight state pairs that are genuinely different positions map to **bit-identical vectors**
under the shipped encoder. That is the concrete form of "the parameterization might be why
the net is weak", and it is gone.

### 2.3 The four required critical fixes, and where they live

| fix | column(s) | file:location |
|---|---|---|
| `active_index` + bench permutation recoverable | `active_index_0..5` (6, one-hot); slot order is party order, **no `BENCH_SORT`** | `llencoder._slot_order`, `Variant.side_blocks["active"]` |
| `last_used_move` including switch target | `lum_kind_*` (5, incl. `unslotted`), `lum_move_slot0..3`, **`lum_switch_slot0..5`**, `lum_unslotted_move` (embedding id) | `llencoder.encode_columnar`, lum block |
| unclipped `active_move_actions`, `times_attacked` | `/64`, no clip. Clipped forms survive only as DERIVED (`*_capped`) | `encode_columnar`, marked `# UNCLIPPED` |
| banked-move identity, all 107 volatiles | `banked_move` embedding id + `has_banked_move`; `vol_*` × **107** | `Variant.side_blocks["vol"]`, `sid` |

---

## 3. ENCODE THROUGHPUT AND FEATURE WIDTHS

Single process, `nice -n 10`, 40,000 pair-A states, best of 3, measured on the **production
path** (4,096-state chunks written into a preallocated array — exactly what `lldataset.py`
does, so the number includes the float16 store).

| encoder | ms/state | states/s | parse | encode | numeric cols | 1M rows f32 |
|---|---|---|---|---|---|---|
| `valuenet/lossless_encoder` (reference) | 0.368 | 2,714 | — | — | 1,863 | 7.5 GB |
| **`llencoder [full]`** | **0.147** | **6,785** | 0.128 | 0.019 | **1,863** | 7.5 GB |
| **`llencoder [lean]`** | **0.144** | **6,927** | 0.129 | 0.015 | **1,206** | 4.8 GB |

* **2.5× the reference; 1M positions in ~2.5 minutes single-core**, ~40 s on 4 cores.
* The **vectorised encode is 0.015–0.019 ms/state (53–67k states/s)**; the remaining 87% is
  string parsing, which is where any further speedup has to come from (a Rust or `re`-free
  tokenizer, or reusing the engine's own parse). Not done — the current speed is not the
  binding constraint on anything in the lab.
* `lldataset.py` end-to-end (gzip + json + parse + encode + store) ran at **4,698 pos/s**
  for `full` and 2,917 pos/s for `lean`; there the bottleneck is `gzip`+`json.loads` on the
  shards, not the encoder.
* Machine variance is real: an earlier uncontended run of the same code measured 0.091
  ms/state. The table reports the slower, conservative figure.

### 3.1 Widths

| | today (v6/v7 shipped) | `lean` | `full` |
|---|---|---|---|
| per-mon numeric | 87 | **68** (all EXACT) | 121 (68 EXACT + 53 DERIVED) |
| per-side numeric | 99 | **186** | 196 (186 + 10) |
| global numeric | 18 | **18** | 19 |
| **total numeric** | **1,260** | **1,206** | **1,863** |
| per-mon embedding ids | 9 | 15 | 15 |
| total embedding ids | — | 184 | 184 |
| provably constant columns on real data | 257 (20.4%) | 0 in the EXACT set by construction | 0 in the EXACT set |

**`lean` is 54 columns narrower than what ships today and is provably lossless.**

### 3.2 float16 storage cost, measured

`lldataset.py` stores `feats` as float16 to fit the 8.6 GB box (`dataset.py` does the same).
The encoder emits float32 and the gate runs on float32. Re-running the round-trip with the
features round-tripped through float16:

```
[full] 2000 states — fields failing under float16: 1
    mon.weight_kg   22000/24000  (91.6667%)
[lean] 2000 states — identical result
```

**Exactly one column** loses exactness: `weight_kg`, on 8.33% of mon slots, at the 1-decimal
resolution PS ships. Every other one of the 101 groups survives float16 bit-exactly.
Mitigations, in order of preference:

1. It costs the *decoder*, not the *net*: on 240,000 occupied pair-A mon slots, species →
   `weight_kg` is a **function** (0 violations, 13 distinct species), so the value is fully
   recoverable from the species embedding. Caveat: this is not true in general — Autotomize,
   Heavy Metal and Light Metal change weight without changing species, and none occur in
   pair A.
2. `LL_DTYPE=float32` makes the cache exact at 2× memory.

---

## 4. THE FLAT ARCHITECTURE

### 4.1 What was removed

The incumbent (`valuenet/train.py:ValueNet`, mirrored bit-exactly by `evallab/labmodel.py`)
computes

```
trunk_in = [ mlp(a1), Σᵢ mlp(b1ᵢ), mlp(a2), Σᵢ mlp(b2ᵢ), sf1, sf2, g ]     # 728 wide
```

Two things die there: the **sum-pool** is permutation-invariant, so nothing downstream can
tell which bench slot a feature came from; and the **shared per-mon MLP** forces one function
to serve "my active", "my bench", "their active" and "their bench". `BENCH_SORT` had already
destroyed party identity before the pool.

`flatnet.FlatValueNet` has **zero pooling operations and zero shared per-mon MLPs**. Slot
identity is positional.

### 4.2 The layout (generated by `python flatnet.py`, so it cannot go stale)

```
FLAT INPUT LAYOUT -- variant 'full'
  numeric block: 1863 columns
    [    0:  121) our active     121 per-slot columns
    [  121:  242) our bench 1    121
    [  242:  363) our bench 2    121
    [  363:  484) our bench 3    121
    [  484:  605) our bench 4    121
    [  605:  726) our bench 5    121
    [  726:  847) their active   121
    [  847:  968) their bench 1  121
    [  968: 1089) their bench 2  121
    [ 1089: 1210) their bench 3  121
    [ 1210: 1331) their bench 4  121
    [ 1331: 1452) their bench 5  121
    [ 1452: 1648) our side block 196
    [ 1648: 1844) their side block 196
    [ 1844: 1863) global          19
  embedding block: 12 x 192 (per-slot, shared TABLES) + 2 x 32 = 2368 columns
  total trunk input: 4231
  pooling operations: 0   shared per-mon MLPs: 0
```

`lean` is the same shape at 68/186/18 → 1,206 numeric, **3,574** trunk input.

Canonical slot order is `our active, our bench 1..5, their active, their bench 1..5, side 1,
side 2, global` — which is *already* `llencoder`'s column order, so the array coming out of
the encoder **is** the input vector: no reshape, no gather, no per-slot module. Bench slots
1..5 are the non-active party members in ascending **party index** order (not species order),
which is what makes the layout a bijection with party index given `active_index`.

Embedding **tables** are shared (a species is a species wherever it stands); the 12 resulting
192-dim vectors are concatenated into fixed positions, so sharing a table is not sharing a
function.

### 4.3 Capacity knob and baseline switch

```bash
WIDTH=512 DEPTH=3 DROPOUT=0 VARIANT=full|lean   # flat arm
```
`build("old", …)` returns `labmodel.LabValueNet` unchanged — `labmodel.assert_baseline_parity`
proves it bit-identical to the shipped `train.ValueNet`. `train_flat.py` picks the arm from
the `.npz` (`feats` key → flat cache, `a1_ids` key → incumbent cache), so both arms run
through one code path, one game-level split, one metric. `PLY_STRIDE` filters on the **ply
column**, so a flat cache built with `POS_STRIDE=4` and the incumbent cache built with
`POS_STRIDE=1` select the *identical* positions without rebuilding either.

---

## 5. SMOKE TRAIN — plumbing only, NOT the experiment

**Labels are the existing single-outcome game results.** That is the objective the lab has
already shown is not the one that decides move quality; the real labels come from the
playout-averaging agent. Nothing below is evidence about move ranking — sibling
discrimination and regret are **not measured here**.

Corpus: `data/el1/A2k`, pair A, `POS_STRIDE=4` → **157,567 train / 17,333 val positions over
18,000 games; 19,416 held-out rows over 2,000 held-out games**. Split by game
(`labenv.is_holdout` + `VAL_FRAC=0.10` inside the train games). Train base rate p = 0.3056.

**Base-rate constant predictor CE (the reference every number below is read against):
train 0.6155 · val 0.6126 · held-out 0.6090.**

| arm | params | best val CE | **held-out CE** | s/epoch | wall |
|---|---|---|---|---|---|
| flat `full` w256 d3 | 1,221,525 | 0.4951 | **0.4905** | 7.4 | 162 s |
| flat `full` w512 d3 | 2,502,037 | 0.4959 | **0.4928** | 10.7 | 169 s |
| flat `full` w512 d4 | 2,764,693 | 0.4926 | **0.4913** | 11.9 | 184 s |
| flat `full` w1024 d3 | 5,456,277 | 0.4962 | **0.4948** | 18.9 | 297 s |
| flat `lean` w512 d3 | 2,165,653 | 0.5375 | **0.5417** | 9.7 | 148 s |
| **baseline** old encoder + pooled | 368,981 | 0.4478 | **0.4439** | 6.4 | 102 s |
| constant predictor | — | 0.6126 | 0.6090 | — | — |

15 epochs, `LR=3e-4`, `BATCH=1024`, `AdamW(wd=1e-4)`, 3 threads, best-val checkpoint.

**The pipeline runs and learns:** every arm beats the base rate by a wide margin (0.609 →
0.44–0.49 held-out), the cache builds in 37 s for 174,900 rows, and an epoch is 7–19 s.

### 5.1 What the numbers say, stated honestly

* **The flat net is worse than the pooled baseline on outcome-CE at this data scale**
  (0.491 vs 0.444 held-out). This is a real result and it is reported as-is.
* **Capacity is not the constraint — data is.** Held-out CE is *monotone in the wrong
  direction* across the width sweep (256 → 0.4905, 512 → 0.4928, 1024 → 0.4948), and the
  per-epoch histories show the larger nets overfitting hard: at w1024, train CE falls to
  0.330 while val CE rises to 0.615. The pooled baseline does not overfit at all
  (train 0.450 / val 0.448). Sum-pooling plus the shared MLP is a strong prior, and with
  157k noisy 0/1 labels the prior wins.
* **The DERIVED columns earn their 657 columns as regularisation.** `lean` (EXACT only)
  overfits fastest and is worst (0.5417): train CE 0.389 against val 0.597 by epoch 14. The
  spec predicted this ("the shipped nets measurably lean on them"); the lab now measures it.
  Losslessness is a property of the EXACT set, so `lean` costs nothing in information — what
  it costs is inductive bias.
* **The plumbing is sound, not merely under-tuned.** A fit check on 3,000 rows drives train
  CE 0.680 → **0.0897** in 60 full-batch steps (constant-predictor CE on that subset: 0.634).
  The flat net can represent and fit the data; the gap is generalisation under a noisy
  objective, not a broken forward pass.
* **This does not decide the architecture question.** The comparison that matters is sibling
  discrimination / regret against the 5M-iteration oracle, with playout-averaged labels. On
  outcome-CE the two arms are being scored on the objective the lab exists to move away from.

---

## 6. ASSUMPTIONS AND LIMITATIONS

1. **The state string is the definition of "the full state."** The 23 struct fields
   `State::serialize` drops (spec §2.1 — `damage_dealt`, `stats_lowered/raised`,
   `active_hp_spread`, the 9 Transform fields, 8 derivable ones) are upstream losses that no
   encoder can recover. 15 are real; fixing them needs a `serialize` change, which is out of
   scope here.
2. **The pair-A acceptance corpus is narrow** (§1.4). 13 fields are constant on it. Their
   round-trip is proven structurally at value 0 by the pair-A gate and at full range by the
   supplementary mixed-corpus gate; `reveal_mask`, `stellar_boosted_types`, `future_sight`,
   `team_preview` and `wish[0] > 1` are **still not exercised at nonzero values by any real
   corpus available locally** — they need synthetic states
   (`valuenet/dump_testvectors.py` is the place).
3. **The fast parser assumes engine-cased (upper) names on the hot path**, falling back to
   `.upper()` on a dict miss. The gate compares every parsed field against the reference
   parser, so a lower-casing producer would cost speed, never correctness.
4. **Vocabulary injectivity is required for losslessness** and is verified, not assumed: 0
   misses on 600,000 pair-A mon slots and on the mixed corpus. `LosslessVocab` records
   misses; the gate treats one as failure.
5. **float16 is a cache STORAGE choice**, not an encoding choice, and it costs exactly one
   column (§3.2).
6. **The flat net has no weight sharing across slots**, so it must see each slot in each
   role. Pair A is a fixed roster — the regime where that is cheapest. **Generalisation
   across rosters is untested and must be re-measured before any of this goes near the
   shipped pipeline.**
7. **No side-swap augmentation** in any arm, matching the rest of the lab.
8. **No parity work.** The Rust mirror (`evaluate_nn.rs`) is untouched and does not implement
   this layout; spec §9 remains the specification for that, and nothing here may be exported
   to a `.bin` until it exists.
9. **The producer bugs the spec identified are NOT fixed** — they are producer-side, not
   encoder-side: `poke_engine_helpers.py:429-430` hard-codes `switch:0` at serve time, and
   `last_move_failed` / `switch_out_move_second_saved_move` are never set by foul-play's state
   builder. The encoder now carries these fields losslessly, which means a train/serve skew
   that used to be hidden by the encoder is now visible to the net. **This matters before
   anything trained on self-play is served on the ladder.**

---

## 7. WHAT IS AND IS NOT SETTLED

**Settled:** the parameterization is no longer a suspect. 101/101 field groups round-trip at
100.000000% over 50,000 pair-A states (36.15M value comparisons) and 80,110 mixed states
(57.92M), in both widths, with zero vocab misses; eight single-field mutations that today's
encoder cannot see are all separated; every switch sibling encodes differently and decodes
back to the exact source state. The gate is 100 s for 50k states and is the permanent
regression check.

**Not settled:** whether the flat architecture is better. The smoke train says it is *worse*
on outcome-CE at 157k positions and that the gap is overfitting, not capacity — but outcome-CE
is the wrong metric and these are the wrong labels. That verdict waits on the
playout-averaging labels and on `evaluate.py`'s sibling discrimination.
