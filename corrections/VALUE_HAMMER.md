# VALUE HAMMER — spec and process

*(Sally's design, 2026-08-15. The value variant of HAMMER_SPEC Part 2:
conform the champion net's evaluation of ruled positions to measured playout
truth, with minimal change to the rest of the function. Empirics from the
v8b_h1 build and the anchor sweep of the same day.)*

## 1. Principle

Each ruling is a constrained optimization:

    minimize   ∫ (f_new − f_old)² dμ          (function-space distance)
    subject to f_new(ruled states) ≈ playout truth

where μ is the distribution of positions the bot actually meets. The loss
that implements it:

    L = BCE(f_new(ruled), target)  +  w · MSE(f_new(anchors), f_old(anchors))

The anchor term is a Monte-Carlo estimate of the integral — a frozen copy of
the source net is the teacher (**self-distillation**). Overfitting the ruled
states is accepted by design; the accumulating ledger dilutes per-example
overfit over time. What is NOT accepted is global drift: without the anchor,
8 same-target rows moved the whole net's output scale by mean |Δlogit| 1.29
in three steps (measured) — the lazy gradient direction is the output bias,
not the position.

## 2. Empirical results (2026-08-15, ruling v1-2665399837-t18)

Target 0.096 (104 playouts, 10W–0D–94L; the evaluator said 0.39). Band 0.11.
Held-out probe: 2,000 archived-game states never used as anchors, split
early-game (decision ≤ 5) / late-game (decision ≥ 15).

| variant | steps | ruled evals | anchored drift | held-out EARLY | held-out LATE |
|---|---|---|---|---|---|
| no anchor, lr 1e-4 | 3 | 0.13 | — | ~1.29 mean (catastrophic) | — |
| 6,000 anchors, w=30, lr 3e-6 | 1401 | 0.099–0.11 | 0.0087 | 0.0232 mean / 0.0958 p99 | 0.0377 / 0.161 |
| 10,310 anchors, w=30, lr 3e-6 | 1604 | 0.099–0.11 | 0.0097 | 0.0202 / 0.0808 | 0.0295 / 0.140 |
| **10,310 anchors, w=100, lr 1e-6** | 4101 | **0.099–0.11** | **0.0045** | **0.0178 / 0.0711** | **0.0270 / 0.120** |

Readings that drive the design:
- **The loophole gap**: anchored points hold 2–4× tighter than held-out
  points. A fixed anchor sample is a fixed quadrature the optimizer threads
  between. More coverage measurably closes it (6k → 10.3k improved every
  held-out number).
- **Drift falls monotonically along (more anchors, higher w, lower lr)** at
  EQUAL conformance — the only cost is steps (1.4k → 4.1k, still minutes on
  CPU). The mechanism: a small enough lr keeps the ruled gradient from
  overshooting between anchor corrections, so the optimizer finds the
  low-norm path to the constraint instead of the fast one. Push this axis
  until wall-time hurts, not before.

**Minibatch follow-up (same day)**: replacing the full-batch anchor pass
with a 512-state minibatch cycled in shuffled epochs (§5.1's design, landed
early) — same ruled conformance in every run, gates PASS:

| recipe (all minibatch-512 except baseline) | wall clock | held-out EARLY | held-out LATE |
|---|---|---|---|
| full-batch w=100 lr=1e-6 (baseline) | 19.2 min | 0.0178 | 0.0270 |
| w=100, lr=1e-6 | 2.8 min | 0.0160 | 0.0237 |
| **w=100, lr=3e-6 (DEFAULT)** | **70 s** | 0.0165 | 0.0241 |
| w=30, lr=3e-6 | 46 s | 0.0195 | 0.0285 |

Readings: minibatching does not cost drift — it slightly improves it
(gradient noise regularizes; epochs still enforce every anchor). At w=100
the lr can rise to 3e-6 for free (drift within noise of the best), so the
speed/correctness trade-off dissolved: 70 s is the frontier. Only weakening
the pin (w=30) actually degrades drift, for 24 s saved — rejected.
- **LATE > EARLY drift is desired**: the carve generalizes along
  wall-endgame features — that is the lesson propagating, not damage. The
  damage gauge is EARLY (unrelated) drift only.
- **Damage scale**: state-dependent drift of 0.02–0.03 logits ≈ 0.005–0.0075
  probability — flips only decisions inside the statistical tie band
  (arm gaps ≤ ~0.005), i.e. decisions that were coin flips. Uniform drift
  components cancel entirely in arm-vs-arm comparison. Independently trained
  seeds differ by far more than this and all laddered fine — seed noise is
  the operational definition of "zero drift."

## 3. The process (per loss)

1. **Diagnose**: from the loss's turn table, identify the decision where the
   eval was wrong (staircase decay while the opponent plays predicted moves
   is the signature). Pull that decision's world states from the archive.
2. **Measure truth**: playout-label the states — N ≈ 100 playouts, 2000
   iters/side, the label recipe (`evallab/playout.py` semantics; ~2 min
   local). N=10 answers only "is the eval wrong?" (SE ±0.095); quoting a
   target needs N ≈ 100 (SE ±0.03).
3. **Rule only where the measurement beats the prior**:
   |old_eval − playout_label| > 3 × label SE. A position where the old net
   was within playout noise belongs in the anchors, not the ledger.
4. **Append to `corrections/value_ledger.jsonl`**:
   `{id, game, decision, target, conform, states, n_playouts, note, ts}` —
   `conform` = target + ~1 label SE (e.g. truth 0.096 → band 0.11).
5. **Run the hammer**: `foul-play/.venv/bin/python corrections/hammer_value.py`
   — trains from the PRISTINE source ckpt (v8b_s1) on ALL ledger rulings
   cumulatively. Generations never stack carve-on-carve; every h<n> is
   "original + all rulings, freshly applied," so abandoning the line is just
   pointing the launcher back at s1.
6. **Gates (all through the real engine, leaf_prof logits)**:
   - every ruled state inside its band (+0.002 tolerance: f32 export
     rounding can land an eval ~0.0005 above a band the fp32 net exactly
     met — observed in the sweep) — HARD gate;
   - held-out drift report (early/late split) — ship bar: EARLY mean ≤ 0.03,
     p99 ≤ 0.10; if exceeded, raise anchor weight / lower lr / widen anchors
     before shipping;
   - one real search on a ruled state: root value should approach the target
     (it will sit somewhat above — successor states are uncarved; if the
     search must also conform, add playout-labeled successor states to the
     ruling).
7. **Ship**: export bin + sidecar (constants inherited from source — a value
   carve does not move sigma_R materially; `_hammered` provenance field
   records the ruling ids), flip the launcher default, commit valuenet +
   foul-play/ladder + the ledger.

## 4. THE PROTOCOL (final, Sally 2026-08-16) — v7 + 4096 anchors

One GPU run, <=20 min guaranteed, any ledger size:

    python hammer_value.py --net <champion>.pt --tag h<n> \
        --anchor-w 30 --w-min 30 --anchor-bs 4096 --device cuda

- **Anchors**: the FULL training corpus, mirror-doubled (both seatings of
  every row, ~4M views), each pinned to the champion's own outputs
  (teacher refs disk-cached per net). Hard w=30, NEVER softened globally.
- **Ruled force**: normalized by ledger size (n/8 scaling) so each carve
  pulls identically at 8 or 8,000 rulings; every ruling ships with its
  measured-target mirror twin (up/down balance by construction).
- **Per-state escalation**: any state outside its +-2SE band (Sally 2026-08-16: conservative bands, protocol-wide) after 1,000
  steps doubles its own weight (cap 64x). Force goes only where needed;
  gentle caps (8x) provably fail extreme carves.
- **Hard wall**: 10-min fine-tune, band checks every 100 steps (per-step
  checks cost a GPU sync = 5x slower), then export + gates ALWAYS run,
  printing any missed bands for triage.
- **Ledger hygiene**: multi-world rulings contribute <=3 representative
  states (8 t18 duplicates at max force were 18% of total carve impulse).
- **Gates**: all bands (+-0.002 engine tolerance) AND frozen-bench Brier
  (bench_v1.jsonl) — measured cost at 88 states: -0.0018, equal to h2's
  cost at 6 states (6x better drift-per-correction than parity anchoring).

Measured reference run (v8c_s1 -> v8c_h1g): 13.5 min total, 8,100 steps,
88/88 bands, ruled mean |eval-truth| 0.350 -> 0.016.

## 4b. Legacy recipe (pre-2026-08-16, parity anchors)

- anchors: all 10,310 parity states, w = 100, lr = 3e-6, minibatch 512
  cycled in shuffled epochs, cap 30,000 steps, train until every ruled
  state ≤ its band (the minibatch sweep winner; these are the script
  defaults, so the bare command in §3.5 IS the recipe). ~70 s end-to-end
  steady-state (anchor encode + source parity logits cached after first
  run). Fallback if a future multi-ruling hammer misses the §3.6 drift
  bars: lr 1e-6 (2.8 min), then full batch (--anchor-bs 0, 19 min).
- Rollback: `RG_NN_WEIGHTS=../valuenet/nets_v8b/v8b_s1.bin` — the original
  is preserved locally, on GitHub (valuenet + foul-play/ladder), and on S3
  including the .pt.

## 5. Roadmap (v2 — the mathematically clean endpoint)

1. **Corpus-cycled anchoring**: replace the fixed parity sample with
   minibatch EPOCHS over the real training corpus (S3 `enc_plc12`, 2M rows;
   download once). Every state pins once per epoch — no fixed quadrature to
   thread between, no sampling bias, no tracking needed. This is the direct
   implementation of the integral in §1.
2. **Truth-labeled extras**: additional playout-labeled positions (wall
   endgames mined from archives) may join the ruled set under the 3×SE rule
   of §3.3 — with the global anchor in place they don't confound; they
   locally replace the "old net was right" prior with a measurement. As the
   ledger grows this loop converges to the principled retrain
   (corpus + hard examples) done incrementally. The systematic version is
   specced in ACTIVE_MINING.md (full-info self-play error mining).
3. **Escalation if a ruling resists** (band unreachable at acceptable
   drift): restrict updates to trunk parameters (freeze embeddings + mon
   MLP) to shrink the perturbation space, or send the ruling class to the
   next full retrain's corpus instead.

## 6. Known limits

- The search reads slightly above the net-level carve (0.239 vs 0.19 on the
  first ruling): successors are uncarved. Expected; see §3.6.
- Anchors pin the sampled distribution; truly novel state classes (never
  archived, not in the corpus) are pinned only by smoothness.
- Each shipped h<n> is duel-unvalidated by design (Sally's accepted risk).
  The drift bars in §3.6 are the substitute; a duel is warranted whenever
  cumulative rulings exceed ~20 or EARLY drift approaches the bar.
