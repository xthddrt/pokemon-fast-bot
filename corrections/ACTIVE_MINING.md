# ACTIVE MINING — self-play error mining for the value net

*(Sally's design, 2026-08-15: instead of waiting for ladder losses to expose
evaluator errors one at a time, mine full-information self-play for states
where the net provably disagrees with playout truth, and hammer only those.
Companion to VALUE_HAMMER.md — this doc defines how rulings are FOUND; that
doc defines how they are APPLIED.)*

## 0. The label player (what "2000 iters/move" means and why it is law)

Every label the v8 net was ever trained on was produced the same way: from a
recorded position, play the game to completion with BOTH sides choosing
moves via a fixed 2,000-iteration MCTS (seeded, iteration-limited —
hardware-independent and reproducible), repeat ~15×, label = fraction won.
That fixed player is the **label player**, and it defines what the net's
output *means*:

    net(s) estimates  P(side-one wins | both sides play the label player)

Not "objective win probability" — win probability under that specific
player. Consequently every truth measurement in this pipeline (screen and
confirm) MUST use the label player. Playouts at a different strength (e.g.
1/5 of ladder time ≈ a 4–5× stronger player) measure a *different quantity*;
a "3σ error" against it could be pure definition gap, and hammering it would
train the net toward a target inconsistent with its own 2M-row corpus.
Full-strength search appears in this pipeline ONLY to generate realistic
trajectories (phase 1) — never to define truth.

When a future full retrain upgrades the teacher/label recipe, all live
ledger targets must be re-measured with the new label player (or retired
into that retrain's corpus). A ledger target is only valid under the label
recipe that produced it.

## 1. Pipeline (one round)

**Phase 1 — trajectories.** G self-play games (default G = 25), engine vs
engine, full information (both teams fully known — one world, no sampling),
champion vs champion at ladder-strength search (4500 ms/decision,
first-decision 14000 ms). Teams drawn from the PS-exact sampler. Record
every decision: full-info state string, the net's raw eval e_t (direct net
call, no search), turn index, and the final game outcome. ~40 decisions per
game → ~1,000 states per round.

*Why full info*: the net only ever evaluates determinized full-info states,
so this measures it exactly on its operating domain, with zero
world-sampling noise contaminating the diagnosis.

**Phase 2 — free prefilter.** The game outcome is a free 1-playout label
for every state on its trajectory. Screen only:
- states with |e_t − outcome| ≥ 0.5 (eval confidently wrong about the
  actual result), plus
- a 20% uniform random sample of the rest (coverage — so the prefilter
  can't blind us to errors the outcome happened to agree with).

This cuts the screening bill ~4–5× at negligible recall cost. Budget
~200 screened states per round.

**Phase 3 — screen (n = 20).** For each screened state: 20 playouts with
the label player. p̂₂₀ = win fraction. Flag if BOTH:
- z = |e − p̂₂₀| / SE_Wilson(p̂₂₀, 20) ≥ 2  (Wilson/Agresti-Coull SE — the
  plain formula degenerates to 0 at p̂ ∈ {0,1}), and
- |e − p̂₂₀| ≥ 0.15 (absolute floor, guards near-terminal states where SE
  collapses).

Expected: real errors + ~5% statistical false positives; both go to
phase 4, which kills the latter.

**Phase 4 — dedup.** Contiguous flagged turns within one game are ONE error
(the same collapse seen 4 times). Keep the earliest state of each contiguous
run (the decision point where fixing the eval could still have changed
play); drop the rest.

**Phase 5 — confirm (n = 100, FRESH seeds).** Re-playout each survivor 100×
with new seeds. Admit to the ledger only if |e − p̂₁₀₀| > max(3 × SE₁₀₀,
0.10). Fresh seeds are mandatory: a state flagged by its own screening
sample has an inflated measured gap (winner's curse); the confirm run gives
the unbiased target. Ledger row as in VALUE_HAMMER §3.4 with
target = p̂₁₀₀ and TWO-SIDED band [p̂₁₀₀ − SE₁₀₀, p̂₁₀₀ + SE₁₀₀]
(underestimates are as real as overestimates; the hammer's band check
extends from ≤ to inside-interval).

**Phase 6 — hammer + gates + ship.** Exactly VALUE_HAMMER §3.5–3.7
(~70 s + gates). All rulings cumulative from pristine s1, engine-gated,
drift bars enforced, shipped as the next h<n>.

**Phase 7 — stopping + validation.**
- Round metric: confirmed rulings per 100 screened states. It should fall
  round over round as errors get fixed (playouts use the label player, not
  the hammered net, so fixed errors stop re-flagging via the eval side of
  the gap). Stop mining when a round confirms ~0 — the "fresh sweep until
  zero findings" rule applied to values.
- Duel trigger (VALUE_HAMMER §6): once cumulative rulings exceed ~20, or
  EARLY drift approaches the bar, run the standard 1000-game 250 ms duel
  vs the pre-mining champion before continuing.
- Every confirmed ruling is also a hard example for the next full retrain's
  corpus (VALUE_HAMMER §5.2) — mining rounds are the incremental form of
  that retrain.

## 2. Cost per round (8-core local; estimates)

| phase | cost |
|---|---|
| 25 full-info games, 8 in parallel | ~10–15 min |
| screen 200 states × 20 playouts | ~60–80 min |
| confirm ~5–15 states × 100 playouts | ~15–30 min |
| hammer + gates | ~2 min |
| **total** | **~1.5–2 h wall, unattended** |

Playout cost basis: ~2 min wall per 100-playout label on 8 cores (measured,
VALUE_HAMMER §3.2). Scale-out: rounds parallelize trivially on EC2 CPU
boxes (the farm harness) if mining goes beyond occasional rounds.

## 3. Known limits

- **Self-confirmation**: playouts are played BY the same net, so a position
  the net misplays can look lost "for real." Bounded — the label player IS
  the value definition — but it means mining sharpens the net toward its own
  play; genuinely new strategic knowledge still arrives via full retrains.
- The prefilter's outcome signal is 1 sample; the 20% random sidecar is the
  hedge, not a proof of full recall.
- Statistical thresholds assume independent playouts; playout draws share
  the seeded engine's chance handling, which is by design (same as corpus).

## 4. Implementation deltas required (small)

1. Trajectory recorder: engine self-play harness (duel infra) emitting
   {state, eval, turn, outcome} per decision.
2. Miner script (screen/dedup/confirm): thin wrapper over the existing
   playout labeler with Wilson-z logic.
3. hammer_value.py: two-sided band support (train + gate on
   inside-interval instead of ≤).
