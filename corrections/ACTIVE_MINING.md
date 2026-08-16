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

**The pinned ruling label player = v8b_s1 @ 2,000 iters.** Ruling v1 (the
t18 target 0.096) was measured with playouts driven by v8b_s1, so every
subsequent ruling uses the same player — and s1 is frozen forever, so
targets stay comparable across ledger generations no matter how many h<n>
carves ship. (The training corpus's own labels were made by the v6-era
label player; that definition gap is accepted — rulings are exceptions
carved against measured self-play truth, not corpus rows.)

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

**Phase 2 — backward endgame-first scan (n = 10; Sally's protocol,
2026-08-15).** Per game, scan decisions BACKWARD from the last turn, 10
label-player playouts per state. Stop at the FIRST state where BOTH:
- z = |e − p̂₁₀| / SE_AC(p̂₁₀) ≥ 2  (Agresti-Coull SE — the plain formula
  degenerates to 0 at p̂ ∈ {0,1}), and
- |e − p̂₁₀| ≥ 0.15 (absolute floor, guards near-terminal states where SE
  collapses).

Why backward: endgame playouts are both CHEAPER (short continuations) and
MORE ACCURATE (less compounding playout-policy noise), so a small n is
trustworthy exactly where accuracy is preferred — the endgame. First-hit-
stop yields at most ONE ruling per game (the latest provably-wrong
decision) and dedups correlated collapse turns for free. Scan depth capped
(default 25 decisions) to bound the no-error case.

**Phase 3 — confirm (n = 30, FRESH seeds).** Re-playout the hit 30× with
new seeds. Admit to the ledger only if |e − p̂₃₀| ≥ 0.10 and z ≥ 3. Fresh
seeds are mandatory: a state flagged by its own screening sample has an
inflated measured gap (winner's curse); the confirm run gives the unbiased
target. Ledger row as in VALUE_HAMMER §3.4 with target = p̂₃₀ and TWO-SIDED
band [p̂₃₀ − SE₃₀, p̂₃₀ + SE₃₀] (underestimates are as real as
overestimates; hammer_value.py accepts "band": [lo, hi]).

**Phase 4 — batch.** Confirmed rulings from all games in the round are
appended together and hammered in ONE run — the hammer is batch-native
(every ledger ruling trains simultaneously from pristine s1), so 5–10 spots
cost one ~70–90 s hammer, not 5–10. Between confirm and hammer sits Sally's
assessment: mine_value.py writes ledger_rows.json and STOPS; rows are
appended to the ledger and hammered on her command.

**Phase 5 — hammer + gates + ship.** Exactly VALUE_HAMMER §3.5–3.7
(~70 s + gates). All rulings cumulative from pristine s1, engine-gated,
drift bars enforced, shipped as the next h<n>.

**Phase 6 — stopping + validation.**
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

| phase | cost (G = 5, 4-way parallel — half the cores) |
|---|---|
| 5 full-info games @ 4500 ms/turn | ~5–10 min |
| backward scans (10 playouts/state, endgame-short) | ~5–25 min |
| confirm hits × 30 playouts | ~2–5 min each |
| hammer + gates (one batch) | ~2 min |
| **total** | **~15–40 min wall, unattended** |

Backward scanning is what keeps this cheap: endgame playouts run seconds,
and the scan stops at the first hit. Scale-out: rounds parallelize
trivially on EC2 CPU boxes (the farm harness) if mining goes beyond
occasional rounds.

## 3. Known limits

- **Self-confirmation**: playouts are played BY the same net, so a position
  the net misplays can look lost "for real." Bounded — the label player IS
  the value definition — but it means mining sharpens the net toward its own
  play; genuinely new strategic knowledge still arrives via full retrains.
- The prefilter's outcome signal is 1 sample; the 20% random sidecar is the
  hedge, not a proof of full recall.
- Statistical thresholds assume independent playouts; playout draws share
  the seeded engine's chance handling, which is by design (same as corpus).

## 4. Implementation (corrections/mine_value.py)

    foul-play/.venv/bin/python corrections/mine_value.py \
        [--games 5] [--ms 4500] [--tag mine1] [--screen-n 10] \
        [--confirm-n 30] [--max-scan 25]

Self-contained: plays the games (audited net via env-isolated
subprocesses), evals every state through leaf_prof, backward-scans with the
pinned label player (s1), confirms, prints the assessment table, writes
<work>/ledger_rows.json. hammer_value.py accepts the two-sided
"band": [lo, hi] rows it emits. Teams from the local holdout corpus
(*.teams.json), ring-paired so no team repeats within a round.
