# pokemon-fast-bot — workspace root

The pipeline code around the two bot repos: engine-conformance checking and
value-net training-data generation / training. Results, corpora, game
archives, and old model artifacts are deliberately NOT here (S3 or
regenerable); this repo is the *process*.

## Workspace assembly

Clone this repo as the workspace root, then the three checkouts inside it:

```bash
git clone https://github.com/xthddrt/pokemon-fast-bot workspace && cd workspace
git clone https://github.com/xthddrt/foul-play
git clone https://github.com/xthddrt/poke-engine
git clone https://github.com/smogon/pokemon-showdown && git -C pokemon-showdown checkout d43fb79
git clone https://github.com/xthddrt/valuenet
git clone https://github.com/xthddrt/truestate
cd foul-play && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
# .env (secrets, never committed): see foul-play/ladder/README.md for the vars
```

Staleness check: `bash foul-play/tools/check_ps_drift.sh` fetches upstream
and reports which layer moved (set data / generator logic / sim mechanics)
with the refresh recipe for each.

`pokemon-showdown @ d43fb79` is the pinned ground truth every conformance
result and the v8 training corpus were built against (`data/random-battles/
gen9/` verified unchanged 2026-08-14). Node ≥ v22 is required to run PS.
The ladder bot itself needs only the two bot repos — see
`foul-play/ladder/README.md` for that quickstart.

## Process 1 — PS-conformance of the engine

Ground truth is `pokemon-showdown/` (READ-ONLY, never edit). The checker
lives in foul-play, fixes land in poke-engine.

1. **Sweep** a fresh corpus of real games:
   `foul-play/sweep_conformance.py <corpus_dir> <out_dir> [workers] [unit]`
   — runs `check_replays.py --damage-tolerance 0` with exact-damage
   membership (`FP_MEMBERSHIP_REPLAY=1`), one process per unit, aggregates
   findings JSON. `aggregate_conformance.py` merges; `audit_selection.py`
   triages.
2. **Triage checker-first**: prove a finding is NOT a checker/harness
   artifact before touching engine code (perspective-asymmetry is the cheap
   test). The checker itself is `foul-play/fp/replay/checker.py` +
   `damage_membership.py` + `comparator.py`.
3. **Targeted fixes**: `corrections/` is the hammer harness — `capture.py`
   captures states, `hammer.py` replays a finding class against the engine,
   `verify_flip.py` proves the fix flips the finding, `acceptance/` holds the
   pinned acceptance net. Damage stays two-branch (alive/dead with correct
   boundary+weights), never 16-roll fans.
4. **Gate**: the engine is "correct" only after TWO consecutive clean fresh
   10k sweeps (0 hard / 0 soft / 0 diverged); any fix resets the streak, and
   the next sweep is always a FRESH corpus, never a re-sweep of a clean one.
   Rebuild + reinstall the wheel before sweeping
   (`pip install foul-play/../poke-engine/poke-engine-py --config-settings=`
   `"build-args=--features poke-engine/terastallization --no-default-features"`).

## Process 2 — training-data generation + training a new net

The v8-line recipe (produced v8b, the current champion). Heavy steps run on
EC2 spot (`evallab/cloud/`, `valuenet/cloud/`); S3 bucket
`pokebot-valuenet-389825051723` holds corpora, encoded caches, and nets.

1. **Corpus**: `evallab/` — `rbpool.py` (team pool) → `generate.py` /
   `corpus_select.py` (positions from games played by the CURRENT champion)
   → `playout.py` (label_p = playout win rate, N playouts at 2000 iters —
   the label that beat single-outcome by miles). `plc1_meta.py` /
   `plc2_stage2.py` show the exact plc runs. Holdout is carved BY TEAM PAIR
   and never re-carved.
2. **Encode**: `evallab/enc_adopted.py encode` — arm A (valuenet/encoder.py
   under labenv's pinned recipe: BENCH_SORT=1 PP_TRUE_MAX=1
   DROP_TIMES_ATTACKED=1) + enc2's 14-col setup block remapped to arm A
   bench order. `enc_adopted.py verify` proves the slot remap.
3. **Train**: `evallab/vt_n.py` → `vt_lib.build_net('old', (128,256), seed,
   add='setup')`. Evaluate on the frozen holdout only (`evallab/evaluate.py`).
4. **Export**: `evallab/export_v8.py ckpt.pt out.bin` → PKNN v8 with vocab +
   max-PP table. The `.constants.json` sidecar is MANDATORY and its values
   are re-MEASURED per net (sigma_R from real searches — see
   `valuenet/nets_v8b/v8b_s1.constants.json` `_measurement` for the method);
   never carried forward.
5. **Validate**: `valuenet/sprt/` — `launch_v8duel.sh` (one spot box,
   sharded `run_duels.py`, sha-pinned preflight), scored by `sprt.py`.
   Fan out by SHARD_INDEX only, one process per shard.
6. **Ship**: promote by updating the launchers' default net
   (`ladder-games/run_game.sh`) — sidecar beside the bin.

Runbooks with the operational detail: `valuenet/GENERATION_RUNBOOK.md`,
`valuenet/CLOUD_PLAYBOOK.md`, `evallab/README.md`, and the farm postmortem
`truestate/farm/POSTMORTEM_RUN1.md` (read before any multi-box fleet —
62% of run-1's lane-time died to one login endpoint).

## What is deliberately absent

`evallab/data/` (12G encoded caches — S3), `ladder-games/games/` (2.7G
archives), `valuenet/m4_*` (superseded model artifacts — S3),
`valuenet/tune/` (sweep results), corpora, ledgers, census files, progress
docs. valuenet/ and truestate/ are their own repos (xthddrt/valuenet,
xthddrt/truestate), cloned alongside. The champion net rides in BOTH
`valuenet/nets_v8b/` (that repo) and
`foul-play/ladder/nets/` so either clone is runnable alone.
