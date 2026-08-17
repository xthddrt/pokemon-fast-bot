# V9 — full execution spec (Sally-approved 2026-08-17)

Goal: v9 = phase-split value net (shared embeddings + per-mon MLP, 3 trunks
blended by hp-mass phase), trained on a fresh decorrelated corpus, shipped
through engine support and a time-equalized SPRT duel. Sally has delegated
execution; canary every stage; 80/20 throughout; supervisors on everything
long-running. GPU instances: ask Sally when needed.

## Locked decisions (do not relitigate)

| decision | value | why |
|---|---|---|
| architecture | 3 trunks, hp-mass blend, hats knots p=0/0.5/1 | probe A/B: beats alive-count on every slice (pp1 0.0331 vs pp2 0.0350 bench) |
| phase variable | p = 1 − Σhp_frac/12 (`phase_of`, mine_value.py) | measured winner; alive-count is display-only |
| label player | v8c_s1.bin @ 2000 iters/side | Sally's call; ckpt_n_s1.pt == v8c_s1 (sha-verified) |
| playouts/row | n=8 early/mid, n=10 late | rows beat playouts at fixed budget; noise cancels across rows |
| rows per game | ≤3: ONE uniform draw per phase bucket | Sally's decorrelation rule; mix ≈ even thirds free |
| game engine | self-play @ 150ms, fresh PS teams, rand-ply 0.15 | competent-play distribution at ~6 core-s/game; NOT full-random (capacity argument) |
| fresh rows target | ~2.0–2.6M (≈ 800k–1M games) — size from canary s/playout to fit $300/1-day | pooled with enc_plc12 (2M) → 4.2–4.6M rows |
| holdout split | BY GAME SEED, never by row | within-game leakage otherwise |
| eval | bench_v2: 10k stratified states, n=50, from a DISJOINT seed range | ship-selection metric; bench_v1 kept for continuity |
| fleet | east-2 5× m6a.24xl ($0.0079, 5–10% band) · east-1 3× c6a.24xl ($0.0119, 5–10%) · west-2 1× c7a.48xl ($0.0154, 10–15%) = 960 of 1056 vCPU (quota 500/300/256) | measured prices + Spot Advisor bands 2026-08-17 |
| budget | ≤$300 total, fleet ~$180–235 conservative | Sally's cap |

## Current state (update as stages land)

- [x] gen pipeline: `mine_value.py gen` (game→1-per-bucket harvest→label→shard); local smokes pass
- [x] cloud MODE=gen with ~30-min chunked S3 uploads (reclaim mitigation)
- [x] PS checkout updated (8ff48ed); vendored gen9 sets byte-identical (509 species)
- [ ] S1 canary (genv9canary, m6a east-2, OLD config — read s/playout + upload path only)
- [ ] S2 fleet run → S3 mining/genv9/<tag>/chunk_*.jsonl.gz
- [ ] S3 bench_v2 (n=50, ~4k games, disjoint seeds ≥ 20,000,000)
- [ ] S4 merge + game-level split + encode (enc_adopted) → S3
- [ ] S5 GPU training (ASK SALLY for 4090) → v9 candidates
- [ ] S6 exams: bench_v2 select, bench_v1 continuity, old 13k-ledger bands
- [ ] S7 engine: 3-trunk export + blend in poke-engine, parity battery, sigma_R remeasure → sidecar
- [ ] S8 SPRT duel vs hz18, time-equalized, blitz settings
- [ ] S9 ship: launcher flip + commits + verification games with turn tables

## Stage detail

### S2 fleet
1. `bash corrections/cloud/pack_mining_code.sh` ONCE (ships new mine_value.py).
   NEVER let 9 launches each repack: concurrent writes race on the same S3
   code.tar.gz key. All launches use SKIP_PACK=1.
2. 9 launches (background, each is its own supervisor: blocks on S3 result,
   EXIT-trap terminates its instance):
   `MODE=gen TAG=genv9-<region><i> SKIP_PACK=1 MS=150 GEN_CHUNK=<from-canary>
    GEN_CHUNKS=<sized> SEED_BASE=<disjoint 10M-spaced> REGION=<r> TYPES=<type>
    TIMEOUT_S=<sized+pad> bash corrections/cloud/launch_mining.sh`
   Seed plan: box k gets SEED_BASE = 10,000,000 + k*1,000,000.
3. Supervisor (local): loop every 10 min — count `aws s3 ls .../genv9/` chunks
   per tag; NO new chunk from a box in 40 min = stall → pull its boot.log,
   diagnose, relaunch ONLY the missing seed range (salvage rule: completed
   chunks are already in S3; never regenerate them).
4. GEN_CHUNK sizing: target ~30 min/chunk on 96 vCPU from canary s/playout.
   Conservative default 750; recompute = (1800s × nproc) / (per-game core-s).

### S3 bench_v2
Same gen pipeline, `--n-early 50 --n-mid 50 --n-late 50`, ~4k games,
SEED_BASE=20,000,000, one m6a box, TAG=benchv2. Produces ~10-12k rows n=50.
Freeze as corrections/bench_v2.jsonl (s, truth, se, n, ph, g, t).

### S4 merge/encode
- Merge chunks; drop rows with missing labels; dedup by (g,t).
- Game-level split: hash(g) % 50 == 0 → holdout (~2%).
- Canary the encode on 10k rows first (compare against bench_enc pathway),
  then full encode via evallab/enc_adopted.py on one m6a box (or GPU box CPU).
- Output layout identical to enc_plc12 (old_*.npy + addon_mon + meta + holdout_i)
  → S3 evallab/enc_v9/.
- ALSO write phase per row into meta for phase-balanced batching.

### S5 training (GPU — ask Sally)
- Corpus = enc_plc12 (2M, old labels) + enc_v9 fresh. Keep sources distinguishable
  in meta (provenance flag) so ablation old-vs-fresh is possible.
- 4 arms: 3 from-scratch seeds + 1 warm-start-from-v8c_s1; phase net per
  corrections/phase_probe.py (PhaseNet, hat blend); swap-aug 50%;
  phase-balanced batches (equal thirds); late-trunk weight decay 2× (first fix
  for the pp1 late overfit; if late still regresses, add late dropout 0.1).
- Select on bench_v2 (floor-adjusted); report bench_v1 + slices + floors.
- Success bar: floor-adjusted bench_v2 beat v8c_s1 overall AND late ≤ baseline.

### S7 engine support (critical path; keep honest)
- Export: new bin format (shared part + 3 trunks + blend spec). Engine computes
  p from hp fractions at leaf, evaluates 1–2 active trunks, blends logits.
- Parity battery: ≥64 states python vs leaf_prof, exact within f32 export
  tolerance; MUST include states straddling p=0.5 knot.
- sigma_R remeasure on v9 (the constants-doc procedure, 24 holdout states,
  12k iters) → PE_TUNE_UCB_* in v9 sidecar. Constants travel with the bin.
- Wheel: release build + PGO as per task #4 recipe; NN cache unchanged.
- Mechanics untouched → no PS conformance resweep needed.

### S8 duel + S9 ship
- SPRT vs hz18, both at ladder settings (4500ms, tera gate 0.25, argmax-only),
  time-equalized. Duel harness: valuenet/sprt/run_duels.py (strips PE_TUNE_*;
  sidecars carry constants — see traps in v8c_s1.constants.json).
- Ship gates: duel non-negative + bench_v2 win + 13k-ledger exam reviewed
  (decide: native fix vs small hammer on top with v9-corpus anchors).
- Ship: RG default → v9 bin, commit valuenet + ladder bundle + this spec update,
  2–3 verification ladder games with turn tables.

## Known traps (all hit once already — do not re-hit)
- zsh does NOT word-split unquoted vars (fetch-script bug). Use while-read.
- pgrep/pkill -f self-match: ALWAYS bracket the pattern (`[p]hase_probe`).
- ssh banner noise: filter `vast-agents|Have fun|Welcome`.
- macOS tar: COPYFILE_DISABLE=1 --no-mac-metadata, exclude `._*`.
- Netless playouts: gen/confirm self-verify `valuenet: loaded` before spending.
- `aws s3 ls | head` broken pipe: redirect to file first.
- Persistent shell cwd: `cd` sticks across calls — use absolute paths.
- Sally's machine: ≤4 local workers; heavy work on cloud.
- Dead workflows: salvage completed pieces first, relaunch only gaps.

## Budget tracker (update)
| item | est | actual |
|---|---|---|
| canaries | $3 | |
| fleet gen + bench_v2 | $180–240 | |
| encode | $10 | |
| GPU training | $5 | |
| duel (CPU box) | $10 | |
| total cap | $300 | |
