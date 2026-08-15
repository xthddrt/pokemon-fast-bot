#!/bin/bash
# evallab end-to-end driver.
#
#   ./run_all.sh canary <dir>          50 games, tiny oracle, 4 cells -- ~5 min
#   ./run_all.sh tensors <dir>         shards -> npz (assumes shards already there)
#   ./run_all.sh cells   <dir>         train + evaluate the 4 cells
#   ./run_all.sh full    <dir>         tensors + cells
#
# <dir> holds A/ B/ C/ shard directories and oracle_*.jsonl, exactly as the
# cloud box lays them out. Everything is capped at 4 cores (EVALLAB_WORKERS /
# EVALLAB_THREADS) -- this machine stays usable.
set -euo pipefail
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${EVALLAB_PY:-$LAB/../foul-play/.venv/bin/python}"
MODE="${1:?canary|tensors|cells|full}"
DIR="${2:?output dir}"
export EVALLAB_WORKERS="${EVALLAB_WORKERS:-4}"
export EVALLAB_THREADS="${EVALLAB_THREADS:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$LAB"
mkdir -p "$DIR"

gate() { "$PY" test_relfeat.py > "$DIR/relfeat_gate.log" 2>&1 && tail -1 "$DIR/relfeat_gate.log"; }

tensors() {
  for p in A B C; do
    ls "$DIR/$p"/shard_${p}_*.jsonl.gz >/dev/null 2>&1 || continue
    echo "== stats $p"; "$PY" stats.py "$DIR/$p/shard_${p}_*.jsonl.gz" 2>&1 | grep -v '^valuenet:' | tee "$DIR/stats_$p.txt"
    echo "== tensors $p"
    n=30000; [ "$p" = A ] || n=6000
    "$PY" dataset.py "$DIR/$p/shard_${p}_*.jsonl.gz" "$DIR/$p" "$n" 2>&1 | grep -v '^valuenet:'
  done
  cat "$DIR"/oracle_*.jsonl > "$DIR/oracle.jsonl" 2>/dev/null || true
  echo "oracle positions: $(wc -l < "$DIR/oracle.jsonl")"
}

cells() {
  for r in 0 1; do for k in 0 "${RANK_W:-0.5}"; do
    tag="rel${r}_rank${k}"
    echo "== train $tag"
    REL=$r RANK_W=$k EPOCHS="${EPOCHS:-10}" "$PY" train_lab.py \
      "$DIR/A_pos.npz" "$DIR/A_sib.npz" "$DIR/net_$tag.pt" 2>&1 | grep -v '^valuenet:' | grep -v '^JSON'
    echo "== evaluate $tag"
    "$PY" evaluate.py "$DIR/net_$tag.pt" "$DIR/oracle.jsonl" "$DIR/[ABC]_hold_pos.npz" \
      "$DIR/res_$tag.json" 2>&1 | grep -v '^valuenet:' | tail -3
  done; done
  {
    "$PY" summarize.py "$DIR"
    echo; echo "### reference brackets"; echo
    "$PY" references.py "$DIR/oracle.jsonl" "$DIR" 2>&1 | grep -v '^valuenet:'
  } | tee "$DIR/SUMMARY.txt"
}

case "$MODE" in
  canary)
    gate
    "$PY" generate.py A 50 "$DIR/A" 0 25000 2>&1 | grep -v '^valuenet:' | tail -2
    "$PY" oracle.py "$DIR/A/shard_A_*.jsonl.gz" "$DIR/oracle_A.jsonl" 12 1000000 2>&1 | grep -v '^valuenet:' | tail -2
    tensors; EPOCHS=4 cells ;;
  tensors) gate; tensors ;;
  cells)   cells ;;
  full)    gate; tensors; cells ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
