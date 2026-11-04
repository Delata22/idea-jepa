#!/bin/bash
# Same evaluation protocol for every model (ours and references): validation (500 questions) at each snapshot,
# snapshot chosen on validation (scripts/babi_pick.py), then the full test set (1000 questions) with outputs.
# Usage: scripts/eval_pick.sh RUN_NAME...   (runs are read from runs/babi/, results written to runs/babi_scale/)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-"uv run python"}
G=runs/babi_scale; mkdir -p $G
for n in "$@"; do
  [ -f runs/babi/$n/final.json ] || { echo "not trained: $n"; continue; }
  [ -f $G/$n.test.json ] && { echo "already evaluated: $n"; continue; }
  V=""; for d in runs/babi/$n@*; do [ -d "$d" ] && V="$V $d"; done
  V="$V runs/babi/$n"
  $PY scripts/babi_eval.py $V --split val --n 500 --show 0 --no_outputs --json $G/$n.val.json > $G/$n.val.out 2>&1
  best=$($PY scripts/babi_pick.py $G/$n.val.json 2> $G/$n.curve.txt)
  $PY scripts/babi_eval.py $best --split test --show 8 --json $G/$n.test.json > $G/$n.test.out 2>&1
  echo "$n: snapshot $best evaluated on test"
done
