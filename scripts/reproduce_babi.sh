#!/bin/bash
# Reproduces the final bAbI table (tasks 1-3, 10k, 3 seeds): Idea-JEPA minimal + size-matched references B0 / B3.
# Every run: 6000 steps x 8192 tokens, snapshots every 1000 steps; evaluation = snapshot chosen on validation, then test.
# The references' learning rate (1e-3 or 3e-4), then dropout (0, 0.1, 0.3), were chosen on qa2 validation, seed 0
# (see the README). On qa3 the references use gradient accumulation (same 8192 tokens per step).
# The final table ran the qa3 references on seed 0 only; this script runs all 3 seeds.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LR_B0=${LR_B0:-1e-3}; LR_B3=${LR_B3:-1e-3}; DP=${DP:-0.3}
train() { local name=$1 cfg=$2; shift 2
          [ -f runs/babi/$name/final.json ] || uv run python -m ideajepa.train --config $cfg --name $name --set "$@"
          bash scripts/eval_pick.sh $name; }
for s in 0 1 2; do for t in 1 2 3; do
  acc0=""; acc3=""
  [ $t = 3 ] && { acc0="train.batch_tokens=1024 train.grad_accum=8"; acc3="train.batch_tokens=2048 train.grad_accum=4"; }
  train bq${t}_min6k_d64_s$s configs/babi_min.yaml       data.path=data/babi/qa${t}_10k train.seed=$s
  train bq${t}_b0dp_s$s      configs/babi_b0_small.yaml  data.path=data/babi/qa${t}_10k train.seed=$s train.lr=$LR_B0 model.dropout=$DP $acc0
  train bq${t}_b3dp_s$s      configs/babi_b3_small.yaml  data.path=data/babi/qa${t}_10k train.seed=$s train.lr=$LR_B3 model.dropout=$DP $acc3
done; done
# "no dropout" rows of the table (first configuration tried), qa1-2 only
for s in 0 1 2; do for t in 1 2; do
  train bq${t}_b0small_s$s   configs/babi_b0_small.yaml  data.path=data/babi/qa${t}_10k train.seed=$s train.lr=$LR_B0
  train bq${t}_b3small_s$s   configs/babi_b3_small.yaml  data.path=data/babi/qa${t}_10k train.seed=$s train.lr=$LR_B3
done; done
uv run python scripts/babi_results_table.py
