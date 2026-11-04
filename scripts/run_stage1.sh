#!/usr/bin/env bash
# Stage 1 (synthetic). Ordered by seed so a preliminary one-seed report exists early:
#   block 1: every config, seed 0  -> diagnostics -> preliminary tables
#   block 2: seeds 1, 2 for the decisive comparisons only
#   block 3: A11 (5.5 GB of VRAM, run alone)
# Parallelism 4: a V-full run peaks at 2.7 GB of VRAM (11.5 GB usable).
set -uo pipefail
cd "$(dirname "$0")/.."
P=${P:-3}
best_lr() {
  uv run python - "$1" <<'PY'
import glob, json, sys, re
best = min(((json.load(open(f))["ce_gen"], re.search(r"lr([0-9.e-]+)/", f).group(1))
            for f in glob.glob(f"runs/stage1/tb_{sys.argv[1]}_lr*/final.json")), default=(0, "1e-3"))
print(best[1])
PY
}
LR_B0=$(best_lr b0); LR_B1=$(best_lr b1)
echo "baseline lr: b0=$LR_B0 b1=$LR_B1"
# seed 0 of B0/B1 = the best run of the LR sweep (same seed, same config)
for m in b0 b1; do
  lr=$([ $m = b0 ] && echo $LR_B0 || echo $LR_B1)
  [ -e runs/stage1/${m}_s0 ] || { [ -f runs/stage1/tb_${m}_lr$lr/final.json ] && ln -s tb_${m}_lr$lr runs/stage1/${m}_s0; }
done
DECISIVE="v_full v_spec b0 b1 b1prime a1 a2 a3 a7 a7b a6b a15"
ALL="$DECISIVE a4 a5 a6 a8 a9 a10 a12"
line() {  # name seed
  local c=$1 s=$2 extra=""
  [ "$c" = b0 ] && extra="train.lr=$LR_B0"
  [ "$c" = b1 ] && extra="train.lr=$LR_B1"
  echo "${c}_s$s|configs/syn_$c.yaml|train.seed=$s $extra"
}
diag_block() {  # run diagnostics in parallel for finished runs lacking diag.json
  ls -d runs/stage1/*_s[0-9] | while read d; do
    [ -f "$d/ckpt.pt" ] && [ ! -f "$d/diag.json" ] && echo "$d"
  done | xargs -P "$P" -I{} bash -c 'uv run python -m ideajepa.eval.diag_synthetic {} --n_docs 800 > {}/diag.log 2>&1 || echo "DIAG FAILED {}"'
  uv run python -m ideajepa.eval.stage1_report runs/stage1 > /dev/null 2>&1 && echo "report updated"
}
: > runs/stage1_block1.txt; for c in $ALL; do line $c 0 >> runs/stage1_block1.txt; done
: > runs/stage1_block2.txt; for s in 1 2; do for c in $DECISIVE; do line $c $s >> runs/stage1_block2.txt; done; done
line a11 0 > runs/stage1_block3.txt
scripts/_sweep.sh "$P" runs/stage1_block1.txt; diag_block; echo "BLOCK1 DONE"
scripts/_sweep.sh "$P" runs/stage1_block2.txt; diag_block; echo "BLOCK2 DONE"
scripts/_sweep.sh 1 runs/stage1_block3.txt; diag_block; echo "STAGE1 DONE"
