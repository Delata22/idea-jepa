#!/usr/bin/env bash
# Evaluate every finished 26/09 run once (breakdown CE; + paraphrase invariance / energy where relevant).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  for f in runs/v2/v2[fghijl]_*/final.json; do
    d=$(dirname "$f"); n=$(basename "$d")
    [ -f "$d/.evaluated" ] && continue
    uv run python scripts/eval_ded_breakdown.py "$d" 40 2>/dev/null | grep -v -i warn | tail -1 >> runs/logs/v2c_breakdown.jsonl
    case "$n" in
      v2g_para*|v2g_noerase_s0|v2h_stageA*) uv run python scripts/eval_para.py "$d" 15 2>/dev/null | grep -v -i warn | tail -1 >> runs/logs/v2_para.jsonl ;;
    esac
    case "$n" in
      v2h_stageA*) uv run python scripts/eval_energy.py "$d" 15 2>/dev/null | grep -v -i warn | tail -1 >> runs/logs/v2_energy.jsonl ;;
    esac
    touch "$d/.evaluated"; echo "$(date +%H:%M) evaluated $n"
  done
  sleep 60
done
