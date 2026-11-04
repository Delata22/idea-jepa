#!/usr/bin/env bash
# usage: scripts/_sweep.sh <parallel> <file with lines: name|config|overrides...>
# Skips runs that already have final.json; retries a run up to 4 times if it died from CUDA OOM
# (the GPU may be shared with other jobs). A run stopped by SIGTERM/SIGINT saves last.pt and exits with 75: reported
# as "interrupted", neither failed nor retried; relaunching the sweep resumes it from last.pt.
P=$1; LIST=$2
[ -f runs/PARALLEL ] && P=$(cat runs/PARALLEL)   # runtime override of the parallelism
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
grep -v '^#' "$LIST" | grep -v '^$' | xargs -P "$P" -I{} bash -c '
  IFS="|" read -r name cfg ovr <<< "{}"
  if ls runs/*/"$name"/final.json > /dev/null 2>&1; then echo "skip $name"; exit 0; fi
  for attempt in 1 2 3 4; do
    chrt -i 0 taskset -c 12-23 uv run python -m ideajepa.train --config "$cfg" --name "$name" --set $ovr >> runs/logs/$name.log 2>&1
    rc=$?
    [ $rc -eq 0 ] && { echo "done $name"; exit 0; }
    [ $rc -eq 75 ] && { echo "interrupted $name (last.pt saved, resumes on relaunch)"; exit 0; }
    tail -n 60 runs/logs/$name.log | grep -q "OutOfMemory" || break
    echo "oom-retry $name ($attempt)"; sleep 90
  done
  echo "FAILED $name"'
