#!/usr/bin/env bash
# Keep training out of the way of interactive use:
#  - SCHED_IDLE + cores 12-23 for all training/eval processes;
#  - at most $(cat runs/PARALLEL) top-level trainers RUNNING at once; extra trainers are paused (SIGSTOP)
#    together with their dataloader workers and resumed (SIGCONT) later, in pid order. No work is lost.
cd "$(dirname "$0")/.."
while true; do
  for p in $(pgrep -f "ideajepa[.]train|eval[_]ded|eval[_]para|eval[_]energy|make[_]dagger|show[_]samples|gen[_]eval"); do
    chrt -i -p 0 "$p" 2>/dev/null; taskset -a -cp 0-23 "$p" >/dev/null 2>&1
  done
  MAXP=$(cat runs/PARALLEL 2>/dev/null || echo 2)
  allp=$(pgrep -f "python3 -m ideajepa[.]train" | sort -n)
  mains=""
  for p in $allp; do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    echo "$allp" | grep -qx "$pp" || mains="$mains $p"
  done
  running=0
  for p in $mains; do
    kids=$(pgrep -P "$p")
    st=$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)
    if [ "$running" -lt "$MAXP" ]; then
      kill -CONT "$p" 2>/dev/null; for k in $kids; do kill -CONT "$k" 2>/dev/null; done
      running=$((running + 1))
    else
      if [ "$st" != "T" ]; then kill -STOP "$p" 2>/dev/null; for k in $kids; do kill -STOP "$k" 2>/dev/null; done; fi
    fi
  done
  sleep 15
done
