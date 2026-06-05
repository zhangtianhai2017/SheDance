#!/bin/bash
# Poll the muscle-rerun log: emit on each new last-line (step change) + a heartbeat every 3min,
# stop on completion / python error / the pipeline process exiting. Covers all terminal states.
prev=""; n=0
while true; do
  cur=$(tail -1 /tmp/muscle_rerun.log 2>/dev/null)
  if [ "$cur" != "$prev" ]; then echo "$(date +%T) | $cur"; prev="$cur"; n=0
  else n=$((n + 1)); [ $((n % 6)) -eq 0 ] && echo "$(date +%T) | (running) $cur"; fi
  printf '%s' "$cur" | grep -qE "PIPELINE DONE|Traceback|Killed|Cannot|No such file" && break
  pgrep -f run_muscle_rerun.sh >/dev/null || { echo "$(date +%T) | pipeline process exited"; break; }
  sleep 30
done
echo "MONITOR-DONE"
