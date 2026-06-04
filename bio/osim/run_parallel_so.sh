#!/bin/bash
# Parallel Static Optimization over pop60.mot with thermal monitoring + auto-throttle + logging.
# - GPU temp via nvidia-smi (accurate); CPU-side ACPI thermal zone via powershell.exe (coarse trend).
# - Concurrency capped (conservative); if HOT: pause new launches + kill newest chunk (auto-reduce).
# - After: concat chunks (discard overlap) + lossless verify + report speedup & temp peaks.
set -u
HERE="$HOME/shedance/osim"
OSPY="$HOME/shedance/osimenv/bin/python"
SRC="/mnt/c/work/2026/Claude/SheDance/bio/osim"
N=16                 # chunks = concurrency cap (16 of 32 cores = conservative)
GPU_MAX=80           # C, nvidia-smi (accurate)
ZONE_MAX=70          # C, ACPI zone (coarse; idles ~28)
SPAN0=0.3; SPAN1=59.7; OV=0.2
RATE=1.14            # measured s/frame (single-chunk: 344s/303f)
RUNLOG="$HERE/parallel_run.log"; TEMPLOG="$HERE/temp_log.csv"
: > "$RUNLOG"; echo "time,gpu_c,zone_c,running,action" > "$TEMPLOG"
rm -f "$HERE/HOT" "$HERE/STOP_MON" "$HERE"/c*_StaticOptimization_*.sto

zone_temp(){ powershell.exe -NoProfile -Command "\$s=(Get-Counter \"\\Thermal Zone Information(*)\\Temperature\").CounterSamples; [math]::Round(\$s[0].CookedValue-273.15,1)" 2>/dev/null | tr -d "\r"; }

# ---- background thermal monitor + auto-reduce ----
(
  while [ ! -f "$HERE/STOP_MON" ]; do
    gpu=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")
    zone=$(zone_temp); run=$(pgrep -fc "so_chunk.py" 2>/dev/null); action="ok"; hot=0
    [ -n "$gpu" ] && [ "$gpu" -ge "$GPU_MAX" ] 2>/dev/null && hot=1
    [ -n "$zone" ] && awk "BEGIN{exit !($zone+0 >= $ZONE_MAX)}" && hot=1
    if [ "$hot" = "1" ]; then
      touch "$HERE/HOT"; newest=$(pgrep -f "so_chunk.py" -n 2>/dev/null)
      [ -n "$newest" ] && kill "$newest" 2>/dev/null && action="HOT_killed_$newest"
    else rm -f "$HERE/HOT"; fi
    echo "$(date +%T),${gpu:-NA},${zone:-NA},${run:-0},$action" >> "$TEMPLOG"
    sleep 10
  done
) & MON=$!

# ---- launch chunks (capped, gated on HOT) ----
W=$(awk "BEGIN{print ($SPAN1-$SPAN0)/$N}")
echo "$(date +%T) launch $N chunks W=${W}s gpu_max=$GPU_MAX zone_max=$ZONE_MAX" | tee -a "$RUNLOG"
T0=$(date +%s); pids=()
for i in $(seq 0 $((N-1))); do
  c0=$(awk "BEGIN{print $SPAN0+$i*$W}"); c1=$(awk "BEGIN{print $SPAN0+($i+1)*$W}")
  s0=$(awk "BEGIN{v=$c0-$OV; if(v<0)v=0; print v}"); s1=$(awk "BEGIN{v=$c1+$OV; if(v>60)v=60; print v}")
  while [ -f "$HERE/HOT" ]; do echo "$(date +%T) HOT pausing c$i" >> "$RUNLOG"; sleep 10; done
  "$OSPY" "$SRC/so_chunk.py" "$s0" "$s1" "c$i" > "$HERE/c$i.out" 2>&1 & pids+=($!)
  echo "$(date +%T) launched c$i [$s0,$s1] pid $!" >> "$RUNLOG"
done
wait "${pids[@]}"
T1=$(date +%s); ELAPSED=$((T1-T0))
touch "$HERE/STOP_MON"; sleep 1; kill "$MON" 2>/dev/null
echo "==== ALL DONE in ${ELAPSED}s ($(awk "BEGIN{printf \"%.1f\",$ELAPSED/60}") min) ====" | tee -a "$RUNLOG"
awk -F, 'NR>1{if($2!="NA"&&$2+0>g)g=$2+0; if($3!="NA"&&$3+0>z)z=$3+0} END{printf "TEMP peaks: GPU %g C, ACPI-zone %g C\n",g,z}' "$TEMPLOG" | tee -a "$RUNLOG"
echo "---- concat + lossless verify ----" | tee -a "$RUNLOG"
"$OSPY" "$SRC/concat_verify.py" "$N" | tee -a "$RUNLOG"
awk "BEGIN{seq=6000*$RATE; printf \"SPEEDUP: seq(extrapolated %.2fs/frame x6000)=%.0fs(%.1f min) | parallel=%ds(%.1f min) | %.1fx\n\",$RATE,seq,seq/60,$ELAPSED,$ELAPSED/60,seq/$ELAPSED}" | tee -a "$RUNLOG"
