#!/bin/bash
# Parallel osmesa render: M frame-chunks through a CAP-wide worker pool (load-balances across
# hybrid P/E cores) -> partial mp4s -> ffmpeg concat (lossless). Thermal monitor + auto-reduce + log.
# osmesa software render is CPU-bound + per-frame independent -> no overlap needed, scales like SO.
set -u
PY=$HOME/shedance/musclemimic/.venv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim
OSIM=$HOME/shedance/osim; REN=$HOME/shedance/renders
STO=$OSIM/pop60_parallel_activation.sto
export RENDER_CACHE=$OSIM/pop60_qpos.npz
export MUJOCO_GL=osmesa
export LP_NUM_THREADS=1            # 1 thread/process = best per-core efficiency
CAP=${CAP:-16}                    # concurrent workers (conservative half of 32 threads)
M=${M:-48}                        # chunks > CAP -> dynamic load balance
GPU_MAX=80; ZONE_MAX=70
OUT=$REN/pop60_parallel.mp4
RUNLOG=$OSIM/render_run.log; TEMPLOG=$OSIM/render_temp.csv
: > "$RUNLOG"; echo "time,gpu_c,zone_c,running,action" > "$TEMPLOG"
rm -f "$OSIM/HOT" "$OSIM/STOP_MON" "$REN"/partial_*.mp4
mkdir -p "$REN"

zone_temp(){ powershell.exe -NoProfile -Command "\$s=(Get-Counter \"\\Thermal Zone Information(*)\\Temperature\").CounterSamples; [math]::Round(\$s[0].CookedValue-273.15,1)" 2>/dev/null | tr -d "\r"; }

# ---- background thermal monitor + auto-reduce ----
(
  while [ ! -f "$OSIM/STOP_MON" ]; do
    gpu=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")
    zone=$(zone_temp); run=$(pgrep -fc "render_one.py" 2>/dev/null); action="ok"; hot=0
    [ -n "$gpu" ] && [ "$gpu" -ge "$GPU_MAX" ] 2>/dev/null && hot=1
    [ -n "$zone" ] && awk "BEGIN{exit !($zone+0 >= $ZONE_MAX)}" && hot=1
    if [ "$hot" = "1" ]; then
      touch "$OSIM/HOT"; newest=$(pgrep -f "render_one.py" -n 2>/dev/null)
      [ -n "$newest" ] && kill "$newest" 2>/dev/null && action="HOT_killed_$newest"
    else rm -f "$OSIM/HOT"; fi
    echo "$(date +%T),${gpu:-NA},${zone:-NA},${run:-0},$action" >> "$TEMPLOG"
    sleep 10
  done
) & MON=$!

# ---- total frames + chunk size ----
TOTAL=$(( $(wc -l < "$STO") - 7 ))
CS=$(( (TOTAL + M - 1) / M ))
echo "$(date +%T) total~$TOTAL frames, M=$M chunks x ~$CS, CAP=$CAP, LP_NUM_THREADS=1" | tee -a "$RUNLOG"

# ---- worker pool: launch M chunks, <=CAP concurrent, gated on HOT ----
T0=$(date +%s); pids=()
for i in $(seq 0 $((M-1))); do
  while [ "$(pgrep -fc render_one.py 2>/dev/null)" -ge "$CAP" ]; do sleep 0.3; done
  while [ -f "$OSIM/HOT" ]; do echo "$(date +%T) HOT pausing chunk $i" >> "$RUNLOG"; sleep 5; done
  a=$((i*CS)); b=$(((i+1)*CS))
  RENDER_FRAMES="$a:$b" "$PY" "$SRC/render_one.py" "$STO" "$REN/partial_$i.mp4" > "$OSIM/rp_$i.out" 2>&1 & pids+=($!)
done
wait "${pids[@]}"            # only workers, NOT the monitor subshell (else deadlock)
T1=$(date +%s); ELAPSED=$((T1-T0))
touch "$OSIM/STOP_MON"; sleep 1; kill "$MON" 2>/dev/null

# ---- re-render any missing/killed chunks serially ----
MISS=0
for i in $(seq 0 $((M-1))); do
  if [ ! -s "$REN/partial_$i.mp4" ]; then
    MISS=$((MISS+1)); a=$((i*CS)); b=$(((i+1)*CS))
    RENDER_FRAMES="$a:$b" "$PY" "$SRC/render_one.py" "$STO" "$REN/partial_$i.mp4" >> "$OSIM/rp_$i.out" 2>&1
  fi
done
[ "$MISS" -gt 0 ] && echo "re-rendered $MISS missing chunks" | tee -a "$RUNLOG"

# ---- concat (lossless) ----
LIST=$OSIM/concat_list.txt; : > "$LIST"
for i in $(seq 0 $((M-1))); do [ -s "$REN/partial_$i.mp4" ] && echo "file '$REN/partial_$i.mp4'" >> "$LIST"; done
ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$OUT" >/dev/null 2>&1
NF=$(ffmpeg -i "$OUT" -f null - 2>&1 | grep -oaE "frame=[ ]*[0-9]+" | tail -1 | grep -oaE "[0-9]+")

echo "==== RENDER DONE in ${ELAPSED}s ($(awk "BEGIN{printf \"%.1f\",$ELAPSED/60}") min) ====" | tee -a "$RUNLOG"
awk -F, 'NR>1{if($2!="NA"&&$2+0>g)g=$2+0; if($3!="NA"&&$3+0>z)z=$3+0} END{printf "TEMP peaks: GPU %g C, ACPI-zone %g C\n",g,z}' "$TEMPLOG" | tee -a "$RUNLOG"
echo "frames in final video: $NF (expected ~$TOTAL) -> $OUT" | tee -a "$RUNLOG"
awk "BEGIN{seq=$TOTAL*0.174; printf \"SPEEDUP: serial-osmesa(0.174s/frame x$TOTAL)=%.0fs(%.1f min) | parallel=%ds(%.1f min) | %.1fx\n\",seq,seq/60,$ELAPSED,$ELAPSED/60,seq/$ELAPSED}" | tee -a "$RUNLOG"
