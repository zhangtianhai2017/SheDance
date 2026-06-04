#!/bin/bash
# Sweep nproc x nthreads for osmesa render to find the memory-bound throughput optimum.
# Short clip, sequential configs, NO background monitor (no deadlock), foreground (direct result).
set -u
PY=$HOME/shedance/musclemimic/.venv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim; OSIM=$HOME/shedance/osim
STO=$OSIM/pop60_parallel_activation.sto
export RENDER_CACHE=$OSIM/pop60_qpos.npz MUJOCO_GL=osmesa
NF=${NF:-480}
TMP=$HOME/shedance/renders/_sweep; mkdir -p "$TMP"
gpu(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d " "; }
echo "sweep on $NF frames (gpu $(gpu)C start)"
run(){
  local np=$1 th=$2; rm -f "$TMP"/s_*.mp4
  local cs=$(( (NF + np - 1)/np )); local S=$(date +%s); local pids=()
  for ((i=0;i<np;i++)); do
    local a=$((i*cs)) b=$(((i+1)*cs)); [ $a -ge $NF ] && break
    LP_NUM_THREADS=$th RENDER_FRAMES="$a:$b" "$PY" "$SRC/render_one.py" "$STO" "$TMP/s_$i.mp4" >/dev/null 2>&1 & pids+=($!)
  done
  wait "${pids[@]}"
  local E=$(date +%s); local d=$((E-S)); [ $d -lt 1 ] && d=1
  printf "nproc=%-3s threads=%-2s  %3ds  %6.1f fps  gpu=%sC\n" "$np" "$th" "$d" "$(awk "BEGIN{printf $NF/$d}")" "$(gpu)"
}
for c in "8 1" "12 1" "16 1" "8 2" "6 2" "4 4"; do run $c; done   # all <=16 procs (avoid OOM)
