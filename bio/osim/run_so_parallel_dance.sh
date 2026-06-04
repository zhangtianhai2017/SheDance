#!/bin/bash
# Generic parallel Static Optimization for any dance .mot (chunk [T0,T1] into N, parallel so_chunk,
# concat). Reuses cyclist_min_reserves.osim + so_chunk.py (CHUNK_MOT env). Lossless (per-frame SO).
# Args: MOT_path T0 T1 [N=16] [prefix=dp] [out_sto]
set -u
OSPY=$HOME/shedance/osimenv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim; OSIM=$HOME/shedance/osim
MOT=$1; T0=$2; T1=$3; N=${4:-16}; PFX=${5:-dp}; OUT=${6:-$OSIM/${PFX}_parallel_activation.sto}
OV=0.1; export CHUNK_MOT=$MOT
rm -f "$OSIM"/${PFX}c*_StaticOptimization_*.sto
W=$(awk "BEGIN{print ($T1-$T0)/$N}")
echo "$(date +%T) parallel SO: $MOT [$T0,$T1] x$N chunks (W=${W}s)"
S=$(date +%s); pids=()
for i in $(seq 0 $((N-1))); do
  c0=$(awk "BEGIN{print $T0+$i*$W}"); c1=$(awk "BEGIN{print $T0+($i+1)*$W}")
  s0=$(awk "BEGIN{v=$c0-$OV; if(v<$T0)v=$T0; print v}"); s1=$(awk "BEGIN{v=$c1+$OV; if(v>$T1)v=$T1; print v}")
  "$OSPY" "$SRC/so_chunk.py" "$s0" "$s1" "${PFX}c$i" > "$OSIM/${PFX}c$i.out" 2>&1 & pids+=($!)
done
wait "${pids[@]}"
echo "$(date +%T) parallel SO done in $(( $(date +%s)-S ))s"
"$OSPY" "$SRC/concat_dance.py" "$PFX" "$T0" "$T1" "$N" "$OUT"
