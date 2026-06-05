#!/bin/bash
# F_max effort/character version: scale muscle max-force -> rebuild reserves model -> parallel SO ->
# render. Weaker body (FMAX<1) tracks the SAME dance with higher activation = redder muscles.
# (SO only re-distributes effort; it does NOT change the motion -- a true physical filter needs
#  forward dynamics. This is the character-knob 'effort' visualization.)  Env: DANCE, FMAX.
set -e
OSPY=$HOME/shedance/osimenv/bin/python
VPY=$HOME/shedance/musclemimic/.venv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim
OSIM=$HOME/shedance/osim
REN=/mnt/c/work/2026/Claude/SheDance/renders
DANCE=${DANCE:-gwa}; FMAX=${FMAX:-0.5}
F=$(awk "BEGIN{printf \"%02d\", $FMAX*100}")

echo "[$(date +%T)] STEP 1/3 build F_max=$FMAX reserves model"
$OSPY $SRC/build_fmax_reserves.py $FMAX

echo "[$(date +%T)] STEP 2/3 parallel Static Optimization (F_max=$FMAX)"
SO_MODEL=$OSIM/cyclist_min_reserves_f${F}.osim \
  bash $SRC/run_so_parallel_dance.sh "$OSIM/${DANCE}_ik.mot" 0.1 11.9 16 "${DANCE}f${F}" "$OSIM/${DANCE}_f${F}_activation.sto"

echo "[$(date +%T)] STEP 3/3 render (F_max=$FMAX, redder = working harder)"
RENDER_CACHE="$OSIM/${DANCE}_refined.npz" COLOR_GAIN=4 RENDER_FPS=60 RENDER_AZ=120 \
  $VPY $SRC/render_one.py "$OSIM/${DANCE}_f${F}_activation.sto" "$REN/${DANCE}_f${F}_muscle.mp4"

echo "[$(date +%T)] FMAX DONE -> $REN/${DANCE}_f${F}_muscle.mp4"
