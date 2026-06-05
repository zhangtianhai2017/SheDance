#!/bin/bash
# Muscle pipeline starting from a READY MyoFullBody cache (e.g. the VedioTo3D adapter output) -- skips
# convert_aist + retarget (the adapter already produced the 89-dof cache). L1/L2 -> TRC -> IK -> SO -> render.
# Env: CACHE (89-dof qpos npz), DANCE (tag), T0=0.1, T1, FPS.
set -e
VPY=$HOME/shedance/musclemimic/.venv/bin/python
OSPY=$HOME/shedance/osimenv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim
OSIM=$HOME/shedance/osim; REN=/mnt/c/work/2026/Claude/SheDance/renders
CACHE=${CACHE:?need CACHE}; DANCE=${DANCE:?need DANCE}; T0=${T0:-0.1}; T1=${T1:-22.6}; FPS=${FPS:-30}
REF=$OSIM/${DANCE}_refined.npz

echo "[$(date +%T)] STEP 1/5 L1/L2 collision refine"
IN_CACHE="$CACHE" OUT_CACHE="$REF" $VPY $SRC/refine_collision.py
echo "[$(date +%T)] STEP 2/5 TRC from MuJoCo"
DANCE=$DANCE DANCE_CACHE="$REF" $VPY $SRC/trc_from_mujoco.py
echo "[$(date +%T)] STEP 3/5 OpenSim IK"
DANCE=$DANCE $OSPY $SRC/add_markers_and_ik.py
echo "[$(date +%T)] STEP 4/5 parallel Static Optimization"
bash $SRC/run_so_parallel_dance.sh "$OSIM/${DANCE}_ik.mot" "$T0" "$T1" 16 "$DANCE" "$OSIM/${DANCE}_par_activation.sto"
echo "[$(date +%T)] STEP 5/5 render muscle video"
RENDER_CACHE="$REF" COLOR_GAIN=4 RENDER_FPS=$FPS $VPY $SRC/render_one.py "$OSIM/${DANCE}_par_activation.sto" "$REN/${DANCE}_muscle.mp4"
echo "[$(date +%T)] PIPELINE DONE -> $REN/${DANCE}_muscle.mp4"
