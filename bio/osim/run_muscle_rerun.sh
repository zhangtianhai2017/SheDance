#!/bin/bash
# Full muscle pipeline re-run on the posture-normalized source:
#   retarget -> L1 collision -> TRC -> OpenSim IK -> parallel SO -> muscle render.
# Two venvs: musclemimic (mujoco/mink/GMR) + osimenv (opensim). Env: SEQ, DANCE (default Waack).
set -e
VPY=$HOME/shedance/musclemimic/.venv/bin/python
OSPY=$HOME/shedance/osimenv/bin/python
SRC=/mnt/c/work/2026/Claude/SheDance/bio/osim
OSIM=$HOME/shedance/osim
REN=/mnt/c/work/2026/Claude/SheDance/renders
SEQ=${SEQ:-gWA_sBM_cAll_d25_mWA0_ch04}
DANCE=${DANCE:-gwa}
NPZ=$HOME/shedance/aist_amass/AIST/$SEQ.npz
CACHE=$HOME/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/$SEQ.npz
REF=$OSIM/${DANCE}_refined.npz

echo "[$(date +%T)] STEP 1/6 retarget ($SEQ, posture-normalized source)"
$VPY $SRC/retarget_smplh.py "$NPZ" "$CACHE" 60

echo "[$(date +%T)] STEP 2/6 L1 collision refine"
IN_CACHE="$CACHE" OUT_CACHE="$REF" $VPY $SRC/refine_collision.py

echo "[$(date +%T)] STEP 3/6 TRC from MuJoCo (refined)"
DANCE=$DANCE DANCE_CACHE="$REF" $VPY $SRC/trc_from_mujoco.py

echo "[$(date +%T)] STEP 4/6 OpenSim IK"
DANCE=$DANCE $OSPY $SRC/add_markers_and_ik.py

echo "[$(date +%T)] STEP 5/6 parallel Static Optimization"
bash $SRC/run_so_parallel_dance.sh "$OSIM/${DANCE}_ik.mot" 0.1 11.9 16 "$DANCE" "$OSIM/${DANCE}_par_activation.sto"

echo "[$(date +%T)] STEP 6/6 render muscle video"
RENDER_CACHE="$REF" COLOR_GAIN=4 RENDER_FPS=60 $VPY $SRC/render_one.py "$OSIM/${DANCE}_par_activation.sto" "$REN/${DANCE}_refined_muscle.mp4"

echo "[$(date +%T)] PIPELINE DONE -> $REN/${DANCE}_refined_muscle.mp4"
