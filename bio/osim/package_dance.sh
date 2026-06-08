#!/bin/bash
# Automatic, CONSISTENT A12B sample package: ONE command, every artifact (skinned FBX, kinematics,
# muscle SO, preview) derived from the SAME collar/spine-fixed adaptive motion -- no manual patching.
# Usage: package_dance.sh <track.npz>   Env: FPS(=30)
set -e
TRACK="$1"; FPS="${FPS:-30}"
PM="$HOME/shedance/musclemimic/.venv/bin/python"   # GMR / pyrender / numpy (no opensim)
POS="$HOME/shedance/osimenv/bin/python"            # OpenSim 4.6 (convert + SO)
B=/mnt/c/work/2026/Claude/SheDance/bio/osim
WF="$HOME/shedance/wf_out/dance1"; O="$HOME/shedance/osim"
echo "### [1/5] kinematic + skinned FBX + GRF  (run_pipeline.sh, fully adaptive)"
FPS="$FPS" bash "$B/run_pipeline.sh" "$TRACK" "$WF"
echo "### [2/5] soul: qpos -> OpenSim ik.mot"
MOT_STEP=1 "$POS" "$B/convert_qpos_to_mot.py" "$WF/qpos.npz" "$O/dance1_ik.mot"
echo "### [3/5] soul: Static Optimization -> activation.sto  (no-GRF pelvis reserves; auto-upgrades when $O/dance1_grf_extloads.xml exists)"
DUR=$("$PM" -c "import numpy as np;z=np.load('$WF/qpos.npz');print(len(z['qpos'])/float(z['frequency']))")
DANCE=dance1 "$POS" "$B/static_opt_osim.py" 0 "$DUR"
echo "### [4/5] preview skin render  (EGL -- osmesa is broken in this pyrender build)"
PYOPENGL_PLATFORM=egl "$PM" "$B/smpl_skin_render.py" "$WF/poses_lock.npz" "$WF/preview_skin.mp4" "$FPS"
echo "### [5/5] assemble + zip"
"$PM" "$B/build_sample.py"
echo "DONE -> SheDance_sample_dance1.zip   (commit + push from Windows git)"
