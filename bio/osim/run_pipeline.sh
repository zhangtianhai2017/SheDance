#!/bin/bash
# SheDance ONE-COMMAND workflow: a source track -> deliverable, fully adaptive (no per-input tuning).
#   video/track -> angle-transfer -> contact foot-lock -> joints -> de-drift -> FBX/joints  (+ soul: qpos -> GRF)
# All thresholds inside each step are derived from the data (figure height, fps, the foot's own speed).
# Usage: run_pipeline.sh <track.npz> <out_dir>   Env: HUM(=our_humanoid.npz) FPS(=30)
set -e
TRACK="$1"; OUT="$2"
HUM="${HUM:-$HOME/shedance/osim/our_humanoid.npz}"; FPS="${FPS:-30}"
PM="$HOME/shedance/musclemimic/.venv/bin/python"          # angle-transfer / lock / GRF (no bpy)
P311="$HOME/shedance/fbxenv311/bin/python"                # FBX export (bpy 4.2, py3.11)
B="/mnt/c/work/2026/Claude/SheDance/bio/osim"
mkdir -p "$OUT"
echo "=== [1/6] angle-transfer (video keypoints -> our humanoid SMPL) ==="
FPS=$FPS "$PM" "$B/source_to_our_poses.py" "$TRACK" "$HUM" "$OUT/poses.npz"
echo "=== [2/6] contact foot-lock (adaptive: height/fps/foot-speed) ==="
HUM="$HUM" "$PM" "$B/footlock_hard.py" "$OUT/poses.npz" "$OUT/poses_lock.npz"
echo "=== [3/6] joints_world (positions + per-joint world rot) ==="
"$PM" "$B/export_joints_world.py" "$OUT/poses_lock.npz" "$OUT/jw.npz"
echo "=== [4/6] vertical de-drift (feet ride Z=0) ==="
"$PM" "$B/foot_ground_vertical.py" "$OUT/jw.npz" "$OUT/joints_world.npz"
echo "=== [5/6] skinned FBX (anim + rest), origin=initial contact, floor Z=0 ==="
"$P311" "$B/fbx_export.py" "$OUT/joints_world.npz" "$HUM" "$OUT/dance.fbx" anim
"$P311" "$B/fbx_export.py" "$OUT/joints_world.npz" "$HUM" "$OUT/dance_tpose.fbx" rest
echo "=== [6/6] SOUL: qpos (mink contact-IK) -> GRF (COM dynamics) ==="
"$PM" "$B/vedioto3d_adapt_contact.py" "$TRACK" "$OUT/qpos.npz" || echo "  (qpos step skipped)"
[ -f "$OUT/qpos.npz" ] && "$PM" "$B/grf_inverse.py" "$OUT/qpos.npz" "$OUT/grf.npz" || true
# TODO soul: OpenSim SO with GRF -> muscle activations; forward FMAX character tiers.
echo "=== DONE -> $OUT ==="
ls -1 "$OUT" | sed "s/^/  /"
