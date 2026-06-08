#!/usr/bin/env python3
"""Canonical packager for the A12B sample deliverable: assemble SheDance_sample_dance1/ from the
ONE-COMMAND workflow outputs + the regenerated soul files, so the whole package is internally
CONSISTENT (FBX, kinematics, muscle SO and preview all derive from the same collar/spine-fixed,
adaptive-locked motion). Sources:
  WF (~/shedance/wf_out/dance1)  : poses_lock / joints_world / qpos / dance.fbx / dance_tpose.fbx / preview_skin.mp4
  O  (~/shedance/osim)           : dance1_ik.mot (qpos->mot) + dance1_so_*_activation.sto (SO)
Run the generators first (run_pipeline.sh, convert_qpos_to_mot, static_opt_osim, smpl_skin_render);
package_dance.sh chains all of it.
"""
import os, shutil, numpy as np

O = os.path.expanduser("~/shedance/osim")
WF = os.path.expanduser("~/shedance/wf_out/dance1")          # one-command workflow output (adaptive, collar-fixed)
TOP = "/mnt/c/work/2026/Claude/SheDance"
S = TOP + "/SheDance_sample_dance1"; DS = TOP + "/SheDance_data_structure"

FBX_README = """SheDance dance1 FBX (skinned SMPL body) - for UE 5.6 import + IK retargeting

Revision : v2 (2026-06-08) - clavicle shrug corrected (partial-aim clavicle), spine articulated,
           adaptive contact foot-lock, vertical de-drift. Same format/specs as v1, motion improved.

dance1.fbx        animation, 683 frames @ 30fps, baked to bones; skinned SMPL body mesh.
dance1_tpose.fbx  rest reference (SMPL zero pose = star/A-pose, NOT a horizontal T-pose);
                  use this as the source retarget pose.

Skeleton : 22 SMPL body bones (idx 0-21), single root 'pelvis', SMPL/SMPL-X body names.
Mesh     : 6890 verts / 13776 faces. SMPL-H skin weights, finger weights folded into wrists
           -> 22 vertex groups. Armature modifier. betas[-2,2] female body. No textures.
Coords   : Z-up, X-forward, meters (apply x100 on import if UE wants cm).
Floor    : FLOOR-ALIGNED. Lowest skinned-mesh contact over the whole clip -> Z=0 (single global
           offset, relative vertical motion preserved: planted frames touch floor, jumps go airborne).
           frame0 pelvis at origin (X=Y=0). Pelvis natural height ~0.89m.
Bone rest: ALL bone rest rotations = Rx(+90deg) (so standard SMPL weights deform correctly).
           Bones therefore look like short +Z stubs and do not visually connect - this is correct;
           LBS deformation and per-bone world rotation (= joint_world_rot) are both correct.
Built    : headless Blender (bpy 4.2) from our joints_world.
Verified : bone head-pos 0.0005mm vs joints_world; skin-deform 0.55mm vs hand-computed SMPL-LBS.

Full notes: comms/msgs/*from-SHEDANCE*skinned* ; data contract: ../README.md (SheDance_DATA_CONTRACT).
"""

mp4 = WF + "/preview_skin.mp4"
sz = os.path.getsize(mp4)
assert sz > 10000, f"skin mp4 looks broken: {sz} bytes"
print("skin preview:", sz, "bytes  OK")
shutil.rmtree(S, ignore_errors=True)
for d in ("data", "ref", "preview", "fbx"):
    os.makedirs(S + "/" + d, exist_ok=True)

# --- kinematics: collar/spine-fixed, adaptive contact-locked, de-drifted (from run_pipeline.sh) ---
shutil.copy(WF + "/poses_lock.npz", S + "/data/dance1_poses.npz")
shutil.copy(WF + "/joints_world.npz", S + "/data/dance1_joints_world.npz")
z = np.load(WF + "/qpos.npz", allow_pickle=True); q = np.asarray(z["qpos"])
np.savez(S + "/data/dance1_qpos.npz", qpos=q, frequency=float(z["frequency"]),
         has_fingers=bool(q.shape[1] >= 129), dof=int(q.shape[1]))   # self-describing flags per the contract

# --- soul (muscle) regenerated from the SAME new qpos: qpos -> ik.mot -> Static Optimization ---
shutil.copy(O + "/dance1_so_StaticOptimization_activation.sto", S + "/data/dance1_activation.sto")
shutil.copy(O + "/dance1_ik.mot", S + "/data/dance1_ik.mot")

# --- skinned FBX (collar-fixed), the A12B artifact ---
shutil.copy(WF + "/dance.fbx", S + "/fbx/dance1.fbx")
shutil.copy(WF + "/dance_tpose.fbx", S + "/fbx/dance1_tpose.fbx")
with open(S + "/fbx/README.txt", "w") as f:
    f.write(FBX_README)

# --- docs / ref / preview ---
shutil.copy(TOP + "/SheDance_DATA_CONTRACT.md", S + "/README.md")
for f in os.listdir(DS + "/ref"):
    shutil.copy(DS + "/ref/" + f, S + "/ref/" + f)
shutil.copy(mp4, S + "/preview/dance1_skin.mp4")

print("sample rebuilt | qpos", q.shape, "has_fingers", bool(q.shape[1] >= 129), "dof", q.shape[1])
for root, _, files in os.walk(S):
    for f in sorted(files):
        print("  ", os.path.relpath(os.path.join(root, f), S))
zpath = shutil.make_archive(TOP + "/SheDance_sample_dance1", "zip", TOP, "SheDance_sample_dance1")
print("ZIPPED ->", zpath, os.path.getsize(zpath), "bytes")
