#!/usr/bin/env python3
"""Export joints_world (T,24,3) = SMPL 24-joint WORLD positions from a poses npz, for downstream
character driving (no smplx needed on their side). Body joints 0-21 via FK; left_hand/right_hand
(22,23) = hand-CENTER points (mean of that hand's SMPL-H finger joints, rigid with the wrist).
`joint_names` is the AUTHORITATIVE list of exactly which joints are present, in order.
Args: in_poses_npz out_npz   Env: FPS    (rest skeleton from ~/shedance/osim/our_humanoid.npz)"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation as R
IN, OUT = sys.argv[1], sys.argv[2]; FPS = float(os.environ.get("FPS", "30"))
RJ = np.asarray(np.load(os.path.expanduser("~/shedance/osim/our_humanoid.npz"), allow_pickle=True)["joints"], float)  # SMPL-H rest, 73
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]   # SMPL-24 (22->l_wrist, 23->r_wrist)
NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
         "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"]
z = np.load(IN, allow_pickle=True); poses = np.asarray(z["poses"], float); trans = np.asarray(z["trans"], float)
lhc = RJ[22:37].mean(0); rhc = RJ[37:52].mean(0)   # hand-center rest = mean of that hand's 15 SMPL-H finger joints
T = len(poses); JW = np.zeros((T, 24, 3)); JWR = np.zeros((T, 24, 4))   # +per-joint WORLD rotation (quat xyzw)
for t in range(T):
    Gr = [None] * 22; J = np.zeros((22, 3))
    for j in range(22):
        Rl = R.from_rotvec(poses[t, 3 * j:3 * j + 3]).as_matrix()
        if PAR[j] == -1: Gr[j] = Rl; J[j] = RJ[j]
        else: Gr[j] = Gr[PAR[j]] @ Rl; J[j] = J[PAR[j]] + Gr[PAR[j]] @ (RJ[j] - RJ[PAR[j]])
    JW[t, :22] = J + trans[t]
    JW[t, 22] = J[20] + trans[t] + Gr[20] @ (lhc - RJ[20])   # left_hand center
    JW[t, 23] = J[21] + trans[t] + Gr[21] @ (rhc - RJ[21])   # right_hand center
    for j in range(22):
        JWR[t, j] = R.from_matrix(Gr[j]).as_quat()           # world rotation, quaternion xyzw, same frame as joints_world
    JWR[t, 22] = R.from_matrix(Gr[20]).as_quat(); JWR[t, 23] = R.from_matrix(Gr[21]).as_quat()   # hands rigid w/ wrists
# REST baseline: SMPL zero-pose => every joint's GLOBAL rotation is identity in canonical (Y-up);
# in our world (Z-up) every joint therefore shares the single Y-up->Z-up alignment Rx(+90deg).
MALIGN = R.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0])).as_quat()        # xyzw
JWRR = np.tile(MALIGN.astype(np.float32), (24, 1))                          # (24,4) joint_world_rot_rest
np.savez(OUT, joints_world=JW.astype(np.float32), joint_world_rot=JWR.astype(np.float32),
         joint_world_rot_rest=JWRR,
         rot_format=np.array("quat_xyzw"), frame=np.array("world_right_handed_Zup_meters"),
         joint_names=np.array(NAMES), parents=np.array(PAR), fps=FPS)
dl = np.linalg.norm(JW[:, 22] - JW[:, 20], axis=1).mean(); dr = np.linalg.norm(JW[:, 23] - JW[:, 21], axis=1).mean()
print(f"joints_world {JW.shape} -> {OUT} | l_hand<->l_wrist {dl*100:.1f}cm  r_hand<->r_wrist {dr*100:.1f}cm")
