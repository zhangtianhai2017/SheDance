#!/usr/bin/env python3
"""52-joint (body + ARTICULATED fingers) variant of export_joints_world, for the HAND calibration FBX.
FK all 52 SMPL-H joints from a (T,156) poses npz; output world positions + per-joint world rotation +
the 52-joint names/parents (so fbx_export builds finger bones). The body export stays 24-joint; this is
the finger-articulated path only. Args: in_poses_npz out_npz   Env: FPS"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation as R
IN, OUT = sys.argv[1], sys.argv[2]; FPS = float(os.environ.get("FPS", "30"))
RJ = np.asarray(np.load(os.path.expanduser("~/shedance/osim/our_humanoid.npz"), allow_pickle=True)["joints"], float)  # SMPL-H rest, 73
N = 52
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26,
       20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]   # SMPL-H kintree
FING = ["index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
        "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]
NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
         "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"] \
        + ["left_" + f for f in FING] + ["right_" + f for f in FING]
z = np.load(IN, allow_pickle=True); poses = np.asarray(z["poses"], float); trans = np.asarray(z["trans"], float)
T = len(poses); JW = np.zeros((T, N, 3)); JWR = np.zeros((T, N, 4))
for t in range(T):
    Gr = [None] * N; J = np.zeros((N, 3))
    for j in range(N):
        Rl = R.from_rotvec(poses[t, 3 * j:3 * j + 3]).as_matrix()
        if PAR[j] == -1:
            Gr[j] = Rl; J[j] = RJ[j]
        else:
            Gr[j] = Gr[PAR[j]] @ Rl; J[j] = J[PAR[j]] + Gr[PAR[j]] @ (RJ[j] - RJ[PAR[j]])
    JW[t] = J + trans[t]
    for j in range(N):
        JWR[t, j] = R.from_matrix(Gr[j]).as_quat()                  # world rotation quat xyzw
MALIGN = R.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0])).as_quat()  # SMPL zero-pose canonical Y-up -> world Z-up
JWRR = np.tile(MALIGN.astype(np.float32), (N, 1))
np.savez(OUT, joints_world=JW.astype(np.float32), joint_world_rot=JWR.astype(np.float32),
         joint_world_rot_rest=JWRR, rot_format=np.array("quat_xyzw"),
         frame=np.array("world_right_handed_Zup_meters"), joint_names=np.array(NAMES), parents=np.array(PAR), fps=FPS)
print("joints_world(hands) %s -> %s  (%d joints incl 30 articulated fingers)" % (JW.shape, OUT, N))
