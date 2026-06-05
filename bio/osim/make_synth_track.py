#!/usr/bin/env python3
"""Make a synthetic VedioTo3D-format track from a known MyoFullBody motion, to round-trip test the
adapter (no real VedioTo3D output yet). Forward the body, emit the used 70 keypoints in WORLD frame,
convert to VedioTo3D camera frame (inverse of the adapter's: world=[cx,cz,-cy] -> cam=[wx,-wz,wy]+Zoff).
Args: in_cache(89-dof qpos) out_track"""
import sys, numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody
IN, OUT = sys.argv[1], sys.argv[2]
c = np.load(IN, allow_pickle=True); q = np.asarray(c["qpos"], float); freq = float(c["frequency"]); T = len(q)
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
d = mujoco.MjData(m); B = mujoco.mjtObj.mjOBJ_BODY
bid = lambda n: mujoco.mj_name2id(m, B, n)
B2K = {"humerus_l": 5, "humerus_r": 6, "ulna_l": 7, "ulna_r": 8, "lunate_l": 62, "lunate_r": 41,
       "femur_l": 9, "femur_r": 10, "tibia_l": 11, "tibia_r": 12, "calcn_l": 13, "calcn_r": 14,
       "toes_l": 15, "toes_r": 18, "head": 3}
kp = np.zeros((T, 70, 3))
for t in range(T):
    d.qpos[:] = q[t]; mujoco.mj_forward(m, d)
    for b, ki in B2K.items():
        kp[t, ki] = d.xpos[bid(b)]
    kp[t, 4] = kp[t, 3]                                          # right_ear ~ head too
    kp[t, 69] = 0.5 * (d.xpos[bid("humerus_l")] + d.xpos[bid("humerus_r")])   # neck
cam = np.stack([kp[:, :, 0], -kp[:, :, 2], kp[:, :, 1]], -1)     # world Z-up -> camera (Y down, Z fwd)
cam[:, :, 2] += 3.0                                              # put 3 m in front (Z>0)
np.savez(OUT, keypoints_3d=cam.astype(np.float32), frames=np.arange(T), fps=freq)
print(f"synth track keypoints_3d {cam.shape} @ {freq}Hz -> {OUT}", flush=True)
