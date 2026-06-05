#!/usr/bin/env python3
"""Step 1 of the corrected flow: express the source motion on OUR humanoid by ANGLE TRANSFER
(not bone scaling, not a second retarget). Per frame, aim each of OUR humanoid's bones along the
source bone direction (swing only; twist dropped -- "don't need it too fine"; premise: human
proportions are close). Output = OUR humanoid's SMPL poses (betas 0,3, female) driving both skin and
muscle downstream -- ONE retarget, at the input.  Args: track_npz our_humanoid_npz out_npz  Env: FPS"""
import os, sys, numpy as np
TRACK, HUM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = float(os.environ.get("FPS", "30"))
tr = np.load(TRACK, allow_pickle=True); KP = np.asarray(tr["keypoints_3d"], float)        # (T,70,3) camera
KP = np.stack([KP[:, :, 0], KP[:, :, 2], -KP[:, :, 1]], -1)                                # -> world Z-up
H = np.load(HUM, allow_pickle=True); restJF = np.asarray(H["joints"], float); restJ = restJF[:24]   # SMPL-H rest joints (full 73 / body 24)
T = len(KP)
# SMPL 24-joint parents + the primary "aim child" (the bone each joint orients)
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
AIM = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
K = {"l_hip": 9, "r_hip": 10, "l_knee": 11, "r_knee": 12, "l_ank": 13, "r_ank": 14, "l_toe": 15,
     "r_toe": 18, "neck": 69, "l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 62, "r_wri": 41,
     "l_ear": 3, "r_ear": 4}
def srcJ(k):           # (24,3) SMPL joint positions for one frame's keypoints k
    mid = lambda a, b: 0.5 * (k[K[a]] + k[K[b]])
    pel = mid("l_hip", "r_hip"); neck = k[K["neck"]]
    J = np.zeros((24, 3))
    J[0] = pel; J[1] = k[K["l_hip"]]; J[2] = k[K["r_hip"]]
    J[3] = pel + 0.25 * (neck - pel); J[6] = pel + 0.55 * (neck - pel); J[9] = pel + 0.82 * (neck - pel)
    J[4] = k[K["l_knee"]]; J[5] = k[K["r_knee"]]; J[7] = k[K["l_ank"]]; J[8] = k[K["r_ank"]]
    J[10] = k[K["l_toe"]]; J[11] = k[K["r_toe"]]; J[12] = neck; J[15] = mid("l_ear", "r_ear")
    J[13] = 0.5 * (neck + k[K["l_sho"]]); J[14] = 0.5 * (neck + k[K["r_sho"]])
    J[16] = k[K["l_sho"]]; J[17] = k[K["r_sho"]]; J[18] = k[K["l_elb"]]; J[19] = k[K["r_elb"]]
    J[20] = k[K["l_wri"]]; J[21] = k[K["r_wri"]]; J[22] = J[20]; J[23] = J[21]
    return J
def align(a, b):       # min rotation matrix taking unit a -> unit b
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1, 0, 0]); ax = ax if np.linalg.norm(ax) > 1e-6 else np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax); return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)
def frame(j0, jr, jl, jup):    # build an orth0normal frame from hip-line + up
    right = jr - jl; right /= np.linalg.norm(right); up = jup - j0; up -= up.dot(right) * right
    up /= np.linalg.norm(up); fwd = np.cross(right, up)
    return np.stack([right, up, fwd], 1)
def mkframe(fwd, up):          # [right|up|fwd] from forward + up (full orientation for head/feet)
    up = up / np.linalg.norm(up); fwd = fwd - fwd.dot(up) * up; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(up, fwd); right /= (np.linalg.norm(right) + 1e-9)
    return np.column_stack([right, up, fwd])
def handframe(fwd, up):        # [right|up|fwd], FWD primary (finger direction reliable; palm normal noisier)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9); up = up - up.dot(fwd) * fwd
    up /= (np.linalg.norm(up) + 1e-9); right = np.cross(up, fwd); right /= (np.linalg.norm(right) + 1e-9)
    return np.column_stack([right, up, fwd])
from scipy.spatial.transform import Rotation as R
order = [3, 6, 9, 12, 15, 1, 4, 7, 10, 2, 5, 8, 11, 13, 16, 18, 20, 14, 17, 19, 21]
poses = np.zeros((T, 156)); trans = np.zeros((T, 3))
restF0 = frame(restJ[0], restJ[2], restJ[1], restJ[3])
for t in range(T):
    sJ = srcJ(KP[t]); k = KP[t]; Rg = [np.eye(3)] * 24
    Rg[0] = frame(sJ[0], sJ[2], sJ[1], sJ[3]) @ restF0.T          # root: rest pelvis frame -> source
    for j in order:
        p = PAR[j]
        if j == 15:                                              # head: face-forward full frame (nose+ears), not swing
            em = 0.5 * (k[3] + k[4]); Rg[j] = mkframe(k[0] - em, em - k[69])
        elif j in (7, 8):                                        # ankle: real foot-plane frame (heel->toe fwd, sole normal up)
            big, sml, heel = (15, 16, 17) if j == 7 else (18, 19, 20)
            n = np.cross(k[big] - k[heel], k[sml] - k[heel])
            if n[2] < 0: n = -n
            Rg[j] = mkframe(k[big] - k[heel], n if np.linalg.norm(n) > 1e-6 else np.array([0.0, 0.0, 1.0]))
        elif j in (20, 21):                                      # wrist: orient hand from source finger knuckles (faithful hand facing)
            if j == 20: sw, smc, si, sp, rw, rmc, ri, rp = k[62], k[[49, 53, 57, 61]], k[49], k[61], restJF[20], restJF[[22, 25, 31, 28]], restJF[22], restJF[28]
            else:       sw, smc, si, sp, rw, rmc, ri, rp = k[41], k[[28, 32, 36, 40]], k[28], k[40], restJF[21], restJF[[37, 40, 46, 43]], restJF[37], restJF[43]
            Fs = handframe(smc.mean(0) - sw, np.cross(si - sw, sp - sw)); Fr = handframe(rmc.mean(0) - rw, np.cross(ri - rw, rp - rw))
            Rg[j] = Fs @ Fr.T
        elif j in AIM:
            c = AIM[j]; rest_dir = restJ[c] - restJ[j]; src_dir = sJ[c] - sJ[j]
            aim = align(Rg[p] @ rest_dir, src_dir); Rg[j] = aim @ Rg[p]
        else:
            Rg[j] = Rg[p]
        poses[t, 3 * j:3 * j + 3] = R.from_matrix(Rg[p].T @ Rg[j]).as_rotvec()
    poses[t, 0:3] = R.from_matrix(Rg[0]).as_rotvec()
    trans[t] = sJ[0]
np.savez(OUT, poses=poses, trans=trans, betas=np.array([0, 3] + [0] * 14, float), gender="female", mocap_framerate=FPS)
print(f"our-humanoid poses {poses.shape} @ {FPS}Hz -> {OUT}", flush=True)
