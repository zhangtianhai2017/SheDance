#!/usr/bin/env python3
"""Convert one AIST++ SMPL motion (.pkl in motions.zip) to the raw SMPL-H npz format the
retarget expects (same keys as ~/shedance/aist_amass: poses(N,156), trans, betas, gender, fps).
AIST++ pkl: smpl_poses (N,24,3) axis-angle, smpl_scaling (scalar), smpl_trans (N,3).
SMPL(24 joints) -> SMPL-H(52): keep body joints 0-21 (66 vals), fingers(22-51)=0.
Args: motions.zip seq_name out_npz"""
import sys, os, io, zipfile, pickle, numpy as np
zpath, seq, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(zpath)
cand = [n for n in z.namelist() if os.path.basename(n) == seq + ".pkl"]
assert cand, f"{seq}.pkl not found in zip"
d = pickle.loads(z.read(cand[0]))
poses = np.asarray(d["smpl_poses"], float).reshape(-1, 72)          # (N,24,3)->(N,72)
scaling = float(np.asarray(d["smpl_scaling"], float).reshape(-1)[0])
trans = np.asarray(d["smpl_trans"], float).reshape(-1, 3) / scaling  # -> meters
# AIST++ coordinate -> aist_amass/AMASS convention: rotate +90deg about world X (make upright).
from scipy.spatial.transform import Rotation as R
_Rx = R.from_euler("x", 90, degrees=True)
poses[:, :3] = (_Rx * R.from_rotvec(poses[:, :3])).as_rotvec()       # root global orient
trans = (_Rx.as_matrix() @ trans.T).T                                # root translation
N = poses.shape[0]
poses156 = np.zeros((N, 156)); poses156[:, :66] = poses[:, :66]      # body+wrists; fingers=0
np.savez(out, poses=poses156, trans=trans, betas=np.zeros(16, float),
         gender="neutral", mocap_framerate=60.0)
print(f"{seq}: {N} frames, scaling={scaling:.3f}, trans-range {np.ptp(trans,0).round(2)} m -> {out}")
