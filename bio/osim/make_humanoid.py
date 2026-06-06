#!/usr/bin/env python3
"""Build OUR humanoid template (rest-pose SMPL-H mesh + joints + faces) at given female betas.
This is the skeleton/shape reference for source_to_our_poses / footlock / export_joints_world.
Args: betas_csv out_npz   e.g.  make_humanoid.py "-2,2" ~/shedance/osim/our_humanoid.npz"""
import os, sys, numpy as np
import general_motion_retargeting.utils.smpl as _sm
_o = _sm.SMPLH_Parser.__init__
def _i(self, *a, **k): k.setdefault("num_betas", 16); _o(self, *a, **k)
_sm.SMPLH_Parser.__init__ = _i
from general_motion_retargeting.utils.smpl import load_smplh_file
BETAS = [float(x) for x in sys.argv[1].split(",")]; OUT = sys.argv[2]
MODELS = os.path.expanduser("~/shedance/smpl_models")
b = np.zeros(16); b[:len(BETAS)] = BETAS
tmp = os.path.expanduser("~/shedance/osim/_zeropose.npz")
np.savez(tmp, poses=np.zeros((1, 156)), trans=np.zeros((1, 3)), betas=b, gender="female", mocap_framerate=30.0)
sd, bm, so, h = load_smplh_file(tmp, MODELS)
V = np.asarray(so.vertices, float); V = V[0] if V.ndim == 3 else V
F = np.asarray(bm.faces)
try:
    J = np.asarray(so.joints, float); J = J[0] if J.ndim == 3 else J
except Exception as e:
    print("so.joints failed:", e, "| so attrs:", [a for a in dir(so) if not a.startswith("_")][:40]); raise
np.savez(OUT, vertices=V, joints=J, faces=F)
print(f"our_humanoid: vertices {V.shape} joints {J.shape} faces {F.shape} betas {BETAS} -> {OUT}")
