#!/usr/bin/env python3
"""SOUL layer step 1: GRF by COM dynamics (master plan 3.2) -- the physical way without force plates.
Total GRF = M*(a_com + g) (Newton on the whole-body COM); distribute to the contacting foot/feet with
CoP at the foot. No soft-contact-penetration artifact (the MuJoCo inverse-with-contact route gave the
stiffness*penetration force, not the real GRF). Output per-foot GRF + CoP for OpenSim SO.

Usage: python grf_inverse.py <qpos_cache.npz> <out_grf.npz>"""
import os, sys, numpy as np, mujoco
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from musclemimic.environments.humanoids import MyoFullBody

IN, OUT = sys.argv[1], sys.argv[2]
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
d = mujoco.MjData(m); Bt = mujoco.mjtObj.mjOBJ_BODY
foot_b = {s: mujoco.mj_name2id(m, Bt, s) for s in ("toes_l", "toes_r", "calcn_l", "calcn_r")}
M = float(m.body_mass.sum()); g = 9.81; weight = M * g

z = np.load(IN, allow_pickle=True); qpos = np.asarray(z["qpos"], float).copy(); freq = float(z["frequency"]); T = len(qpos)
def gsmooth(x, s=2.5, r=6):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum(); xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
qpos = gsmooth(qpos); qn = qpos[:, 3:7]; qpos[:, 3:7] = qn / np.linalg.norm(qn, axis=1, keepdims=True)

# per-frame COM + foot positions
com = np.zeros((T, 3)); footP = {s: np.zeros((T, 3)) for s in foot_b}
mass = m.body_mass.copy()
for t in range(T):
    d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
    com[t] = (mass[:, None] * d.xipos).sum(0) / M
    for s, b in foot_b.items(): footP[s][t] = d.xpos[b]
# de-drift the feet to a Z=0 floor for contact detection + CoP
toeZ_l = footP["toes_l"][:, 2]; toeZ_r = footP["toes_r"][:, 2]
floor = gaussian_filter1d(minimum_filter1d(np.minimum(toeZ_l, toeZ_r), 15, mode="nearest"), 4, mode="nearest")

# COM acceleration -> total GRF (Newton)
a = np.zeros((T, 3)); a[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) * (freq ** 2)
a = gsmooth(a, 2.0, 5)
GRF_tot = M * (a + np.array([0, 0, g]))                       # (T,3)

# contact per foot (toe near floor) + distribute total GRF to contacting feet
def contact(tz):
    return (tz - floor < 0.06)
cL, cR = contact(toeZ_l), contact(toeZ_r)
GRF = {"l": np.zeros((T, 3)), "r": np.zeros((T, 3))}; CoP = {"l": np.zeros((T, 3)), "r": np.zeros((T, 3))}
for t in range(T):
    if cL[t] and cR[t]:                                       # double support -> split 50/50
        GRF["l"][t] = GRF_tot[t] * 0.5; GRF["r"][t] = GRF_tot[t] * 0.5
        CoP["l"][t] = footP["toes_l"][t] * [1, 1, 0]; CoP["r"][t] = footP["toes_r"][t] * [1, 1, 0]
    else:                                                     # body is always supported -> lower foot bears it
        lf, tn = ("l", "toes_l") if toeZ_l[t] <= toeZ_r[t] else ("r", "toes_r")
        GRF[lf][t] = GRF_tot[t]; CoP[lf][t] = footP[tn][t] * [1, 1, 0]

np.savez(OUT, grf_l=GRF["l"].astype(np.float32), grf_r=GRF["r"].astype(np.float32),
         cop_l=CoP["l"].astype(np.float32), cop_r=CoP["r"].astype(np.float32),
         grf_total=GRF_tot.astype(np.float32), frequency=freq)
st = cL | cR
print("body weight = %.0f N" % weight)
print("GRF_total_z: mean %.0f  median %.0f  min %.0f  max %.0f N  (站立相均值应≈体重 %.0f)" % (
    GRF_tot[st, 2].mean(), np.median(GRF_tot[st, 2]), GRF_tot[:, 2].min(), GRF_tot[:, 2].max(), weight))
print("接触帧: 左 %d 右 %d 双支撑 %d 腾空 %d /%d" % (int(cL.sum()), int(cR.sum()), int((cL & cR).sum()), int((~st).sum()), T))
print("腾空帧 GRF_total_z 均值 %.0f N (应≈0=自由落体)" % (GRF_tot[~st, 2].mean() if (~st).any() else 0))
