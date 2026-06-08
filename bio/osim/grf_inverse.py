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
WF = max(5, int(round(0.5 * freq))); SF = max(1.0, 0.13 * freq)     # ADAPTIVE floor window/smooth (0.5s / 0.13s)
SQ = max(1.0, 0.08 * freq); SA = max(1.0, 0.06 * freq)             # ADAPTIVE qpos / accel smooth (sec x freq)
def gsmooth(x, s=2.5, r=6):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum(); xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
qpos = gsmooth(qpos, SQ, int(3 * SQ)); qn = qpos[:, 3:7]; qpos[:, 3:7] = qn / np.linalg.norm(qn, axis=1, keepdims=True)
d.qpos[:] = qpos[0]; mujoco.mj_forward(m, d); HEIGHT = float(np.ptp(d.xpos[:, 2])); BAND_C = 0.04 * HEIGHT   # ADAPTIVE contact band ~ 4% of figure height

# per-frame COM + foot positions
com = np.zeros((T, 3)); footP = {s: np.zeros((T, 3)) for s in foot_b}
mass = m.body_mass.copy()
for t in range(T):
    d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
    com[t] = (mass[:, None] * d.xipos).sum(0) / M
    for s, b in foot_b.items(): footP[s][t] = d.xpos[b]
# de-drift to a Z=0 floor; a foot contacts via its LOWEST part (toe OR heel) -- heel-stance (toe up) is still support
toeZ_l = footP["toes_l"][:, 2]; toeZ_r = footP["toes_r"][:, 2]
heelZ_l = footP["calcn_l"][:, 2]; heelZ_r = footP["calcn_r"][:, 2]
footZ_l = np.minimum(toeZ_l, heelZ_l); footZ_r = np.minimum(toeZ_r, heelZ_r)   # each foot's ground clearance (lowest part)
floor = gaussian_filter1d(minimum_filter1d(np.minimum(footZ_l, footZ_r), WF, mode="nearest"), SF, mode="nearest")

# COM acceleration -> total GRF (Newton): the physical truth of total vertical support
a = np.zeros((T, 3)); a[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) * (freq ** 2)
a = gsmooth(a, SA, int(3 * SA))
GRF_tot = M * (a + np.array([0, 0, g]))                       # (T,3)

# AUTOMATIC airborne resolution (general, not per-clip): geometric foot-near-floor MISSES real support
# (heel down/toe up, or monocular float), which is why "airborne" frames showed nonzero GRF. COM dynamics
# arbitrate -- the body is TRULY airborne only when total vertical support is ~0 (free fall); any frame that
# is geometrically off-floor but physically supported is reassigned to the lower foot. Self-consistent for
# any input: a standing dance yields ~0 true-airborne frames, a real jump yields airborne frames with GRF~0.
AIRBORNE_FRAC = float(os.environ.get("AIRBORNE_FRAC", "0.1"))                  # total support < 10% body weight = effectively free fall
cL = (footZ_l - floor < BAND_C); cR = (footZ_r - floor < BAND_C)               # geometric contact (toe OR heel near floor)
airborne = (~cL & ~cR) & (GRF_tot[:, 2] < AIRBORNE_FRAC * weight)              # no contact AND COM ~free fall = real flight
lowL = footZ_l <= footZ_r
def downpart(side, t):                                                          # CoP at whichever foot-part is on the floor
    return (footP["toes_" + side] if footP["toes_" + side][t, 2] <= footP["calcn_" + side][t, 2] else footP["calcn_" + side])[t] * [1, 1, 0]
GRF = {"l": np.zeros((T, 3)), "r": np.zeros((T, 3))}; CoP = {"l": np.zeros((T, 3)), "r": np.zeros((T, 3))}
rescued = 0
for t in range(T):
    if airborne[t]:
        continue                                                              # genuine flight -> both feet 0
    cl, cr = bool(cL[t]), bool(cR[t])
    if not (cl or cr):                                                        # physics says supported but geometry missed -> lower foot
        cl, cr = (True, False) if lowL[t] else (False, True); rescued += 1
    if cl and cr:                                                            # double support -> split 50/50
        GRF["l"][t] = GRF_tot[t] * 0.5; GRF["r"][t] = GRF_tot[t] * 0.5
        CoP["l"][t] = downpart("l", t); CoP["r"][t] = downpart("r", t)
    elif cl:
        GRF["l"][t] = GRF_tot[t]; CoP["l"][t] = downpart("l", t)
    else:
        GRF["r"][t] = GRF_tot[t]; CoP["r"][t] = downpart("r", t)

np.savez(OUT, grf_l=GRF["l"].astype(np.float32), grf_r=GRF["r"].astype(np.float32),
         cop_l=CoP["l"].astype(np.float32), cop_r=CoP["r"].astype(np.float32),
         grf_total=GRF_tot.astype(np.float32), frequency=freq)
sup = ~airborne
print("body weight = %.0f N" % weight)
print("GRF_total_z: 支撑相 mean %.0f median %.0f  全程 min %.0f max %.0f N  (支撑相均值应≈体重 %.0f)" % (
    GRF_tot[sup, 2].mean(), np.median(GRF_tot[sup, 2]), GRF_tot[:, 2].min(), GRF_tot[:, 2].max(), weight))
print("接触帧: 左 %d 右 %d 双支撑 %d | 真腾空(物理 GRF≈0) %d | 几何漏判→物理救回下脚 %d /%d" % (
    int(cL.sum()), int(cR.sum()), int((cL & cR).sum()), int(airborne.sum()), rescued, T))
print("真腾空帧 GRF_total_z 均值 %.0f N (现按物理定义,应≈0=自由落体)" % (GRF_tot[airborne, 2].mean() if airborne.any() else 0))
