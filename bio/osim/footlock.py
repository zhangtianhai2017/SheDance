#!/usr/bin/env python3
"""Foot-lock by ROOT correction. Trust the leg angles (the dance -- untouched); correct only the
global root trans so a planted foot stops sliding. Removes the monocular-capture foot-skate WITHOUT
changing any pose: the dance is 100% preserved, only WHERE the body sits in world space shifts.
Per frame: contact weight w_f by foot height (low=in contact); correction velocity = -(weighted foot
horizontal velocity); integrate -> root offset; trans[:, xy] += offset.
Args: in_poses_npz our_humanoid_npz out_poses_npz   Env: BAND(0.08 contact ramp m), SMOOTH(2 frames), POINT(toe|ankle), FPS"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
IN, HUM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BAND = float(os.environ.get("BAND", "0.08")); SMOOTH = float(os.environ.get("SMOOTH", "2")); FPS = float(os.environ.get("FPS", "30"))
POINT = os.environ.get("POINT", "toe"); LF, RF = (10, 11) if POINT == "toe" else (7, 8)
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
P = np.load(IN, allow_pickle=True); poses = np.asarray(P["poses"], float); trans0 = np.asarray(P["trans"], float)
restJ = np.asarray(np.load(HUM, allow_pickle=True)["joints"], float)[:24]
T = len(poses); up = 2; hz = [0, 1]
def fk_feet(tr):
    out = np.zeros((T, 2, 3))
    for t in range(T):
        Gr = [None] * 24; Jw = np.zeros((24, 3))
        for j in range(24):
            Rl = R.from_rotvec(poses[t, 3 * j:3 * j + 3]).as_matrix()
            if PAR[j] == -1: Gr[j] = Rl; Jw[j] = restJ[j]
            else: Gr[j] = Gr[PAR[j]] @ Rl; Jw[j] = Jw[PAR[j]] + Gr[PAR[j]] @ (restJ[j] - restJ[PAR[j]])
        out[t, 0] = Jw[LF] + tr[t]; out[t, 1] = Jw[RF] + tr[t]
    return out
F0 = fk_feet(trans0)
zz = F0[:, :, up]; ground = np.percentile(zz, 5)
if os.environ.get("HARD"):                                    # one-hot with HYSTERESIS: pin one foot, switch only on clear lift-off
    w = np.zeros((T, 2)); anchor = int(zz[0, 1] < zz[0, 0])
    for t in range(T):
        oth = 1 - anchor
        if zz[t, anchor] > ground + BAND and zz[t, oth] < ground + BAND: anchor = oth          # current lifted, other planted
        elif zz[t, oth] < zz[t, anchor] - 0.04 and zz[t, oth] < ground + BAND: anchor = oth     # other clearly lower & planted
        if zz[t, anchor] < ground + BAND: w[t, anchor] = 1.0
else:
    w = np.clip((ground + BAND - zz) / BAND, 0, 1)            # (T,2) contact weight by height
vel = np.zeros((T, 2, 2)); vel[1:] = F0[1:][:, :, hz] - F0[:-1][:, :, hz]
wsum = w.sum(1); corr = np.zeros((T, 2))
for c in range(2):
    corr[:, c] = np.where(wsum > 1e-3, -(w[:, 0] * vel[:, 0, c] + w[:, 1] * vel[:, 1, c]) / np.maximum(wsum, 1e-3), 0.0)
if SMOOTH > 0: corr = gaussian_filter1d(corr, SMOOTH, axis=0)
ACCAP = float(os.environ.get("ACCAP", "0"))                   # slew-limit the correction velocity -> bound root accel (jerk) with only a tiny local slide
if ACCAP > 0:
    cap = ACCAP / 100.0
    for t in range(1, T):
        dv = corr[t] - corr[t - 1]; n = np.linalg.norm(dv)
        if n > cap: corr[t] = corr[t - 1] + dv * (cap / n)
offset = np.cumsum(corr, axis=0)
trans = trans0.copy(); trans[:, hz] += offset
def stance_speed(F):
    z = F[:, :, up]; g = np.percentile(z, 5); out = []
    for f in range(2):
        st = z[:, f] < g + 0.04; sp = np.linalg.norm(np.diff(F[:, f][:, hz], axis=0), axis=1) * FPS
        out.append(np.median(sp[st[:-1]]) * 100 if st[:-1].sum() > 5 else float("nan"))
    return out
b = stance_speed(F0); a = stance_speed(fk_feet(trans))
acc = np.linalg.norm(np.diff(offset, 2, axis=0), axis=1) * 100
print(f"footlock POINT={POINT} BAND={BAND} SMOOTH={SMOOTH} HARD={bool(os.environ.get('HARD'))}")
print(f"  root offset: max {np.abs(offset).max() * 100:.1f}cm  mean {np.abs(offset).mean() * 100:.1f}cm  | accel p99 {np.percentile(acc, 99):.1f} max {acc.max():.1f} cm/fr^2 (jerk)")
print(f"  stance horiz speed L/R:  before {b[0]:.0f}/{b[1]:.0f} cm/s  ->  after {a[0]:.0f}/{a[1]:.0f} cm/s")
np.savez(OUT, poses=poses, trans=trans, betas=np.asarray(P["betas"]), gender=str(P["gender"]), mocap_framerate=FPS)
print(f"  poses UNCHANGED (dance preserved); only root trans corrected -> {OUT}", flush=True)
