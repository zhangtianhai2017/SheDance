#!/usr/bin/env python3
"""HARD foot-lock with proper contact detection (the decisive clean step). Detect plants by VERTICAL
behaviour (toe near LOCAL floor AND not lifting -> catches sliding plants, distinguishes from steps),
then for each contact segment FORCE the ankle to its plant-ENTRY XY via analytic 2-bone leg IK
(swing-only, twist preserved), slerp-blended at segment edges. Removes the monocular foot-skate exactly.
Only hip+knee axis-angle change; vertical drift is left to the export's floor handling.

Usage: python footlock_hard.py <in_poses.npz> <out_poses.npz>   Env: BAND=0.05 VTHR=0.25
"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import minimum_filter1d, binary_opening, binary_closing

IN, OUT = sys.argv[1], sys.argv[2]
BAND = float(os.environ.get("BAND", "0.05")); VTHR = float(os.environ.get("VTHR", "0.25"))
RJ = np.asarray(np.load(os.path.expanduser("~/shedance/osim/our_humanoid.npz"), allow_pickle=True)["joints"], float)
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
z = np.load(IN, allow_pickle=True); poses = np.asarray(z["poses"], float).copy(); trans = np.asarray(z["trans"], float)
T = len(poses); FPS = float(z["mocap_framerate"]) if "mocap_framerate" in z.files else 30.0
LEGS = [(1, 4, 7, 10), (2, 5, 8, 11)]                 # hip, knee, ankle, toe


def fk(pt, tr):
    Gr = [None] * 22; J = np.zeros((22, 3))
    for j in range(22):
        Rl = Rot.from_rotvec(pt[3 * j:3 * j + 3]).as_matrix()
        if PAR[j] == -1: Gr[j] = Rl; J[j] = RJ[j]
        else: Gr[j] = Gr[PAR[j]] @ Rl; J[j] = J[PAR[j]] + Gr[PAR[j]] @ (RJ[j] - RJ[PAR[j]])
    return Gr, J + tr


def swing(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.clip(np.dot(a, b), -1, 1)); s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0: return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, p); return Rot.from_rotvec(np.pi * ax / np.linalg.norm(ax)).as_matrix()
    return Rot.from_rotvec(np.arccos(c) * v / s).as_matrix()


# FK once for detection
J_all = np.array([fk(poses[t], trans[t])[1] for t in range(T)])      # (T,22,3)
toeZ = J_all[:, [10, 11], 2]


def contact(zf):                                       # near LOCAL floor AND not lifting
    lf = minimum_filter1d(zf, 21, mode="nearest"); vz = np.abs(np.r_[0.0, np.diff(zf)] * FPS)
    c = (zf - lf < BAND) & (vz < VTHR)
    return binary_opening(binary_closing(c, np.ones(3)), np.ones(3))
CMASK = [contact(toeZ[:, 0]), contact(toeZ[:, 1])]
print("contact frames L=%d R=%d /%d" % (int(CMASK[0].sum()), int(CMASK[1].sum()), T))


def segments(mask):
    segs = []; s = None
    for t in range(len(mask)):
        if mask[t] and s is None: s = t
        elif not mask[t] and s is not None: segs.append((s, t)); s = None
    if s is not None: segs.append((s, len(mask)))
    return segs


nseg = 0
for li, (hip, knee, ankle, toe) in enumerate(LEGS):
    L1 = np.linalg.norm(RJ[knee] - RJ[hip]); L2 = np.linalg.norm(RJ[ankle] - RJ[knee])
    for (s, e) in segments(CMASK[li]):
        if e - s < 3: continue
        nseg += 1
        pin = J_all[s, ankle, :2].copy()               # plant-ENTRY ankle XY (exact anchor)
        ramp = max(1, min(4, (e - s) // 3))
        for t in range(s, e):
            w = min(1.0, (t - s + 1) / ramp, (e - t) / ramp)
            Gr, J = fk(poses[t], trans[t]); H, K, A = J[hip], J[knee], J[ankle]
            Tt = np.array([pin[0], pin[1], A[2]])       # lock XY, keep current height
            HT = Tt - H; d = np.linalg.norm(HT)
            if d < 1e-6: continue
            dirHT = HT / d
            if d >= L1 + L2 - 1e-4:
                Kp = H + L1 * dirHT
            else:
                pole = (K - H) - np.dot(K - H, dirHT) * dirHT
                pole = pole / np.linalg.norm(pole) if np.linalg.norm(pole) > 1e-6 else np.array([0, 0, 1.0])
                ca = np.clip((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1)
                Kp = H + L1 * (np.cos(np.arccos(ca)) * dirHT + np.sin(np.arccos(ca)) * pole)
            Ghip_new = swing(K - H, Kp - H) @ Gr[hip]
            Knew = H + Ghip_new @ (RJ[knee] - RJ[hip])
            Gknee_inh = Ghip_new @ Rot.from_rotvec(poses[t, 3 * knee:3 * knee + 3]).as_matrix()
            Anew = Knew + Gknee_inh @ (RJ[ankle] - RJ[knee])
            Gknee_new = swing(Anew - Knew, Tt - Knew) @ Gknee_inh
            loc_hip = Rot.from_matrix(Gr[0].T @ Ghip_new); loc_knee = Rot.from_matrix(Ghip_new.T @ Gknee_new)
            for jid, new in ((hip, loc_hip), (knee, loc_knee)):
                old = Rot.from_rotvec(poses[t, 3 * jid:3 * jid + 3])
                poses[t, 3 * jid:3 * jid + 3] = Slerp([0, 1], Rot.concatenate([old, new]))(w).as_rotvec()

out = {k: z[k] for k in z.files}; out["poses"] = poses
np.savez(OUT, **out)
print("HARD foot-lock: %d segments, %d frames -> %s" % (nseg, T, OUT))

# verdict: per-foot slide DURING its detected contact (did the hard lock hold each foot?)
Jo = np.array([fk(poses[t], trans[t])[1] for t in range(T)])
for li, (hip, knee, ankle, toe) in enumerate(LEGS):
    sp = [np.linalg.norm(Jo[t, toe, :2] - Jo[t - 1, toe, :2]) * FPS * 100 for t in range(1, T) if CMASK[li][t] and CMASK[li][t - 1]]
    sp = np.array(sp or [0.0])
    print("  leg%d (%s) 接触相脚滑 median %.1f mean %.1f max %.1f cm/s" % (li, "L" if li == 0 else "R", np.median(sp), sp.mean(), sp.max()))
