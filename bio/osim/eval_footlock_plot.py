#!/usr/bin/env python3
"""Top-down footprint check for foot-lock: plot toe paths + stance points (before|after). A clean
lock collapses smeared stance points into tight footprint clusters. Args: before_npz after_npz humanoid out_png"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
A, B, HUM, OUT = sys.argv[1:5]
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
restJ = np.asarray(np.load(HUM, allow_pickle=True)["joints"], float)[:24]
def fk(npz):
    P = np.load(npz, allow_pickle=True); po = np.asarray(P["poses"], float); tr = np.asarray(P["trans"], float)
    T = len(po); J = np.zeros((T, 24, 3))
    for t in range(T):
        Gr = [None] * 24; Jw = np.zeros((24, 3))
        for j in range(24):
            Rl = R.from_rotvec(po[t, 3 * j:3 * j + 3]).as_matrix()
            if PAR[j] == -1: Gr[j] = Rl; Jw[j] = restJ[j]
            else: Gr[j] = Gr[PAR[j]] @ Rl; Jw[j] = Jw[PAR[j]] + Gr[PAR[j]] @ (restJ[j] - restJ[PAR[j]])
        J[t] = Jw + tr[t]
    return J
JA, JB = fk(A), fk(B)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, J, title in [(axes[0], JA, "BEFORE (raw source trans) - feet smear"), (axes[1], JB, "AFTER foot-lock - feet cluster")]:
    z = J[:, [10, 11], 2]; g = np.percentile(z, 5)
    for foot, col, nm in [(10, "tab:blue", "L"), (11, "tab:red", "R")]:
        p = J[:, foot, :2]; st = J[:, foot, 2] < g + 0.04
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.5, alpha=0.35)
        ax.scatter(p[st, 0], p[st, 1], s=16, color=col, alpha=0.7, label=f"toe {nm} (stance)")
    pel = J[:, 0, :2]; ax.plot(pel[:, 0], pel[:, 1], "k--", lw=1, alpha=0.5, label="pelvis")
    ax.set_title(title); ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT, flush=True)
