#!/usr/bin/env python3
"""Plot reference vs F_max archetypes for key joints: shows the weak body under-shooting peaks
(the 'embodied imitator' — same intent, different bodies). Output PNG."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OD = os.path.expanduser("~/shedance/osim")
CACHE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
t0, t1 = 2.0, 4.0


def smooth(x, s=2.5, r=6):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, "valid") for j in range(x.shape[1])], 1)


c = np.load(CACHE, allow_pickle=True); freq = float(c["frequency"])
ref = smooth(np.asarray(c["qpos"], float))[int(t0 * freq):int(t1 * freq)]
arch = {f: np.load(os.path.join(OD, f"cmcq_{t0:.1f}_{t1:.1f}_f{f:.2f}.npy")) for f in (2.5, 1.0, 0.7)}
n = min([ref.shape[0]] + [a.shape[0] for a in arch.values()])
t = np.arange(n) / freq

joints = [("shoulder_elv_r", 36), ("elbow_flex_r", 39), ("hip_flexion_r", 61), ("elv_angle_r", 35)]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (nm, qi) in zip(axes.ravel(), joints):
    ax.plot(t, ref[:n, qi], "k-", lw=2.2, label="reference (intent)")
    for f, col in zip((2.5, 1.0, 0.7), ("tab:blue", "tab:green", "tab:red")):
        ax.plot(t, arch[f][:n, qi], col, lw=1.3, label=f"Fmax x{f}")
    ax.set_title(nm); ax.set_xlabel("s"); ax.set_ylabel("rad"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Embodied imitator: same intent, different bodies (weak under-shoots)", fontsize=13)
fig.tight_layout()
out = os.path.join(OD, "archetypes_joints.png"); fig.savefig(out, dpi=90)
print("wrote", out)
# also report peak-reach per joint per archetype
for nm, qi in joints:
    amp_ref = ref[:n, qi].max() - ref[:n, qi].min()
    amps = {f: arch[f][:n, qi].max() - arch[f][:n, qi].min() for f in (2.5, 1.0, 0.7)}
    print(f"{nm}: ref range={amp_ref:.3f}  " + " ".join(f"x{f}={amps[f]/amp_ref*100:.0f}%" for f in (2.5, 1.0, 0.7)))
