#!/usr/bin/env python3
"""Generate an OpenSim .trc marker file from the MuJoCo MyoFullBody motion, using shared-body
ORIGINS (= joint centers) as markers. Both models share these body names/frames (same source),
so the markers correspond. IK on these markers solves the OpenSim coords in their OWN convention
-> fundamentally fixes the name-mapping convention bugs (left-shoulder sign, pro_sup offset...).
MuJoCo Z-up -> OpenSim Y-up: (x,y,z) -> (x, z, -y).
"""
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
OUT = os.path.expanduser("~/shedance/osim/pop.trc")
MARK = ["pelvis", "lumbar5", "femur_r", "femur_l", "tibia_r", "tibia_l",
        "talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l",
        "humerus_r", "humerus_l", "ulna_r", "ulna_l", "radius_r", "radius_l"]


def gaussian_smooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gaussian_smooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    env = MyoFullBody(disable_fingers=True)
    m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
    d = mujoco.MjData(m)
    bid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in MARK}
    miss = [nm for nm, i in bid.items() if i < 0]
    assert not miss, f"missing bodies: {miss}"
    T = qpos.shape[0]
    pos = np.zeros((T, len(MARK), 3))
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        for j, nm in enumerate(MARK):
            x, y, z = d.xpos[bid[nm]]
            pos[t, j] = (x, z, -y)   # MuJoCo Z-up -> OpenSim Y-up

    with open(OUT, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(OUT)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{freq:.1f}\t{freq:.1f}\t{T}\t{len(MARK)}\tm\t{freq:.1f}\t1\t{T}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(MARK) + "\t\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(MARK))) + "\t\n")
        f.write("\n")
        for t in range(T):
            row = [str(t + 1), f"{t/freq:.5f}"]
            for j in range(len(MARK)):
                row += [f"{pos[t,j,0]:.6f}", f"{pos[t,j,1]:.6f}", f"{pos[t,j,2]:.6f}"]
            f.write("\t".join(row) + "\n")
    print(f"wrote {OUT}: {T} frames x {len(MARK)} markers @ {freq:.0f}Hz")
    print("markers:", ", ".join(MARK))


if __name__ == "__main__":
    main()
