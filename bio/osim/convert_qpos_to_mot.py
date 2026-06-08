#!/usr/bin/env python3
"""Convert the retargeted MyoFullBody qpos trajectory -> OpenSim coordinates .mot.

MyoFullBody (MuJoCo, MyoConverter-derived) and cyclistFullBodyMuscle.osim share source
models (Rajagopal legs / Saul arms / lumbar), so PRIMARY coordinates map 1:1 by name.
Shoulder-girdle & knee coupler DOFs are CONSTRAINT-dependent in the .osim (auto-computed),
so we don't map them. First pass: pelvis orientation upright (isolate joint-convention test);
pelvis translation taken from qpos with MuJoCo Z-up -> OpenSim Y-up axis swap.
"""
import os, sys, numpy as np
import opensim as osim

# general: take qpos npz + output mot as args (falls back to the original AIST probe cache)
CACHE = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
MODEL = os.path.expanduser("~/shedance/osim/cyclistFullBodyMuscle.osim")
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/shedance/osim/pop.mot")

# OpenSim coord name -> MyoFullBody qpos index (primary joints only)
MAP = {
    "hip_flexion_r": 61, "hip_adduction_r": 62, "hip_rotation_r": 63, "knee_angle_r": 66,
    "knee_angle_r_beta": 74, "ankle_angle_r": 69, "subtalar_angle_r": 70, "mtp_angle_r": 71,
    "hip_flexion_l": 75, "hip_adduction_l": 76, "hip_rotation_l": 77, "knee_angle_l": 80,
    "knee_angle_l_beta": 88, "ankle_angle_l": 83, "subtalar_angle_l": 84, "mtp_angle_l": 85,
    "L4_L5_FE": 13, "L4_L5_LB": 14, "L4_L5_AR": 15, "L3_L4_FE": 16, "L3_L4_LB": 17, "L3_L4_AR": 18,
    "L2_L3_FE": 19, "L2_L3_LB": 20, "L2_L3_AR": 21, "L1_L2_FE": 22, "L1_L2_LB": 23, "L1_L2_AR": 24,
    "L5_S1_FE": 7, "L5_S1_LB": 8, "L5_S1_AR": 9,   # MyoFullBody flex_extension/lat_bending/axial_rotation
    "elv_angle_r": 35, "shoulder_elv_r": 36, "shoulder_rot_r": 38, "elbow_flex_r": 39,
    "pro_sup_r": 40, "wrist_flex_r": 42, "wrist_dev_r": 41,
    "elv_angle_l": 53, "shoulder_elv_l": 54, "shoulder_rot_l": 56, "elbow_flex_l": 57,
    "pro_sup_l": 58, "wrist_flex_l": 60, "wrist_dev_l": 59,
}


def gaussian_smooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gaussian_smooth(np.asarray(c["qpos"], float))
    freq = float(c["frequency"]); T = qpos.shape[0]
    model = osim.Model(MODEL); model.initSystem()
    cs = model.getCoordinateSet()
    coords = [cs.get(i).getName() for i in range(cs.getSize())]
    n_mapped = sum(1 for cn in coords if cn in MAP)
    print(f"T={T} freq={freq}  opensim coords={len(coords)}  mapped={n_mapped}")

    # pelvis: orientation upright (0), translation from qpos (Z-up -> Y-up)
    px, py, pz = qpos[:, 0], qpos[:, 1], qpos[:, 2]
    pel = {"pelvis_tx": px, "pelvis_ty": pz, "pelvis_tz": -py,
           "pelvis_tilt": np.zeros(T), "pelvis_list": np.zeros(T), "pelvis_rotation": np.zeros(T)}

    # build columns in coord order
    cols = []
    for cn in coords:
        if cn in pel:
            cols.append(pel[cn])
        elif cn in MAP:
            cols.append(qpos[:, MAP[cn]])
        else:
            cols.append(np.zeros(T))   # constrained/unmapped -> neutral
    data = np.array(cols).T   # (T, ncoord)
    t = np.arange(T) / freq
    STEP = int(os.environ.get("MOT_STEP", "3"))   # downsample 100Hz -> ~33Hz (SO speed; 6Hz lowpass anyway)
    idx = list(range(0, T, STEP))

    with open(OUT, "w") as f:
        f.write("pop\nversion=1\n")
        f.write(f"nRows={len(idx)}\nnColumns={len(coords)+1}\ninDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(coords) + "\n")
        for i in idx:
            f.write(f"{t[i]:.5f}\t" + "\t".join(f"{v:.6f}" for v in data[i]) + "\n")
    print(f"wrote {OUT}  ({len(idx)} rows x {len(coords)+1} cols, step={STEP} ~{freq/STEP:.0f}fps)")
    print("sample mapped coords @ frame", T // 2, ":")
    for cn in ("hip_flexion_r", "knee_angle_r", "elbow_flex_r", "shoulder_elv_r", "elv_angle_r", "L4_L5_FE"):
        print(f"  {cn:16s} = {qpos[T//2, MAP[cn]]:+.3f} rad")


if __name__ == "__main__":
    main()
