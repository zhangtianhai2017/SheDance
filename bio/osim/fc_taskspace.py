#!/usr/bin/env python3
"""Contact-aware physics tracking, iteration 1 (R&D). Soul-correct mechanism: track the FEET in SPACE
(not joint angles). When a foot is on the floor, contact friction CANNOT be slid past by capacity-limited
muscle -> the impossible slide is digested by the body (like FMAX digests impossible forces). Joint-angle
tracking (old cmc) instead forced the foot via the kinematic chain, dragging it through the floor.

Control: gravity/coriolis compensation + TRUNK joint-PD toward ref + LEG operational-space task pulling
each foot toward its (de-drifted) reference world position via J^T F + muscle capacity clamp (FMAX).
Root: vertical+orientation prescribed (simple tier), horizontal free with damping (anti-blowup). Contact ON.

Args: [t0 t1 fmax]. Out: ~/shedance/osim/fct_<...>.npy
"""
import sys, os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser("~/shedance/osim/ourdance1_locked_cache.npz")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
KP, KD = 200.0, 60.0           # trunk joint PD
KPF, KDF = 800.0, 80.0         # foot task (operational space)
KD_ROOT = 80.0                 # horizontal root damping (anti-blowup)
CAP_MIN = 15.0
OUT = os.path.expanduser(f"~/shedance/osim/fct_{t0:.1f}_{t1:.1f}_f{FMAX:.2f}.npy")


def gsmooth(x, s=2.5, r=6):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def cap_envelope(m, d, fmax):
    d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(m.nu, m.nv)
    d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
    cap = (full - passive) * fmax; base = R.T @ passive
    return base + np.minimum(R.T * cap[None, :], 0).sum(1), base + np.maximum(R.T * cap[None, :], 0).sum(1)


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gsmooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    env = MyoFullBody(disable_fingers=True)
    m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
    for i in range(m.ngeom):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        if "floor" in nm or any(k in nm for k in ("foot", "calcn", "toes", "phalanx", "bofoot")):
            m.geom_friction[i][0] = 2.0
    nv = m.nv; dt = m.opt.timestep; d = mujoco.MjData(m)
    FB = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("calcn_l", "calcn_r", "toes_l", "toes_r")}

    # de-drift pelvis z (feet near floor)
    T = qpos.shape[0]; soleZ = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d); soleZ[t] = min(d.xpos[FB[b]][2] for b in ("toes_l", "toes_r", "calcn_l", "calcn_r"))
    floor = gaussian_filter1d(minimum_filter1d(soleZ, 15, mode="nearest"), 4, mode="nearest")
    qpos[:, 2] = qpos[:, 2] - floor

    # reference foot world positions (de-drifted) per frame
    footref = {b: np.zeros((T, 3)) for b in ("calcn_l", "calcn_r")}
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        for b in ("calcn_l", "calcn_r"): footref[b][t] = d.xpos[FB[b]]
    stance_thr = min(footref["calcn_l"][:, 2].min(), footref["calcn_r"][:, 2].min()) + 0.06
    stance = {b: footref[b][:, 2] < stance_thr for b in ("calcn_l", "calcn_r")}   # planted when foot is low

    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 / freq, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) * (freq / 2)

    # leg vs trunk muscle DOFs (foot task drives legs; joint-PD drives trunk)
    legkw = ("hip", "knee", "ankle", "subtalar", "mtp", "walker", "femur", "tibia")
    legdof, trunkdof = [], []
    for j in range(6, nv):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[j]) or "").lower()
        (legdof if any(k in nm for k in legkw) else trunkdof).append(j)
    legdof, trunkdof = np.array(legdof), np.array(trunkdof)

    f0 = int(t0 * freq); d.qpos[:] = qpos[f0]; d.qvel[:] = qvel[f0]; mujoco.mj_forward(m, d)
    lo1, hi1 = cap_envelope(m, d, 1.0)
    musc = np.array([j for j in range(6, nv) if (hi1[j] - lo1[j]) > CAP_MIN])
    print(f"legDOF={len(legdof)} trunkDOF={len(trunkdof)} musc={len(musc)} floor[{floor.min():.3f},{floor.max():.3f}]", flush=True)

    nsteps = int((t1 - t0) / dt); achieved = []; qe = np.zeros(nv); Jp = np.zeros((3, nv)); Jr = np.zeros((3, nv))
    anchor = {"calcn_l": None, "calcn_r": None}
    for s in range(nsteps):
        tt = t0 + s * dt; rf = min(T - 2, max(1, int(round(tt * freq))))
        mujoco.mj_differentiatePos(m, qe, 1.0, d.qpos, qpos[rf])
        qdd = np.zeros(nv); qdd[trunkdof] = qacc[rf][trunkdof] + KP * qe[trunkdof] + KD * (qvel[rf][trunkdof] - d.qvel[trunkdof])
        d.qacc[:] = qdd; mujoco.mj_inverse(m, d); tau = d.qfrc_inverse.copy()       # bias + trunk-PD inertial torque
        # leg foot tasks: STANCE -> hold the planted anchor (no slide); SWING -> track reference foot
        for b in ("calcn_l", "calcn_r"):
            mujoco.mj_jacBody(m, d, Jp, Jr, FB[b])
            fpos = d.xpos[FB[b]]; fvel = Jp @ d.qvel
            if stance[b][rf]:
                if anchor[b] is None: anchor[b] = fpos.copy()                       # plant: lock current position
                tgt = anchor[b]
            else:
                anchor[b] = None; tgt = footref[b][rf]
            F = KPF * (tgt - fpos) - KDF * fvel
            tau[legdof] += (Jp.T @ F)[legdof]
        lo, hi = cap_envelope(m, d, FMAX)
        tau[musc] = np.clip(tau[musc], lo[musc], hi[musc])
        tau[0:6] = 0.0; tau[0:2] = -KD_ROOT * d.qvel[0:2]                           # root: free x,y w/ damping
        d.qfrc_applied[:] = tau; d.ctrl[:] = 0.0; d.act[:] = 0.0
        mujoco.mj_step(m, d)
        d.qpos[2:7] = qpos[rf][2:7]; d.qvel[2:6] = qvel[rf][2:6]                    # prescribe z + orientation
        if s % int(round(1 / freq / dt)) == 0: achieved.append(d.qpos.copy())
    achieved = np.array(achieved); np.save(OUT, achieved)

    P = np.zeros((len(achieved), 4, 3))
    for i, qp in enumerate(achieved):
        d.qpos[:] = qp; mujoco.mj_forward(m, d)
        for k, b in enumerate(("toes_l", "toes_r", "calcn_l", "calcn_r")): P[i, k] = d.xpos[FB[b]]
    zmin = P[:, :, 2].min(); sp = []
    for i in range(1, len(P)):
        bi = np.argmin(P[i, :, 2])
        if P[i, bi, 2] < zmin + 0.05: sp.append(np.linalg.norm(P[i, bi, :2] - P[i - 1, bi, :2]) * freq * 100)
    sp = np.array(sp or [0.0]); xy = achieved[:, :2]
    print(f"FMAX={FMAX} saved {achieved.shape} -> {OUT}", flush=True)
    print(f"OUTPUT 站立相脚滑 median {np.median(sp):.1f} max {sp.max():.1f} cm/s (ref ~38) | 盆骨 x,y 跨度 {np.ptp(xy[:,0]):.2f}x{np.ptp(xy[:,1]):.2f}m 行程 {np.linalg.norm(np.diff(xy,axis=0),axis=1).sum():.2f}m", flush=True)


if __name__ == "__main__":
    main()
