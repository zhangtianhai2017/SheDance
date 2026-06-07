#!/usr/bin/env python3
"""Forward tracking WITH foot-ground contact ON, so planted feet don't slide (physics, not kinematic
patch). Simple tier per user: prescribe pelvis VERTICAL (z, de-drifted so feet reach the floor) +
ORIENTATION (keep upright); FREE pelvis HORIZONTAL (x,y) -> contact friction anchors the planted foot
while the muscle-tracked legs move the body over it. Primary joints muscle-driven (PD -> inverse ->
clamp to FMAX capacity), same as cmc_mujoco.

Args: [t0 t1 fmax]. Out: ~/shedance/osim/fc_<t0>_<t1>_f<fmax>.npy  (achieved qpos @ ref fps)
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
KP, KD, CAP_MIN = 200.0, 60.0, 15.0
OUT = os.path.expanduser(f"~/shedance/osim/fc_{t0:.1f}_{t1:.1f}_f{FMAX:.2f}.npy")


def gsmooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def cap_envelope(m, d, fmax):
    d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(m.nu, m.nv)
    d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
    cap = (full - passive) * fmax; base = R.T @ passive
    lo = base + np.minimum(R.T * cap[None, :], 0).sum(1); hi = base + np.maximum(R.T * cap[None, :], 0).sum(1)
    return lo, hi


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gsmooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    env = MyoFullBody(disable_fingers=True)
    m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
    # NOTE: contact stays ENABLED (do NOT disable). Floor plane geom is at Z=0 in the model.
    for i in range(m.ngeom):                              # crank foot<->floor friction so planted foot grips
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        if "floor" in nm or any(k in nm for k in ("foot", "calcn", "toes", "phalanx", "bofoot")):
            m.geom_friction[i][0] = 5.0
    nv = m.nv; dt = m.opt.timestep; d = mujoco.MjData(m)
    toes = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("toes_l", "toes_r", "calcn_l", "calcn_r")]

    # de-drift pelvis z: feet ride the floor (so contact engages when z is prescribed)
    T = qpos.shape[0]; soleZ = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        soleZ[t] = min(d.xpos[b][2] for b in toes)
    floor = gaussian_filter1d(minimum_filter1d(soleZ, size=15, mode="nearest"), sigma=4, mode="nearest")
    qpos[:, 2] = qpos[:, 2] - floor - 0.012   # lower the body so feet press into Z=0 (solid contact -> friction)

    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 / freq, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) * (freq / 2)

    f0 = int(t0 * freq)
    d.qpos[:] = qpos[f0]; d.qvel[:] = qvel[f0]; mujoco.mj_forward(m, d)
    lo1, hi1 = cap_envelope(m, d, 1.0)
    musc = np.array([j for j in range(6, nv) if (hi1[j] - lo1[j]) > CAP_MIN])
    print(f"muscle DOFs={len(musc)} dt={dt} floor[min,max]=[{floor.min():.3f},{floor.max():.3f}]", flush=True)

    nsteps = int((t1 - t0) / dt); achieved = []; qe = np.zeros(nv)
    for s in range(nsteps):
        tt = t0 + s * dt; rf = min(T - 2, max(1, int(round(tt * freq))))
        mujoco.mj_differentiatePos(m, qe, 1.0, d.qpos, qpos[rf])
        qdd = qacc[rf] + KP * qe + KD * (qvel[rf] - d.qvel)
        qdd[2:6] = qacc[rf][2:6]            # prescribe pelvis z + orientation accel; x,y (0,1) left to physics
        d.qacc[:] = qdd; mujoco.mj_inverse(m, d); tau = d.qfrc_inverse.copy()
        lo, hi = cap_envelope(m, d, FMAX); tau_apply = tau.copy()
        tau_apply[musc] = np.clip(tau[musc], lo[musc], hi[musc]); tau_apply[0:6] = 0.0
        d.qfrc_applied[:] = tau_apply; d.ctrl[:] = 0.0; d.act[:] = 0.0
        mujoco.mj_step(m, d)
        d.qpos[2:7] = qpos[rf][2:7]; d.qvel[2:6] = qvel[rf][2:6]    # re-prescribe z+orientation; KEEP physics x,y
        if s % int(round(1 / freq / dt)) == 0:
            achieved.append(d.qpos.copy())
    achieved = np.array(achieved); np.save(OUT, achieved)

    # quick verdict: foot slide + body x,y travel
    P = np.zeros((len(achieved), len(toes), 3))
    for i, qp in enumerate(achieved):
        d.qpos[:] = qp; mujoco.mj_forward(m, d)
        for k, b in enumerate(toes): P[i, k] = d.xpos[b]
    zmin = P[:, :, 2].min(); sp = []
    for i in range(1, len(P)):
        bi = np.argmin(P[i, :, 2])
        if P[i, bi, 2] < zmin + 0.05: sp.append(np.linalg.norm(P[i, bi, :2] - P[i - 1, bi, :2]) * freq * 100)
    sp = np.array(sp or [0.0])
    xy = achieved[:, :2]
    print(f"FMAX={FMAX} saved {achieved.shape} -> {OUT}", flush=True)
    print(f"OUTPUT 站立相脚滑 median {np.median(sp):.1f} max {sp.max():.1f} cm/s (ref qpos was ~38) | 盆骨 x,y 行程 {np.linalg.norm(np.diff(xy,axis=0),axis=1).sum():.2f}m 跨度 {np.ptp(xy[:,0]):.2f}x{np.ptp(xy[:,1]):.2f}m", flush=True)


if __name__ == "__main__":
    main()
