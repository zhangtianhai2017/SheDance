#!/usr/bin/env python3
"""B, done properly: MUSCLE-SPACE forward capacity-limited tracking (CMC) in MuJoCo.

v1 clamped each DOF's torque independently -> inconsistent (muscles are shared) -> oscillation.
v2 solves ONE activation vector a in [0,1] (SO-style QP, no reserves) that best produces the
PD-desired joint torque across ALL muscle DOFs at once, then applies the resulting CONSISTENT
muscle forces. Where muscles can't reach the demand, the achieved torque is the best feasible
-> clean deviation (the embodied imitator), no limit-cycle wobble. F_max = the character.

Args: [t0 t1 fmax]. Pelvis prescribed; muscles drive the joints. Forward-integrated.
"""
import sys, os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco
from scipy.optimize import lsq_linear
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
KP, KD = 200.0, 60.0
CTRL_HZ = 100.0
OUT = os.path.expanduser(f"~/shedance/osim/cmcq_{t0:.1f}_{t1:.1f}_f{FMAX:.2f}.npy")


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
    nv, nu = m.nv, m.nu; dt = m.opt.timestep
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    T = qpos.shape[0]
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 / freq, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) * (freq / 2)

    f0 = int(t0 * freq)
    d.qpos[:] = qpos[f0]; d.qvel[:] = qvel[f0]; mujoco.mj_forward(m, d)
    musc = np.arange(6, nv)             # joints (muscle-controlled); pelvis(0:6) prescribed
    K = max(1, int(round((1.0 / CTRL_HZ) / dt)))
    nsteps = int((t1 - t0) / dt)
    achieved = []
    qe = np.zeros(nv)
    a = np.zeros(nu)

    for s in range(nsteps):
        tt = t0 + s * dt
        rf = min(T - 2, max(1, int(round(tt * freq))))
        if s % K == 0:                  # solve activations at control rate
            mujoco.mj_differentiatePos(m, qe, 1.0, d.qpos, qpos[rf])
            qdd = qacc[rf] + KP * qe + KD * (qvel[rf] - d.qvel)
            qdd[0:6] = qacc[rf][0:6]
            d.qacc[:] = qdd; mujoco.mj_inverse(m, d)
            tau = d.qfrc_inverse.copy()
            d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(nu, nv)
            d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
            cap = (full - passive) * FMAX
            A = R[:, musc].T * cap[None, :]
            b = tau[musc] - R[:, musc].T @ passive
            a = lsq_linear(A, b, bounds=(0.0, 1.0), max_iter=40, tol=1e-2).x
        d.act[:] = a; d.ctrl[:] = a       # consistent muscle forces
        d.qfrc_applied[:] = 0.0
        mujoco.mj_step(m, d)
        d.qpos[0:7] = qpos[rf][0:7]; d.qvel[0:6] = qvel[rf][0:6]   # prescribe pelvis
        if s % int(round(1 / freq / dt)) == 0:
            achieved.append(d.qpos.copy())

    achieved = np.array(achieved)
    np.save(OUT, achieved)
    ref = qpos[f0:f0 + achieved.shape[0]]; n = min(len(ref), len(achieved))
    err = np.zeros(nv)
    for i in range(n):
        mujoco.mj_differentiatePos(m, qe, 1.0, achieved[i], ref[i]); err += np.abs(qe)
    err /= n
    print(f"FMAX={FMAX} muscDOF-MAE={err[6:].mean():.4f} rad meanAct={a.mean():.3f} satFrac={(a>0.95).mean():.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
