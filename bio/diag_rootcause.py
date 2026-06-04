#!/usr/bin/env python3
"""Decisive controlled test: WHY do required torques exceed muscle capacity?

2x2: smoothing {sigma2.5, sigma10} x capacity {at-motion-velocity, isometric qvel=0}.
- If heavy smoothing fixes feasibility -> torque side (Pop too fast / retarget noise).
- If isometric capacity fixes it    -> capacity side (muscle FV crushed by motion speed).
Reports mean #infeasible DOFs (of 68) over sampled frames under each condition.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody

c = np.load(os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz"), allow_pickle=True)
qpos_raw = np.asarray(c["qpos"], float); dt = 1 / float(c["frequency"]); T = qpos_raw.shape[0]
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
nv, nu = m.nv, m.nu


def smooth(x, sigma):
    r = int(max(6, sigma * 3)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    y = np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
    q = y[:, 3:7]; y[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True); return y


def jn(dd):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[dd]) or ""


musc_dof = np.array([dd for dd in range(6, nv)
                     if not (jn(dd).startswith("knee_angle_") and jn(dd) not in ("knee_angle_l", "knee_angle_r"))])
d = mujoco.MjData(m)
frames = list(range(50, T - 50, 20))   # ~55 sampled frames


def derivs(qpos):
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 * dt, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)
    return qvel, qacc


def feasibility(sigma, isometric):
    qpos = smooth(qpos_raw, sigma); qvel, qacc = derivs(qpos)
    infeas_tot = 0.0; tau_tot = 0.0
    for t in frames:
        d.qpos[:] = qpos[t]; d.qvel[:] = qvel[t]; d.qacc[:] = qacc[t]; mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse[musc_dof].copy()
        d.qvel[:] = 0.0 if isometric else qvel[t]
        d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(nu, nv)
        d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
        A = R[:, musc_dof].T * (full - passive)[None, :]
        b = tau - R[:, musc_dof].T @ passive
        lo = np.minimum(A, 0).sum(1); hi = np.maximum(A, 0).sum(1)
        infeas_tot += ((b < lo - 1) | (b > hi + 1)).sum()
        tau_tot += np.linalg.norm(tau)
    return infeas_tot / len(frames), tau_tot / len(frames)


print(f"sampled frames: {len(frames)}  matched DOFs: {len(musc_dof)}")
print("condition                         mean#infeasible/68   mean||tau||")
for sigma in (2.5, 10.0):
    for iso in (False, True):
        nif, tn = feasibility(sigma, iso)
        tag = f"sigma={sigma:<4} {'isometric' if iso else 'at-velocity'}"
        print(f"  {tag:32s} {nif:6.1f}            {tn:7.1f}")
