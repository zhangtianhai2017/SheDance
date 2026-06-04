#!/usr/bin/env python3
"""Per-DOF feasibility: which matched DOFs the muscles physically cannot produce in a in [0,1]."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody

c = np.load(os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz"), allow_pickle=True)
qpos = np.asarray(c["qpos"], float); dt = 1 / float(c["frequency"]); T = qpos.shape[0]
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
nv, nu = m.nv, m.nu
k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); k /= k.sum()
xp = np.pad(qpos, ((6, 6), (0, 0)), mode="edge")
qpos = np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(qpos.shape[1])], 1)
q = qpos[:, 3:7]; qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
d = mujoco.MjData(m)
qvel = np.zeros((T, nv))
for t in range(1, T - 1):
    v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 * dt, qpos[t - 1], qpos[t + 1]); qvel[t] = v
qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)

def jn(dd):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[dd]) or ""

musc_dof = np.array([dd for dd in range(6, nv)
                     if not (jn(dd).startswith("knee_angle_") and jn(dd) not in ("knee_angle_l", "knee_angle_r"))])
t = T // 2
d.qpos[:] = qpos[t]; d.qvel[:] = qvel[t]; d.qacc[:] = qacc[t]; mujoco.mj_inverse(m, d)
tau = d.qfrc_inverse[musc_dof].copy()
d.qvel[:] = qvel[t]
d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(nu, nv)
d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
ac = full - passive
Ra = R[:, musc_dof].T; A = Ra * ac[None, :]; b = tau - Ra @ passive
lo = np.minimum(A, 0).sum(1); hi = np.maximum(A, 0).sum(1)   # achievable torque range over a in [0,1]
nspan = (np.abs(A) > 0.5).sum(1)
infeas = (b < lo - 1) | (b > hi + 1)
print(f"matched DOFs: {len(musc_dof)}  infeasible (b outside achievable): {int(infeas.sum())}")
print("DOF                          b(Nm)   achiev[lo,hi]        #musc  feasible")
for dd in np.argsort(np.abs(b))[::-1][:20]:
    flag = "NO" if infeas[dd] else "yes"
    print(f"  {jn(musc_dof[dd]):26s} {b[dd]:7.1f}  [{lo[dd]:8.1f},{hi[dd]:8.1f}] {nspan[dd]:4d}   {flag}")
