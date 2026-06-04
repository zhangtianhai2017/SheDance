#!/usr/bin/env python3
"""Stage 2: per-frame static optimization (muscle redundancy) with reserve actuators.

For each frame: inverse dynamics -> required joint generalized force; then solve for muscle
activations a in [0,1] that best produce it (min ||a||^2), with reserve actuators absorbing
what muscles can't (no GRF available -> root/leg torques partly covered by reserves).

Per-frame problem (actuated DOFs only):
    minimize  w_res * || A a - b ||^2  +  || a ||^2     over a in [0,1]
  where A = R[:,act].T @ diag(active_capacity)   (n_dof x nu)
        b = tau_req - R[:,act].T @ passive_force  (torque muscles' ACTIVE part must make)
  reserves = b - A a  (penalized via w_res). Solved as box-constrained least squares.

Muscle force model is queried from MuJoCo directly (ctrl=0 -> passive, ctrl=1 -> max),
so we don't reimplement Hill dynamics. Outputs activation trajectory + diagnostics.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
W_RES = 100.0   # reserve penalty weight (higher -> prefer muscles over reserves)
OUT = os.path.expanduser("~/shedance/bio_pop_activations.npz")


def get_model(env):
    for a in ("_model", "model", "_mjmodel", "mjmodel"):
        mm = getattr(env, a, None)
        if isinstance(mm, mujoco.MjModel):
            return mm
    for a in dir(env):
        try:
            mm = getattr(env, a)
        except Exception:
            continue
        if isinstance(mm, mujoco.MjModel):
            return mm
    raise RuntimeError("no MjModel")


def gaussian_smooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = np.asarray(c["qpos"], dtype=np.float64)
    dt = 1.0 / float(c["frequency"])
    T, nq = qpos.shape
    env = MyoFullBody(disable_fingers=True)
    m = get_model(env)
    nv, nu = m.nv, m.nu
    qpos = gaussian_smooth(qpos)
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    # qvel/qacc (quat-aware)
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 * dt, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)

    d = mujoco.MjData(m)

    # --- GRF estimate (clears the floating-base leg-torque inflation) ---
    G = np.array([0.0, 0.0, -9.81]); M_total = float(m.body_mass.sum())
    foot_bodies = [i for i in range(m.nbody)
                   if any(kk in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
                          for kk in ("calcn", "talus", "toes"))]
    com = np.zeros((T, 3)); footz = np.zeros((T, len(foot_bodies)))
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        com[t] = (m.body_mass[:, None] * d.xipos).sum(0) / M_total
        footz[t] = [d.xipos[bi, 2] for bi in foot_bodies]
    a_com = np.zeros((T, 3)); a_com[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
    grf = M_total * (a_com - G[None, :])
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # GRF supplied analytically
    jacp = np.zeros((3, nv))
    # only match torques on PRIMARY actuated DOFs. Exclude free-joint (root) AND the knee
    # coupler DOFs (knee_angle_translation*/beta*/rotation2,3) — these carry huge CONSTRAINT
    # forces (the rolling-sliding knee mechanism), NOT muscle torques, and the joint's own
    # coupling handles them in playback. Keep only the primary knee flexion knee_angle_l/r.
    def jname(dd):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[dd]) or ""
    musc_dof = []
    for dd in range(6, nv):
        nm = jname(dd)
        if nm.startswith("knee_angle_") and nm not in ("knee_angle_l", "knee_angle_r"):
            continue   # knee coupler DOF
        musc_dof.append(dd)
    musc_dof = np.array(musc_dof)
    n_dof = len(musc_dof)
    print(f"matched DOFs: {n_dof} / {nv} (排除根 + 膝耦合约束DOF)")
    acts = np.zeros((T, nu))
    reserve_norm = np.zeros(T)
    musc_tau_frac = np.zeros(T)          # fraction of required torque met by muscles

    frames = range(2, T - 2)
    for t in frames:
        # inverse dynamics for required generalized force (feet floating: root residual accepted)
        d.qpos[:] = qpos[t]; d.qvel[:] = qvel[t]; d.qacc[:] = qacc[t]
        mujoco.mj_inverse(m, d)
        # subtract GRF generalized force (J^T F) -> physiological leg torques
        z = footz[t]; zmin = z.min()
        w = np.maximum(0.0, 0.08 - (z - zmin)); w = w / (w.sum() + 1e-9)
        qgrf = np.zeros(nv)
        for kk, bi in enumerate(foot_bodies):
            F = w[kk] * grf[t]
            p = d.xipos[bi].copy(); p[2] = 0.0
            mujoco.mj_jac(m, d, jacp, None, p, bi)
            qgrf += jacp.T @ F
        tau = (d.qfrc_inverse - qgrf)[musc_dof].copy()

        # muscle force model at this state: passive (ctrl=0) and max (ctrl=1)
        d.qvel[:] = qvel[t]
        d.act[:] = 0.0; mujoco.mj_forward(m, d)
        passive = d.actuator_force.copy()
        R = np.array(d.actuator_moment).reshape(nu, nv)
        d.act[:] = 1.0; mujoco.mj_forward(m, d)
        full = d.actuator_force.copy()
        active_cap = full - passive          # per-muscle active force capacity at this state

        Ra = R[:, musc_dof].T                 # (n_dof x nu)
        A = Ra * active_cap[None, :]          # contribution of activation a to joint torque
        b = tau - Ra @ passive               # active part muscles must produce

        # box-constrained least squares: min ||[sqrt(w)A; I] a - [sqrt(w)b; 0]||^2, a in [0,1]
        sw = np.sqrt(W_RES)
        C = np.vstack([sw * A, np.eye(nu)])
        dvec = np.concatenate([sw * b, np.zeros(nu)])
        sol = lsq_linear(C, dvec, bounds=(0.0, 1.0), max_iter=50, tol=1e-3)
        a = sol.x
        acts[t] = a
        res = b - A @ a
        reserve_norm[t] = np.linalg.norm(res)
        denom = np.linalg.norm(tau) + 1e-9
        musc_tau_frac[t] = 1.0 - np.linalg.norm(res) / denom

    fr = list(frames)
    print("=== static optimization results ===")
    print(f"  frames solved: {len(fr)}  muscles: {nu}")
    print(f"  mean activation: {acts[fr].mean():.3f}  | frac muscles>0.05: {(acts[fr] > 0.05).mean():.2f}")
    print(f"  reserve |residual| mean: {reserve_norm[fr].mean():.1f}  (越小越好/越像肌肉能做)")
    print(f"  muscle-met torque fraction mean: {np.clip(musc_tau_frac[fr],-1,1).mean():.2f} (1=肌肉全包)")
    np.savez(OUT, activations=acts, qpos=qpos, frequency=float(c["frequency"]),
             reserve_norm=reserve_norm, root_drop=0.0)
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
