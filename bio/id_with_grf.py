#!/usr/bin/env python3
"""ID with estimated ground reaction force (GRF) — clears the floating-base hurdle.

Naive inverse dynamics of a floating-base humanoid is dynamically inconsistent without
ground support: feet-floating -> torques ~bodyweight*levers (huge); contact-ID -> spring
blowup. Standard biomechanics fix when no force plate: estimate GRF from whole-body COM
dynamics (F_grf = M*(a_com - g)) and apply it on the contacting feet, THEN do ID.

This script validates that GRF-ID drops the required joint torques to physiological ranges
(hundreds of N·m) so muscle extraction (Stage 2) becomes feasible.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
G = np.array([0.0, 0.0, -9.81])


def get_model(env):
    for a in ("_model", "model", "_mjmodel", "mjmodel"):
        mm = getattr(env, a, None)
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
    T = qpos.shape[0]
    env = MyoFullBody(disable_fingers=True)
    m = get_model(env)
    nv = m.nv
    qpos = gaussian_smooth(qpos)
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    # identify foot bodies (the segment that touches ground)
    foot_keys = ("calcn", "talus", "foot", "toes", "bofoot")
    foot_bodies = []
    for i in range(m.nbody):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
        if any(k in nm for k in foot_keys):
            foot_bodies.append(i)
    # keep left/right ankle-ish: pick the two lowest-in-tree distinct sides by name
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in foot_bodies]
    print("foot bodies:", list(zip(foot_bodies, names)))
    M_total = float(m.body_mass.sum())
    print(f"total mass = {M_total:.1f} kg  -> bodyweight = {M_total*9.81:.0f} N")

    d = mujoco.MjData(m)

    # COM trajectory (mass-weighted body COMs)
    com = np.zeros((T, 3))
    footz = np.zeros((T, len(foot_bodies)))
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        com[t] = (m.body_mass[:, None] * d.xipos).sum(0) / M_total
        footz[t] = [d.xipos[b, 2] for b in foot_bodies]
    a_com = np.zeros((T, 3)); a_com[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
    grf = M_total * (a_com - G[None, :])    # (T,3) total ground reaction force

    # qvel/qacc
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 * dt, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)

    # disable contact: GRF is supplied analytically (subtracted via Jacobian), no double count
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # mj_inverse IGNORES xfrc_applied -> instead subtract GRF's generalized force J^T F manually.
    qfrc = np.zeros((T, nv))
    jacp = np.zeros((3, nv))
    for t in range(1, T - 1):
        d.qpos[:] = qpos[t]; d.qvel[:] = qvel[t]; d.qacc[:] = qacc[t]
        mujoco.mj_inverse(m, d)                  # net required force (body treated unsupported)
        qf = d.qfrc_inverse.copy()
        # distribute GRF across feet by how planted each is (lower z -> more load)
        z = footz[t]; zmin = z.min()
        w = np.maximum(0.0, 0.08 - (z - zmin))   # within 8cm of lowest foot shares load
        w = w / (w.sum() + 1e-9)
        qgrf = np.zeros(nv)
        for k, b in enumerate(foot_bodies):
            F = w[k] * grf[t]
            p = d.xipos[b].copy(); p[2] = 0.0    # contact point ~ under foot at floor (COP est)
            mujoco.mj_jac(m, d, jacp, None, p, b)
            qgrf += jacp.T @ F                   # generalized force produced by this foot's GRF
        qfrc[t] = qf - qgrf                      # remove GRF -> what muscles/residual must supply

    sl = slice(2, T - 2)
    rootn = np.linalg.norm(qfrc[sl, :6], axis=1)
    rootF = np.linalg.norm(qfrc[sl, :3], axis=1)     # residual FORCE (N)
    rootM = np.linalg.norm(qfrc[sl, 3:6], axis=1)    # residual MOMENT (N·m, COP error)
    jointn = np.linalg.norm(qfrc[sl, 6:], axis=1)
    print("=== ID-with-GRF results ===")
    print(f"  GRF |F| mean={np.linalg.norm(grf[sl],axis=1).mean():.0f} N (应≈体重{M_total*9.81:.0f})")
    print(f"  root residual FORCE mean={rootF.mean():.1f} N (GRF力生效则↓)")
    print(f"  root residual MOMENT mean={rootM.mean():.1f} N·m (COP没解准则高)")
    print(f"  root residual mean={rootn.mean():.1f} max={rootn.max():.1f}  (应大幅下降)")
    print(f"  joint torque |tau| mean={jointn.mean():.1f} max={jointn.max():.1f}  (生理量级=几百)")
    print(f"  per-DOF |tau| mean top6: {np.round(np.sort(np.abs(qfrc[sl,6:]).mean(0))[::-1][:6],1)}")


if __name__ == "__main__":
    main()
