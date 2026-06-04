#!/usr/bin/env python3
"""Forward capacity-limited tracking (CMC-style) in MuJoCo — the 'embodied imitator'.

Pelvis + coupler DOFs (knee/shoulder rhythm) are PRESCRIBED from the reference (muscles don't
drive them). The PRIMARY joints (hip/knee/ankle/lumbar/shoulder/elbow) are muscle-driven:
a PD controller toward the reference produces a desired joint torque, CLAMPED to the muscle
capacity envelope (scaled by FMAX = the character's strength). Where impossible (jitter /
beyond this body), torque saturates -> the body deviates -> output = what THIS body can dance.

FMAX: 1.0 normal, >1 strong, <1 delicate. Args: [t0 t1 fmax]. Forward-integrated (scales with time).
"""
import sys, os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
KP, KD = 200.0, 60.0   # overdamped (zeta~2) to suppress clamp/PD limit-cycle overshoot
CAP_MIN = 15.0   # DOFs with muscle torque range > this are muscle-controlled; rest prescribed
OUT = os.path.expanduser(f"~/shedance/osim/cmcq_{t0:.1f}_{t1:.1f}_f{FMAX:.2f}.npy")


def gaussian_smooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def cap_envelope(m, d, fmax):
    d.act[:] = 0.0; mujoco.mj_forward(m, d); passive = d.actuator_force.copy(); R = np.array(d.actuator_moment).reshape(m.nu, m.nv)
    d.act[:] = 1.0; mujoco.mj_forward(m, d); full = d.actuator_force.copy()
    cap = (full - passive) * fmax
    base = R.T @ passive
    lo = base + np.minimum(R.T * cap[None, :], 0).sum(1)
    hi = base + np.maximum(R.T * cap[None, :], 0).sum(1)
    return lo, hi


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gaussian_smooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
    q = qpos[:, 3:7]; qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    env = MyoFullBody(disable_fingers=True)
    m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
    nv = m.nv; dt = m.opt.timestep
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # isolate controller from foot-ground forces
    d = mujoco.MjData(m)
    T = qpos.shape[0]
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv); mujoco.mj_differentiatePos(m, v, 2 / freq, qpos[t - 1], qpos[t + 1]); qvel[t] = v
    qacc = np.zeros((T, nv)); qacc[1:-1] = (qvel[2:] - qvel[:-2]) * (freq / 2)

    f0 = int(t0 * freq)
    d.qpos[:] = qpos[f0]; d.qvel[:] = qvel[f0]; mujoco.mj_forward(m, d)
    # which DOFs to muscle-control: those with meaningful muscle torque range (FMAX=1 structural)
    lo1, hi1 = cap_envelope(m, d, 1.0)
    musc = np.array([j for j in range(6, nv) if (hi1[j] - lo1[j]) > CAP_MIN])
    presc = np.array([j for j in range(nv) if j not in set(musc.tolist())])  # pelvis + couplers
    print(f"muscle-controlled DOFs={len(musc)} prescribed={len(presc)} dt={dt}", flush=True)

    nsteps = int((t1 - t0) / dt)
    achieved = []
    qe = np.zeros(nv)
    for s in range(nsteps):
        tt = t0 + s * dt
        rf = min(T - 2, max(1, int(round(tt * freq))))
        mujoco.mj_differentiatePos(m, qe, 1.0, d.qpos, qpos[rf])   # ref - current (config diff)
        qdd = qacc[rf] + KP * qe + KD * (qvel[rf] - d.qvel)        # PD toward ref
        qdd[0:6] = qacc[rf][0:6]                                   # pelvis: ref accel only (prescribed, no feedback)
        d.qacc[:] = qdd
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse.copy()
        lo, hi = cap_envelope(m, d, FMAX)
        tau_apply = tau.copy()
        tau_apply[musc] = np.clip(tau[musc], lo[musc], hi[musc])   # clamp ONLY muscle DOFs to capacity
        tau_apply[0:6] = 0.0                                       # pelvis prescribed, not torque-driven
        d.qfrc_applied[:] = tau_apply
        d.ctrl[:] = 0.0; d.act[:] = 0.0                            # force only from qfrc_applied (cap_envelope left act=1)
        mujoco.mj_step(m, d)
        d.qpos[0:7] = qpos[rf][0:7]; d.qvel[0:6] = qvel[rf][0:6]   # prescribe pelvis (stable; not muscle-actuated)
        if s % int(round(1 / freq / dt)) == 0:
            achieved.append(d.qpos.copy())

    achieved = np.array(achieved)
    np.save(OUT, achieved)
    ref = qpos[f0:f0 + achieved.shape[0]]
    n = min(len(ref), len(achieved))
    err = np.zeros(nv)
    for i in range(n):
        mujoco.mj_differentiatePos(m, qe, 1.0, achieved[i], ref[i]); err += np.abs(qe)
    err /= n
    print(f"FMAX={FMAX} steps={nsteps} saved {achieved.shape} muscDOF-MAE={err[musc].mean():.4f} rad -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
