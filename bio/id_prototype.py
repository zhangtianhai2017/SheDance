#!/usr/bin/env python3
"""Stage 1 of the biomechanics-inverse route: inverse dynamics on the retargeted Pop.

qpos trajectory -> qvel,qacc (finite diff, quaternion-aware) -> mj_inverse -> required
generalized forces. Reports the actuated-joint torques vs the root (free-joint) residual
(the residual tells us how dynamically self-consistent the reference is -> RRA need).
Pure CPU, no training. Also peeks at the moment-arm matrix + muscle force scale for Stage 2.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")


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


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = np.asarray(c["qpos"], dtype=np.float64)      # (T, nq)
    freq = float(c["frequency"])
    dt = 1.0 / freq
    T, nq = qpos.shape

    env = MyoFullBody(disable_fingers=True)
    m = get_model(env)
    d = mujoco.MjData(m)
    nv = m.nv
    print(f"T={T} nq={nq} nv={nv} dt={dt:.4f}s  nu(muscles)={m.nu}")
    assert nq == m.nq, (nq, m.nq)

    # --- smooth qpos before differentiating (raw finite-diff amplifies noise) ---
    def gaussian_smooth(x, sigma=2.5, radius=6):
        k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        k /= k.sum()
        xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
        return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)

    qpos = gaussian_smooth(qpos)
    # renormalize the free-joint quaternion (qpos[3:7]) after smoothing
    q = qpos[:, 3:7]
    qpos[:, 3:7] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    # --- ground the feet: retarget left them ~2cm above floor (body floats -> bad ID).
    # drop whole trajectory's root z so the lowest foot-geom point plants (small penetration).
    foot_ids = [i for i in range(m.ngeom)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower().__contains__("foot")]
    floor_z = 0.0
    min_foot = 9.9
    for t in range(0, T, 20):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        for gi in foot_ids:
            min_foot = min(min_foot, d.geom_xpos[gi, 2])
    drop = min_foot - floor_z + 0.005   # 5mm penetration to engage contact
    qpos[:, 2] -= drop
    print(f"grounding: lowest foot was {min_foot:.3f}m above floor -> dropped root z by {drop:.3f}m")

    # qvel via mj_differentiatePos (handles the free-joint quaternion correctly)
    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        v = np.zeros(nv)
        mujoco.mj_differentiatePos(m, v, 2 * dt, qpos[t - 1], qpos[t + 1])
        qvel[t] = v
    # qacc via central diff of qvel
    qacc = np.zeros((T, nv))
    qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)

    # inverse dynamics per frame
    qfrc = np.zeros((T, nv))
    ncon = np.zeros(T)
    for t in range(1, T - 1):
        d.qpos[:] = qpos[t]
        d.qvel[:] = qvel[t]
        d.qacc[:] = qacc[t]
        mujoco.mj_inverse(m, d)
        qfrc[t] = d.qfrc_inverse
        ncon[t] = d.ncon
    print(f"contacts during ID: mean ncon={ncon[2:T-2].mean():.1f} (脚踩实=多于1)")

    sl = slice(2, T - 2)  # drop noisy ends
    root = qfrc[sl, :6]          # free-joint residual (not muscle-actuatable)
    joints = qfrc[sl, 6:]        # actuated DOFs
    rootn = np.linalg.norm(root, axis=1)
    jointn = np.linalg.norm(joints, axis=1)
    print("=== inverse dynamics results ===")
    print(f"  root residual |force| (N, N·m mixed): mean={rootn.mean():.1f} max={rootn.max():.1f}")
    print(f"  actuated joint torque |tau|:          mean={jointn.mean():.1f} max={jointn.max():.1f}")
    print(f"  per-joint-DOF |tau| mean (top 6): "
          f"{np.round(np.sort(np.abs(joints).mean(0))[::-1][:6],1)}")
    print("  -> 大 root 残差 = 参考动力学不自洽(需 RRA);小 = 基本自洽")

    # Stage-2 peek: moment-arm matrix + muscle force scale
    d2 = mujoco.MjData(m)
    d2.qpos[:] = qpos[T // 2]
    mujoco.mj_forward(m, d2)
    M = np.array(d2.actuator_moment).reshape(m.nu, nv) if np.array(d2.actuator_moment).size == m.nu * nv else None
    print("=== stage-2 pieces ===")
    print(f"  actuator_moment (R) shape ok: {None if M is None else M.shape}")
    fmax = np.array([m.actuator_gainprm[i, 2] for i in range(m.nu)])
    print(f"  muscle peak force F0: mean={fmax.mean():.0f} max={fmax.max():.0f} N (nu={m.nu})")
    print("  -> Stage2: 每帧解 min Σa² s.t. R[:, 6:].T @ (a*F0*FLV + Fpassive) = qfrc[6:], a∈[0,1]")


if __name__ == "__main__":
    main()
