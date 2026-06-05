#!/usr/bin/env python3
"""MuJoCo hand Static Optimization: solve the 62 finger-muscle activations for a finger motion on the
129-dof MyoFullBody. Per frame:
  - inverse dynamics (mj_inverse) -> generalized force each finger dof needs (gravity+inertia+passive),
  - muscle force is LINEAR in activation at a fixed pose: force = gain*act + bias (gain via act=0/1),
  - moment arms from actuator_moment (valid only with a normalized root quaternion!),
  - solve act by regularized bounded least squares on the finger dofs: min ||C act - b||^2 + LAM|act|^2.
TEST mode synthesizes a finger open/close to validate (flexors hot on close, extensors on open).
Args: out_npz   Env: TEST=1 (synthesize) | IN_QPOS=<npz with qpos(N,129)>; LAM=0.05"""
import os, sys, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from scipy.optimize import lsq_linear
from musclemimic.environments.humanoids import MyoFullBody

def getm(df):
    e = MyoFullBody(disable_fingers=df)
    return next(o for o in (getattr(e, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
m = getm(False); m1 = getm(True)
d = mujoco.MjData(m); A, J = mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT
OUT = sys.argv[1]; LAM = float(os.environ.get("LAM", "0.05"))

body_acts = set(mujoco.mj_id2name(m1, A, i) for i in range(m1.nu))
fing = [i for i in range(m.nu) if mujoco.mj_id2name(m, A, i) not in body_acts]      # 62 finger muscles
fnames = [mujoco.mj_id2name(m, A, i) for i in fing]
fjoint = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, J, j) or "").endswith(("_r", "_l"))
          and any(k in (mujoco.mj_id2name(m, J, j) or "") for k in ("mcp", "pip", "dip", "cmc", "ip"))]
fqadr = np.array([int(m.jnt_qposadr[j]) for j in fjoint])
fdof = np.array([int(m.jnt_dofadr[j]) for j in fjoint])
flexq = np.array([int(m.jnt_qposadr[j]) for j in fjoint if "flexion" in (mujoco.mj_id2name(m, J, j) or "")])

freq = 60.0
if os.environ.get("TEST"):
    N = 120; tt = np.arange(N) / freq
    flex = 0.6 * (1 - np.cos(2 * np.pi * tt / 0.5))                  # period 0.5s, amp 1.2 -> fast clench, high qacc
    if m.nkey > 0:
        base = m.key_qpos[0].copy()                 # natural, constraint-satisfying pose
    else:
        base = np.zeros(m.nq); base[2] = 0.95; base[3:7] = [1, 0, 0, 0]
    print(f"base pose: {'keyframe' if m.nkey > 0 else 'default-stand'} (nkey={m.nkey})", flush=True)
    qpos = np.tile(base, (N, 1)); qpos[:, flexq] = flex[:, None]
else:
    c = np.load(os.environ["IN_QPOS"], allow_pickle=True)
    qpos = np.asarray(c["qpos"], float); freq = float(c["frequency"]); N = len(qpos)
    if qpos.shape[1] != m.nq:
        raise SystemExit(f"need 129-dof qpos, got {qpos.shape[1]}")

dt = 1.0 / freq
act = np.zeros((N, len(fing))); qfrc_max = 0.0
for k in range(N):
    d.qpos[:] = qpos[k]; d.qvel[:] = 0; d.qacc[:] = 0
    if 0 < k < N - 1:
        d.qvel[fdof] = (qpos[k + 1, fqadr] - qpos[k - 1, fqadr]) / (2 * dt)
        d.qacc[fdof] = (qpos[k + 1, fqadr] - 2 * qpos[k, fqadr] + qpos[k - 1, fqadr]) / dt ** 2
    mujoco.mj_inverse(m, d); qfrc = d.qfrc_inverse[fdof].copy()
    qfrc_max = max(qfrc_max, float(np.abs(qfrc).max()))
    d.act[:] = 0; mujoco.mj_forward(m, d); f0 = d.actuator_force.copy()
    d.act[:] = 1; mujoco.mj_forward(m, d); f1 = d.actuator_force.copy()
    gain = (f1 - f0)[fing]; bias = f0[fing]
    Mfd = np.asarray(d.actuator_moment).reshape(m.nu, m.nv)[np.ix_(fing, fdof)]   # (62, n_fdof)
    C = (Mfd * gain[:, None]).T                                                   # (n_fdof, 62)
    b = qfrc - (Mfd * bias[:, None]).sum(0)
    Caug = np.vstack([C, np.sqrt(LAM) * np.eye(len(fing))])
    baug = np.concatenate([b, np.zeros(len(fing))])
    act[k] = lsq_linear(Caug, baug, bounds=(0, 1), max_iter=80).x

np.savez(OUT, finger_act=act, finger_names=np.array(fnames), frequency=freq)
print(f"hand-SO: {N} frames, {len(fing)} finger muscles, {len(fdof)} finger dofs -> {OUT}", flush=True)
print(f"max finger torque {qfrc_max:.4f} Nm | act: max {act.max():.3f}, mean {act.mean():.4f}, >0.05 share {(act>0.05).mean()*100:.1f}%", flush=True)

if os.environ.get("TEST"):    # validate: flexors hot while closing, extensors while opening
    vel = np.gradient(qpos[:, flexq].mean(1))
    closing = vel > 1e-4; opening = vel < -1e-4
    isflex = np.array([n.startswith(("FDS", "FDP", "FPL", "FDM")) for n in fnames])
    isext = np.array([n.startswith(("ED", "EI", "EP")) for n in fnames])
    fl_close = act[closing][:, isflex].mean(); fl_open = act[opening][:, isflex].mean()
    ex_close = act[closing][:, isext].mean(); ex_open = act[opening][:, isext].mean()
    print(f"屈肌激活: 合拢时={fl_close:.3f}  张开时={fl_open:.3f}  (合拢应更高)", flush=True)
    print(f"伸肌激活: 合拢时={ex_close:.3f}  张开时={ex_open:.3f}  (张开应更高)", flush=True)
    print(f"判定: {'PASS 屈伸分工正确' if fl_close > fl_open and ex_open > ex_close else 'CHECK 分工不明显'}", flush=True)
