#!/usr/bin/env python3
"""De-risk the MuJoCo hand-SO: confirm we can read the muscle force model + moment arms + inverse
dynamics on the 129-dof MyoFullBody. Muscle force is LINEAR in activation at a fixed state:
force = gain*act + bias, so gain = force(act=1) - force(act=0), bias = force(act=0) (robust, no API
guessing). Then a finger flexor's moment on a finger-flexion dof should be non-zero; mj_inverse gives
the generalized force each dof needs."""
import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
env = MyoFullBody(disable_fingers=False)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
d = mujoco.MjData(m)
A, J = mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT

# small finger flex so muscles are off their slack
for j in range(m.njnt):
    nm = mujoco.mj_id2name(m, J, j) or ""
    if nm.endswith("_r") and "flexion" in nm and any(k in nm for k in ("mcp", "pip", "dip", "cmc", "ip")):
        d.qpos[m.jnt_qposadr[j]] = 0.4
mujoco.mj_forward(m, d)

# gain/bias via act=0 / act=1 (force linear in act at fixed pose)
d.act[:] = 0; mujoco.mj_forward(m, d); f0 = d.actuator_force.copy()
d.act[:] = 1; mujoco.mj_forward(m, d); f1 = d.actuator_force.copy()
gain = f1 - f0; bias = f0

for tgt in ("FDP3", "FDS3", "EDC3"):
    fi = next((i for i in range(m.nu) if (mujoco.mj_id2name(m, A, i) or "") == tgt), None)
    if fi is None: continue
    print(f"{tgt}: gaintype={m.actuator_gaintype[fi]} (3=muscle)  gain={gain[fi]:.1f}  bias={bias[fi]:.1f}", flush=True)

# moment arm matrix
am = np.asarray(d.actuator_moment).reshape(m.nu, m.nv)   # flat (nu*nv) -> dense (nu, nv)
print(f"actuator_moment dense shape={am.shape} (nu={m.nu} x nv={m.nv})", flush=True)
fi = next(i for i in range(m.nu) if (mujoco.mj_id2name(m, A, i) or "") == "FDP3")
jf = next(j for j in range(m.njnt) if (mujoco.mj_id2name(m, J, j) or "") == "mcp3_flexion_r")
dof = int(m.jnt_dofadr[jf])
print(f"FDP3 moment on mcp3_flexion_r (dof {dof}) = {am[fi, dof]:.5f}  (nonzero -> it crosses the joint)", flush=True)

# inverse dynamics: generalized force each dof needs (qacc=0 here -> gravity+passive)
d.act[:] = 0; mujoco.mj_forward(m, d); d.qacc[:] = 0
mujoco.mj_inverse(m, d)
print(f"qfrc_inverse[mcp3_flexion_r] = {d.qfrc_inverse[dof]:.4f}", flush=True)
print("OK: muscle force model + moment arms + inverse dynamics all readable", flush=True)
