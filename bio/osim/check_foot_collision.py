#!/usr/bin/env python3
"""Measure foot-foot proximity/penetration over a motion (decide if collision handling is warranted).
Reports min L-R foot body distance + MuJoCo contact penetration (if collision geoms active).
Env: RENDER_CACHE = the qpos cache."""
import os, numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody
import render_core as rc

c = np.load(rc.CACHE, allow_pickle=True)
qpos = rc.gaussian_smooth(np.asarray(c["qpos"], float))
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
d = mujoco.MjData(m)
B = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in ("calcn_r", "calcn_l", "toes_r", "toes_l")}
print("ngeom", m.ngeom, "(collision geoms present?)")
mc, mt, npen, deepest = 1e9, 1e9, 0, 0.0
for t in range(qpos.shape[0]):
    d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
    mc = min(mc, float(np.linalg.norm(d.xpos[B["calcn_r"]] - d.xpos[B["calcn_l"]])))
    mt = min(mt, float(np.linalg.norm(d.xpos[B["toes_r"]] - d.xpos[B["toes_l"]])))
    for i in range(d.ncon):
        if d.contact[i].dist < -1e-4:
            npen += 1; deepest = min(deepest, float(d.contact[i].dist))
print(f"min foot-foot dist: calcn {mc:.3f} m, toes {mt:.3f} m  (feet ~0.08-0.10m wide)")
print(f"MuJoCo contacts penetrating(<-0.1mm): {npen} frame-contacts, deepest {deepest*1000:.1f} mm")
