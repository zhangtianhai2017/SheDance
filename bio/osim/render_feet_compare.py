#!/usr/bin/env python3
"""Feet-zoom before/after at the worst-penetration frame (verify L1 removed the overlap).
Left = original (overlapping), Right = refined. Env: ORIG, REF, OUT"""
import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
from PIL import Image
orig = np.asarray(np.load(os.environ["ORIG"], allow_pickle=True)["qpos"], float)
ref = np.asarray(np.load(os.environ["REF"], allow_pickle=True)["qpos"], float)
out = os.environ["OUT"]
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
B = mujoco.mjtObj.mjOBJ_BODY
G = mujoco.mjtObj.mjOBJ_GEOM
cl, cr = mujoco.mj_name2id(m, B, "calcn_l"), mujoco.mj_name2id(m, B, "calcn_r")
gl = [mujoco.mj_name2id(m, G, n) for n in ("l_foot_col1", "l_foot_col3", "l_foot_col4", "l_bofoot_col1", "l_bofoot_col2")]
gr = [mujoco.mj_name2id(m, G, n) for n in ("r_foot_col1", "r_foot_col3", "r_foot_col4", "r_bofoot_col1", "r_bofoot_col2")]
d = mujoco.MjData(m)
def footdist(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    return min(mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a in gl for b in gr)
do = [footdist(orig[t]) for t in range(len(orig))]
dr = [footdist(ref[t]) for t in range(len(ref))]
cands = [(dr[t] - do[t], t) for t in range(len(orig)) if do[t] < -0.005 and dr[t] > 0]
frame = max(cands)[1] if cands else int(np.argmin(do))
print(f"demo frame {frame}: orig geom-dist {do[frame]*1000:.1f}mm (penetrating) -> refined {dr[frame]*1000:.1f}mm (resolved)", flush=True)
m.vis.global_.offwidth = 480; m.vis.global_.offheight = 560
rend = mujoco.Renderer(m, height=560, width=480); opt = mujoco.MjvOption()
def shot(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    mid = (d.xpos[cl] + d.xpos[cr]) / 2
    cam = mujoco.MjvCamera(); cam.lookat[:] = mid; cam.distance = 0.75; cam.elevation = -18; cam.azimuth = 90
    rend.update_scene(d, cam, opt); return rend.render().copy()
Image.fromarray(np.hstack([shot(orig[frame]), shot(ref[frame])])).save(out)
print("saved", out, flush=True)
