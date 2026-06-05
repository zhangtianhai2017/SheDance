#!/usr/bin/env python3
"""Render OUR humanoid's OWN body volume (MyoFullBody collision geoms = limb/torso/head capsules),
flesh-colored & solid, driven by the muscle-driven qpos. This is the skin that follows our muscle
model (not VedioTo3D's source mesh). Args: cache out frame1 frame2 ...  Env: AZ"""
import os, sys, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
from PIL import Image
cache, out = sys.argv[1], sys.argv[2]; frames = [int(x) for x in sys.argv[3:]]
q = np.asarray(np.load(cache, allow_pickle=True)["qpos"], float)
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
# show only the body-volume collision geoms (flesh, solid); hide bones + muscle wrapping
nshown = 0
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        m.geom_rgba[g] = [0.90, 0.66, 0.58, 1.0]; nshown += 1
    else:
        m.geom_rgba[g] = [0, 0, 0, 0]
print(f"showing {nshown} body-volume collision geoms", flush=True)
d = mujoco.MjData(m)
W, H = 360, 620; m.vis.global_.offwidth = W; m.vis.global_.offheight = H
r = mujoco.Renderer(m, height=H, width=W); opt = mujoco.MjvOption()   # tendons off by default
def shot(qq):
    d.qpos[:] = qq; mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.9]
    cam.distance = 2.8; cam.elevation = -8; cam.azimuth = float(os.environ.get("AZ", "120"))
    r.update_scene(d, cam, opt); return r.render().copy()
Image.fromarray(np.hstack([shot(q[f]) for f in frames])).save(out)
print(f"saved {out}: frames {frames}", flush=True)
