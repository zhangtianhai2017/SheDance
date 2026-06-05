#!/usr/bin/env python3
"""Side-by-side full-body skeleton at one frame from two MyoFullBody caches (qpos). Each body is
camera-centered so poses are comparable regardless of world position. Args: cacheA cacheB frame out"""
import os, sys, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
from PIL import Image
A, Bc, frame, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
qA = np.asarray(np.load(A, allow_pickle=True)["qpos"], float)
qB = np.asarray(np.load(Bc, allow_pickle=True)["qpos"], float)
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
d = mujoco.MjData(m)
W, H = 480, 640; m.vis.global_.offwidth = W; m.vis.global_.offheight = H
r = mujoco.Renderer(m, height=H, width=W); opt = mujoco.MjvOption()
def shot(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.9]
    cam.distance = 2.7; cam.elevation = -10; cam.azimuth = 120
    r.update_scene(d, cam, opt); return r.render().copy()
Image.fromarray(np.hstack([shot(qA[frame]), shot(qB[frame])])).save(out)
print(f"saved {out} (left={os.path.basename(A)}, right={os.path.basename(Bc)}, frame {frame})", flush=True)
