#!/usr/bin/env python3
"""Validate MyoFullBody finger muscles render + drive: 129-dof, curl the right-hand fingers, color
the 62 hand muscles (flexors hot on a curl, extensors cool), render a right-hand close-up.
Env: OUT, CURL (rad), AZ, EL, DIST."""
import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
from PIL import Image

OUT = os.environ.get("OUT", "/mnt/c/work/2026/Claude/SheDance/renders/hand_test.png")
CURL = float(os.environ.get("CURL", "1.0"))
def getm(df):
    e = MyoFullBody(disable_fingers=df)
    return next(o for o in (getattr(e, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
m = getm(False); m1 = getm(True)
d = mujoco.MjData(m)
A = mujoco.mjtObj.mjOBJ_ACTUATOR; J = mujoco.mjtObj.mjOBJ_JOINT; B = mujoco.mjtObj.mjOBJ_BODY

body_acts = set(mujoco.mj_id2name(m1, A, i) for i in range(m1.nu))
fing = [(i, mujoco.mj_id2name(m, A, i)) for i in range(m.nu) if mujoco.mj_id2name(m, A, i) not in body_acts]

# curl the right-hand finger flexion joints
ncurl = 0
for j in range(m.njnt):
    nm = mujoco.mj_id2name(m, J, j) or ""
    if nm.endswith("_r") and "flexion" in nm and any(k in nm for k in ("mcp", "pip", "dip", "cmc", "ip")):
        d.qpos[m.jnt_qposadr[j]] = CURL; ncurl += 1
mujoco.mj_forward(m, d)

def act_for(nm):
    if any(nm.startswith(p) for p in ("FDS", "FDP", "FPL", "FDM")): return 0.85   # flexors -> do the curl
    if any(nm.startswith(p) for p in ("ED", "EI", "EP")): return 0.10             # extensors (lengthen)
    return 0.40                                                                    # intrinsics
for i, nm in fing:
    t = int(m.actuator_trnid[i, 0])
    if 0 <= t < m.ntendon:
        a = act_for(nm)
        m.tendon_rgba[t] = [0.2 + 0.8 * a, 0.15 + 0.15 * a, 0.4 * (1 - a), 0.55 + 0.45 * a]

hb = mujoco.mj_name2id(m, B, "thirdmc_r")
W, Hh = 760, 760; m.vis.global_.offwidth = W; m.vis.global_.offheight = Hh
r = mujoco.Renderer(m, height=Hh, width=W)
opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
cam = mujoco.MjvCamera(); cam.lookat[:] = d.xpos[hb]
cam.distance = float(os.environ.get("DIST", "0.28"))
cam.elevation = float(os.environ.get("EL", "-12"))
cam.azimuth = float(os.environ.get("AZ", "90"))
r.update_scene(d, cam, opt)
Image.fromarray(r.render()).save(OUT)
print(f"finger muscles {len(fing)}, curled {ncurl} joints, hand close-up -> {OUT}", flush=True)
