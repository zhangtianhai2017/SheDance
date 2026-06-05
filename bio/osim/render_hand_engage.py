#!/usr/bin/env python3
"""Kinematic-engagement viz of the finger muscles. Free finger motion needs ~0 active force, so
coloring by FORCE is cold; instead color each finger muscle by how fast its tendon is SHORTENING
(contracting = 'engaged'/driving the move). Generate a finger open/close, render a right-hand
close-up video + save a closing-moment still. Honest: this is engagement (length-change), not force.
Env: OUT (mp4), STILL (png), PERIOD s, AZ/EL/DIST, FPS."""
import os, numpy as np, mujoco, subprocess
os.environ.setdefault("MUJOCO_GL", "osmesa")
from musclemimic.environments.humanoids import MyoFullBody
from PIL import Image

def getm(df):
    e = MyoFullBody(disable_fingers=df)
    return next(o for o in (getattr(e, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
m = getm(False); m1 = getm(True); d = mujoco.MjData(m)
A, J, B = mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
OUT = os.environ.get("OUT", "/mnt/c/work/2026/Claude/SheDance/renders/hand_engage.mp4")
STILL = os.environ.get("STILL", "/mnt/c/work/2026/Claude/SheDance/renders/hand_engage_close.png")
FPS = int(os.environ.get("FPS", "60")); PERIOD = float(os.environ.get("PERIOD", "0.8"))

body_acts = set(mujoco.mj_id2name(m1, A, i) for i in range(m1.nu))
fing = [i for i in range(m.nu) if mujoco.mj_id2name(m, A, i) not in body_acts]
ften = [int(m.actuator_trnid[i, 0]) for i in fing]
flexj = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, J, j) or "").endswith("_r")
         and "flexion" in (mujoco.mj_id2name(m, J, j) or "") and any(k in (mujoco.mj_id2name(m, J, j) or "") for k in ("mcp", "pip", "dip", "cmc", "ip"))]
flexq = np.array([int(m.jnt_qposadr[j]) for j in flexj]); flexd = np.array([int(m.jnt_dofadr[j]) for j in flexj])

freq = FPS; N = 180; t = np.arange(N) / freq
flex = 0.45 * (1 - np.cos(2 * np.pi * t / PERIOD))            # rhythmic open/close, amp 0.9
base = m.key_qpos[0].copy() if m.nkey > 0 else np.zeros(m.nq)
if m.nkey == 0: base[2] = 0.95; base[3:7] = [1, 0, 0, 0]
qpos = np.tile(base, (N, 1)); qpos[:, flexq] = flex[:, None]
dt = 1.0 / freq

# pass 1: engagement = tendon shortening rate (contracting muscle = driving the motion)
eng = np.zeros((N, len(fing)))
for k in range(N):
    d.qpos[:] = qpos[k]; d.qvel[:] = 0
    if 0 < k < N - 1:
        d.qvel[flexd] = (qpos[k + 1, flexq] - qpos[k - 1, flexq]) / (2 * dt)
    mujoco.mj_forward(m, d)
    eng[k] = -np.asarray(d.ten_velocity)[ften]               # shortening -> positive engagement
pos = eng[eng > 0]
SCALE = float(np.percentile(pos, 92)) if pos.size else 1.0
print(f"finger muscles {len(fing)}, engagement scale {SCALE:.4f} m/s", flush=True)

# pass 2: render
W, H = 760, 760; m.vis.global_.offwidth = W; m.vis.global_.offheight = H
r = mujoco.Renderer(m, height=H, width=W); opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
hb = mujoco.mj_name2id(m, B, "thirdmc_r")
cam = mujoco.MjvCamera(); cam.distance = float(os.environ.get("DIST", "0.28"))
cam.elevation = float(os.environ.get("EL", "-12")); cam.azimuth = float(os.environ.get("AZ", "90"))
dim = np.array([0.25, 0.25, 0.3, 0.4])
m.tendon_rgba[:] = dim                                        # non-finger tendons stay dim
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
flexvel = np.gradient(flex); kclose = int(np.argmax(flexvel))
for k in range(N):
    d.qpos[:] = qpos[k]; mujoco.mj_forward(m, d)
    e = np.clip(eng[k] / SCALE, 0, 1)
    for idx, ti in enumerate(ften):
        a = e[idx]; m.tendon_rgba[ti] = [0.2 + 0.8 * a, 0.15 + 0.15 * a, 0.4 * (1 - a), 0.55 + 0.45 * a]
    cam.lookat[:] = d.xpos[hb]; r.update_scene(d, cam, opt); img = r.render()
    ff.stdin.write(img.tobytes())
    if k == kclose:
        Image.fromarray(img).save(STILL)
ff.stdin.close(); ff.wait()
print(f"wrote {OUT}; closing-moment still -> {STILL}", flush=True)
