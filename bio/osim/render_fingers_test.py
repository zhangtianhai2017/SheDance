#!/usr/bin/env python3
"""A-proof: drive the WITH-finger MyoFullBody (129 qpos) = real body dance (mapped 89->129 by
joint name) + SYNTHETIC finger curl, render via osmesa. Proves the finger drive+render endpoint
works (independent of the GMR config / real finger data, which come in B)."""
import os, subprocess, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
from musclemimic.environments.humanoids import MyoFullBody
import render_core as rc


def getm(df):
    env = MyoFullBody(disable_fingers=df)
    return next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel"))
                if isinstance(o, mujoco.MjModel))


ZOOM = os.environ.get("ZOOM")
OUT = "/mnt/c/work/2026/Claude/SheDance/renders/" + ("fingers_zoom.mp4" if ZOOM else "fingers_test.mp4")
NF = 200
m89 = getm(True); m129 = getm(False); d = mujoco.MjData(m129)
c = np.load(rc.CACHE, allow_pickle=True)
q89 = rc.gaussian_smooth(np.asarray(c["qpos"], float))
NQ = {0: 7, 1: 4, 2: 1, 3: 1}


def n2a(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            for j in range(m.njnt)}


a89 = n2a(m89); a129 = n2a(m129)
body = [nm for nm in a89 if nm in a129]
finger_flex = [nm for nm in a129 if nm not in a89 and "flexion" in nm]
print(f"mapped body joints: {len(body)}; synthetic-curl finger DOFs: {len(finger_flex)}", flush=True)

cam = mujoco.MjvCamera(); cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 120; cam.lookat[:] = [0, 0, 0.95]
hand_bid = -1
if ZOOM:
    for cand in ("lunate_r", "capitate_r", "secondmc_r", "firstmc_r", "hand_r"):
        hand_bid = mujoco.mj_name2id(m129, mujoco.mjtObj.mjOBJ_BODY, cand)
        if hand_bid >= 0:
            print("zoom tracking body:", cand, flush=True); break
    cam.distance = 0.32; cam.elevation = -10
m129.vis.global_.offwidth = rc.W; m129.vis.global_.offheight = rc.H
rend = mujoco.Renderer(m129, height=rc.H, width=rc.W); opt = mujoco.MjvOption()
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rc.W}x{rc.H}",
                       "-framerate", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for fi in range(min(NF, q89.shape[0])):
    d.qpos[:] = 0.0
    for nm in body:
        a0, t = a89[nm]; b0, _ = a129[nm]; nq = NQ[t]
        d.qpos[b0:b0 + nq] = q89[fi, a0:a0 + nq]
    curl = 0.5 * (1 - np.cos(2 * np.pi * 3 * fi / NF)) * 1.1   # 0..1.1 rad, 3 cycles
    for nm in finger_flex:
        d.qpos[a129[nm][0]] = curl
    mujoco.mj_forward(m129, d)
    if ZOOM and hand_bid >= 0:
        cam.lookat[:] = d.xpos[hand_bid]; cam.azimuth = 120 + 0.5 * fi
    else:
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.95]
    rend.update_scene(d, cam, opt)
    ff.stdin.write(rend.render().tobytes())
ff.stdin.close(); ff.wait(); rend.close()
print("wrote", OUT, flush=True)
