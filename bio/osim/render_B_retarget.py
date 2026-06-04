#!/usr/bin/env python3
"""B: prove the SMPL-H hand-pose -> MyoFullBody finger-qpos retarget capability.
Synthetic SMPL-H hand input authored as MANO axis-angle (open->fist->point->open), mapped via the
MANO-joint -> MyoFullBody finger-DOF correspondence, then rendered. The 'point' pose (index extended
while the rest curl) proves the PER-FINGER mapping is correct (not a uniform curl).
NOTE: flexion ~ ||axis-angle|| (rotation magnitude) — good for gross "motion-in-place"; real data
would refine with signed flex/abduction decomposition. Body comes from the real dance (89->129)."""
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
OUT = "/mnt/c/work/2026/Claude/SheDance/renders/" + ("B_retarget_zoom.mp4" if ZOOM else "B_retarget.mp4")
NF = 240
m89 = getm(True); m129 = getm(False); d = mujoco.MjData(m129)
c = np.load(rc.CACHE, allow_pickle=True); q89 = rc.gaussian_smooth(np.asarray(c["qpos"], float))
NQ = {0: 7, 1: 4, 2: 1, 3: 1}


def n2a(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            for j in range(m.njnt)}


a89 = n2a(m89); a129 = n2a(m129); body = [nm for nm in a89 if nm in a129]

# MyoFullBody finger DOF base -> (MANO finger, joint#, component f=flex a=abduct)
DOF2MANO = {
    "mcp2_flexion": ("Index", 1, "f"), "mcp2_abduction": ("Index", 1, "a"), "pm2_flexion": ("Index", 2, "f"), "md2_flexion": ("Index", 3, "f"),
    "mcp3_flexion": ("Middle", 1, "f"), "mcp3_abduction": ("Middle", 1, "a"), "pm3_flexion": ("Middle", 2, "f"), "md3_flexion": ("Middle", 3, "f"),
    "mcp4_flexion": ("Ring", 1, "f"), "mcp4_abduction": ("Ring", 1, "a"), "pm4_flexion": ("Ring", 2, "f"), "md4_flexion": ("Ring", 3, "f"),
    "mcp5_flexion": ("Pinky", 1, "f"), "mcp5_abduction": ("Pinky", 1, "a"), "pm5_flexion": ("Pinky", 2, "f"), "md5_flexion": ("Pinky", 3, "f"),
    "cmc_flexion": ("Thumb", 1, "f"), "cmc_abduction": ("Thumb", 1, "a"), "mp_flexion": ("Thumb", 2, "f"), "ip_flexion": ("Thumb", 3, "f"),
}


def hand_theta(fi):
    """Per-finger flexion theta(t): open(0-60)->fist(60-120)->point[index out](120-180)->open(180-240)."""
    def ramp(a, b, t): return float(np.clip((t - a) / (b - a), 0, 1))
    fingers = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
    if fi < 60: g = 0.0
    elif fi < 120: g = ramp(60, 120, fi)
    elif fi < 180: g = 1.0
    else: g = 1 - ramp(180, 240, fi)
    th = {f: 1.3 * g for f in fingers}
    if 120 <= fi < 180:                         # POINT: index straight, others fisted
        th["Index"] = 0.0
    th["Thumb"] = min(th["Thumb"], 0.7)
    return th


def synth_mano_aa(theta_finger):                # SMPL-H hand pose as axis-angle (flex about x), ||aa||=theta
    return {(f, j): np.array([theta_finger[f], 0.0, 0.0]) for f in theta_finger for j in (1, 2, 3)}


def map_fingers_to_qpos(theta_finger, side):    # the retarget: MANO aa -> MyoFullBody finger DOF values
    aa = synth_mano_aa(theta_finger); out = {}
    for base, (f, j, comp) in DOF2MANO.items():
        dof = base + ("_r" if side == "R" else "_l")
        out[dof] = float(np.linalg.norm(aa[(f, j)])) if comp == "f" else 0.0
    return out


cam = mujoco.MjvCamera(); cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 120; cam.lookat[:] = [0, 0, 0.95]
hand_bid = -1
if ZOOM:
    hand_bid = mujoco.mj_name2id(m129, mujoco.mjtObj.mjOBJ_BODY, "lunate_r")
    cam.distance = 0.34; cam.elevation = -10
m129.vis.global_.offwidth = rc.W; m129.vis.global_.offheight = rc.H
rend = mujoco.Renderer(m129, height=rc.H, width=rc.W); opt = mujoco.MjvOption()
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rc.W}x{rc.H}",
                       "-framerate", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for fi in range(NF):
    d.qpos[:] = 0.0
    for nm in body:
        a0, t = a89[nm]; b0, _ = a129[nm]; nq = NQ[t]
        d.qpos[b0:b0 + nq] = q89[fi % q89.shape[0], a0:a0 + nq]
    th = hand_theta(fi)
    for side in ("R", "L"):
        for dof, val in map_fingers_to_qpos(th, side).items():
            if dof in a129:
                d.qpos[a129[dof][0]] = val
    mujoco.mj_forward(m129, d)
    if ZOOM and hand_bid >= 0:
        cam.lookat[:] = d.xpos[hand_bid]; cam.azimuth = 120
    else:
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.95]
    rend.update_scene(d, cam, opt)
    ff.stdin.write(rend.render().tobytes())
ff.stdin.close(); ff.wait(); rend.close()
print("wrote", OUT, flush=True)
