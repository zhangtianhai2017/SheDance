#!/usr/bin/env python3
"""C: GENRE-conditioned hand-motion generation for inputs that LACK finger mocap (the original
SheDance idea: an imitator improvises plausible hand motion by dance style). Finger flexion =
base + energy-coupling(qvel) + beat oscillation, parameterized per genre. Body from the real dance.
Env: GENRE (Pop/Lock/Ballet), ZOOM, NF (frame limit)."""
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


ZOOM = os.environ.get("ZOOM"); GENRE = os.environ.get("GENRE", "Pop")
OUT = f"/mnt/c/work/2026/Claude/SheDance/renders/C_{GENRE}{'_zoom' if ZOOM else ''}.mp4"
m89 = getm(True); m129 = getm(False); d = mujoco.MjData(m129)
c = np.load(rc.CACHE, allow_pickle=True)
q89 = rc.gaussian_smooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
qvel = np.asarray(c["qvel"], float)
NF = int(os.environ.get("NF", q89.shape[0]))
NF = min(NF, q89.shape[0])

# --- genre profiles: how the "imitator" improvises hands per dance style ---
GENRE_PROF = {
    "Pop":    dict(base=0.15, e_gain=0.55, beat_hz=2.6, beat_amp=0.35, thumb=0.5, sharp=True),
    "Lock":   dict(base=0.10, e_gain=0.70, beat_hz=3.2, beat_amp=0.45, thumb=0.5, sharp=True),
    "Ballet": dict(base=0.04, e_gain=0.20, beat_hz=0.0, beat_amp=0.08, thumb=0.3, sharp=False),
}
P = GENRE_PROF.get(GENRE, GENRE_PROF["Pop"])

e = np.linalg.norm(qvel, axis=1)
e = np.convolve(e, np.ones(7) / 7, mode="same")
e_norm = np.clip(e / (np.percentile(e, 90) + 1e-6), 0, 1)

NQ = {0: 7, 1: 4, 2: 1, 3: 1}
def n2a(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            for j in range(m.njnt)}
a89 = n2a(m89); a129 = n2a(m129); body = [nm for nm in a89 if nm in a129]
flex_dofs = [nm for nm in a129 if nm not in a89 and "flexion" in nm]
print(f"genre={GENRE} flex DOFs={len(flex_dofs)} frames={NF}", flush=True)


def gen_flex(fi):
    t = fi / freq
    beat = max(0.0, np.sin(2 * np.pi * P["beat_hz"] * t)) if P["beat_hz"] > 0 else 0.0
    if P["sharp"]:
        beat = beat ** 2
    return P["base"] + P["e_gain"] * e_norm[fi] + P["beat_amp"] * beat


cam = mujoco.MjvCamera(); cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 120; cam.lookat[:] = [0, 0, 0.95]
hand_bid = -1
if ZOOM:
    hand_bid = mujoco.mj_name2id(m129, mujoco.mjtObj.mjOBJ_BODY, "lunate_r"); cam.distance = 0.34; cam.elevation = -10
m129.vis.global_.offwidth = rc.W; m129.vis.global_.offheight = rc.H
rend = mujoco.Renderer(m129, height=rc.H, width=rc.W); opt = mujoco.MjvOption()
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rc.W}x{rc.H}",
                       "-framerate", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for fi in range(NF):
    d.qpos[:] = 0.0
    for nm in body:
        a0, t = a89[nm]; b0, _ = a129[nm]; nq = NQ[t]
        d.qpos[b0:b0 + nq] = q89[fi, a0:a0 + nq]
    bf = gen_flex(fi)
    for ki, nm in enumerate(flex_dofs):
        val = bf * (P["thumb"] if any(s in nm for s in ("cmc", "mp_", "ip_")) else 1.0)
        val *= 1.0 + 0.08 * np.sin(0.7 * ki + 2 * np.pi * 0.5 * fi / freq)   # slight per-finger stagger
        d.qpos[a129[nm][0]] = float(np.clip(val, 0, 1.4))
    mujoco.mj_forward(m129, d)
    cam.lookat[:] = d.xpos[hand_bid] if (ZOOM and hand_bid >= 0) else [d.qpos[0], d.qpos[1], 0.95]
    rend.update_scene(d, cam, opt)
    ff.stdin.write(rend.render().tobytes())
ff.stdin.close(); ff.wait(); rend.close()
print("wrote", OUT, flush=True)
