#!/usr/bin/env python3
"""C v2: genre GESTURE LIBRARY + KINEMATIC BEAT detection + beat-aligned hand sequencing.
- Beat: articulation-velocity local minima (the dancer's holds/hits). AIST++-style kinematic beat.
  (No audio on disk + no librosa -> can't sync to the actual music track; kinematic beats track the
   music because the dance is on-beat. True audio sync needs the audio file + librosa.)
- Gesture library: per-finger (flexion, abduction) key-poses; per-genre sequence; snap(sharp)/ease(soft).
Env: GENRE (Pop/Lock/Ballet), ZOOM, NF."""
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
OUT = f"/mnt/c/work/2026/Claude/SheDance/renders/C2_{GENRE}{'_zoom' if ZOOM else ''}.mp4"
m89 = getm(True); m129 = getm(False); d = mujoco.MjData(m129)
c = np.load(rc.CACHE, allow_pickle=True)
q89 = rc.gaussian_smooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
qvel = np.asarray(c["qvel"], float)
NF = min(int(os.environ.get("NF", q89.shape[0])), q89.shape[0])

# ---- kinematic beats = articulation-velocity local minima (holds/hits) ----
artic = np.convolve(np.linalg.norm(qvel[:, 6:], axis=1), np.ones(5) / 5, mode="same")
thr = np.percentile(artic[:NF], 55); gap = int(0.22 * freq); beats = []; last = -10 ** 9
for i in range(2, NF - 2):
    if artic[i] <= artic[i - 1] and artic[i] < artic[i + 1] and artic[i] < thr and i - last >= gap:
        beats.append(i); last = i
print(f"genre={GENRE}  kinematic beats={len(beats)} = {len(beats)/(NF/freq)*60:.0f}/min over {NF/freq:.1f}s", flush=True)

# ---- gesture library: per-finger (flexion, abduction) ----
FING = ["thumb", "index", "middle", "ring", "pinky"]
KP = {
    "OPEN":   {f: (0.0, 0.0) for f in FING},
    "FIST":   {f: (1.35, 0.0) for f in FING},
    "POINT":  {"thumb": (0.7, 0.3), "index": (0.0, 0.0), "middle": (1.35, 0.0), "ring": (1.35, 0.0), "pinky": (1.35, 0.0)},
    "GUN":    {"thumb": (0.0, 0.5), "index": (0.0, 0.0), "middle": (1.35, 0.0), "ring": (1.35, 0.0), "pinky": (1.35, 0.0)},
    "SPREAD": {f: (0.1, 0.45) for f in FING},
    "RELAX":  {f: (0.35, 0.05) for f in FING},
}
SEQ = {"Pop": ["FIST", "OPEN", "POINT", "FIST", "SPREAD", "OPEN"],
       "Lock": ["FIST", "POINT", "FIST", "OPEN", "GUN", "FIST"],
       "Ballet": ["OPEN", "RELAX", "SPREAD", "RELAX"]}
TRANS = {"Pop": 0.10, "Lock": 0.07, "Ballet": 0.45}
seq = SEQ.get(GENRE, SEQ["Pop"]); trans = TRANS.get(GENRE, 0.12)

FDOF = {
    "thumb": (["cmc_flexion", "mp_flexion", "ip_flexion"], "cmc_abduction"),
    "index": (["mcp2_flexion", "pm2_flexion", "md2_flexion"], "mcp2_abduction"),
    "middle": (["mcp3_flexion", "pm3_flexion", "md3_flexion"], "mcp3_abduction"),
    "ring": (["mcp4_flexion", "pm4_flexion", "md4_flexion"], "mcp4_abduction"),
    "pinky": (["mcp5_flexion", "pm5_flexion", "md5_flexion"], "mcp5_abduction"),
}
allbases = set()
for flist, ab in FDOF.values():
    allbases |= set(flist) | {ab}


def kp_vector(name):
    out = {}
    for f, (flist, ab) in FDOF.items():
        fl, a = KP[name][f]
        for fd in flist:
            out[fd] = fl
        out[ab] = a
    return out


beat_idx_at = np.zeros(NF, int); gi, bi = 0, 0
for fi in range(NF):
    if bi < len(beats) and fi >= beats[bi]:
        gi += 1; bi += 1
    beat_idx_at[fi] = gi

NQ = {0: 7, 1: 4, 2: 1, 3: 1}
def n2a(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            for j in range(m.njnt)}
a89 = n2a(m89); a129 = n2a(m129); body = [nm for nm in a89 if nm in a129]
cur = {b: 0.0 for b in allbases}
alpha = min(1.0, (1.0 / freq) / max(trans, 1e-3))

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
    target = kp_vector(seq[beat_idx_at[fi] % len(seq)])
    for b in allbases:
        cur[b] += (target[b] - cur[b]) * alpha
        for side in ("_r", "_l"):
            dof = b + side
            if dof in a129:
                d.qpos[a129[dof][0]] = cur[b]
    mujoco.mj_forward(m129, d)
    cam.lookat[:] = d.xpos[hand_bid] if (ZOOM and hand_bid >= 0) else [d.qpos[0], d.qpos[1], 0.95]
    rend.update_scene(d, cam, opt)
    ff.stdin.write(rend.render().tobytes())
ff.stdin.close(); ff.wait(); rend.close()
print("wrote", OUT, flush=True)
