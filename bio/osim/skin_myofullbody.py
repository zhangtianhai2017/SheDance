#!/usr/bin/env python3
"""Skin OUR humanoid (the female good-figure body, betas 0,3, posture-fixed) and drive it by the
MUSCLE-driven motion: bind our body mesh to the MyoFullBody skeleton (blended LBS) in a SHARED T-pose
(our humanoid's SMPL-H T-pose retargeted to MyoFullBody), then deform by the dance MyoFullBody qpos.
Muscle -> body -> skin (SheDance's own logic). NOT the source dancer's mesh.
Args: dance_cache humanoid_npz(verts+joints+faces) myo_tpose_npz out  [frame...]   Env: FPS, AZ."""
import os, sys, numpy as np, mujoco
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from musclemimic.environments.humanoids import MyoFullBody
DANCE, HUM, MYOT, OUT = sys.argv[1:5]
frames_arg = [int(x) for x in sys.argv[5:]]; FPS = int(os.environ.get("FPS", "30"))
q = np.asarray(np.load(DANCE, allow_pickle=True)["qpos"], float)
Hd = np.load(HUM, allow_pickle=True); V = np.asarray(Hd["vertices"], float); J = np.asarray(Hd["joints"], float); F = np.asarray(Hd["faces"])
qt = np.asarray(np.load(MYOT, allow_pickle=True)["qpos"], float)[0]
env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
B = mujoco.mjtObj.mjOBJ_BODY; d = mujoco.MjData(m)
seg = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])})
seg = [b for b in seg if b != 0]
# SMPL-H joint index -> MyoFullBody body name (standard SMPL body joints 0-21)
CORR = {"pelvis": 0, "head": 15, "humerus_l": 16, "humerus_r": 17, "ulna_l": 18, "ulna_r": 19,
        "lunate_l": 20, "lunate_r": 21, "femur_l": 1, "femur_r": 2, "tibia_l": 4, "tibia_r": 5,
        "calcn_l": 7, "calcn_r": 8, "toes_l": 10, "toes_r": 11}
d.qpos[:] = qt; mujoco.mj_forward(m, d)
src, dst = [], []
for body, ji in CORR.items():
    bid = mujoco.mj_name2id(m, B, body)
    if bid >= 0:
        src.append(J[ji]); dst.append(d.xpos[bid].copy())
src, dst = np.array(src), np.array(dst)
mus, mud = src.mean(0), dst.mean(0)
Hm = (src - mus).T @ (dst - mud) / len(src); U, D, Vt = np.linalg.svd(Hm); Rm = Vt.T @ U.T
if np.linalg.det(Rm) < 0: Vt[-1] *= -1; Rm = Vt.T @ U.T
s = D.sum() / ((src - mus) ** 2).sum() * len(src); t = mud - s * Rm @ mus
V0a = s * (V @ Rm.T) + t
print(f"align our-humanoid T-pose -> MyoFullBody T-pose: scale {s:.3f}, residual {np.linalg.norm((s*(src@Rm.T)+t)-dst,axis=1).mean()*1000:.0f}mm", flush=True)
# blended LBS bind in the shared T-pose
segpos = np.array([d.xpos[b] for b in seg]); segmat0 = np.array([d.xmat[b].reshape(3, 3) for b in seg])
K = 4
d2 = ((V0a[:, None, :] - segpos[None, :, :]) ** 2).sum(-1); idx = np.argsort(d2, axis=1)[:, :K]
wts = 1.0 / (np.take_along_axis(d2, idx, 1) + 1e-6); wts /= wts.sum(1, keepdims=True)
vloc = np.einsum("nkij,nkj->nki", segmat0[idx].transpose(0, 1, 3, 2), V0a[:, None, :] - segpos[idx])

W, Hh = 480, 760; r = pyrender.OffscreenRenderer(W, Hh)
FLESH = np.array([0.90, 0.66, 0.58]); az = np.radians(float(os.environ.get("AZ", "150"))); el = np.radians(10.0)
def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f); sx = np.cross(f, up); sx /= np.linalg.norm(sx); u = np.cross(sx, f)
    M = np.eye(4); M[:3, 0] = sx; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye; return M
L1 = np.array([0.3, 1.0, 0.8]); L1 /= np.linalg.norm(L1)
def deform(qx):
    d.qpos[:] = qx; mujoco.mj_forward(m, d)
    R = np.array([d.xmat[b].reshape(3, 3) for b in seg]); T = np.array([d.xpos[b] for b in seg])
    vk = np.einsum("nkij,nkj->nki", R[idx], vloc) + T[idx]
    return (wts[..., None] * vk).sum(1)
def render(Vw):
    ctr = Vw.mean(0); up = np.array([0, 0, 1.0])
    dv = np.array([np.cos(az), np.sin(az), 0]) + up * np.sin(el); dv /= np.linalg.norm(dv)
    tm = trimesh.Trimesh(Vw, F, process=True); N = tm.vertex_normals; sh = 0.3 + 0.8 * np.clip(N @ L1, 0, 1)
    tm.visual.vertex_colors = (np.concatenate([np.clip(FLESH * sh[:, None], 0, 1), np.ones((len(N), 1))], 1) * 255).astype(np.uint8)
    sc = pyrender.Scene(bg_color=[0.12, 0.13, 0.17, 1.0], ambient_light=[1.0, 1.0, 1.0])
    sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True)); sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.6), pose=look_at(ctr + dv * 2.6, ctr, up))
    return r.render(sc)[0]
if frames_arg:
    from PIL import Image
    Image.fromarray(np.hstack([render(deform(q[f])) for f in frames_arg])).save(OUT); print(f"saved {OUT}: {frames_arg}", flush=True)
else:
    import subprocess
    ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-framerate", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for x in range(len(q)): ff.stdin.write(render(deform(q[x])).tobytes())
    ff.stdin.close(); ff.wait(); print(f"wrote {OUT}", flush=True)
r.delete()
