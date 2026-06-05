#!/usr/bin/env python3
"""Render the SMPL-H SOURCE skin (the real dancer body) of a dance -> clear, real feet.
(The foot-overlap was a MyoFullBody retarget artifact; the source body doesn't have it.)
Args: raw_smplh_npz out  [fps]   Env: SAMPLE=<frame> (single PNG), CAM_AZ/CAM_EL deg."""
import os, sys, subprocess, numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import trimesh, pyrender
import general_motion_retargeting.utils.smpl as _sm
_o = _sm.SMPLH_Parser.__init__
def _i(self, *a, **k): k.setdefault("num_betas", 16); _o(self, *a, **k)
_sm.SMPLH_Parser.__init__ = _i
from general_motion_retargeting.utils.smpl import load_smplh_file

raw, out = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
MODELS = os.path.expanduser("~/shedance/smpl_models")
GENDER = os.environ.get("GENDER"); BETAS = os.environ.get("BETAS"); ZEROPOSE = os.environ.get("ZEROPOSE")
if GENDER or BETAS or ZEROPOSE:
    d = dict(np.load(raw, allow_pickle=True))
    if GENDER:
        d["gender"] = GENDER
    if BETAS:
        b = np.array([float(x) for x in BETAS.split(",")], float)
        bb = np.zeros(16, float); bb[:len(b)] = b; d["betas"] = bb
    if ZEROPOSE:
        p = np.asarray(d["poses"], float); p[:, 3:66] = 0.0
        p[:, 0:3] = [np.pi / 2, 0, 0]   # clean upright standing root (no dance tilt) -> pure rest SHAPE
        d["poses"] = p
    if os.environ.get("POSTURE_DEBIAS"):   # remove systematic spine/neck/head posture bias, keep dance dynamics
        amt = float(os.environ.get("POSTURE_DEBIAS"))
        p = np.asarray(d["poses"], float); bj = p[:, :66].reshape(len(p), 22, 3)
        for j in (3, 6, 9, 12, 15):        # spine1/2/3, neck, head
            bj[:, j, :] -= amt * bj[:, j, :].mean(0, keepdims=True)
        p[:, :66] = bj.reshape(len(p), 66); d["poses"] = p
    HEAD = float(os.environ.get("POSTURE_HEAD", "0"))   # active head retraction (deg, chin tuck)
    if HEAD:
        p = np.asarray(d["poses"], float); bj = p[:, :66].reshape(len(p), 22, 3)
        bj[:, 12, 0] -= np.radians(HEAD)               # neck extension (X-) -> head slides back
        bj[:, 15, 0] += np.radians(HEAD)               # head flexion (X+) -> re-level face (chin tuck)
        p[:, :66] = bj.reshape(len(p), 66); d["poses"] = p
    raw = "/tmp/_skin_patched.npz"; np.savez(raw, **d)
    print(f"override gender={GENDER} betas={BETAS} zeropose={bool(ZEROPOSE)}", flush=True)
sd, bm, so, h = load_smplh_file(raw, MODELS)
V = np.asarray(so.vertices, float); F = np.asarray(bm.faces)
W, Hh = 480, 700
exts = V.max(1) - V.min(1)                       # (N,3) per-frame extents
up_ax = int(np.argmax(np.median(exts, 0)))       # body-height axis (movement axes vary per-frame, height is steady)
size = float(np.median(exts[:, up_ax]))          # ~ body height (~1.7m)
up = np.zeros(3); up[up_ax] = 1.0
horiz = [a for a in range(3) if a != up_ax]
az = np.radians(float(os.environ.get("CAM_AZ", "0"))); el = np.radians(float(os.environ.get("CAM_EL", "8")))
dir3 = np.zeros(3); dir3[horiz[0]] = np.cos(az); dir3[horiz[1]] = np.sin(az); dir3 += up * np.sin(el)
dir3 /= np.linalg.norm(dir3)

def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M
r = pyrender.OffscreenRenderer(W, Hh)
side = np.cross(dir3, up); side /= np.linalg.norm(side)
L1 = 0.25 * dir3 + 1.0 * side + 0.75 * up; L1 /= np.linalg.norm(L1)   # key (side+top)
L2 = 0.5 * dir3 - 1.0 * side; L2 /= np.linalg.norm(L2)                # fill (opposite side)
FLESH = np.array([0.90, 0.66, 0.58])
# GL diffuse lighting is broken in this pyrender/pyopengl build -> bake Lambert shading into vertex
# colors and render fully-ambient (vertex colors don't need a GL light).
def shade(N):
    s = 0.28 + 0.80 * np.clip(N @ L1, 0, 1) + 0.22 * np.clip(N @ L2, 0, 1)
    c = np.clip(FLESH[None, :] * s[:, None], 0, 1)
    return (np.concatenate([c, np.ones((len(c), 1))], 1) * 255).astype(np.uint8)
def frame(i):
    ctr = V[i].mean(0)                           # follow the body (remove dance translation -> stays centered)
    campose = look_at(ctr + dir3 * size * 2.2, ctr, up)
    tm = trimesh.Trimesh(V[i], F, process=True)
    tm.visual.vertex_colors = shade(tm.vertex_normals)
    sc = pyrender.Scene(bg_color=[0.12, 0.13, 0.17, 1.0], ambient_light=[1.0, 1.0, 1.0])
    sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.6), pose=campose)
    col, _ = r.render(sc); return col

print(f"frames {len(V)} up_axis {up_ax} size {size:.2f}m", flush=True)
S = os.environ.get("SAMPLE")
if S:
    from PIL import Image
    Image.fromarray(frame(int(S))).save(out); print("sample ->", out); r.delete(); sys.exit()
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                       "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for i in range(len(V)):
    ff.stdin.write(frame(i).tobytes())
ff.stdin.close(); ff.wait(); r.delete(); print("wrote", out)
