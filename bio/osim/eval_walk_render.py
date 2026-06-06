#!/usr/bin/env python3
"""Foot-lock eval: WORLD-FIXED camera + checker GROUND so foot-vs-floor sliding is visible (the normal
renders hide it: body-centered / camera-follows-body). Renders one motion; call twice + hstack to compare.
Args: npz out  [fps]   Env: CAM_AZ, CAM_EL, SAMPLE=<frame> (single PNG)."""
import os, sys, subprocess, numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
import general_motion_retargeting.utils.smpl as _sm
_o = _sm.SMPLH_Parser.__init__
def _i(self, *a, **k): k.setdefault("num_betas", 16); _o(self, *a, **k)
_sm.SMPLH_Parser.__init__ = _i
from general_motion_retargeting.utils.smpl import load_smplh_file
npz, out = sys.argv[1], sys.argv[2]; fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MODELS = os.path.expanduser("~/shedance/smpl_models")
sd, bm, so, h = load_smplh_file(npz, MODELS)
V = np.asarray(so.vertices, float); F = np.asarray(bm.faces)
exts = V.max(1) - V.min(1); up_ax = int(np.argmax(np.median(exts, 0))); horiz = [a for a in range(3) if a != up_ax]
up = np.zeros(3); up[up_ax] = 1.0
level = float(V[:, :, up_ax].min())                         # ground = lowest vertex over the dance
bc = V.mean(1); gctr = bc.mean(0).copy(); gctr[up_ax] = level + 0.9
travel = max(np.ptp(bc[:, horiz[0]]), np.ptp(bc[:, horiz[1]])); fit = max(travel, 1.8) * 1.35
az = np.radians(float(os.environ.get("CAM_AZ", "5"))); el = np.radians(float(os.environ.get("CAM_EL", "10")))
dir3 = np.zeros(3); dir3[horiz[0]] = np.cos(az); dir3[horiz[1]] = np.sin(az); dir3 += up * np.sin(el); dir3 /= np.linalg.norm(dir3)
def look_at(eye, tgt, u):
    f = tgt - eye; f /= np.linalg.norm(f); s = np.cross(f, u); s /= np.linalg.norm(s); uu = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = uu; M[:3, 2] = -f; M[:3, 3] = eye; return M
campose = look_at(gctr + dir3 * fit * 1.4, gctr, up)
side = np.cross(dir3, up); side /= np.linalg.norm(side)
L1 = 0.25 * dir3 + 1.0 * side + 0.75 * up; L1 /= np.linalg.norm(L1)
FLESH = np.array([1.0, 0.80, 0.72])
def shade(N): s = 0.85 + 0.45 * np.clip(N @ L1, 0, 1); return (np.concatenate([np.clip(FLESH * s[:, None], 0, 1), np.ones((len(N), 1))], 1) * 255).astype(np.uint8)
def make_ground(n=26):
    a, b = horiz; us = np.linspace(gctr[a] - fit, gctr[a] + fit, n + 1); vs = np.linspace(gctr[b] - fit, gctr[b] + fit, n + 1)
    verts = []; faces = []; cols = []; vi = 0
    for i in range(n):
        for j in range(n):
            def P(u, v): p = np.zeros(3); p[a] = u; p[b] = v; p[up_ax] = level; return p
            verts += [P(us[i], vs[j]), P(us[i + 1], vs[j]), P(us[i + 1], vs[j + 1]), P(us[i], vs[j + 1])]
            faces += [[vi, vi + 1, vi + 2], [vi, vi + 2, vi + 3]]
            c = [0.74, 0.76, 0.82] if (i + j) % 2 == 0 else [0.44, 0.46, 0.52]; cols += [c] * 4; vi += 4
    tm = trimesh.Trimesh(np.array(verts), np.array(faces), process=False); tm.visual.vertex_colors = (np.array(cols) * 255).astype(np.uint8); return tm
ground = make_ground()
W, Hh = 420, 640; r = pyrender.OffscreenRenderer(W, Hh)
def frame(i):
    tm = trimesh.Trimesh(V[i], F, process=True); tm.visual.vertex_colors = shade(tm.vertex_normals)
    sc = pyrender.Scene(bg_color=[0.10, 0.11, 0.14, 1.0], ambient_light=[1, 1, 1])
    sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True)); sc.add(pyrender.Mesh.from_trimesh(ground, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.6), pose=campose); return r.render(sc)[0]
print(f"frames {len(V)} up_ax {up_ax} travel {travel:.2f}m ground {level:.2f}", flush=True)
S = os.environ.get("SAMPLE")
if S:
    from PIL import Image
    Image.fromarray(frame(int(S))).save(out); print("sample ->", out); r.delete(); sys.exit()
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-framerate", str(fps),
                       "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for i in range(len(V)): ff.stdin.write(frame(i).tobytes())
ff.stdin.close(); ff.wait(); r.delete(); print("wrote", out, flush=True)
