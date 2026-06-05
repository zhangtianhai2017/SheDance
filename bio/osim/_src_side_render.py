#!/usr/bin/env python3
"""Throwaway: render VedioTo3D source mesh at given frames + side angle, individual pngs, to compare
faithfulness against OUR humanoid. Args: outdir az frame...   (camera frame Y-down -> world Z-up)."""
import os, sys, numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from PIL import Image
OUTDIR = sys.argv[1]; AZ = float(sys.argv[2]); frames = [int(x) for x in sys.argv[3:]]
D = os.path.expanduser("~/shedance/vedio_dance1/sam3d_body_dance1_pkg/data")
z = np.load(D + "/tracks/track_00_smooth.npz", allow_pickle=True)
V = np.asarray(z["vertices"], float); F = np.asarray(np.load(D + "/faces.npy"))
V = np.stack([V[:, :, 0], V[:, :, 2], -V[:, :, 1]], -1)        # world Z-up
az = np.radians(AZ); el = np.radians(8.0); up = np.array([0, 0, 1.0])
FLESH = np.array([0.90, 0.66, 0.58])
def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f); s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye; return M
dir3 = np.array([np.cos(az), np.sin(az), 0]) + up * np.sin(el); dir3 /= np.linalg.norm(dir3)
L1 = np.array([0.3, 1.0, 0.8]); L1 /= np.linalg.norm(L1)
W, Hh = 480, 760; r = pyrender.OffscreenRenderer(W, Hh)
for fr in frames:
    ctr = V[fr].mean(0); size = float((V[fr].max(0) - V[fr].min(0)).max())
    tm = trimesh.Trimesh(V[fr], F, process=True); N = tm.vertex_normals; sh = 0.3 + 0.8 * np.clip(N @ L1, 0, 1)
    tm.visual.vertex_colors = (np.concatenate([np.clip(FLESH * sh[:, None], 0, 1), np.ones((len(N), 1))], 1) * 255).astype(np.uint8)
    sc = pyrender.Scene(bg_color=[0.12, 0.13, 0.17, 1.0], ambient_light=[1, 1, 1])
    sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True)); sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.6), pose=look_at(ctr + dir3 * size * 2.2, ctr, up))
    Image.fromarray(r.render(sc)[0]).save(f"{OUTDIR}/_src{fr}.png")
r.delete(); print("saved source frames", frames, flush=True)
