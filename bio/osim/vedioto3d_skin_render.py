#!/usr/bin/env python3
"""Render VedioTo3D's OWN skin mesh (the real dancer's body surface) for a dance -> complete skin video.
Uses per-frame vertices + shared faces directly (no SMPL, no retarget). Camera frame (Y-down) -> world
Z-up [X,Z,-Y]; baked Lambert shading into vertex colors (GL diffuse is dead in this pyrender build);
per-frame body-centered camera.  Args: track_npz faces_npy out_mp4 [fps]   Env: CAM_AZ, CAM_EL."""
import os, sys, subprocess, numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
TRACK, FACES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
V = np.asarray(np.load(TRACK, allow_pickle=True)["vertices"], float)   # (T,18439,3) camera meters
F = np.asarray(np.load(FACES))                                         # (36874,3)
V = np.stack([V[:, :, 0], V[:, :, 2], -V[:, :, 1]], -1)                # camera (Y-down) -> world Z-up

exts = V.max(1) - V.min(1)
up_ax = int(np.argmax(np.median(exts, 0))); size = float(np.median(exts[:, up_ax]))
up = np.zeros(3); up[up_ax] = 1.0
horiz = [a for a in range(3) if a != up_ax]
az = np.radians(float(os.environ.get("CAM_AZ", "0"))); el = np.radians(float(os.environ.get("CAM_EL", "8")))
dir3 = np.zeros(3); dir3[horiz[0]] = np.cos(az); dir3[horiz[1]] = np.sin(az); dir3 += up * np.sin(el)
dir3 /= np.linalg.norm(dir3)
side = np.cross(dir3, up); side /= np.linalg.norm(side)
L1 = 0.25 * dir3 + 1.0 * side + 0.75 * up; L1 /= np.linalg.norm(L1)
L2 = 0.5 * dir3 - 1.0 * side; L2 /= np.linalg.norm(L2)
FLESH = np.array([0.90, 0.66, 0.58])
def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f); s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye; return M
def shade(N):
    s = 0.28 + 0.80 * np.clip(N @ L1, 0, 1) + 0.22 * np.clip(N @ L2, 0, 1)
    c = np.clip(FLESH[None, :] * s[:, None], 0, 1)
    return (np.concatenate([c, np.ones((len(c), 1))], 1) * 255).astype(np.uint8)
W, Hh = 480, 760
r = pyrender.OffscreenRenderer(W, Hh)
print(f"frames {len(V)} up_axis {up_ax} size {size:.2f}m", flush=True)
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                       "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for i in range(len(V)):
    ctr = V[i].mean(0); campose = look_at(ctr + dir3 * size * 2.2, ctr, up)
    tm = trimesh.Trimesh(V[i], F, process=True)
    tm.visual.vertex_colors = shade(tm.vertex_normals)
    sc = pyrender.Scene(bg_color=[0.12, 0.13, 0.17, 1.0], ambient_light=[1.0, 1.0, 1.0])
    sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.6), pose=campose)
    ff.stdin.write(r.render(sc)[0].tobytes())
ff.stdin.close(); ff.wait(); r.delete()
print(f"wrote {OUT}", flush=True)
