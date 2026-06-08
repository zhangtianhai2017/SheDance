#!/usr/bin/env python3
"""Synthesize a 2nd, DIFFERENT input to stress-test workflow generality (no 2nd real video available).
We perturb exactly the axes the pipeline must adapt to WITHOUT re-tuning:
  - REVERSE the choreography      -> different contact timing / pose order (a genuinely different sequence)
  - RESAMPLE 30 -> 50 fps         -> the seconds-based windows must change (0.7s = 21 frames -> 35)
  - PLAY 1.3x FASTER              -> foot speeds rise -> the data-derived VTHR must rise
Source SCALE and global YAW are deliberately NOT used: angle-transfer is scale-invariant and a global
yaw is absorbed by the root, so neither would actually test anything. Run the workflow on this at FPS=50.
Usage: make_variant.py <in_track.npz> <out_track.npz>   Env: SPEED(=1.3) FPS_NEW(=50) FPS_OLD(=30)
"""
import os, sys, numpy as np
IN, OUT = sys.argv[1], sys.argv[2]
SPEED = float(os.environ.get("SPEED", "1.3")); FPS_NEW = float(os.environ.get("FPS_NEW", "50")); FPS_OLD = float(os.environ.get("FPS_OLD", "30"))
tr = np.load(IN, allow_pickle=True); d = {k: tr[k] for k in tr.files}
KP = np.asarray(tr["keypoints_3d"], float)[::-1]              # REVERSE choreography
T = KP.shape[1] if False else len(KP)
newT = int(round((len(KP) / FPS_OLD / SPEED) * FPS_NEW))     # frames at new fps, 1.3x faster
t_old = np.linspace(0.0, 1.0, len(KP)); t_new = np.linspace(0.0, 1.0, newT)
KPn = np.empty((newT, KP.shape[1], 3))
for j in range(KP.shape[1]):
    for c in range(3):
        KPn[:, j, c] = np.interp(t_new, t_old, KP[:, j, c])  # smooth temporal resample
d["keypoints_3d"] = KPn
idx = np.round(np.linspace(0, len(KP) - 1, newT)).astype(int)
for k in list(d.keys()):                                      # keep other T-first arrays consistent (reversed + resampled, nearest)
    a = d[k]
    if k != "keypoints_3d" and isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == len(KP):
        d[k] = np.asarray(a)[::-1][idx]
np.savez(OUT, **d)
print("variant: %d frames @30fps -> %d frames @%gfps (reversed, %gx faster) -> %s" % (len(KP), newT, FPS_NEW, SPEED, OUT))
