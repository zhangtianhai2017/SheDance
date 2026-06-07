#!/usr/bin/env python3
"""Remove monocular VERTICAL floor-drift so planted feet ride Z=0 (instead of floating ~20cm).
Single-global-offset alignment grounds only the one lowest instant; the source's per-frame vertical
depth drift then leaves the support foot floating. Here we estimate the drifting floor as a smoothed
LOWER-ENVELOPE of the lower toe height (window-min -> gaussian) and subtract it per frame: the support
foot rides 0, while brief excursions above the envelope (real jumps) are preserved. Vertical only;
joint_world_rot and XY untouched. Re-export the FBX from the output.

Usage: python foot_ground_vertical.py <in_joints_world.npz> <out_joints_world.npz> [win] [sigma]
"""
import sys, numpy as np
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
IN, OUT = sys.argv[1], sys.argv[2]
WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 15
SIG = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
z = np.load(IN, allow_pickle=True); d = {k: z[k] for k in z.files}
JW = z["joints_world"].astype(np.float32); N = [str(x) for x in z["joint_names"]]
L, R = N.index("left_foot"), N.index("right_foot")
lowerToe = np.minimum(JW[:, L, 2], JW[:, R, 2])                 # lower foot toe-joint height per frame
floor = gaussian_filter1d(minimum_filter1d(lowerToe, size=WIN, mode="nearest"), sigma=SIG, mode="nearest")
JW[:, :, 2] = JW[:, :, 2] - floor[:, None]                     # ride the drifting floor down to ~0
d["joints_world"] = JW
np.savez(OUT, **d)
print("vertical floor-drift removed (win=%d sigma=%.1f): floor est min %.3f max %.3f mean %.3f -> %s"
      % (WIN, SIG, floor.min(), floor.max(), floor.mean(), OUT))
