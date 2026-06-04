#!/usr/bin/env python3
"""Build a ~60s test motion by ping-ponging the clean 12s pop_ik.mot (continuous at turns).
For PERFORMANCE testing of parallel SO (content = repeated Pop; per-frame work is real)."""
import os, numpy as np
HERE = os.path.expanduser("~/shedance/osim")
IN = os.path.join(HERE, "pop_ik.mot"); OUT = os.path.join(HERE, "pop60.mot")
TARGET = 6000   # frames @100Hz = 60s

L = open(IN).read().splitlines()
hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
hdr = L[hi + 1]
data = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
fwd = data[:, 1:]                              # drop time col
cycle = np.vstack([fwd, fwd[-2:0:-1]])         # ping-pong: forward + reversed (smooth turn)
reps = int(np.ceil(TARGET / len(cycle)))
allf = np.vstack([cycle] * reps)[:TARGET]
T = len(allf)
t = (np.arange(T) / 100.0)[:, None]
out = np.hstack([t, allf])
with open(OUT, "w") as f:
    f.write("pop60\nversion=1\n")
    f.write(f"nRows={T}\nnColumns={out.shape[1]}\ninDegrees=yes\nendheader\n")
    f.write(hdr + "\n")
    for r in range(T):
        f.write("\t".join(f"{v:.6f}" for v in out[r]) + "\n")
print(f"wrote {OUT}: {T} frames = {T/100:.1f}s")
