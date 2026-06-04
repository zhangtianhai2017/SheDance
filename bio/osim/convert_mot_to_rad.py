#!/usr/bin/env python3
"""Convert IK output pop_ik.mot (inDegrees=yes) -> pop_ik_rad.mot (inDegrees=no) for Moco.
Rotational coords deg->rad; translations (pelvis_tx/ty/tz) unchanged."""
import os, numpy as np
HERE = os.path.expanduser("~/shedance/osim")
IN = os.path.join(HERE, "pop_ik.mot"); OUT = os.path.join(HERE, "pop_ik_rad.mot")
TRANS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}

L = open(IN).read().splitlines()
hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
hdr = L[hi + 1].split("\t")
data = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
for j, nm in enumerate(hdr):
    if j == 0 or nm in TRANS:
        continue
    data[:, j] = np.radians(data[:, j])
T = data.shape[0]
with open(OUT, "w") as f:
    f.write("pop_ik_rad\nversion=1\n")
    f.write(f"nRows={T}\nnColumns={len(hdr)}\ninDegrees=no\nendheader\n")
    f.write("\t".join(hdr) + "\n")
    for i in range(T):
        f.write("\t".join(f"{v:.6f}" for v in data[i]) + "\n")
print(f"wrote {OUT}: {T} rows, {len(hdr)} cols (rad)")
