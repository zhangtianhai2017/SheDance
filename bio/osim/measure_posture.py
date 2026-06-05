#!/usr/bin/env python3
"""Report mean local rotation (deg) of upper-body joints in an SMPL-H npz, to see residual posture bias.
Arg: npz"""
import sys, numpy as np
P = np.load(sys.argv[1])["poses"]; b = P[:, :66].reshape(-1, 22, 3)
deb = {3, 6, 9, 12, 15}
for j, nm in [(3, "spine1"), (6, "spine2"), (9, "spine3"), (12, "neck"), (15, "head"),
              (13, "collarL"), (14, "collarR"), (16, "shoulderL"), (17, "shoulderR")]:
    m = np.degrees(b[:, j, :].mean(0)).round(1)
    tag = "debiased->~0" if j in deb else "NOT debiased"
    print(f"{nm:9s} mean[x,y,z]={m}   {tag}")
root = np.degrees(P[:, 0:3].mean(0)).round(1)
print(f"root      mean={root}   dev-from-upright[90,0,0]={(root - np.array([90., 0, 0])).round(1)}")
