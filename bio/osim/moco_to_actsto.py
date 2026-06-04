#!/usr/bin/env python3
"""Convert a MocoTrack solution (popB_*.sto, columns /forceset/<m>/activation) into an
SO-style activation .sto (bare muscle-name columns) that render_muscle.py can color from.
Args: in_popB.sto out_act.sto"""
import sys, numpy as np

IN, OUT = sys.argv[1], sys.argv[2]
L = open(IN).read().splitlines()
hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
hdr = L[hi + 1].split("\t")
data = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)

# pick activation state columns: /forceset/<muscle>/activation -> bare <muscle>
cols = [(i, h.split("/")[-2]) for i, h in enumerate(hdr) if h.endswith("/activation")]
tcol = hdr.index("time")
names = [nm for _, nm in cols]
T = data.shape[0]
with open(OUT, "w") as f:
    f.write("activation\nversion=1\n")
    f.write(f"nRows={T}\nnColumns={len(names)+1}\ninDegrees=no\nendheader\n")
    f.write("time\t" + "\t".join(names) + "\n")
    for r in range(T):
        row = [f"{data[r, tcol]:.5f}"] + [f"{data[r, i]:.5f}" for i, _ in cols]
        f.write("\t".join(row) + "\n")
print(f"wrote {OUT}: {T} frames, {len(names)} muscle activations")
