#!/usr/bin/env python3
"""Concat parallel-SO chunk activations for a dance (drop overlap). Args: prefix T0 T1 N out_sto"""
import sys, os, numpy as np
HERE = os.path.expanduser("~/shedance/osim")
pfx, T0, T1, N, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
W = (T1 - T0) / N


def read(p):
    L = open(p).read().splitlines(); hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    return L[hi + 1].split("\t"), np.array([x.split("\t") for x in L[hi + 2:] if x.strip()], float)


rows, hdr = [], None
for i in range(N):
    f = os.path.join(HERE, f"{pfx}c{i}_StaticOptimization_activation.sto")
    if not os.path.exists(f):
        print("missing chunk", i); continue
    h, d = read(f); hdr = hdr or h
    c0, c1 = T0 + i * W, T0 + (i + 1) * W
    inc = (d[:, 0] >= c0 - 1e-6) & ((d[:, 0] < c1 - 1e-6) if i < N - 1 else (d[:, 0] <= c1 + 1e-6))
    rows.append(d[inc])
alld = np.vstack(rows); alld = alld[np.argsort(alld[:, 0])]
with open(out, "w") as f:
    f.write("activation\nversion=1\n")
    f.write(f"nRows={alld.shape[0]}\nnColumns={alld.shape[1]}\ninDegrees=no\nendheader\n")
    f.write("\t".join(hdr) + "\n")
    for r in alld:
        f.write("\t".join(f"{v:.6f}" for v in r) + "\n")
print(f"concat {alld.shape[0]} frames -> {out}")
