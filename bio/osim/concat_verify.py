#!/usr/bin/env python3
"""Concatenate parallel SO chunk outputs (discard overlap) + verify chunking is LOSSLESS
(a boundary frame solved in two adjacent chunks must match -> proves per-frame independence).
Args: N (chunk count). Chunk math must match run_parallel_so.sh."""
import sys, os, glob, numpy as np
HERE = os.path.expanduser("~/shedance/osim")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SPAN0, SPAN1, OV = 0.3, 59.7, 0.2
W = (SPAN1 - SPAN0) / N


def read_sto(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    return hdr, d


chunks = []
missing = []
for i in range(N):
    f = os.path.join(HERE, f"c{i}_StaticOptimization_activation.sto")
    if not os.path.exists(f):
        missing.append(i); continue
    hdr, d = read_sto(f)
    chunks.append((i, hdr, d))
if missing:
    print(f"WARNING: missing chunks {missing} (killed by thermal auto-reduce or failed)")

# verify boundary consistency (adjacent chunks' overlap frame must match)
maxdiff = 0.0; ncmp = 0
cmap = {i: (hdr, d) for i, hdr, d in chunks}
for i in range(N - 1):
    if i not in cmap or i + 1 not in cmap:
        continue
    tb = SPAN0 + (i + 1) * W
    (h0, d0), (h1, d1) = cmap[i], cmap[i + 1]
    k0 = np.argmin(np.abs(d0[:, 0] - tb)); k1 = np.argmin(np.abs(d1[:, 0] - tb))
    if abs(d0[k0, 0] - d1[k1, 0]) < 1e-4:   # same time frame present in both
        diff = np.abs(d0[k0, 1:] - d1[k1, 1:]).max()
        maxdiff = max(maxdiff, diff); ncmp += 1

# concat: keep each chunk's core range [core0, core1)
rows = []
hdr_ref = chunks[0][1] if chunks else None
for i, hdr, d in chunks:
    c0 = SPAN0 + i * W; c1 = SPAN0 + (i + 1) * W
    inc = (d[:, 0] >= c0 - 1e-6) & (d[:, 0] < c1 - 1e-6 if i < N - 1 else d[:, 0] <= c1 + 1e-6)
    rows.append(d[inc])
alld = np.vstack(rows)
alld = alld[np.argsort(alld[:, 0])]
OUT = os.path.join(HERE, "pop60_parallel_activation.sto")
with open(OUT, "w") as f:
    f.write("activation\nversion=1\n")
    f.write(f"nRows={alld.shape[0]}\nnColumns={alld.shape[1]}\ninDegrees=no\nendheader\n")
    f.write("\t".join(hdr_ref) + "\n")
    for r in alld:
        f.write("\t".join(f"{v:.6f}" for v in r) + "\n")
print(f"concatenated {alld.shape[0]} frames -> {OUT}")
print(f"LOSSLESS check: {ncmp} boundaries compared, max activation diff = {maxdiff:.2e} (≈0 = exact chunking)")
