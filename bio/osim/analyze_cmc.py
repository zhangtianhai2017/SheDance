#!/usr/bin/env python3
"""Compare CMC F_max archetypes: deviation of the achieved motion (states.sto coord values)
from the reference (.mot). The 'embodied imitator' vision -> weaker body (lower F_max) should
deviate MORE, especially at high-demand joints. Args: list of fmax values (default 0.7 1.0 2.5).
"""
import sys, os, glob, numpy as np

HERE = os.path.expanduser("~/shedance/osim")
MOT = os.path.join(HERE, "pop.mot")
fmaxes = [float(x) for x in sys.argv[1:]] or [0.7, 1.0, 2.5]


def read_sto(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    return hdr, d


# reference
rh, rd = read_sto(MOT)
rt = rd[:, 0]
ref = {rh[i]: rd[:, i] for i in range(1, len(rh))}


def region(nm):
    if any(k in nm for k in ("hip_", "knee_", "ankle_", "subtalar")): return "leg"
    if any(k in nm for k in ("shoulder_", "elv_angle", "elbow", "pro_sup")): return "arm"
    return "trunk/other"


print(f"{'fmax':>5} {'frames':>7} {'t0..t1':>12} {'overall_dev(deg)':>16}   per-region mean dev (deg)")
for fm in fmaxes:
    f = os.path.join(HERE, f"popCMC_f{fm:.2f}_states.sto")
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        print(f"{fm:5.1f}  (no/empty states yet)"); continue
    hdr, d = read_sto(f)
    t = d[:, 0]
    valcols = {hdr[i].split("/")[-2]: i for i in range(len(hdr)) if hdr[i].endswith("/value")}
    regdev = {"leg": [], "arm": [], "trunk/other": []}
    alld = []
    for cn, ci in valcols.items():
        if cn not in ref or "beta" in cn or "pelvis" in cn:
            continue
        ach = d[:, ci]
        refi = np.interp(t, rt, ref[cn])
        dev = np.degrees(np.abs(ach - refi))
        alld.append(dev.mean()); regdev[region(cn)].append(dev.mean())
    od = np.mean(alld) if alld else 0
    rs = " ".join(f"{r}={np.mean(v):.2f}" for r, v in regdev.items() if v)
    print(f"{fm:5.1f} {len(t):7d} {t[0]:.2f}..{t[-1]:.2f} {od:16.2f}   {rs}")
