#!/usr/bin/env python3
"""Analyze a MocoTrack solution (popB_*.sto): deviation of achieved coords from the reference
(.mot) per region + muscle activation. Vision: weaker F_max -> more deviation at high-demand frames.
Args: solution .sto path(s)."""
import sys, os, numpy as np

HERE = os.path.expanduser("~/shedance/osim")
MOT = os.path.join(HERE, "pop_ik_rad.mot")


def read_sto(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    return hdr, d


rh, rd = read_sto(MOT); rt = rd[:, 0]
ref = {rh[i]: rd[:, i] for i in range(1, len(rh))}


def region(nm):
    if any(k in nm for k in ("hip_", "knee_", "ankle_", "subtalar")): return "leg"
    if any(k in nm for k in ("shoulder_", "elv_angle", "elbow", "pro_sup")): return "arm"
    return "trunk"


for f in sys.argv[1:]:
    if not os.path.exists(f):
        print(f"{f}: missing"); continue
    hdr, d = read_sto(f); t = d[:, 0]
    val = {hdr[i].split("/")[-2]: i for i in range(len(hdr)) if hdr[i].endswith("/value")}
    reg = {"leg": [], "arm": [], "trunk": []}
    worst = []
    for cn, ci in val.items():
        if cn not in ref or "beta" in cn or "pelvis" in cn:
            continue
        dev = np.degrees(np.abs(d[:, ci] - np.interp(t, rt, ref[cn])))
        reg[region(cn)].append(dev.mean()); worst.append((cn, dev.mean(), dev.max()))
    allm = np.mean([m for v in reg.values() for m in v]) if any(reg.values()) else 0
    rs = " ".join(f"{r}={np.mean(v):.2f}" for r, v in reg.items() if v)
    print(f"{os.path.basename(f)}: t={t[0]:.2f}..{t[-1]:.2f} frames={len(t)} | overall dev={allm:.2f}deg | {rs}")
    for cn, m, mx in sorted(worst, key=lambda x: -x[1])[:4]:
        print(f"     {cn:18s} mean={m:.2f} max={mx:.2f} deg")
    # muscle activations (controls in /forceset/ not res_)
    mc = [i for i in range(len(hdr)) if "/forceset/" in hdr[i] and "res_" not in hdr[i] and "reserve" not in hdr[i]]
    if mc:
        A = np.abs(d[:, mc]); print(f"     muscle act: mean={A.mean():.3f} frac>0.05={np.mean(A>0.05):.2f} max={A.max():.2f}")
