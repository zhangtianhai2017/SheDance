#!/usr/bin/env python3
"""Analyze OpenSim Static Optimization output: muscle activations + reserve forces by region.

Decisive feasibility read: pelvis residuals expected large (no GRF); JOINT (leg/lumbar/arm)
reserves should be small if muscles produce the motion (= OpenSim's proper constraint handling
resolved the redundancy my hand-rolled solver couldn't). Uses _force.sto (actual forces N/N*m)
for reserves and _activation.sto for muscle activations.
"""
import sys, os, glob, numpy as np

HERE = os.path.expanduser("~/shedance/osim")
act_f = sorted(glob.glob(os.path.join(HERE, "*StaticOptimization_activation.sto")))
frc_f = sorted(glob.glob(os.path.join(HERE, "*StaticOptimization_force.sto")))


def read_sto(path):
    lines = open(path).read().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    hdr = lines[hi + 1].split("\t")
    data = np.array([l.split("\t") for l in lines[hi + 2:] if l.strip()], float)
    return hdr, data


def region(name):
    s = name.lower()
    if "pelvis" in s: return "pelvis(residual)"
    if any(k in s for k in ("hip", "knee", "ankle", "subtalar", "mtp")): return "leg"
    if any(k in s for k in ("l5_s1", "l4_l5", "l3_l4", "l2_l3", "l1_l2", "abs")): return "lumbar"
    if any(k in s for k in ("shoulder", "elv", "elbow", "pro_sup", "sternum")): return "arm/torso"
    return "other"


if act_f:
    hdr, data = read_sto(act_f[-1])
    musc = [c for c in hdr if c != "time" and not c.startswith("reserve")]
    res = [c for c in hdr if c.startswith("reserve")]
    A = np.abs(data[:, [hdr.index(c) for c in musc]])
    print(f"activation.sto: {data.shape[0]} frames, {len(musc)} muscles, {len(res)} reserves")
    print(f"MUSCLE activation: mean={A.mean():.3f}  frac>0.05={np.mean(A>0.05):.2f}  frac>0.5={np.mean(A>0.5):.3f}  max={A.max():.2f}")

if frc_f:
    hdr, data = read_sto(frc_f[-1])
    res = [c for c in hdr if c.lower().startswith("reserve")]
    print(f"\nforce.sto reserves: {len(res)}")
    groups = {}
    for c in res:
        f = np.abs(data[:, hdr.index(c)])
        groups.setdefault(region(c), []).append((c, f.mean(), f.max()))
    print("RESERVE force/torque by region (N / N*m):")
    for g in ("pelvis(residual)", "leg", "lumbar", "arm/torso", "other"):
        if g not in groups: continue
        items = groups[g]
        print(f"  {g:18s} n={len(items):3d}  mean|res|={np.mean([m for _,m,_ in items]):7.1f}  max|res|={max(mx for _,_,mx in items):7.1f}")
        for c, m, mx in sorted(items, key=lambda x: -x[1])[:3]:
            print(f"      {c.replace('reserve_',''):24s} mean={m:7.1f} max={mx:7.1f}")
    print("\n判定: 关节(leg/lumbar/arm)储备小 => 肌肉能产出该区动作; 仅 pelvis 残差大=缺GRF(已知,可后补)")
