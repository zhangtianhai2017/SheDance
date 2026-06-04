#!/usr/bin/env python3
"""Analyze a MocoInverse solution .sto: muscle activations + reserve/residual usage by region.

The decisive feasibility read: pelvis residuals are expected large (no GRF -> reserves carry
body support), but JOINT (muscle-driven) reserves should be SMALL if muscles produce the motion.
Reserve force/torque = control * optimal_force (ModOpAddReserves optimal force, default 250 here).
"""
import sys, os, numpy as np

STO = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/shedance/osim/pop_moco_2.00_2.50.sto")
RES_OPT = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0

lines = open(STO).read().splitlines()
hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
hdr = lines[hi + 1].split("\t")
data = np.array([l.split("\t") for l in lines[hi + 2:] if l.strip()], float)
print(f"{os.path.basename(STO)}: {data.shape[0]} rows x {len(hdr)} cols")

cols = {n: i for i, n in enumerate(hdr)}
act_cols = [n for n in hdr if n.endswith("/activation")]
# reserves are CoordinateActuators added by ModOpAddReserves -> name contains 'reserve'
res_cols = [n for n in hdr if "reserve" in n.lower()]
print(f"muscle activation cols: {len(act_cols)}  reserve cols: {len(res_cols)}")

if act_cols:
    A = np.abs(data[:, [cols[c] for c in act_cols]])
    print(f"\nMUSCLE activations: mean={A.mean():.3f}  frac>0.05={np.mean(A>0.05):.2f}  max={A.max():.2f}")

# group reserves by body region from coordinate name
def region(name):
    s = name.lower()
    if "pelvis" in s: return "pelvis(residual)"
    if "hip" in s or "knee" in s or "ankle" in s or "subtalar" in s or "mtp" in s: return "leg"
    if any(k in s for k in ("l5_s1","l4_l5","l3_l4","l2_l3","l1_l2","abs","lumbar")): return "lumbar"
    if any(k in s for k in ("shoulder","elv","elbow","pro_sup","arm")): return "arm"
    return "other"

groups = {}
for c in res_cols:
    f = np.abs(data[:, cols[c]]) * RES_OPT      # reserve generalized force/torque
    groups.setdefault(region(c), []).append((c, f.mean(), f.max()))

print("\nRESERVE usage by region (force N / torque N*m = control x optimal_force):")
for g in ("pelvis(residual)", "leg", "lumbar", "arm", "other"):
    if g not in groups: continue
    items = groups[g]
    allmean = np.mean([m for _, m, _ in items]); allmax = max(mx for _, _, mx in items)
    print(f"  {g:18s} n={len(items):3d}  mean|res|={allmean:7.1f}  max|res|={allmax:7.1f}")
    for c, m, mx in sorted(items, key=lambda x: -x[1])[:4]:
        short = c.split("/")[-1] if "/" in c else c
        print(f"      {short:32s} mean={m:7.1f} max={mx:7.1f}")
print("\n判定: leg/lumbar/arm 储备小 => 肌肉能产出该区域动作(OpenSim 正确约束处理生效)")
