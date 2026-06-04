#!/usr/bin/env python3
"""How low can muscle count go? Compute, per muscle, which coordinates it spans (moment arm
>1mm). Then for several 'kept coordinate' sets (rigid torso / one lumbar DOF / full lumbar),
count muscles that actuate at least one kept coord. Muscles spanning ONLY dropped coords are
removable. Robust (no locking/welding side effects). Rigid torso (leg+arm only) = the floor.
"""
import os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclistFullBodyMuscle.osim")

WELD = ["mtp_r", "mtp_l", "radius_hand_r", "radius_hand_l", "r1R_sterR_jnt", "T1_head_neck"]
for i in range(1, 13):
    WELD += [f"T{i}_r{i}R_CVjnt", f"T{i}_r{i}L_CVjnt"]
for a, b in [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5"), ("T5", "T6"), ("T6", "T7"),
             ("T7", "T8"), ("T8", "T9"), ("T9", "T10"), ("T10", "T11"), ("T11", "T12"), ("T12", "L1")]:
    WELD.append(f"{a}_{b}_IVDjnt")

weldvec = osim.StdVectorString()
for w in WELD:
    weldvec.append(w)
mp = osim.ModelProcessor(MODEL); mp.append(osim.ModOpReplaceJointsWithWelds(weldvec))
m = mp.process(); state = m.initSystem(); m.realizePosition(state)
cs = m.getCoordinateSet()
coords = [cs.get(i).getName() for i in range(cs.getSize())]


def grp(nm):
    if any(k in nm for k in ("hip_", "knee_", "ankle_", "subtalar")): return "LEG"
    if any(k in nm for k in ("shoulder_", "elv_angle", "elbow", "pro_sup")): return "ARM"
    if nm.startswith("L5_S1"): return "L5S1"
    if any(nm.startswith(p) for p in ("Abs", "L1_L2", "L2_L3", "L3_L4", "L4_L5")): return "LUMBAR_REST"
    if "pelvis" in nm: return "PELVIS"
    return "OTHER"


cg = {c: grp(c) for c in coords}
print("coord groups:", {g: sum(1 for c in coords if cg[c] == g) for g in set(cg.values())})

# per-muscle spanned coords
muscles = m.getMuscles()
span = {}
for i in range(muscles.getSize()):
    mu = muscles.get(i); nm = mu.getName(); s = set()
    for ci in range(cs.getSize()):
        try:
            if abs(mu.computeMomentArm(state, cs.get(ci))) >= 1e-3:
                s.add(cg[coords[ci]])
        except Exception:
            pass
    span[nm] = s

KEEPSETS = [
    ("rigid torso (LEG+ARM only)", {"LEG", "ARM"}),
    ("+1 lumbar DOF (L5_S1)", {"LEG", "ARM", "L5S1"}),
    ("+L5_S1 +Abs region", {"LEG", "ARM", "L5S1", "LUMBAR_REST"}),  # = full lumbar here
]
print(f"\nstart muscles (after base weld): {len(span)}")
for name, keep in KEEPSETS:
    kept = [nm for nm, s in span.items() if s & keep]
    leg = sum(1 for nm in kept if "LEG" in span[nm] and "ARM" not in span[nm])
    arm = sum(1 for nm in kept if "ARM" in span[nm] and "LEG" not in span[nm])
    print(f"  keep {name:32s} -> {len(kept):4d} muscles")
