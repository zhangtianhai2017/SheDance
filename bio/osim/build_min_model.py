#!/usr/bin/env python3
"""Build the minimal CMC model (~149 muscles): weld ribs/thoracic/wrist/mtp, then KEEP only
muscles that actuate a leg or arm coordinate (remove the ~200 multi-segment spine muscles).
Lumbar/pelvis stay as free coords driven by reserve/residual actuators (torso follows the
reference, not rigid). Output: cyclist_min.osim — gait+arm scale, CMC-tractable.
"""
import os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclistFullBodyMuscle.osim")
OUT = os.path.join(HERE, "cyclist_min.osim")

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
limb = [cs.get(i) for i in range(cs.getSize())
        if any(k in cs.get(i).getName() for k in ("hip_", "knee_", "ankle_", "subtalar",
                                                  "shoulder_", "elv_angle", "elbow", "pro_sup"))]
print(f"after weld: DOF={cs.getSize()} muscles={m.getMuscles().getSize()} limb_coords={len(limb)}")

muscles = m.getMuscles()
remove = []
for i in range(muscles.getSize()):
    mu = muscles.get(i); mx = 0.0
    for c in limb:
        try:
            mx = max(mx, abs(mu.computeMomentArm(state, c)))
        except Exception:
            pass
    if mx < 1e-3:
        remove.append(mu.getName())
fs = m.updForceSet()
for nm in remove:
    idx = fs.getIndex(nm)
    if idx >= 0:
        fs.remove(idx)
m.finalizeConnections(); m.initSystem()
print(f"removed {len(remove)} non-limb muscles -> kept {m.getMuscles().getSize()}")
m.printToXML(OUT)
print("wrote", OUT)
