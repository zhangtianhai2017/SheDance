#!/usr/bin/env python3
"""Prune the cyclistFullBodyMuscle model for MocoTrack feasibility.

(1) Weld locked/irrelevant joints (ribs, fine thoracic, neck, wrist, mtp, sternum).
(2) Remove 'dead' muscles: those with ~0 moment arm on EVERY remaining (free) coordinate
    (i.e. muscles that only spanned welded segments). Keeps leg/lumbar/shoulder/elbow muscles.
Output: cyclist_pruned.osim — far fewer muscles -> Moco becomes tractable.
"""
import os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclistFullBodyMuscle.osim")
OUT = os.path.join(HERE, "cyclist_pruned.osim")

WELD = ["mtp_r", "mtp_l", "radius_hand_r", "radius_hand_l", "r1R_sterR_jnt", "T1_head_neck"]
for i in range(1, 13):
    WELD += [f"T{i}_r{i}R_CVjnt", f"T{i}_r{i}L_CVjnt"]
for a, b in [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5"), ("T5", "T6"), ("T6", "T7"),
             ("T7", "T8"), ("T8", "T9"), ("T9", "T10"), ("T10", "T11"), ("T11", "T12"), ("T12", "L1")]:
    WELD.append(f"{a}_{b}_IVDjnt")

weldvec = osim.StdVectorString()
for w in WELD:
    weldvec.append(w)
mp = osim.ModelProcessor(MODEL)
mp.append(osim.ModOpReplaceJointsWithWelds(weldvec))
model = mp.process()
state = model.initSystem()
model.realizePosition(state)

cs = model.getCoordinateSet()
coords = [cs.get(i) for i in range(cs.getSize())]
print(f"after weld: coords={len(coords)} muscles={model.getMuscles().getSize()}")

muscles = model.getMuscles()
dead = []
for i in range(muscles.getSize()):
    mu = muscles.get(i)
    maxma = 0.0
    for c in coords:
        try:
            maxma = max(maxma, abs(mu.computeMomentArm(state, c)))
        except Exception:
            pass
    if maxma < 1e-3:           # < 1 mm on every free coord -> dead
        dead.append(mu.getName())

fs = model.updForceSet()
for nm in dead:
    idx = fs.getIndex(nm)
    if idx >= 0:
        fs.remove(idx)
model.finalizeConnections()
model.initSystem()
print(f"removed {len(dead)} dead muscles -> kept {model.getMuscles().getSize()}")
model.printToXML(OUT)
print("wrote", OUT)
