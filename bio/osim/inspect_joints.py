#!/usr/bin/env python3
"""List joints, their coordinates, and locked status -> decide which joints to weld for Moco."""
import os
import opensim as o

m = o.Model(os.path.expanduser("~/shedance/osim/cyclistFullBodyMuscle.osim")); m.initSystem()
js = m.getJointSet(); cs = m.getCoordinateSet()
locked = {cs.get(i).getName(): cs.get(i).getDefaultLocked() for i in range(cs.getSize())}
print("total coords", cs.getSize(), " locked", sum(locked.values()))

# essential joints to KEEP free (legs, lumbar, glenohumeral+elbow, pelvis-ground)
KEEP_HINT = ("ground_pelvis", "hip_", "walker_knee", "knee_", "ankle_", "subtalar_",
             "L5_S1", "L4_L5", "L3_L4", "L2_L3", "L1_L2", "Abs", "flexext",
             "shoulder", "elbow", "radioulnar", "acromial", "GlenoHumeral", "unrot")
weld = []
print("=== joints ===")
for j in range(js.getSize()):
    J = js.get(j); nm = J.getName(); nc = J.numCoordinates()
    cnames = []
    anylock = False
    for k in range(nc):
        cn = J.get_coordinates(k).getName(); lk = locked.get(cn, False)
        cnames.append(cn + ("*" if lk else "")); anylock = anylock or lk
    note = ""
    if nc == 0:
        note = "(already weld)"
    elif anylock:
        note = "<-- has locked"
    print("  %-24s n=%d [%s] %s" % (nm, nc, ", ".join(cnames), note))
    if anylock:
        weld.append(nm)
print("=== joints with locked coords (must weld for Moco):", len(weld))
print(weld)
