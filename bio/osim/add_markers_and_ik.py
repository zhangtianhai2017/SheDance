#!/usr/bin/env python3
"""Add joint-center markers to the OpenSim model (at body origins matching pop.trc), then run
Inverse Kinematics -> coordinates .mot in the model's OWN convention (cures the name-mapping
convention bugs). IK fits each model body origin to the corresponding MuJoCo-derived marker.
"""
import os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
DANCE = os.environ.get("DANCE", "pop")
MODEL = os.path.join(HERE, "cyclist_min.osim")
TRC = os.path.join(HERE, f"{DANCE}.trc")
MODELM = os.path.join(HERE, "cyclist_min_markers.osim")
OUTMOT = os.path.join(HERE, f"{DANCE}_ik.mot")
MARK = ["pelvis", "lumbar5", "femur_r", "femur_l", "tibia_r", "tibia_l",
        "talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l",
        "humerus_r", "humerus_l", "ulna_r", "ulna_l", "radius_r", "radius_l"]

model = osim.Model(MODEL); model.initSystem()
model.updMarkerSet().clearAndDestroy()   # drop inherited gait markers (some on welded bodies -> dangling)
for nm in MARK:
    body = model.getBodySet().get(nm)
    mk = osim.Marker(nm, body, osim.Vec3(0, 0, 0))
    model.addMarker(mk)
model.finalizeConnections(); model.initSystem()
model.printToXML(MODELM)
print(f"added {len(MARK)} markers -> {MODELM}")

# IK task set (weight each marker)
taskset = osim.IKTaskSet()
for nm in MARK:
    t = osim.IKMarkerTask(); t.setName(nm); t.setApply(True); t.setWeight(1.0)
    taskset.cloneAndAppend(t)

# read trc time range
trc = osim.MarkerData(TRC)
t0, t1 = trc.getStartFrameTime(), trc.getLastFrameTime()
print(f"trc time {t0:.2f}..{t1:.2f}")

ik = osim.InverseKinematicsTool()
ik.setModel(model)
ik.set_IKTaskSet(taskset)
ik.setMarkerDataFileName(TRC)
ik.setStartTime(t0); ik.setEndTime(t1)
ik.setOutputMotionFileName(OUTMOT)
ik.set_report_errors(True)
print("running IK...", flush=True)
ik.run()
print("IK done ->", OUTMOT, flush=True)
