#!/usr/bin/env python3
"""B via OpenSim CMC (Computed Muscle Control): forward muscle-driven tracking with the
field's stability/consistency handling. Muscles (capacity = F_max) try to track the reference;
where they saturate, tracking degrades -> deviation = the embodied imitator. F_max = character.

Builds: CMC model (pruned + reserve/residual actuators), CMC_Tasks (track each coordinate),
runs CMCTool on a short segment. Args: [t0 t1 fmax].
"""
import sys, os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MOT = os.path.join(HERE, "pop.mot")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 7.5
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 7.8
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
PRUNED = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "cyclist_min.osim")

# --- build CMC model: pruned + reserve/residual CoordinateActuators ---
model = osim.Model(PRUNED)
if FMAX != 1.0:
    for i in range(model.getMuscles().getSize()):
        mu = model.getMuscles().get(i)
        mu.setMaxIsometricForce(mu.getMaxIsometricForce() * FMAX)
cs = model.getCoordinateSet()
coords = [cs.get(i).getName() for i in range(cs.getSize())]
for nm in coords:
    if "beta" in nm:
        continue                         # knee_angle_*_beta is constraint-coupled (dependent) -> no actuator
    ca = osim.CoordinateActuator(nm); ca.setName("res_" + nm)
    is_limb = any(k in nm for k in ("hip_", "knee_", "ankle_", "subtalar", "shoulder_", "elv_angle", "elbow", "pro_sup"))
    of = 150.0 if is_limb else 1500.0   # limb: moderate (muscles lead); pelvis+lumbar(no muscles): strong reserve-driven
    ca.setOptimalForce(of); ca.setMinControl(-1e4); ca.setMaxControl(1e4)
    model.addForce(ca)
model.finalizeConnections(); model.initSystem()
CMCMODEL = os.path.join(HERE, f"cmc_model_f{FMAX:.2f}.osim")
model.printToXML(CMCMODEL)

# --- CMC task set: track every coordinate ---
taskset = osim.CMC_TaskSet()
for nm in coords:
    if "beta" in nm:
        continue                         # dependent (coupled) coordinate -> no tracking task
    task = osim.CMC_Joint(nm)
    task.setName(nm)
    task.setActive(True, False, False)
    task.setKP(100.0); task.setKV(20.0)
    task.setWeight(1.0, 1.0, 1.0)
    taskset.cloneAndAppend(task)
TASKS = os.path.join(HERE, f"cmc_tasks_f{FMAX:.2f}.xml")
taskset.printToXML(TASKS)
print(f"CMC model muscles={model.getMuscles().getSize()} coords={len(coords)} tasks written", flush=True)

# --- CMCTool ---
cmc = osim.CMCTool()
cmc.setName(f"popCMC_f{FMAX:.2f}")
cmc.setModelFilename(CMCMODEL)
cmc.setDesiredKinematicsFileName(MOT)
cmc.setTaskSetFileName(TASKS)
cmc.setInitialTime(t0); cmc.setFinalTime(t1)
cmc.setLowpassCutoffFrequency(6.0)
cmc.setTimeWindow(0.01)
cmc.setUseFastTarget(False)   # slow target = soft tracking (minimize error) -> allows deviation where muscles can't (the vision) + more robust
cmc.setResultsDir(HERE)
SETUP = os.path.join(HERE, f"cmc_setup_f{FMAX:.2f}.xml")
cmc.printToXML(SETUP)
print("running CMC (reloaded from setup)...", flush=True)
osim.CMCTool(SETUP).run()
print("CMC done -> outputs in", HERE, flush=True)
