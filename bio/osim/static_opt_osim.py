#!/usr/bin/env python3
"""OpenSim Static Optimization fallback (per-frame muscle redundancy, proper constraints).

Unlike Moco, SO handles locked coordinates natively (no welding) and scales better to large
muscle counts (per-frame QP). No GRF -> add reserve/residual CoordinateActuators:
pelvis residuals strong (carry body support), joint reserves moderate. Output: activations .sto
+ reserve forces, which we read by region to judge muscle feasibility.

Args: [t0 t1]  defaults 2.0 2.5
"""
import sys, os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
DANCE = os.environ.get("DANCE", "pop")
FMAX = float(os.environ.get("FMAX_SCALE", "1.0"))         # <1.0 = weaker body (effort/vision version)
TAG = "" if abs(FMAX - 1.0) < 1e-9 else f"_f{int(round(FMAX*100)):02d}"
MODEL = os.path.join(HERE, "cyclist_min.osim")
MOT = os.path.join(HERE, f"{DANCE}_ik.mot")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

model = osim.Model(MODEL)

# --- add reserve / residual actuators on unconstrained coordinates ---
cs = model.getCoordinateSet()
for i in range(cs.getSize()):
    c = cs.get(i)
    if c.getDefaultLocked():
        continue
    nm = c.getName()
    ca = osim.CoordinateActuator(nm)
    ca.setName("reserve_" + nm)
    s = nm.lower()
    if "pelvis" in s:
        of = 1000.0          # residuals cheap: must carry body support (no GRF available)
    else:
        of = 1.0             # joint reserves EXPENSIVE: force muscles to do the work
    ca.setOptimalForce(of)
    ca.setMinControl(-1e4); ca.setMaxControl(1e4)
    model.addForce(ca)

if abs(FMAX - 1.0) > 1e-9:                # effort version: weaker muscles -> higher activation to track
    mus = model.getMuscles()
    for i in range(mus.getSize()):
        mu = mus.get(i); mu.setMaxIsometricForce(mu.getMaxIsometricForce() * FMAX)

model.initSystem()
print(f"FMAX_SCALE={FMAX} coords={cs.getSize()} muscles={model.getMuscles().getSize()} forces={model.getForceSet().getSize()}", flush=True)
MODELRES = os.path.join(HERE, f"{DANCE}{TAG}_reserves.osim")
model.printToXML(MODELRES)

# --- Static Optimization via AnalyzeTool loaded from FILE (robust path) ---
so = osim.StaticOptimization()
so.setStartTime(t0); so.setEndTime(t1)
so.setUseModelForceSet(True)
so.setActivationExponent(2.0)
so.setUseMusclePhysiology(True)

tool = osim.AnalyzeTool()
tool.setName(f"{DANCE}{TAG}_so")
tool.setModelFilename(MODELRES)
tool.setInitialTime(t0); tool.setFinalTime(t1)
tool.setCoordinatesFileName(MOT)
_extl = os.path.join(HERE, f"{DANCE}_grf_extloads.xml")
if os.path.exists(_extl):
    tool.setExternalLoadsFileName(_extl)   # GRF -> physiological leg muscles, zero pelvis residual
tool.setLowpassCutoffFrequency(6.0)
tool.setResultsDir(HERE)
tool.getAnalysisSet().cloneAndAppend(so)
SETUP = os.path.join(HERE, "so_setup.xml")
tool.printToXML(SETUP)
print("running Static Optimization (reloaded from setup XML)...", flush=True)
tool2 = osim.AnalyzeTool(SETUP)   # reload from XML -> triggers full model+coords->states load
tool2.run()
print("done. outputs in", HERE, "(pop_so_StaticOptimization_activation.sto / _force.sto)", flush=True)
