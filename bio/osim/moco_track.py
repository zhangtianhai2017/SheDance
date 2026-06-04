#!/usr/bin/env python3
"""B = capacity-limited tracking (MocoTrack): the simulated body tries to follow the reference,
but muscle F_max bounds what it achieves -> impossible/jittery demands -> natural deviation.
This is the 'embodied imitator': output = what THIS body can do, not an exact copy.

F_max scale = the character (1.0 normal, >1 strong dancer, <1 delicate). Args: [t0 t1 fmax mesh].
Fail-fast test first (tiny segment, coarse mesh).
"""
import sys, os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclist_min.osim")   # 149-muscle minimal model (was cyclist_pruned=377)
MOT = os.path.join(HERE, "pop_ik_rad.mot")
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 7.5
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 7.75
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
mesh = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
tag = f"{t0:.2f}_{t1:.2f}_f{FMAX:.2f}"
OUT = os.path.join(HERE, f"popB_{tag}.sto")
print(f"=== MocoTrack t=[{t0},{t1}] Fmax x{FMAX} mesh={mesh} ===", flush=True)

mp = osim.ModelProcessor(MODEL)
mp.append(osim.ModOpIgnoreTendonCompliance())
mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
if FMAX != 1.0:
    mp.append(osim.ModOpScaleMaxIsometricForce(FMAX))   # <-- the character's strength
mp.append(osim.ModOpAddReserves(250.0))
m = mp.process(); m.initSystem()
print(f"coords={m.getCoordinateSet().getSize()} muscles={m.getMuscles().getSize()}", flush=True)

tp = osim.TableProcessor(MOT)
tp.append(osim.TabOpLowPassFilter(6.0))
tp.append(osim.TabOpUseAbsoluteStateNames())

track = osim.MocoTrack()
track.setName("popB")
track.setModel(mp)
track.setStatesReference(tp)
track.set_states_global_tracking_weight(1.0)
track.set_allow_unused_references(True)
track.set_track_reference_position_derivatives(True)
track.set_control_effort_weight(0.01)
track.set_initial_time(t0)
track.set_final_time(t1)
track.set_mesh_interval(mesh)
print("initialize + solve...", flush=True)
study = track.initialize()
solver = study.updSolver()
solver.resetProblem(study.getProblem())
sol = study.solve()
if not sol.success():
    sol.unseal()
sol.write(OUT)
print(f"done success={sol.success()} obj={sol.getObjective():.3f} -> {OUT}", flush=True)
