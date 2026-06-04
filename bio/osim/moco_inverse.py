#!/usr/bin/env python3
"""MocoInverse: motion (.mot) -> muscle activations on cyclistFullBodyMuscle.osim.

The mature tool for "given kinematics, find muscle activations" — handles muscle dynamics,
kinematic constraints (coupled knee / shoulder rhythm) and adds reserve/residual actuators
automatically. No GRF available -> pelvis residual carried by reserves (expected large);
the TEST is whether the JOINT (muscle-driven) reserves stay small -> muscles produce the motion.

Args: [t0 t1 mesh]  (defaults 2.0 2.5 0.05) — run short first, then scale.
"""
import sys, os
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclistFullBodyMuscle.osim")
MOT = os.path.join(HERE, "pop.mot")

t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
mesh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
tag = f"{t0:.2f}_{t1:.2f}"
OUT = os.path.join(HERE, f"pop_moco_{tag}.sto")

print(f"=== MocoInverse  t=[{t0},{t1}] mesh={mesh} ===", flush=True)

# --- joints to weld (locked-coord joints Moco rejects + irrelevant-for-dance segments) ---
WELD = ["mtp_r", "mtp_l", "radius_hand_r", "radius_hand_l", "r1R_sterR_jnt", "T1_head_neck"]
for i in range(1, 13):                       # 24 rib costovertebral joints (locked Y/Z)
    WELD += [f"T{i}_r{i}R_CVjnt", f"T{i}_r{i}L_CVjnt"]
for a, b in [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5"), ("T5", "T6"), ("T6", "T7"),
             ("T7", "T8"), ("T8", "T9"), ("T9", "T10"), ("T10", "T11"), ("T11", "T12"), ("T12", "L1")]:
    WELD.append(f"{a}_{b}_IVDjnt")           # rigidify thoracic spine (keep lumbar free)
weldvec = osim.StdVectorString()
for w in WELD:
    weldvec.append(w)

# --- model prep for Moco ---
mp = osim.ModelProcessor(MODEL)
mp.append(osim.ModOpReplaceJointsWithWelds(weldvec))
mp.append(osim.ModOpIgnoreTendonCompliance())
mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
mp.append(osim.ModOpAddReserves(250.0))   # reserve/residual actuators (optimal force 250)
model = mp.process()
model.initSystem()
print(f"coords={model.getCoordinateSet().getSize()} muscles={model.getMuscles().getSize()} "
      f"forces={model.getForceSet().getSize()}", flush=True)

# --- kinematics input ---
tp = osim.TableProcessor(MOT)
tp.append(osim.TabOpLowPassFilter(6.0))
tp.append(osim.TabOpUseAbsoluteStateNames())

inverse = osim.MocoInverse()
inverse.setModel(mp)
inverse.setKinematics(tp)
inverse.set_initial_time(t0)
inverse.set_final_time(t1)
inverse.set_mesh_interval(mesh)
inverse.set_kinematics_allow_extra_columns(True)
inverse.set_convergence_tolerance(1e-2)
inverse.set_constraint_tolerance(1e-3)

print("solving...", flush=True)
sol = inverse.solve()
ok = sol.getMocoSolution().success() if hasattr(sol.getMocoSolution(), "success") else True
mocosol = sol.getMocoSolution()
mocosol.write(OUT)
print(f"solved. wrote {OUT}", flush=True)
