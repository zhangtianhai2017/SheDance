#!/usr/bin/env python3
"""L1 hard non-penetration refinement (kinematic, margin=0) via mink.
Per frame: PostureTask holds the retargeted pose; CollisionAvoidanceLimit (min_dist=0) pushes
overlapping geom-pairs apart -> minimal change that removes interpenetration. Body stays physical
(like F_max: a body constraint shaping output), input intent preserved as the posture target.
Env: IN_CACHE, OUT_CACHE, REFINE_FRAMES (a:b optional), ITERS (default 15).
v1 pairs: left-foot vs right-foot collision geoms (the visible problem). Extend GROUPS for hands/body."""
import os, sys, numpy as np, mujoco, mink
from musclemimic.environments.humanoids import MyoFullBody

IN = os.environ["IN_CACHE"]; OUT = os.environ["OUT_CACHE"]
ITERS = int(os.environ.get("ITERS", "30"))
RELAX = float(os.environ.get("RELAX", "-0.005"))   # negative => actively push apart existing penetration
c = np.load(IN, allow_pickle=True); qpos = np.asarray(c["qpos"], float); freq = float(c["frequency"])
fr = os.environ.get("REFINE_FRAMES")
sl = slice(*[int(x) if x else None for x in fr.split(":")]) if fr else slice(None)

env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
BODY, GEOM = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM

def col_geoms(bodies):
    names = []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, BODY, int(m.geom_bodyid[g]))
        gn = mujoco.mj_id2name(m, GEOM, g)
        if bn in bodies and gn and (m.geom_contype[g] or m.geom_conaffinity[g]):
            names.append(gn)
            m.geom_conaffinity[g] = 1; m.geom_contype[g] = 1   # make this pair eligible in mink's filter
    return names

LF = col_geoms({"calcn_l", "toes_l", "talus_l"})
RF = col_geoms({"calcn_r", "toes_r", "talus_r"})
print(f"left-foot geoms {LF}\nright-foot geoms {RF}", flush=True)
geom_pairs = [(LF, RF)]

config = mink.Configuration(m)
posture = mink.PostureTask(m, cost=1.0)
collision = mink.CollisionAvoidanceLimit(m, geom_pairs, minimum_distance_from_collisions=0.0,
                                         collision_detection_distance=0.1, gain=0.85,
                                         bound_relaxation=RELAX)
limits = [collision]
solver = next((s for s in ("quadprog", "osqp", "daqp", "cvxopt", "scs")
               if __import__("qpsolvers").available_solvers and s in __import__("qpsolvers").available_solvers), "quadprog")
print("solver:", solver, flush=True)

# foot-foot geom distance helper (min over the LF x RF col-geom pairs)
gid = {n: mujoco.mj_name2id(m, GEOM, n) for n in LF + RF}
def min_foot_dist(d):
    mn = 1e9
    for a in LF:
        for b in RF:
            mn = min(mn, mujoco.mj_geomDistance(m, d, gid[a], gid[b], 1.0, None))
    return mn

idx = np.arange(len(qpos))[sl]
refined = qpos.copy()
d0 = mujoco.MjData(m)
pen_before = []; pen_after = []; dev = []; fails = 0
for t in idx:
    config.update(qpos[t]); posture.set_target(qpos[t])
    d0.qpos[:] = qpos[t]; mujoco.mj_forward(m, d0); pen_before.append(min_foot_dist(d0))
    for _ in range(ITERS):
        try:
            vel = mink.solve_ik(config, [posture], 1.0, solver, limits=limits, safety_break=False, damping=1e-6)
        except Exception:
            fails += 1; break   # QP failed this frame -> keep partial refine, don't crash the run
        config.integrate_inplace(vel, 1.0)
    refined[t] = config.q
    pen_after.append(min_foot_dist(config.data))
    dev.append(float(np.degrees(np.abs(refined[t][7:] - qpos[t][7:]).max())))

np.savez(OUT, qpos=refined, frequency=freq)
pb, pa, dv = np.array(pen_before), np.array(pen_after), np.array(dev)
print(f"frames {len(idx)}  foot-dist before: min {pb.min()*1000:.1f}mm (penetrating={(pb<0).sum()})"
      f"  -> after: min {pa.min()*1000:.1f}mm (penetrating={(pa<0).sum()})", flush=True)
print(f"max joint deviation (refined vs input): {dv.max():.2f} deg, mean {dv.mean():.2f} deg  (QP-fail frames: {fails}) -> {OUT}", flush=True)
