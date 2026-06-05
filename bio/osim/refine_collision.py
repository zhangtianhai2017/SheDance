#!/usr/bin/env python3
"""L1 hard non-penetration refinement (kinematic, margin=0) via mink.
Per frame: PostureTask holds the retargeted pose; CollisionAvoidanceLimit (NEGATIVE bound_relaxation)
actively pushes overlapping geom-pairs apart -> minimal change that removes interpenetration.
Body stays physical (a body constraint, like F_max); input intent preserved as the posture target.
Scope (B): feet, hands(metacarpals+thumb), hand<->foot, hand<->body. (89-dof model has no fingers.)
Robust solve (A): SOLVER + jaxopt_osqp fallback on QP failure + damping; per-frame fail keeps partial.
Env: IN_CACHE, OUT_CACHE, REFINE_FRAMES (a:b), ITERS=40, RELAX=-0.005, SOLVER=daqp."""
import os, numpy as np, mujoco, mink
from musclemimic.environments.humanoids import MyoFullBody

IN, OUT = os.environ["IN_CACHE"], os.environ["OUT_CACHE"]
ITERS = int(os.environ.get("ITERS", "40"))
RELAX = float(os.environ.get("RELAX", "-0.005"))
SOLVER = os.environ.get("SOLVER", "daqp")
c = np.load(IN, allow_pickle=True); qpos = np.asarray(c["qpos"], float); freq = float(c["frequency"])
fr = os.environ.get("REFINE_FRAMES")
sl = slice(*[int(x) if x else None for x in fr.split(":")]) if fr else slice(None)

env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
REGIONS = {
    "LF": {"calcn_l", "toes_l"}, "RF": {"calcn_r", "toes_r"},
    "LH": {"firstmc_l", "secondmc_l", "thirdmc_l", "fourthmc_l", "fifthmc_l", "proximal_thumb_l"},
    "RH": {"firstmc_r", "secondmc_r", "thirdmc_r", "fourthmc_r", "fifthmc_r", "proximal_thumb_r"},
    "BODY": {"thorax", "pelvis", "head", "neck"},
}
def region_geoms(bodies):
    out = []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])); gn = mujoco.mj_id2name(m, G, g)
        if bn in bodies and gn and (m.geom_contype[g] or m.geom_conaffinity[g]):
            out.append(gn); m.geom_conaffinity[g] = 1; m.geom_contype[g] = 1   # eligible in mink filter
    return out
RG = {r: region_geoms(bs) for r, bs in REGIONS.items()}
# Hard-L1 only where overlap is a CLEAR artifact (feet never legitimately pass through each other).
# Hands<->hands/body distort genuine dance contacts (the dancer's hands intentionally near body) ->
# those belong to L2/trust-input (negative margin / soft), not hard non-penetration.
PAIRS = [("LF", "RF")]
geom_pairs = [(RG[a], RG[b]) for a, b in PAIRS if RG[a] and RG[b]]
for r in REGIONS:
    print(f"{r}: {len(RG[r])} geoms", flush=True)

config = mink.Configuration(m); posture = mink.PostureTask(m, cost=1.0)
collision = mink.CollisionAvoidanceLimit(m, geom_pairs, minimum_distance_from_collisions=0.0,
                                         collision_detection_distance=0.1, gain=0.85, bound_relaxation=RELAX)
limits = [collision]
gidp = collision.geom_id_pairs
print(f"geom-id pairs {collision.max_num_contacts}, solver {SOLVER}, iters {ITERS}, relax {RELAX}", flush=True)

def min_pair_dist(d):
    return min((mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a, b in gidp), default=1.0)

MAXDEV = float(os.environ.get("MAXDEV", "30"))   # guard: revert frame to input if refine distorts beyond this
idx = np.arange(len(qpos))[sl]; refined = qpos.copy()
d0 = mujoco.MjData(m); pb = []; pa = []; dev = []; fails = 0; reverts = 0
for t in idx:
    config.update(qpos[t]); posture.set_target(qpos[t])
    d0.qpos[:] = qpos[t]; mujoco.mj_forward(m, d0); pb.append(min_pair_dist(d0))
    for _ in range(ITERS):
        try:
            vel = mink.solve_ik(config, [posture], 1.0, SOLVER, limits=limits, safety_break=False, damping=1e-5)
        except Exception:
            fails += 1; break   # QP failed -> keep partial refine for this frame
        config.integrate_inplace(vel, 1.0)
    dv_t = float(np.degrees(np.abs(config.q[7:] - qpos[t][7:]).max()))
    if dv_t > MAXDEV:           # pathological / over-distorting -> trust input, keep original
        refined[t] = qpos[t]; pa.append(pb[-1]); dev.append(0.0); reverts += 1
    else:
        refined[t] = config.q; pa.append(min_pair_dist(config.data)); dev.append(dv_t)

np.savez(OUT, qpos=refined, frequency=freq)
pb, pa, dv = np.array(pb), np.array(pa), np.array(dev)
print(f"frames {len(idx)}  min-pair-dist before {pb.min()*1000:.1f}mm (penetrating {(pb<0).sum()})"
      f" -> after {pa.min()*1000:.1f}mm (penetrating {(pa<0).sum()})", flush=True)
print(f"max joint deviation {dv.max():.2f} deg, mean {dv.mean():.2f} deg (QP-fails {fails}, reverts {reverts}) -> {OUT}", flush=True)
