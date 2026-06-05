#!/usr/bin/env python3
"""L1+L2 kinematic collision refinement via mink (margin = the quantifiable TRUST knob).
Per frame: PostureTask holds the retargeted pose; CollisionAvoidanceLimit (negative bound_relaxation)
resolves interpenetration toward a per-region-pair MARGIN (minimum signed distance):
  margin = 0  -> hard non-penetration (touch ok, no overlap).  Feet never legitimately overlap.
  margin < 0  -> allow contact/overlap up to |margin| (TRUST the dancer's intended contact); only
                 resolve GROSS penetration deeper than that.  Hands<->hands/body/feet (a dance puts
                 hands on the body on purpose; hard separation distorts the motion -- L1 proved it).
A MAXDEV cap trades non-penetration against the input when they conflict (revert frame = trust input).
Residual penetration past the margin is reported PER PAIR = the 'unexpected collision' feedback signal.
Env: IN_CACHE, OUT_CACHE, REFINE_FRAMES (a:b), ITERS=40, RELAX=-0.005, MAXDEV=30,
     MARGIN_FEET/HANDS/HANDBODY/HANDFOOT (metres) override the trust knobs."""
import os, numpy as np, mujoco, mink
from musclemimic.environments.humanoids import MyoFullBody

IN, OUT = os.environ["IN_CACHE"], os.environ["OUT_CACHE"]
ITERS = int(os.environ.get("ITERS", "40"))
RELAX = float(os.environ.get("RELAX", "-0.005"))
MAXDEV = float(os.environ.get("MAXDEV", "30"))
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

def mg(name, default):
    return float(os.environ.get(name, default))
PAIRS = [   # (regionA, regionB, trust-margin metres);  0 = hard,  <0 = trust contact / fix gross only
    ("LF", "RF", mg("MARGIN_FEET", "0.0")),
    ("LH", "RH", mg("MARGIN_HANDS", "-0.02")),
    ("LH", "BODY", mg("MARGIN_HANDBODY", "-0.03")), ("RH", "BODY", mg("MARGIN_HANDBODY", "-0.03")),
    ("LH", "LF", mg("MARGIN_HANDFOOT", "-0.02")), ("LH", "RF", mg("MARGIN_HANDFOOT", "-0.02")),
    ("RH", "LF", mg("MARGIN_HANDFOOT", "-0.02")), ("RH", "RF", mg("MARGIN_HANDFOOT", "-0.02")),
]
config = mink.Configuration(m); posture = mink.PostureTask(m, cost=1.0)
limits = []; pinfo = []   # (label, margin, geom_id_pairs) one CollisionAvoidanceLimit per pair
for a, b, margin in PAIRS:
    if not (RG[a] and RG[b]):
        continue
    lim = mink.CollisionAvoidanceLimit(m, [(RG[a], RG[b])], minimum_distance_from_collisions=margin,
                                       collision_detection_distance=0.1, gain=0.85, bound_relaxation=RELAX)
    limits.append(lim); pinfo.append((f"{a}-{b}", margin, lim.geom_id_pairs))
for lbl, margin, gidp in pinfo:
    print(f"{lbl:9s} margin {margin*1000:+.0f}mm  ({len(gidp)} geom-pairs)", flush=True)
print(f"solver {SOLVER}, iters {ITERS}, relax {RELAX}, MAXDEV {MAXDEV} deg", flush=True)

def min_dist(d, gidp):
    return min((mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a, b in gidp), default=1.0)

idx = np.arange(len(qpos))[sl]; refined = qpos.copy()
d0 = mujoco.MjData(m); dev = []; fails = 0; reverts = 0
resid = {lbl: {"frames": 0, "worst": 0.0} for lbl, _, _ in pinfo}   # gross penetration past the margin
for t in idx:
    config.update(qpos[t]); posture.set_target(qpos[t])
    for _ in range(ITERS):
        try:
            vel = mink.solve_ik(config, [posture], 1.0, SOLVER, limits=limits, safety_break=False, damping=1e-5)
        except Exception:
            fails += 1; break   # QP failed -> keep partial refine for this frame
        config.integrate_inplace(vel, 1.0)
    dv_t = float(np.degrees(np.abs(config.q[7:] - qpos[t][7:]).max()))
    if dv_t > MAXDEV:                 # over-distorting -> trust input, keep original (residual reported)
        refined[t] = qpos[t]; dev.append(0.0); reverts += 1
    else:
        refined[t] = config.q; dev.append(dv_t)
    d0.qpos[:] = refined[t]; mujoco.mj_forward(m, d0)
    for lbl, margin, gidp in pinfo:
        dmin = min_dist(d0, gidp)
        if dmin < margin - 1e-4:      # still penetrating past the trust margin = unexpected collision
            resid[lbl]["frames"] += 1
            resid[lbl]["worst"] = min(resid[lbl]["worst"], dmin - margin)

np.savez(OUT, qpos=refined, frequency=freq)
dv = np.array(dev)
print(f"frames {len(idx)}  max dev {dv.max():.2f} deg, mean {dv.mean():.2f} deg "
      f"(QP-fails {fails}, reverts {reverts}) -> {OUT}", flush=True)
print("--- unexpected-collision feedback (frames still penetrating past the trust margin) ---", flush=True)
for lbl, margin, _ in pinfo:
    r = resid[lbl]
    flag = "" if r["frames"] == 0 else f"  worst {r['worst']*1000:.0f}mm past margin"
    print(f"  {lbl:9s} margin {margin*1000:+.0f}mm: {r['frames']:3d}/{len(idx)} frames{flag}", flush=True)
