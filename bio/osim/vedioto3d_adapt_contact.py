#!/usr/bin/env python3
"""Contact-constrained version of vedioto3d_adapt: same mink IK of the 70 keypoints onto MyoFullBody,
but when a foot is in CONTACT its IK target is ANCHORED (held at plant XY + a single floor Z) instead of
chasing the (monocular, sliding) toe keypoint. Removes foot-skate + vertical drift at the source while
keeping swing motion. Contact is detected by VERTICAL behaviour (foot low AND not lifting) -- a sliding
plant is fast horizontally but slow vertically, so this catches it (the crux the earlier attempt missed).

Args: track_npz out_cache_npz   Env: ITERS=40 POSTURE=0.05 FPS BAND=0.06 VTHR=0.30 FOOTW=60"""
import os, sys, json, numpy as np, mujoco, mink
from scipy.ndimage import binary_opening, binary_closing, minimum_filter1d
from musclemimic.environments.humanoids import MyoFullBody

TRACK, OUT = sys.argv[1], sys.argv[2]
ITERS = int(os.environ.get("ITERS", "40")); POSTURE = float(os.environ.get("POSTURE", "0.05"))
BAND = float(os.environ.get("BAND", "0.06")); VTHR = float(os.environ.get("VTHR", "0.30"))
FOOTW = float(os.environ.get("FOOTW", "60"))   # foot task weight while anchored (hold it firmly)
FPS = float(os.environ.get("FPS", "30"))

KP = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
      "left_elbow","right_elbow","left_hip","right_hip","left_knee","right_knee","left_ankle",
      "right_ankle","left_big_toe_tip","left_small_toe_tip","left_heel","right_big_toe_tip",
      "right_small_toe_tip","right_heel"]
KP += ["rh"]*21 + ["lh"]*21 + ["left_olecranon","right_olecranon","left_cubital_fossa",
       "right_cubital_fossa","left_acromion","right_acromion","neck"]
IK = {n: i for i, n in enumerate(KP)}

def landmarks(k):
    mid = lambda a, b: 0.5 * (k[IK[a]] + k[IK[b]])
    hipmid = mid("left_hip", "right_hip"); neck = k[IK["neck"]]
    return {"pelvis": hipmid + 0.15 * (neck - hipmid), "spine1": hipmid + 0.40 * (neck - hipmid),
            "head": mid("left_ear", "right_ear"),
            "left_shoulder": k[IK["left_shoulder"]], "right_shoulder": k[IK["right_shoulder"]],
            "left_elbow": k[IK["left_elbow"]], "right_elbow": k[IK["right_elbow"]],
            "left_wrist": k[62], "right_wrist": k[41],
            "left_hip": k[IK["left_hip"]], "right_hip": k[IK["right_hip"]],
            "left_knee": k[IK["left_knee"]], "right_knee": k[IK["right_knee"]],
            "left_ankle": k[IK["left_ankle"]], "right_ankle": k[IK["right_ankle"]],
            "left_foot": k[IK["left_big_toe_tip"]], "right_foot": k[IK["right_big_toe_tip"]]}

cfg = json.load(open(os.path.expanduser("~/shedance/musclemimic/loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody.json")))
MAP = {body: (v[0], float(v[1])) for body, v in cfg["ik_match_table1"].items()}
for b in ("femur_l", "femur_r", "tibia_l", "tibia_r"):
    if b in MAP and MAP[b][1] == 0: MAP[b] = (MAP[b][0], 30.0)
for b in ("humerus_l", "humerus_r"):
    if b in MAP: MAP[b] = (MAP[b][0], 40.0)
MAP = {b: (s, w) for b, (s, w) in MAP.items() if w > 0}
FOOT_BODIES = {s: b for b, (s, w) in MAP.items() if s in ("left_foot", "right_foot")}   # body tracking each foot
print("foot bodies:", FOOT_BODIES, flush=True)

env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
B = mujoco.mjtObj.mjOBJ_BODY; d0 = mujoco.MjData(m)
base = m.key_qpos[0].copy() if m.nkey > 0 else np.zeros(m.nq)
if m.nkey == 0: base[2] = 0.95; base[3:7] = [1, 0, 0, 0]
d0.qpos[:] = base; mujoco.mj_forward(m, d0)
MODEL_H = float(d0.xpos[mujoco.mj_name2id(m, B, "head")][2] - d0.xpos[mujoco.mj_name2id(m, B, "calcn_l")][2])

tr = np.load(TRACK, allow_pickle=True); kp3 = np.asarray(tr["keypoints_3d"], float); T = len(kp3)
W = np.stack([kp3[:, :, 0], kp3[:, :, 2], -kp3[:, :, 1]], -1)
head_w = 0.5 * (W[:, IK["left_ear"]] + W[:, IK["right_ear"]]); ank_w = 0.5 * (W[:, IK["left_ankle"]] + W[:, IK["right_ankle"]])
subj_h = float(np.median(np.linalg.norm(head_w - ank_w, axis=1)))
scale = 1.0 if os.environ.get("NOSCALE") else MODEL_H / max(subj_h, 1e-6); W *= scale
print(f"frames {T}, subj {subj_h:.2f}->model {MODEL_H:.2f} (scale {scale:.3f})", flush=True)

# --- contact detection by VERTICAL behaviour (low AND not lifting); anchor removes horizontal slide ---
def contact_mask(toeZ):
    localfloor = minimum_filter1d(toeZ, 21, mode="nearest")            # tracks the drifting floor (handles vertical drift)
    vz = np.abs(np.r_[0.0, np.diff(toeZ)] * FPS)
    raw = ((toeZ - localfloor) < BAND) & (vz < VTHR)                   # near LOCAL floor AND not lifting = planted
    raw = binary_closing(raw, structure=np.ones(3)); raw = binary_opening(raw, structure=np.ones(3))
    return raw, float(np.percentile(toeZ, 8))
cL, fzL = contact_mask(W[:, IK["left_big_toe_tip"], 2]); cR, fzR = contact_mask(W[:, IK["right_big_toe_tip"], 2])
floorZ = min(fzL, fzR)
print(f"contact frames L={int(cL.sum())} R={int(cR.sum())} /{T}  floorZ={floorZ:.3f}", flush=True)

config = mink.Configuration(m); config.update(base)
posture = mink.PostureTask(m, cost=POSTURE)
LMD = float(os.environ.get("LMD", "0.1"))
tasks = {b: mink.FrameTask(b, "body", position_cost=w, orientation_cost=0.0, lm_damping=LMD) for b, (s, w) in MAP.items()}
KNEE = {"left_foot": ["femur_l", "tibia_l"], "right_foot": ["femur_r", "tibia_r"]}      # knee/shin bodies per leg
qpos = np.zeros((T, m.nq)); anchor = {"left_foot": None, "right_foot": None}
for t in range(T):
    lm = landmarks(W[t])
    for s, cm in (("left_foot", cL), ("right_foot", cR)):
        kpi = IK["left_big_toe_tip"] if s == "left_foot" else IK["right_big_toe_tip"]
        if cm[t]:
            if anchor[s] is None: anchor[s] = np.array([W[t, kpi, 0], W[t, kpi, 1]])   # plant: lock XY only
            lm[s] = np.array([anchor[s][0], anchor[s][1], W[t, kpi, 2]])               # held XY, keypoint Z
            tasks[FOOT_BODIES[s]].position_cost = FOOTW                                # hold the foot firmly
            for kb in KNEE[s]:                                                         # knee/shin: stop tracking the SLIDING ref
                if kb in tasks: tasks[kb].position_cost = 1.0                          # -> leg = anchored foot + hip (IK fills knee)
        else:
            anchor[s] = None
            tasks[FOOT_BODIES[s]].position_cost = MAP[FOOT_BODIES[s]][1]
            for kb in KNEE[s]:
                if kb in tasks: tasks[kb].position_cost = MAP[kb][1]                   # restore tracking
    posture.set_target(config.q)
    for b, (s, w) in MAP.items():
        tasks[b].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lm[s]))
    for _ in range(ITERS):
        try: vel = mink.solve_ik(config, [posture] + list(tasks.values()), 1.0, "daqp", safety_break=False, damping=1e-4)
        except Exception: break
        config.integrate_inplace(vel, 1.0)
    qpos[t] = config.q

zmin = np.inf
for t in range(T):
    d0.qpos[:] = qpos[t]; mujoco.mj_forward(m, d0)
    zmin = min(zmin, float(min(d0.xpos[mujoco.mj_name2id(m, B, n)][2] for n in ("calcn_l", "calcn_r", "toes_l", "toes_r"))))
qpos[:, 2] -= zmin
np.savez(OUT, qpos=qpos, frequency=FPS)

# verdict: slide DURING detected contact (did the anchor hold the planted feet?)
Pl = np.zeros((T, 3)); Pr = np.zeros((T, 3))
for t in range(T):
    d0.qpos[:] = qpos[t]; mujoco.mj_forward(m, d0)
    Pl[t] = d0.xpos[mujoco.mj_name2id(m, B, "toes_l")]; Pr[t] = d0.xpos[mujoco.mj_name2id(m, B, "toes_r")]
spL = [np.linalg.norm(Pl[t, :2] - Pl[t - 1, :2]) * FPS * 100 for t in range(1, T) if cL[t] and cL[t - 1]]
spR = [np.linalg.norm(Pr[t, :2] - Pr[t - 1, :2]) * FPS * 100 for t in range(1, T) if cR[t] and cR[t - 1]]
sp = np.array(spL + spR or [0.0])
print(f"adapted+contact -> {OUT} (qpos {qpos.shape} @ {FPS}Hz)", flush=True)
print(f"OUTPUT 接触相脚滑(检出接触帧) median {np.median(sp):.1f} mean {sp.mean():.1f} max {sp.max():.1f} cm/s  (ref ~38)", flush=True)
