#!/usr/bin/env python3
"""Adapter: VedioTo3D (SAM 3D Body / MHR) per-frame output -> MyoFullBody 89-dof cache (qpos), so any
video can drive the SheDance muscle pipeline. We track the 70 semantic keypoints (positions) -- the
SOUL's 'track spatial keypoints, not joint angles' -- via mink IK onto MyoFullBody, reusing the GMR
smplh_to_myofullbody body<->landmark mapping + weights. Coordinate: VedioTo3D camera (X right, Y DOWN,
Z fwd, meters) -> SheDance world Z-up [X, Z, -Y]; height-normalized; grounded.
Args: track_npz out_cache_npz   Env: ITERS=12, POSTURE=0.05, FPS (override meta)"""
import os, sys, json, numpy as np, mujoco, mink
from musclemimic.environments.humanoids import MyoFullBody

TRACK, OUT = sys.argv[1], sys.argv[2]
ITERS = int(os.environ.get("ITERS", "40")); POSTURE = float(os.environ.get("POSTURE", "0.05"))

# --- 70-keypoint index by name (DATA_FORMAT.md §6.1) ---
KP = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
      "left_elbow","right_elbow","left_hip","right_hip","left_knee","right_knee","left_ankle",
      "right_ankle","left_big_toe_tip","left_small_toe_tip","left_heel","right_big_toe_tip",
      "right_small_toe_tip","right_heel"]  # 0-20; hands 21-62 then olecranon/acromion/neck 63-69
KP += ["rh"]*21 + ["lh"]*21 + ["left_olecranon","right_olecranon","left_cubital_fossa",
       "right_cubital_fossa","left_acromion","right_acromion","neck"]
IK = {n: i for i, n in enumerate(KP)}
# the SMPL-H landmark names GMR's config tracks -> how to get them from VedioTo3D keypoints
def landmarks(k):   # k: (70,3) one frame -> {smplh_name: xyz}
    mid = lambda a, b: 0.5 * (k[IK[a]] + k[IK[b]])
    hipmid = mid("left_hip", "right_hip"); neck = k[IK["neck"]]
    return {
        "pelvis": hipmid + 0.15 * (neck - hipmid),    # SMPL-H pelvis sits ABOVE the hip-midpoint (up the spine)
        "spine1": hipmid + 0.40 * (neck - hipmid),
        "head": mid("left_ear", "right_ear"),
        "left_shoulder": k[IK["left_shoulder"]], "right_shoulder": k[IK["right_shoulder"]],
        "left_elbow": k[IK["left_elbow"]], "right_elbow": k[IK["right_elbow"]],
        "left_wrist": k[62], "right_wrist": k[41],
        "left_hip": k[IK["left_hip"]], "right_hip": k[IK["right_hip"]],
        "left_knee": k[IK["left_knee"]], "right_knee": k[IK["right_knee"]],
        "left_ankle": k[IK["left_ankle"]], "right_ankle": k[IK["right_ankle"]],
        "left_foot": k[IK["left_big_toe_tip"]], "right_foot": k[IK["right_big_toe_tip"]],
    }

cfg = json.load(open(os.path.expanduser("~/shedance/musclemimic/loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody.json")))
MAP = {body: (v[0], float(v[1])) for body, v in cfg["ik_match_table1"].items()}   # body -> (smplh_name, pos_weight)
for b in ("femur_l", "femur_r", "tibia_l", "tibia_r"):        # position-only IK is underdetermined without
    if b in MAP and MAP[b][1] == 0:                           # the hips/knees (GMR uses orientations; we track positions)
        MAP[b] = (MAP[b][0], 30.0)
for b in ("humerus_l", "humerus_r"):                          # pin the shoulders -- config's weight 1 is too
    if b in MAP:                                              # low for position-only (GMR positions them via orientation)
        MAP[b] = (MAP[b][0], 40.0)
MAP = {b: (s, w) for b, (s, w) in MAP.items() if w > 0}                            # keep tracked ones

env = MyoFullBody(disable_fingers=True)
m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
B = mujoco.mjtObj.mjOBJ_BODY
d0 = mujoco.MjData(m)
base = m.key_qpos[0].copy() if m.nkey > 0 else np.zeros(m.nq)
if m.nkey == 0:
    base[2] = 0.95; base[3:7] = [1, 0, 0, 0]
d0.qpos[:] = base; mujoco.mj_forward(m, d0)
# model height (head body z - foot z) for scale normalization
hz = d0.xpos[mujoco.mj_name2id(m, B, "head")][2]; fz = d0.xpos[mujoco.mj_name2id(m, B, "calcn_l")][2]
MODEL_H = float(hz - fz)

tr = np.load(TRACK, allow_pickle=True)
kp3 = np.asarray(tr["keypoints_3d"], float)                       # (T,70,3) camera meters
T = len(kp3)
# coordinate: camera (X right, Y down, Z fwd) -> world Z-up
W = np.stack([kp3[:, :, 0], kp3[:, :, 2], -kp3[:, :, 1]], -1)     # [X, Z, -Y]
# height-normalize to the model (use median head<->ankle over frames)
head_w = 0.5 * (W[:, IK["left_ear"]] + W[:, IK["right_ear"]])
ank_w = 0.5 * (W[:, IK["left_ankle"]] + W[:, IK["right_ankle"]])
subj_h = float(np.median(np.linalg.norm(head_w - ank_w, axis=1)))
scale = 1.0 if os.environ.get("NOSCALE") else MODEL_H / max(subj_h, 1e-6)
W *= scale
print(f"frames {T}, subj height {subj_h:.2f}m -> model {MODEL_H:.2f}m (scale {scale:.3f}), tracking {len(MAP)} bodies", flush=True)

config = mink.Configuration(m); config.update(base)
posture = mink.PostureTask(m, cost=POSTURE)
tasks = {b: mink.FrameTask(b, "body", position_cost=w, orientation_cost=0.0, lm_damping=0.1) for b, (s, w) in MAP.items()}
solver = "daqp"
qpos = np.zeros((T, m.nq))
for t in range(T):
    lm = landmarks(W[t])
    posture.set_target(config.q)                                 # regularize toward previous frame (temporal)
    for b, (s, w) in MAP.items():
        tasks[b].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lm[s]))
    for _ in range(ITERS):
        try:
            vel = mink.solve_ik(config, [posture] + list(tasks.values()), 1.0, solver, safety_break=False, damping=1e-4)
        except Exception:
            break
        config.integrate_inplace(vel, 1.0)
    qpos[t] = config.q

# ground: shift root z so the lowest foot sits at 0 over the clip
zmin = np.inf
for t in range(T):
    d0.qpos[:] = qpos[t]; mujoco.mj_forward(m, d0)
    zmin = min(zmin, float(min(d0.xpos[mujoco.mj_name2id(m, B, n)][2] for n in ("calcn_l", "calcn_r", "toes_l", "toes_r"))))
qpos[:, 2] -= zmin
freq = float(os.environ.get("FPS") or json.load(open(os.path.join(os.path.dirname(TRACK), "..", "meta.json")))["fps"]) if os.path.exists(os.path.join(os.path.dirname(TRACK), "..", "meta.json")) else float(os.environ.get("FPS", "30"))
np.savez(OUT, qpos=qpos, frequency=freq)
print(f"adapted -> {OUT} (qpos {qpos.shape} @ {freq}Hz)", flush=True)
