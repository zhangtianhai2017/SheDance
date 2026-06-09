#!/usr/bin/env python3
"""Build a retargeting CALIBRATION pose set on OUR humanoid: a standard rest + single-joint isolation
poses, each HELD static for HOLD frames so every action occupies a clear frame/time SEGMENT. A12B runs
the FBX through the SAME IK Retargeter and verifies every bone maps to the right MetaHuman bone with the
right axis/sign. Each isolation = ONE joint, ONE local axis, +ISO_DEG.

BATCH selects the joint set (env BATCH): arm (collar/shoulder/elbow/wrist), leg (hip/knee/ankle/foot),
torso (pelvis/spine1-3/neck/head), or all. Outputs: poses npz (-> export_joints_world -> fbx_export)
+ manifest CSV (machine) + timeline TXT (human list: frame range / time -> action).
Args: out_poses.npz out_manifest.csv [out_timeline.txt]   Env: BATCH(=arm) ISO_DEG(=35) HOLD(=30) FPS(=30)
"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation as Rsc
OUT, MAN = sys.argv[1], sys.argv[2]
TL = sys.argv[3] if len(sys.argv) > 3 else MAN.rsplit(".", 1)[0] + "_timeline.txt"
BATCH = os.environ.get("BATCH", "arm")
ISO = np.radians(float(os.environ.get("ISO_DEG", "35"))); ISO_D = float(np.degrees(ISO))
APOSE = np.radians(45.0)
HOLD = int(os.environ.get("HOLD", "30")); FPS = float(os.environ.get("FPS", "30"))
ROOT = np.array([np.pi / 2.0, 0.0, 0.0]); R_ROOT = Rsc.from_rotvec(ROOT)   # stand canonical Y-up upright in world Z-up
EN = {0: "pelvis", 1: "left_hip", 2: "right_hip", 3: "spine1", 4: "left_knee", 5: "right_knee",
      6: "spine2", 7: "left_ankle", 8: "right_ankle", 9: "spine3", 10: "left_foot", 11: "right_foot",
      12: "neck", 13: "left_collar", 14: "right_collar", 15: "head", 16: "left_shoulder",
      17: "right_shoulder", 18: "left_elbow", 19: "right_elbow", 20: "left_wrist", 21: "right_wrist"}
CN = {0: "盆骨", 1: "左髋", 2: "右髋", 3: "脊1", 4: "左膝", 5: "右膝", 6: "脊2", 7: "左踝", 8: "右踝",
      9: "脊3", 10: "左脚", 11: "右脚", 12: "颈", 13: "左锁骨", 14: "右锁骨", 15: "头", 16: "左肩",
      17: "右肩", 18: "左肘", 19: "右肘", 20: "左腕", 21: "右腕"}
FING = ["index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
        "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]
FCN = {"index": "食指", "middle": "中指", "pinky": "小指", "ring": "无名指", "thumb": "拇指"}
for _side, _scn, _base in (("left", "左", 22), ("right", "右", 37)):   # SMPL-H 30 finger joints (15 / hand)
    for _k, _f in enumerate(FING):
        EN[_base + _k] = "%s_%s" % (_side, _f); CN[_base + _k] = "%s%s%s" % (_scn, FCN[_f[:-1]], _f[-1])
BATCHES = {"arm": [13, 14, 16, 17, 18, 19, 20, 21], "leg": [1, 2, 4, 5, 7, 8, 10, 11],
           "torso": [0, 3, 6, 9, 12, 15], "all": list(range(22)),
           "lhand": list(range(22, 37)), "rhand": list(range(37, 52)), "hand": list(range(22, 52))}
JOINTS = BATCHES[BATCH]
AX = [("X", 0), ("Y", 1), ("Z", 2)]

uniq = []                                            # (joint_idx, joint_name, axis, deg, desc, pose156)
p = np.zeros(156); uniq.append((-1, "REST_TPOSE", "-", 0.0, "T-pose 标准静止(rest 基准)", p))
if BATCH in ("arm", "all"):                          # A-pose reference is arm-specific (align vs MetaHuman A-pose)
    ap = np.zeros(156); ap[3 * 16 + 2] = -APOSE; ap[3 * 17 + 2] = +APOSE
    uniq.append((-1, "REST_APOSE", "Z", 45.0, "A-pose 双臂下垂 45°", ap))
for j in JOINTS:
    for axn, ax in AX:
        q = np.zeros(156)
        if j == 0:                                   # root: compose +ISO about local axis ON TOP of the standing ROOT
            e = np.zeros(3); e[ax] = ISO
            q[0:3] = (R_ROOT * Rsc.from_rotvec(e)).as_rotvec()
        else:
            q[3 * j + ax] = ISO
        uniq.append((j, EN[j], axn, ISO_D, "%s 绕局部 %s 轴 +%d°" % (CN[j], axn, int(round(ISO_D))), q))

frames = []; rows = []                               # rows: pose, fs, fe, ts, te, jidx, jname, axis, deg, desc
for P, (jid, jn, ax, deg, desc, p156) in enumerate(uniq):
    pp = p156.copy()
    if jid != 0:                                     # non-root frames stand via ROOT; root frames already hold the composed root
        pp[0:3] = ROOT
    fs = len(frames); frames.extend([pp] * HOLD); fe = len(frames) - 1
    rows.append((P, fs, fe, fs / FPS, (fe + 1) / FPS, jid, jn, ax, deg, desc))
poses = np.stack(frames); T = len(poses)
np.savez(OUT, poses=poses, trans=np.zeros((T, 3)), betas=np.array([-2, 2] + [0] * 14, float),
         gender="female", mocap_framerate=FPS)

with open(MAN, "w", encoding="utf-8") as f:
    f.write("pose,frame_start,frame_end,t_start_s,t_end_s,joint_idx,joint_name,axis,angle_deg,desc\n")
    for r in rows:
        f.write("%d,%d,%d,%.2f,%.2f,%d,%s,%s,%.1f,%s\n" % r)
with open(TL, "w", encoding="utf-8") as f:
    f.write("SheDance 标定姿势时间线 — 批次: %s\n" % BATCH)
    f.write("FBX %gfps,每个姿势静止保持 %d 帧 = %.1fs,共 %d 帧 / %.1fs\n\n" % (FPS, HOLD, HOLD / FPS, T, T / FPS))
    for r in rows:
        f.write("帧 %3d–%3d   (%5.1f–%5.1fs)   %s\n" % (r[1], r[2], r[3], r[4], r[9]))

print("[%s] calib poses %s -> %s | manifest+timeline" % (BATCH, poses.shape, OUT))
for r in rows:
    print("帧 %3d–%3d  (%5.1f–%5.1fs)  %s" % (r[1], r[2], r[3], r[4], r[9]))
