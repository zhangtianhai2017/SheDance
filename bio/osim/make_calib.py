#!/usr/bin/env python3
"""Build a retargeting CALIBRATION pose set on OUR humanoid: a standard rest + single-joint isolation
poses, each HELD static for HOLD frames so every action occupies a clear frame/time SEGMENT (easy to
scrub + document). A12B runs this one FBX through the IK Retargeter and verifies every bone maps to the
right MetaHuman bone with the right axis/sign. Each isolation = ONE joint, ONE local axis, +ISO_DEG.

This batch = the ARM/HAND chain (collar, shoulder, elbow, wrist). Outputs: poses npz (-> export_joints_world
-> fbx_export) + manifest CSV (machine) + timeline TXT (human list: frame range / time -> action).
Args: out_poses.npz out_manifest.csv [out_timeline.txt]   Env: ISO_DEG(=35) HOLD(=30) FPS(=30)
"""
import os, sys, numpy as np
OUT, MAN = sys.argv[1], sys.argv[2]
TL = sys.argv[3] if len(sys.argv) > 3 else MAN.rsplit(".", 1)[0] + "_timeline.txt"
ISO = np.radians(float(os.environ.get("ISO_DEG", "35"))); ISO_D = float(np.degrees(ISO))
APOSE = np.radians(45.0)
HOLD = int(os.environ.get("HOLD", "30")); FPS = float(os.environ.get("FPS", "30"))
EN = {13: "left_collar", 14: "right_collar", 16: "left_shoulder", 17: "right_shoulder",
      18: "left_elbow", 19: "right_elbow", 20: "left_wrist", 21: "right_wrist"}
CN = {13: "左锁骨", 14: "右锁骨", 16: "左肩", 17: "右肩", 18: "左肘", 19: "右肘", 20: "左腕", 21: "右腕"}
JOINTS = [13, 14, 16, 17, 18, 19, 20, 21]            # arm/hand chain (this batch)
AX = [("X", 0), ("Y", 1), ("Z", 2)]
ROOT = np.array([np.pi / 2.0, 0.0, 0.0])             # stand the canonical Y-up body upright in world Z-up

uniq = []                                            # (joint_idx, joint_name, axis, deg, desc, pose156)
p = np.zeros(156); uniq.append((-1, "REST_TPOSE", "-", 0.0, "T-pose 标准静止(rest 基准)", p))
ap = np.zeros(156); ap[3 * 16 + 2] = -APOSE; ap[3 * 17 + 2] = +APOSE
uniq.append((-1, "REST_APOSE", "Z", 45.0, "A-pose 双臂下垂 45°", ap))
for j in JOINTS:
    for axn, ax in AX:
        q = np.zeros(156); q[3 * j + ax] = ISO
        uniq.append((j, EN[j], axn, ISO_D, "%s 绕局部 %s 轴 +%d°" % (CN[j], axn, int(round(ISO_D))), q))

frames = []; rows = []                               # rows: pose, fs, fe, ts, te, jidx, jname, axis, deg, desc
for P, (jid, jn, ax, deg, desc, p156) in enumerate(uniq):
    pp = p156.copy(); pp[0:3] = ROOT
    fs = len(frames)
    frames.extend([pp] * HOLD)
    fe = len(frames) - 1
    rows.append((P, fs, fe, fs / FPS, (fe + 1) / FPS, jid, jn, ax, deg, desc))
poses = np.stack(frames); T = len(poses)
np.savez(OUT, poses=poses, trans=np.zeros((T, 3)), betas=np.array([-2, 2] + [0] * 14, float),
         gender="female", mocap_framerate=FPS)

with open(MAN, "w", encoding="utf-8") as f:
    f.write("pose,frame_start,frame_end,t_start_s,t_end_s,joint_idx,joint_name,axis,angle_deg,desc\n")
    for r in rows:
        f.write("%d,%d,%d,%.2f,%.2f,%d,%s,%s,%.1f,%s\n" % r)
with open(TL, "w", encoding="utf-8") as f:
    f.write("SheDance 手臂/手 标定姿势 — 时间线清单\n")
    f.write("FBX %gfps,每个姿势静止保持 %d 帧 = %.1fs,共 %d 帧 / %.1fs\n\n" % (FPS, HOLD, HOLD / FPS, T, T / FPS))
    for r in rows:
        f.write("帧 %3d–%3d   (%4.1f–%4.1fs)   %s\n" % (r[1], r[2], r[3], r[4], r[9]))

print("calib poses %s -> %s\nmanifest -> %s\ntimeline -> %s\n" % (poses.shape, OUT, MAN, TL))
for r in rows:
    print("帧 %3d–%3d  (%4.1f–%4.1fs)  %s" % (r[1], r[2], r[3], r[4], r[9]))
