#!/usr/bin/env python3
"""Emit ONE browsable Markdown frame/time -> action table covering all calibration batches, built from
their manifests so it stays in sync. Args: [calib_dir] [out.md]"""
import sys, csv, os
C = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/shedance/wf_out/calib")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/work/2026/Claude/SheDance/SheDance_calib_timeline.md"
BATCHES = [("arm", "calib_arm.fbx"), ("leg", "calib_leg.fbx"), ("torso", "calib_torso.fbx"),
           ("lhand", "calib_lhand.fbx"), ("rhand", "calib_rhand.fbx")]
L = ["# SheDance 标定 — 帧 / 时间 → 动作表", "",
     "每个姿势静止保持 **1 秒**(30 帧 @30fps);第 K 个姿势 = 第 K 秒 = 帧 `30K`–`30K+29`。",
     "把 `SheDance_calib.zip` 里对应的 `calib_<batch>.fbx` 用**和 dance1.fbx 同一个 IK Retargeter** 导入并 scrub ——",
     "MetaHuman 上应是**同一根骨、同方向**在动;若动错骨或反向,即该骨映射/轴错了。", ""]
for b, fbx in BATCHES:
    man = os.path.join(C, "calib_%s_manifest.csv" % b)
    rows = list(csv.reader(open(man, encoding="utf-8")))[1:]
    nf = int(rows[-1][2]) + 1
    L += ["## %s — `%s`  (%d 帧 / %.0fs)" % (b, fbx, nf, nf / 30.0), "",
          "| 帧段 | 时间 | 动作 |", "|---|---|---|"]
    for r in rows:                                   # pose,fs,fe,ts,te,jidx,jname,axis,deg,desc
        L.append("| %s–%s | %s–%s s | %s |" % (r[1], r[2], r[3], r[4], r[9]))
    L.append("")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote %s  (%d batches)" % (OUT, len(BATCHES)))
