#!/usr/bin/env python3
"""Re-import an exported FBX into a clean bpy scene and report what's actually in it:
bone count + names, animation frame range, fps, and a couple of sampled pelvis positions
vs the source joints_world (ratio -> confirms units/self-consistency). Read-only sanity check.
Usage: blender_python fbx_verify.py <fbx> <joints_world.npz>"""
import sys, numpy as np, bpy
FBX, JW_NPZ = sys.argv[-2], sys.argv[-1]
JW = np.load(JW_NPZ, allow_pickle=True)["joints_world"].astype(float)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
bones = [b.name for b in arm.data.bones]
print("BONES n=%d : %s ..." % (len(bones), bones[:6]))
print("ROOT bone (no parent):", [b.name for b in arm.data.bones if b.parent is None])
act = arm.animation_data.action if arm.animation_data else None
if act:
    fr = act.frame_range
    print("ANIM frame_range=%s  n_fcurves=%d  scene_fps=%d" % (tuple(fr), len(act.fcurves), bpy.context.scene.render.fps))
    T = JW.shape[0]
    for t in (0, T // 2, T - 1):
        bpy.context.scene.frame_set(t)
        bpy.context.view_layer.update()
        pb = arm.pose.bones["pelvis"]
        w = arm.matrix_world @ pb.head
        src = JW[t, 0]
        print("  f%4d pelvis FBX=(%.3f,%.3f,%.3f)  src=(%.3f,%.3f,%.3f)  |FBX|/|src|=%.3f"
              % (t, w.x, w.y, w.z, src[0], src[1], src[2], (w.length / (np.linalg.norm(src) + 1e-9))))
else:
    print("ANIM none")
