#!/usr/bin/env python3
"""Export a generic SMPL-body FBX (skeleton + baked animation) from our joints_world data, via headless bpy.
World-driving: each bone's WORLD matrix per frame = (joint_world_rot, joints_world) -> exact match to our data,
independent of bone rest roll. 22 body bones (SMPL 0-21, names = SMPL-X body names). No fingers (dance1).

Usage: blender_python fbx_export.py <joints_world.npz> <our_humanoid.npz> <out.fbx> <mode>
  mode = test | anim | rest
Coords in: world right-handed Z-up meters (our data). FBX exported Z-up / X-forward; units documented by caller.
"""
import sys, numpy as np
import bpy
from mathutils import Matrix, Quaternion, Vector

JW_NPZ, HUM_NPZ, OUT_FBX, MODE = sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
N = 22                                              # body bones only (SMPL 0-21)
RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # +90deg about X: canonical Y-up -> world Z-up

z = np.load(JW_NPZ, allow_pickle=True)
JW = z["joints_world"].astype(float)               # (T,24,3) world positions
JWR = z["joint_world_rot"].astype(float)           # (T,24,4) world rot, xyzw
NAMES = [str(x) for x in z["joint_names"]][:N]
PAR = [int(x) for x in z["parents"]][:N]
FPS = float(z["fps"]); T = JW.shape[0]
restJ = np.load(HUM_NPZ, allow_pickle=True)["joints"].astype(float)[:N]   # canonical rest joints (Y-up)
rest_world = (RX90 @ restJ.T).T                     # standing rest skeleton in world Z-up

def quat_wxyz(q_xyzw):                              # our xyzw -> mathutils wxyz
    return Quaternion((q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]))

# ---- fresh scene + armature ----
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = int(round(FPS))
arm = bpy.data.armatures.new("SMPL"); obj = bpy.data.objects.new("SMPL", arm)
bpy.context.scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj; obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
ebs = []
for j in range(N):
    eb = arm.edit_bones.new(NAMES[j])
    eb.head = Vector(rest_world[j])
    # tail: toward first child if any, else continue parent->this direction (leaf stub)
    children = [c for c in range(N) if PAR[c] == j]
    if children:
        eb.tail = Vector(rest_world[children[0]])
    else:
        d = rest_world[j] - rest_world[PAR[j]] if PAR[j] >= 0 else np.array([0, 0, 0.1])
        nd = d / (np.linalg.norm(d) + 1e-9)
        eb.tail = Vector(rest_world[j] + nd * 0.10)
    ebs.append(eb)
for j in range(N):
    if PAR[j] >= 0:
        ebs[j].parent = ebs[PAR[j]]
bpy.ops.object.mode_set(mode="POSE")
for pb in obj.pose.bones:
    pb.rotation_mode = "QUATERNION"

def pose_frame(t):
    for j in range(N):                              # hierarchy order (parents first; PAR[j] < j in SMPL)
        M = Matrix.LocRotScale(Vector(JW[t, j]), quat_wxyz(JWR[t, j]), Vector((1, 1, 1)))
        obj.pose.bones[NAMES[j]].matrix = M
        bpy.context.view_layer.update()             # child must see parent's NEW matrix, else chain compounds wrong

if MODE == "rest":
    pass                                            # edit rest IS the standing rest skeleton; export as-is
elif MODE == "test":
    errs = []
    for t in (0, T // 2, T - 1):
        pose_frame(t)
        for j in range(N):
            errs.append(np.linalg.norm(np.array(obj.pose.bones[NAMES[j]].head) - JW[t, j]))
    print("TEST max head-pos error vs joints_world: %.4f mm" % (max(errs) * 1000))
    print("TEST bones=%d frames-checked=3 names0..3=%s" % (N, NAMES[:4]))
else:                                               # anim: bake all frames
    bpy.context.scene.frame_start = 0; bpy.context.scene.frame_end = T - 1
    for t in range(T):
        bpy.context.scene.frame_set(t)
        pose_frame(t)
        for j in range(N):
            pb = obj.pose.bones[NAMES[j]]
            pb.keyframe_insert("location", frame=t)
            pb.keyframe_insert("rotation_quaternion", frame=t)

if MODE != "test":
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=OUT_FBX, use_selection=True, object_types={"ARMATURE"},
        add_leaf_bones=False, bake_anim=(MODE == "anim"),
        bake_anim_use_all_bones=True, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
        axis_up="Z", axis_forward="X", apply_unit_scale=True, global_scale=1.0,
        use_armature_deform_only=False, primary_bone_axis="Y", secondary_bone_axis="X")
    print("EXPORTED %s mode=%s frames=%d fps=%g bones=%d" % (OUT_FBX, MODE, T, FPS, N))
