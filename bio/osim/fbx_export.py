#!/usr/bin/env python3
"""Export a SKINNED SMPL-body FBX (skeleton + skinned mesh + baked animation) from our joints_world,
via headless bpy. UE needs a skinned mesh to make a SkeletalMesh + to run the IK Retargeter (source_mesh
is mandatory), so skeleton-only is not importable.

22 body bones (SMPL 0-21, SMPL-X body names). Bone REST rotation = Rx(+90deg) for ALL bones, because SMPL
zero-pose has identity joint global rotation in canonical (Y-up) -> Rx90 in world; this makes the standard
SMPL skinning weights deform correctly (LBS: M[j] = G_pose[j] @ G_rest[j]^-1, G_rest rotation must be Rx90).
Animation = world-driving (pose_bone.matrix per frame from joint_world_rot/joints_world); positions exact.
Mesh = our_humanoid verts/faces (betas [-2,2] female) + SMPL-H weights folded fingers->wrists -> 22 groups.

Usage: blender_python fbx_export.py <joints_world.npz> <our_humanoid.npz> <out.fbx> <test|anim|rest>
"""
import sys, numpy as np, bpy
from mathutils import Matrix, Quaternion, Vector

JW_NPZ, HUM_NPZ, OUT_FBX, MODE = sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
PKL = "/mnt/c/work/2026/Claude/SheDance/backup/smpl/SMPLH_NEUTRAL.pkl"   # for skin weights (6890x52)
RX90_np = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)            # canonical Y-up -> world Z-up
RX90_M = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

z = np.load(JW_NPZ, allow_pickle=True)
JW = z["joints_world"].astype(float); JWR = z["joint_world_rot"].astype(float)
_an = [str(x) for x in z["joint_names"]]; N = len(_an) if len(_an) > 24 else 22   # 22 body, or 52 with articulated fingers
NAMES = _an[:N]; PAR = [int(x) for x in z["parents"]][:N]
FPS = float(z["fps"]); T = JW.shape[0]
hum = np.load(HUM_NPZ, allow_pickle=True)
restJ = hum["joints"].astype(float)[:N]
verts = hum["vertices"].astype(float); faces = hum["faces"].astype(int)
rest_world = (RX90_np @ restJ.T).T
verts_world = (RX90_np @ verts.T).T

import pickle
w52 = np.asarray(pickle.load(open(PKL, "rb"), encoding="latin1")["weights"], float)   # (6890,52)
w22 = w52[:, :N].copy()
if N <= 24:                                         # body rig: fold fingers into the wrists (the 52-bone rig keeps them articulated)
    w22[:, 20] += w52[:, 22:37].sum(1)              # left fingers (22-36) -> left_wrist
    w22[:, 21] += w52[:, 37:52].sum(1)              # right fingers (37-51) -> right_wrist

# ---- floor-align: lowest SKINNED-MESH point over the clip -> Z=0, + frame0 horizontal -> origin.
# LBS is equivariant under a global rigid shift of (bind skel, bind mesh, pose skel), so shifting all
# three by the same s grounds the animation cleanly. dz from the posed mesh (not joints) => no sole sink.
def _q2m(q):                                         # (M,4) xyzw -> (M,3,3)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1-2*(y*y+z*z); m[:, 0, 1] = 2*(x*y-z*w); m[:, 0, 2] = 2*(x*z+y*w)
    m[:, 1, 0] = 2*(x*y+z*w); m[:, 1, 1] = 1-2*(x*x+z*z); m[:, 1, 2] = 2*(y*z-x*w)
    m[:, 2, 0] = 2*(x*z-y*w); m[:, 2, 1] = 2*(y*z+x*w); m[:, 2, 2] = 1-2*(x*x+y*y); return m
vrest_h = np.concatenate([verts_world, np.ones((verts_world.shape[0], 1))], axis=1)
Grest = np.tile(np.eye(4), (N, 1, 1)); Grest[:, :3, :3] = RX90_np; Grest[:, :3, 3] = rest_world
Grest_inv = np.linalg.inv(Grest)
if MODE in ("anim", "test"):
    minz = 1e18
    for t in range(T):
        Gp = np.tile(np.eye(4), (N, 1, 1)); Gp[:, :3, :3] = _q2m(JWR[t, :N]); Gp[:, :3, 3] = JW[t, :N]
        row2 = (Gp @ Grest_inv)[:, 2, :]                     # (N,4) Z-row of each bone LBS transform
        minz = min(minz, float(((vrest_h @ row2.T) * w22).sum(1).min()))
    lf0 = 10 if JW[0, 10, 2] < JW[0, 11, 2] else 11        # initial-contact foot (lower toe at frame 0)
    SH = np.array([-float(JW[0, lf0, 0]), -float(JW[0, lf0, 1]), -minz])   # origin = floor(Z=0) + initial contact point
    JW = JW + SH
else:                                                # rest: ground the bind mesh itself
    SH = np.array([-float(rest_world[0, 0]), -float(rest_world[0, 1]), -float(verts_world[:, 2].min())])
rest_world = rest_world + SH; verts_world = verts_world + SH
print("FLOOR shift dx=%.3f dy=%.3f dz=%.3f (mode %s)" % (SH[0], SH[1], SH[2], MODE))

def quat_wxyz(q):
    return Quaternion((q[3], q[0], q[1], q[2]))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = int(round(FPS))

# ---- armature: Rx90-rest bones ----
arm = bpy.data.armatures.new("SMPL"); arm_obj = bpy.data.objects.new("SMPL", arm)
bpy.context.scene.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj; arm_obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
ebs = []
for j in range(N):
    eb = arm.edit_bones.new(NAMES[j]); eb.use_connect = False
    eb.head = Vector(rest_world[j]); eb.tail = Vector(rest_world[j]) + Vector((0, 0, 0.10))
    M = RX90_M.copy(); M.translation = Vector(rest_world[j]); eb.matrix = M   # enforce Rx90 rest rotation
    ebs.append(eb)
for j in range(N):
    if PAR[j] >= 0:
        ebs[j].parent = ebs[PAR[j]]
bpy.ops.object.mode_set(mode="OBJECT")

# ---- skinned mesh ----
mesh = bpy.data.meshes.new("SMPL_body")
mesh.from_pydata([Vector(v) for v in verts_world], [], [list(map(int, f)) for f in faces])
mesh.update()
mo = bpy.data.objects.new("SMPL_body", mesh); bpy.context.scene.collection.objects.link(mo)
vgs = [mo.vertex_groups.new(name=NAMES[j]) for j in range(N)]
for v in range(verts_world.shape[0]):
    for j in range(N):
        wv = w22[v, j]
        if wv > 1e-4:
            vgs[j].add([v], float(wv), "REPLACE")
mo.parent = arm_obj
mod = mo.modifiers.new("Armature", "ARMATURE"); mod.object = arm_obj

bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="POSE")
for pb in arm_obj.pose.bones:
    pb.rotation_mode = "QUATERNION"

def pose_frame(t):
    for j in range(N):
        M = Matrix.LocRotScale(Vector(JW[t, j]), quat_wxyz(JWR[t, j]), Vector((1, 1, 1)))
        arm_obj.pose.bones[NAMES[j]].matrix = M
        bpy.context.view_layer.update()

if MODE == "test":
    # 1) head positions exact?  2) skinning matches my own SMPL-LBS?
    G_rest = [Matrix.LocRotScale(Vector(rest_world[j]), RX90_M.to_quaternion(), Vector((1, 1, 1))) for j in range(N)]
    herr = []; derr = []
    dg = bpy.context.evaluated_depsgraph_get()
    for t in (0, T // 2, T - 1):
        pose_frame(t)
        for j in range(N):
            herr.append((Vector(arm_obj.pose.bones[NAMES[j]].head) - Vector(JW[t, j])).length)
        dg.update()
        meval = mo.evaluated_get(dg).data
        Gp = [Matrix.LocRotScale(Vector(JW[t, j]), quat_wxyz(JWR[t, j]), Vector((1, 1, 1))) for j in range(N)]
        Mj = [Gp[j] @ G_rest[j].inverted() for j in range(N)]
        for v in range(0, verts_world.shape[0], 343):          # ~20 sample verts
            ref = Vector((0, 0, 0)); vr = Vector(verts_world[v])
            for j in range(N):
                if w22[v, j] > 1e-4:
                    ref += float(w22[v, j]) * (Mj[j] @ vr)
            derr.append((meval.vertices[v].co - ref).length)
    print("TEST head-pos max err = %.4f mm | skin-deform max err vs SMPL-LBS = %.4f mm" % (max(herr) * 1000, max(derr) * 1000))
    print("TEST bones=%d verts=%d faces=%d" % (N, verts_world.shape[0], faces.shape[0]))
elif MODE == "rest":
    pass                                                        # bind pose = standing rest skinned body
else:
    bpy.context.scene.frame_start = 0; bpy.context.scene.frame_end = T - 1
    for t in range(T):
        bpy.context.scene.frame_set(t)
        pose_frame(t)
        for j in range(N):
            pb = arm_obj.pose.bones[NAMES[j]]
            pb.keyframe_insert("location", frame=t); pb.keyframe_insert("rotation_quaternion", frame=t)

if MODE != "test":
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.data.objects:
        o.select_set(o in (arm_obj, mo))
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_scene.fbx(
        filepath=OUT_FBX, use_selection=True, object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False, bake_anim=(MODE == "anim"), bake_anim_use_all_bones=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0, mesh_smooth_type="FACE",
        use_mesh_modifiers=False, axis_up="Z", axis_forward="X",
        apply_unit_scale=True, global_scale=1.0, primary_bone_axis="Y", secondary_bone_axis="X")
    print("EXPORTED %s mode=%s frames=%d fps=%g bones=%d verts=%d" % (OUT_FBX, MODE, T, FPS, N, verts_world.shape[0]))
