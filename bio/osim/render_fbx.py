#!/usr/bin/env python3
"""Quick visual sanity render of a skinned FBX: import it, auto-frame the animation, workbench-render
N evenly-spaced frames to PNGs. For eyeballing that the skinned body deforms correctly.
Usage: blender_python render_fbx.py <fbx> <out_dir> [n_frames]"""
import sys, os, bpy
from mathutils import Vector
FBX, OUT_DIR = sys.argv[-3], sys.argv[-2]
NF = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 12
os.makedirs(OUT_DIR, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX)
mesh = [o for o in bpy.data.objects if o.type == "MESH"][0]
sc = bpy.context.scene
act = [o for o in bpy.data.objects if o.type == "ARMATURE"][0].animation_data
f0, f1 = (int(act.action.frame_range[0]), int(act.action.frame_range[1])) if act else (1, 1)

# bbox over sampled frames (world space, evaluated)
dg = bpy.context.evaluated_depsgraph_get()
mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
for f in range(f0, f1 + 1, max(1, (f1 - f0) // 8)):
    sc.frame_set(f); dg.update()
    me = mesh.evaluated_get(dg)
    for c in me.bound_box:
        w = me.matrix_world @ Vector(c)
        mn = Vector((min(mn[i], w[i]) for i in range(3))); mx = Vector((max(mx[i], w[i]) for i in range(3)))
center = (mn + mx) / 2.0; size = max((mx - mn)) or 2.0

cam_data = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cam_data); sc.collection.objects.link(cam)
cam.location = center + Vector((size * 0.35, -size * 1.9, size * 0.45))
d = center - cam.location
cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
sun = bpy.data.objects.new("S", bpy.data.lights.new("S", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 3.0; sun.rotation_euler = (0.6, 0.2, 0.5)
bpy.ops.mesh.primitive_plane_add(size=max(8.0, size * 4), location=(center.x, center.y, 0.0))   # ground at Z=0
gp = bpy.context.active_object; gm = bpy.data.materials.new("ground"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.30, 0.34, 1)
gp.data.materials.append(gm)

mat = bpy.data.materials.new("body"); mat.use_nodes = True
b = mat.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (0.8, 0.55, 0.5, 1); b.inputs["Roughness"].default_value = 0.6
mesh.data.materials.append(mat)
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.06, 0.06, 0.08, 1)
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 16    # CPU: no GL needed (headless WSL)
sc.render.resolution_x = 512; sc.render.resolution_y = 640; sc.render.film_transparent = False

frames = [f0 + round(i * (f1 - f0) / max(1, NF - 1)) for i in range(NF)]
for k, f in enumerate(frames):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(OUT_DIR, "f%03d_frame%04d.png" % (k, f))
    bpy.ops.render.render(write_still=True)
print("RENDERED %d frames -> %s (body center %s size %.2f)" % (len(frames), OUT_DIR, tuple(round(x,2) for x in center), size))
