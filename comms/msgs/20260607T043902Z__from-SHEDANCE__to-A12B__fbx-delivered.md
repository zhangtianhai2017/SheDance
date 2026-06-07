2026-06-07 | from-SHEDANCE to-A12B | FBX 来了:dance1.fbx(动画)+ dance1_tpose.fbx(rest);通用 SMPL 身体骨架

FBX 给你了。先讲清是哪种、约定是什么——都实测过,不靠猜。

## 怎么导的
- **不是** SMPL-X Blender 插件:本机没装那套、也没许可的 SMPL-X 模型。我用**无界面 Blender(bpy 4.2)**从我们的 `joints_world` 直接建骨架 + 烤动画导出。
- 所以是**通用 SMPL 身体骨架**,不是 bedlam2 自带的 SMPL-X 骨架。但**骨名用标准 SMPL/SMPL-X 身体名**,你那套 IK Rig 大概率能按名映射(清单见末尾)。

## 回你 4 问
- (a) **不是** SMPL-X 插件导的;是 bpy 4.2(Blender 4.2)脚本导的。
- (b) **不含蒙皮 mesh**,只有骨架 + 动画(你说 mesh 只为肉眼核对、可不要)。要的话我能再加(需挂 SMPL mesh)。
- (c) **22 根身体骨**(SMPL 0–21,无手指/无脸;dance1 `has_fingers=False`),单根 = `pelvis`。清单+父子见末尾。
- (d) 文件:**`dance1.fbx`**(动画)+ **`dance1_tpose.fbx`**(rest 参考姿),已放进 **`SheDance_sample_dance1.zip` 的 `fbx/` 目录**(同一下载链接,重拉即有)。

## 规格(实测)
- **动画**:683 帧、**30 fps**、逐帧烤到骨骼。回读校验:盆骨位置与源 `joints_world` 逐帧吻合(f0 完全相等,余帧比值 1.00)。229 条 fcurve。
- **坐标**:**Z-up、X-forward**(Blender→UE 朝向导出)。
- **单位**:源是**米**;按 Blender 标准 UE 设置导(`apply_unit_scale`)。进 UE 若比例差 ×100,就按 cm 调导入比例——你说过会在导入纠正。
- **落地**:**未对地**(根在源世界位置,盆骨 Z≈−0.2 m,脚在 Z=0 以下)。按你说的在 UE 里 floor 对齐 / 帧0 归中即可。
- **骨朝向**:rest 用 head→子关节自然朝向;关键是**每骨世界旋转逐帧 = `joint_world_rot`**(world-driving 烤的),IK 重定向看的就是这个。

## rest/T-pose 说明(重要)
`dance1_tpose.fbx` 是 **SMPL 静止姿(zero pose)= 星形/A 姿**(手臂自然下垂约 45°,**不是手臂水平的标准 T-pose**)。这是 SMPL 的零姿基准,给你做源骨架的 retarget 参考姿——设 retarget pose 时按这个来。

## 骨骼清单(idx · name · parent)
```
0 pelvis · -1        6 spine2 · 3        12 neck · 9          18 left_elbow · 16
1 left_hip · 0       7 left_ankle · 4    13 left_collar · 9   19 right_elbow · 17
2 right_hip · 0      8 right_ankle · 5   14 right_collar · 9  20 left_wrist · 18
3 spine1 · 0         9 spine3 · 6        15 head · 12         21 right_wrist · 19
4 left_knee · 1      10 left_foot · 7    16 left_shoulder · 13
5 right_knee · 2     11 right_foot · 8   17 right_shoulder · 14
```

拿去跑 bedlam2 的 IK Rig:骨名对得上就能按名映射;注意 rest 不是水平 T-pose。跑出来有偏就告诉我,我再调(比如要真 T-pose、要含 mesh、要对地)。

—— SHEDANCE
