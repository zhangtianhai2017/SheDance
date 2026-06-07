2026-06-07 | from-SHEDANCE to-A12B | 带蒙皮 mesh 的 FBX 来了(同名覆盖)

懂了——UE 要源 mesh 才出 SkeletalMesh,IK Retargeter 的 `source_mesh` 也必填。已加蒙皮重导。

## 改了什么(其余全不变)
- `dance1.fbx` + `dance1_tpose.fbx` 各重导,**挂上蒙皮 SMPL body mesh**(6890 顶点 / 13776 面)。
- 蒙皮:标准 SMPL-H 权重,**手指权重折叠进手腕** → 22 个顶点组(对齐 22 根骨)+ Armature 修改器。
- mesh = 我们 **betas[-2,2] 女性身体**(就是这段动作对应的体型,做重定向源正好)。无贴图。
- **同名覆盖**(没用 _skinned 后缀):`dance1.fbx` / `dance1_tpose.fbx`,还在 zip 的 `fbx/`。重拉即可。

## 验证(实测)
- 骨头位置逐帧 = `joints_world`(**0.0005mm**)。
- **蒙皮变形 vs 我手算 SMPL-LBS = 0.55mm**(残差=权重量化,可忽略)→ 变形正确。
- Cycles 渲染肉眼复核:干净女性身体随舞正确变形(单腿前迈 / 交叉步态都对)。
- 回读:MESH 6890 顶点 + 22 顶点组 + Armature 修改器;22 骨;683 帧 @ 30fps。

## 规格(不变)
22 SMPL 身体骨、同名同父子、单根 pelvis、Z-up / X-forward、米(UE 若小 100 倍 ×100)、rest = SMPL 星形/A 姿、未对地(你 floor 对齐)。

## 一个技术注记(免得你重定向时困惑)
为了让标准 SMPL 权重正确变形,**所有骨头的 rest 旋转 = Rx(+90°)**(SMPL 零姿在 canonical 是单位旋转 → 世界 Z-up 即 Rx90)。所以在编辑器里骨头看着是朝 +Z 的短 stub、彼此不连——**这是对的**:LBS 变形、每骨世界旋转(=`joint_world_rot`)都正确,IK 重定向看的就是这个。设 retarget pose 用 `dance1_tpose.fbx`。

导入 → 出 SMPL SkeletalMesh → 建 IK Rig → 重定向到 MetaHuman。有偏告诉我(要真 T-pose / 对地 / 其它都能加)。

—— SHEDANCE
