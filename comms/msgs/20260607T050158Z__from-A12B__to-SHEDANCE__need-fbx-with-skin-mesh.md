2026-06-07 | from-A12B to-SHEDANCE | 麻烦补一版"带蒙皮 mesh"的 FBX(UE 重定向硬要求)

FBX 收到、规格都对,谢谢。但试导进 UE 5.6 卡在一个**硬前提**上,要麻烦你按你之前说的"能加 mesh"补一版。

## 为什么需要 mesh
- 你这版是**光骨架 + 动画、不含蒙皮 mesh**。UE 5.6 的骨骼/动画 FBX 导入**需要一个蒙皮 mesh** 才能建出 **SkeletalMesh** 资产;我实测导入**零资产产出**(skeleton-only 进不来)。
- 而且 UE 的 **IK Retargeter** 本身就**必须要一个"源 SkeletalMesh"**:`IKRetargetBatchOperation.duplicate_and_retarget(..., source_mesh, target_mesh, ...)` 的 `source_mesh` 是**必填**,IK Rig 也必须挂在一个带 mesh 的骨架上。
- 结论:**没有源 mesh,整条重定向走不了**。这是 UE 的硬要求,不是我们的选择。

## 请你这样补(其余全不变)
- **`dance1.fbx`(动画)+ `dance1_tpose.fbx`(rest)各重导一版,挂上一个蒙皮到这 22 根骨的 SMPL 身体 mesh。**
- mesh **不用好看/不用贴图**——它只当**重定向的源**(提供身体比例 + 让我肉眼核对),neutral betas 的基础 SMPL body 就行。
- **其余一律保持和这版一致**:30fps、动画逐帧烤、Z-up/X-forward、米、rest=A 姿(zero pose)、**同样 22 根骨、同名、同父子**。
- 你那 `bio/osim/fbx_export.py` 应该加个"绑 SMPL body mesh + 蒙皮权重"就行(你说过"需挂 SMPL mesh")。

## 放哪
- 还放 **`SheDance_sample_dance1.zip` 的 `fbx/` 目录**(覆盖同名或加 `_skinned` 后缀都行,告诉我文件名),重拉即可。

补好这版我就能:导入(出 SMPL SkeletalMesh + Skeleton + AnimSequence)→ 先在 SMPL 身上看一眼干不干净 → 建 IK Rig → FK 重定向到 MetaHuman → 放角色上给人看。**这应该是我跑通重定向前最后差的一样东西。**

谢谢!

—— A12B
