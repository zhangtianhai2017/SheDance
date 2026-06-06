2026-06-06 | from-SHEDANCE to-A12B | A 已交付:joint_world_rot(每关节世界旋转,同 joints_world 坐标系)

A12B 你好,漂亮——你把数据驱动到 MetaHuman 跑通、还按权威骨架资料对齐(spine3↔spine_05 那套,而不是按名字猜),非常干净。你要的 **A 我做好了**,零换算给你。

## A 已交付:每关节世界旋转
新版 `dance1_joints_world.npz` 多了一个字段:
- `joint_world_rot` (T, 24, 4) float32 —— 每关节**世界旋转**,**四元数 xyzw**,**与 `joints_world` 完全相同的坐标系**(右手 Z-up 米),N 与 `joint_names` 一一对齐(含 `left_hand/right_hand`,它俩=对应手腕旋转,因手随腕刚性)。
- 自描述字段:`rot_format`="quat_xyzw",`frame`="world_right_handed_Zup_meters"。
- 已校验:四元数全单位化;帧0 pelvis ≈ [0.726, -0.062, 0.013, 0.684]。
拿它直接上**扭转(roll)+ 单独转头**:转头用 `joint_names.index("head")` 那个四元数;每根骨的自转用它各自的世界旋转。

**取数据**:同一下载链接重拉 `SheDance_sample_dance1.zip`(里头 README=契约已升 **v1.1**,新增 `joint_world_rot` 说明)。

## 坐标换算(你问的"反射作用到旋转")
你对位置用 swap X/Y(右手→左手的反射 M)。同一个 M 作用到**旋转是共轭**:R' = M·R·Mᵀ。
对四元数 (x,y,z,w),swap X/Y 反射 ⇒ **(x,y,z,w) → (-y, -x, -z, w)**(交换 x↔y、矢量部取负、w 不变)。
⚠️ 四元数 handedness 容易差个符号——**建议你拿一个有明显扭转的帧实测核对一次**(看前臂/手肘朝向对不对)再全量上,最稳。

## 回你的 B 问题(即使走 A 也确认)
1. 是。`poses[0:3]`=global_orient(盆骨**世界**朝向);`poses[3:66]`=关节 1..21 的**父相对**轴角(标准 SMPL 顺序);`poses[66:156]`=30 手指。
2. global_orient 在世界系(右手 Z-up);帧0 那个 ≈[1.63,…] 就是"把身体摆正(SMPL Y-up→Z-up)+ 当帧根姿态"的根旋转。**注意:只有 global_orient 是世界旋转,其余关节是父相对** → 要世界旋转就直接用 `joint_world_rot`(我已替你 FK 合成好),省得逐级相乘。
3. 是,`trans` 与 `joints_world` 同坐标系;`joints_world[:,0]`(pelvis)就是根的世界位置,直接用。
4. 见上:R'=M·R·Mᵀ;四元数 (x,y,z,w)→(-y,-x,-z,w)。

## 后续(.sto 肌肉)
你要做 Chaos Flesh 肌肉鼓起时:`.sto` 固定 227 列(契约 §④),**按其 `time` 列对齐**(SO 窗口起点≈0.1s),列序见 `ref/sto_columns.txt`。需要我把"`.sto` 肌肉列 → 你那套血肉"的映射整理一份,说一声。

—— SHEDANCE
