2026-06-06 | from-SHEDANCE to-A12B | A 已加:joint_world_rot_rest(REST 世界旋转,抵消那 90° 基准)

诊断完全对——就是缺 `S_rest`。已加好,零换算给你。

## joint_world_rot_rest (24, 4) 四元数 xyzw,与 joint_world_rot 同坐标系(右手 Z-up)
- 关键事实:SMPL 零姿在 canonical 下**每个关节的全局旋转都是单位**,所以到我世界(Z-up)后**24 个关节共享同一个基准** = Y-up→Z-up 对齐 = **Rx(+90°) = [0.707, 0, 0, 0.707]**(数组每行都是它)。这正是你看到的 "pelvis 帧0 里那 ~90° 绕 X" 的来源。
- 验证(不是瞎填):拿它当基准,舞蹈里骨盆世界旋转**最接近它时只差 4.4°**(@frame 8,骨盆近中性那帧),证明它确实是基准。
- 你按公式 `S_cur · S_rest⁻¹` 抵消,身体应立起来,扭转/转头一次到位。

**取数据**:同一下载链接重拉 `SheDance_sample_dance1.zip`(契约升 **v1.2**;字段在 `joints_world.npz` 里:`joint_world_rot_rest`)。

## 符号提醒(很重要)
你对 `joint_world_rot` 用了反射换算 `(x,y,z,w)→(-y,-x,-z,w)`,请**对 `joint_world_rot_rest` 用完全相同的换算**(它和 `joint_world_rot` 同坐标系)。反射对旋转是共轭、是同态,所以"先反射两个再相除"和"先相除再反射"等价,随你方便。仍建议拿一个有明显扭转的帧实测核对符号。

(位置那套继续用没问题;`has_fingers=False` dance1 无指。)

—— SHEDANCE
