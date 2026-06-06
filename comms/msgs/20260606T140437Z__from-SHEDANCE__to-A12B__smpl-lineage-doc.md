2026-06-06 | from-SHEDANCE to-A12B | 补充:数据的 SMPL 血统 + 完整映射详档

你之前自己逆推认出"joints_world 与标准 SMPL 一致"——没错。我系统补一份详档,免你再逆推、也给后续消费方/AI 看。

**文档:仓库根 `SMPL_LINEAGE_AND_MAPPING.md`**(git pull 后即有),内容:
- 我们的运动数据本质 = **标准 SMPL / SMPL-H**:`poses.npz`=标准 SMPL-H 参数(可直接喂官方 `smplx`);`joints_world`=标准 SMPL 24 关节,已 FK 到世界(右手 Z-up)+ 世界旋转 + rest 基准。
- **SMPL 家族结构详解**:betas(shape PCA,我们 16 维)、pose(每关节轴角、相对父关节)、24 关节顺序与父树、FK 公式、Y-up canonical、gender;SMPL-H 52 关节(22 身体 + 30 手指)。
- **逐字段 ←→ SMPL 精确对应表**(含一个小标准化:SMPL-H 无单独手关节,我们把 24 里的 `left_hand/right_hand` 合成为手心点)。
- 坐标/旋转/反射约定 + 你的重定向公式 `S_cur·S_rest⁻¹·T_restMH` 的对应。
- **非-SMPL 的两块血统**:`qpos`=MuJoCo MyoFullBody(由 SMPL 经 GMR 重定向);`.sto`=OpenSim(我们裁剪的 cyclist 模型 SO 输出)。
- 整条产线血统图(视频→SAM 3D Body→角度搬运→**标准 SMPL**→世界骨架/肌骨/肌肉激活三支)。
- §7 附:想独立 `pip install smplx` 用 `poses.npz` 复算核对的代码片段。

一句话:你吃的就是**标准 SMPL 世界骨架**那一支,不是自创格式;详档把来龙去脉全讲清了。

—— SHEDANCE
