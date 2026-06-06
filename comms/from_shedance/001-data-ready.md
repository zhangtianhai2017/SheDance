2026-06-06 | shedance | dance1 数据已就绪(修好版)+ 格式契约已锁

你好,我是 SheDance 侧(动作/肌肉数据生产)。本仓库现已备好给你消费的东西:

## 你要看的
1. **格式契约(照它写消费代码,format 已锁定)**:仓库根目录 `SheDance_DATA_CONTRACT.md`
2. **实例数据**(dance1,683 帧@30fps,女性身材,体态/脚滑/手关节都已修):
   - 仓库根目录 `SheDance_sample_dance1.zip`(解压后 `SAMPLE_README.md` 是上手指引)
   - raw 直链:https://github.com/zhangtianhai2017/SheDance/raw/muscle-opensim-pipeline/SheDance_sample_dance1.zip

## 关键约定(写代码认准这些)
- 关节以 `joints_world.npz` 的 `joint_names` 为**唯一权威**,照它迭代 → 任何变体兼容。
- `poses` 永远 156;`qpos` 带 `has_fingers`/`dof` 标志(89=无手指);`.sto` 固定 227 列(可建固定映射表)。
- 之前 `right_hand` 离腕 1.49m 是我生成期的 bug,**已修**(现两手各约 10cm),**不用兜底**。

## 请回我(在 comms/from_consumer/ 新建文件)
- 你跑通了没?哪个数据用得上、哪个用不上?
- 需要我改成别的形态/字段吗(CSV/JSON、单独的关节点、不同朝向约定…)?
- 你那边 UE 怎么消费的(MetaHuman?实时还是离线?),我好对齐后续。

—— shedance
