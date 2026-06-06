# SheDance 数据格式契约 (v1)

> 给消费方:**以此为准写消费代码**。本文锁定的是**格式(format)**;具体数值(betas、体态、某段动作)后续会更新,但下面的字段结构、命名、单位、稳定性承诺**不变**。

## 0. 唯一权威
**`<dance>_joints_world.npz` 的 `joint_names` 是"本段有哪些关节"的唯一权威。** 它永远存在、如实列出全部关节及其顺序(逐元素对应 `joints_world[:, k]`)。**请完全照它迭代,不要硬编码关节数量或下标**——这样任何变体(身体-only / 含逐指)都自动兼容。

## 1. 文件与字段(结构已锁定)

**① `<dance>_poses.npz` — SMPL-H 人形动作(驱动首选)**
- `poses` (T,156) float64 —— 52 关节 × 3 轴角。**永远 156**(不会出现窄版)。
  - `[:, 0:3]` 全局朝向 · `[:, 3:66]` 21 个身体关节 · `[:, 66:156]` 30 个手指关节槽
- `trans` (T,3) float64 —— 根位移(世界系,米)
- `betas` (16,) float64 —— 体型 · `gender` str · `mocap_framerate` float

**② `<dance>_joints_world.npz` — 关节世界坐标(开箱即用,无需 smplx)**
- `joints_world` (T,N,3) float32 —— 世界系米,Z 轴朝上(N = `joint_names` 长度)
- `joint_names` (N,) str —— ★**权威**。当前 N=24:身体 0–21(骨盆…双腕)+ `left_hand`/`right_hand`(手心点)
- `parents` (N,) int —— 父索引(建层级/连骨架用) · `fps` float

**③ `<dance>_cache.npz` — MyoFullBody qpos(肌骨模型位形)**
- `qpos` (T,nq) float64 —— nq = **89(无手指)/ 129(含手指)**。`qpos[0:7]` = 根自由关节 (x,y,z, 四元数 w,x,y,z)
- `frequency` float · `has_fingers` bool · `dof` int —— (后两个标志位即将随数据带上;在此之前按 `qpos` 宽度判)

**④ `<dance>_activation.sto` — 肌肉激活(OpenSim 文本)**
- **固定 227 列**(`time` + 226 肌肉/储备/残差),**顺序固定、与带不带手无关**。肌肉激活 ∈ [0,1]。列顺序见随附 `ref/sto_columns.txt`。

**⑤ `<dance>_ik.mot` — OpenSim IK 广义坐标**
- 49 列,`inDegrees=yes`(列见 `ref/mot_columns.txt`)。一般驱动用不上,完整起见提供。

## 2. 全局约定
- 世界系 **Z 轴朝上**,右手系。单位 **米 / 弧度 / 秒**。MuJoCo 四元数为 **w,x,y,z**。
- 旋转表示:`poses`=轴角;`qpos` 根=四元数;`.mot`=角度(度)。
- 帧对齐:`poses / joints_world / qpos / .mot` 按**行号**对齐(同帧数、同 fps);`.sto` 按其 **`time` 列**对齐(SO 时间窗,起点约 0.1s)。

## 3. 变体判定(一套代码吃两种)
| 要判断 | 依据 |
|---|---|
| 有哪些关节、什么顺序 | **`joints_world.joint_names`**(迭代它) |
| 是否带手指(姿态层) | `poses` 恒 156;`np.any(poses[:,66:156] != 0)` |
| qpos 维度 | 宽度 89/129(或 `has_fingers` 标志) |
| 肌肉列 | 固定 227 列同序,可建固定"肌肉→血肉"映射表 |

"含手"会同时体现在:`poses` 手指槽非零 + `qpos`→129 +(可能)`joints_world` 追加逐指点(都会在 `joint_names` 里命名)。`.sto` 不随之改变。

## 4. 稳定性承诺
- `joint_names` 永远是权威的自描述清单;新增关节(如逐指)会**追加并命名**,身体 22 个恒在。
- `poses` 永远 156;`.sto` 永远 227 列同序。若将来更换 OpenSim 模型导致列变化,会加 `model_version` 字段并**提前通知**,绝不静默改。
- 坐标系/单位/旋转约定不变。

## 5. 已知并已修
- `right_hand` 曾恒离右腕约 1.49m(像攥杆):是数据**生成期的命名/骨架错配 bug**(SMPL-24 命名套了 SMPL-H 骨架),**已修复**——现在 `left_hand`/`right_hand` 是真正的手心点,各离同侧手腕约 10cm。**非动作问题、非个例,消费侧无需对此做兜底。** 修正数据将随下一版一并提供。
