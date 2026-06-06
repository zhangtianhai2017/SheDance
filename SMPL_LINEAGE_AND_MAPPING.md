# SheDance 数据的 SMPL 血统与精确映射（消费方 / AI 详读）

> 目的:把"我们的数据是由广泛使用的 **SMPL** 家族派生/标准化而来"讲透,给出**逐字段对应关系** +
> **SMPL 本身的数据结构**,让你(或你的 AI)无需逆推就能完全理解、并可选择直接用官方 `smplx` 复算。
> 配套:格式契约见 `SheDance_DATA_CONTRACT.md`;本文讲"血统与换算",契约讲"字段与稳定性"。

---

## 0. TL;DR（先看这一段）
- **运动数据本质就是标准 SMPL / SMPL-H**,不是自创格式:
  - `poses.npz` = **标准 SMPL-H 参数**(betas + 轴角 pose + gender)→ 可直接喂官方 `smplx`。
  - `joints_world.npz` = **标准 SMPL 24 关节骨架**,我们替你做好了**正运动学(FK)→ 世界坐标**(右手 Z-up),
    并附**世界旋转**与**rest 基准**。等于"标准 SMPL 骨架 + 开箱即用的世界姿态",免你装 smplx。
- **不是 SMPL 的两块**(它们是另外两个标准,见 §5):
  - `qpos.npz` = MuJoCo **MyoFullBody** 肌骨模型位形(89-dof),由 SMPL 经 **GMR** 重定向而来。
  - `.sto` = **OpenSim** 肌肉激活(我们裁剪的 cyclist 模型 Static Optimization 输出)。
- 你已自验 `joints_world` 与官方 `vchoutas/smplx` 一致 —— 没错,它就是标准 SMPL 24 关节。

---

## 1. 原始标准:SMPL 家族(Max Planck;官方包 `smplx`,github.com/vchoutas/smplx)

**SMPL**(Loper et al. 2015):参数化人体模型。
- 输入:**shape** `betas`(PCA 系数,常见 10 维,可扩到 16/300;我们用 **16**)+ **pose**(每关节**轴角** axis-angle)。
- 输出:**6890 顶点网格** + **24 个关节**(可经 J_regressor 再多出若干 landmark)。
- 关节数 **24**,标准顺序(索引→名):
  `0 pelvis,1 left_hip,2 right_hip,3 spine1,4 left_knee,5 right_knee,6 spine2,7 left_ankle,8 right_ankle,9 spine3,10 left_foot,11 right_foot,12 neck,13 left_collar,14 right_collar,15 head,16 left_shoulder,17 right_shoulder,18 left_elbow,19 right_elbow,20 left_wrist,21 right_wrist,22 left_hand,23 right_hand`
- 父索引(运动学树,-1=根):
  `[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]`
- **正运动学(FK)**:每关节全局旋转 `G_j = G_parent · R(pose_j)`(根 `G_0 = R(global_orient)`);
  全局位置 `p_j = p_parent + G_parent · (restJ_j − restJ_parent)`。这正是我们 `joints_world` 的算法。
- **pose 表示 = 轴角**:每关节 3 个数,模长=转角(弧度),方向=转轴;**相对父关节**(根 `global_orient` 除外,它是世界朝向)。
- **零姿(rest/canonical)**:所有 pose=0,模型直立。约定上 SMPL 模板**纵轴朝 +Y(Y-up)**;
  零姿下**每个关节的全局旋转都是单位阵**(关节本地轴与世界轴对齐)。**这一点很关键**(见 §4 的 rest 基准)。
- **gender**:male / female / neutral 三套模型(形体不同);我们用 **female**。

**SMPL-H**(SMPL + MANO 双手):
- 关节 **52** = 22 身体(0–21,**没有单独的 22/23 手关节**)+ 30 手指(每手 15:左 22–36、右 37–51)。
- pose 维度 **156 = 52×3**。手指顺序见契约/`ref`。
- 我们的 `poses.npz` 就是 SMPL-H(156)。

**SMPL-X**:再加下颌/眼/表情。**我们没用**(提一句,免混淆)。

---

## 2. 逐字段:我们的数据 ←→ SMPL 的精确对应

| 我们的字段 | 是不是标准 SMPL | 精确关系 |
|---|---|---|
| `poses.npz: poses (T,156)` | **是,标准 SMPL-H 轴角** | `[0:3]`=global_orient(根**世界**朝向);`[3:66]`=身体关节 1..21 的**父相对**轴角;`[66:156]`=30 手指。直接喂 `smplx(model_type="smplh", num_betas=16)`。 |
| `poses.npz: betas (16,)` | **是,标准 SMPL shape** | PCA 形状系数(我们=female、`[-2,2,0,…]`)。 |
| `poses.npz: gender` | 是 | "female"。 |
| `poses.npz: trans (T,3)` | 是,标准根平移 | 世界系米(右手 Z-up)。 |
| `joints_world.npz: joints_world (T,24,3)` | **标准 SMPL 24 关节**,FK 到世界 | = 用上面 poses 跑 SMPL FK 得到的前 24 个关节的**世界位置**(右手 Z-up,米)。`joint_names`/`parents` = 标准 SMPL(§1)。 |
| `joints_world.npz: joint_world_rot (T,24,4)` | 标准 FK 的副产物 | 每关节**世界旋转**(四元数 xyzw)= FK 里的 `G_j`,我们替你合成好(省得你逐级相乘)。同坐标系。 |
| `joints_world.npz: joint_world_rot_rest (24,4)` | 派生常量 | **零姿**每关节世界旋转 = 重定向基准 `S_rest`。因 SMPL 零姿在 canonical 下全局旋转皆为单位,转到我们世界(Z-up)后**24 关节共享同一个** = Y-up→Z-up 对齐 = `Rx(+90°)`=[0.707,0,0,0.707]。 |

**一个我们做的小标准化**:SMPL-H 没有"单独的手关节",但**标准 SMPL(非 H)有 22=left_hand、23=right_hand**。
为了让 `joints_world` 回到标准 24 关节,我们把 22/23 合成为**手心点**(该手 15 个 MANO 手指关节的均值,随手腕刚性)。
即:`joints_world` 是 **SMPL-24 视角**,`poses.npz` 是 **SMPL-H(52)视角**,二者同一具身体、不同粒度。

---

## 3. 你有两种消费方式(任选)
- **(A) 直接用世界空间骨架(最省事,你现在走的就是这条)**:`joints_world` + `joint_world_rot`(+ `joint_world_rot_rest` 做基准抵消)。不必装 smplx。
- **(B) 自己跑官方 SMPL(要网格/更强控制时)**:装 `pip install smplx`,用 `poses.npz` 的 betas+poses+gender 复算,得到 6890 顶点网格 + 52 关节。与我们 `joints_world` 的前 24 关节一致(差一个 §4 的世界对齐)。

---

## 4. 坐标 / 约定 / 易错点(务必看)
- **世界系**:右手、**Z 轴朝上**、米。SMPL canonical 是 **Y-up** → 我们用 `Rx(+90°)` 转到 Z-up(这就是 `joint_world_rot_rest` 的来源)。
- **旋转表示**:`poses`=轴角(SMPL 原生);`joint_world_rot`/`_rest`=四元数 **xyzw**;MuJoCo `qpos` 根=四元数 **wxyz**;`.mot`=度。
- **FK 合成**:`joint_world_rot` 已是世界旋转(`G_j`);`poses` 里除 global_orient 外都是**父相对**——要世界旋转就用 `joint_world_rot`,别拿 poses 的关节轴角当世界量。
- **右手→左手(给 UE)**:位置常用 swap X/Y;同一反射 `M` 作用到旋转是**共轭** `R'=M·R·Mᵀ`,四元数 `(x,y,z,w)→(-y,-x,-z,w)`;**对 `joint_world_rot` 和 `joint_world_rot_rest` 用同一个换算**;反射是同态,先换算两个再相除、或先相除再换算,等价。符号易错,建议拿一帧实测核对。
- **重定向到你骨架**(你已在做):`T_cur[骨] = S_cur · S_rest⁻¹ · T_restMH`,其中 `S_cur`=`joint_world_rot`、`S_rest`=`joint_world_rot_rest`、`T_restMH`=你骨架静止姿世界朝向。

---

## 5. 非 SMPL 的两块(另外两个标准,血统说明)
- **`qpos.npz`(MyoFullBody)**:MuJoCo 的肌骨人体模型(89 自由度,disable_fingers;129 含手指)。由 SMPL-H 经 **GMR(General Motion Retargeting)** 重定向得到 —— 即"把 SMPL 动作搬到 MyoFullBody 骨架"。关节顺序见 `ref/myofullbody_layout.txt`。**不是 SMPL,别用 SMPL 约定解读它。**
- **`.sto`(OpenSim)**:逐肌肉激活,来自我们**裁剪过的 OpenSim 模型**(完整 `cyclistFullBodyMuscle` ~350 肌 → 我们裁的 `cyclist_min` ~149 肌 + 储备/残差作动器)经 Static Optimization。固定 227 列,见 `ref/sto_columns.txt`。**这是 OpenSim 标准,不是 SMPL。**

---

## 6. 整条产线血统(从视频到这些数据)
```
任意单人舞蹈视频
  → VedioTo3D / SAM 3D Body(Meta;输出 MHR 127 关节 + 70 个 COCO-WholeBody 关键点 + 网格)
  → 角度搬运(把 70 关键点的骨向套到“我们的 SMPL-H 人形”上,女 betas[-2,2],修好体态/脚锁定)
  → 【SMPL-H 姿态】= poses.npz  ← 这是“回到标准 SMPL”的关键节点
        ├─ FK → joints_world.npz（标准 SMPL 24 关节世界姿态 + 旋转 + rest 基准）   ← 你现在用的
        ├─ GMR 重定向 → MyoFullBody qpos.npz（MuJoCo 肌骨）
        └─ OpenSim Static Optimization → *.sto（逐肌肉激活）
```
一句话:**视频 → 标准 SMPL → (世界骨架 / 肌骨 / 肌肉激活) 三种下游**。你吃的是"标准 SMPL 世界骨架"那支。

---

## 7. 想自查一致性?
```python
import numpy as np, smplx, torch
d = np.load("dance1_poses.npz", allow_pickle=True)
m = smplx.create("<SMPL-H 模型目录>", model_type="smplh", gender="female", num_betas=16, use_pca=False, flat_hand_mean=True)
out = m(betas=torch.tensor(d["betas"][None],dtype=torch.float32),
        global_orient=torch.tensor(d["poses"][:1,0:3],dtype=torch.float32),
        body_pose=torch.tensor(d["poses"][:1,3:66],dtype=torch.float32),
        transl=torch.tensor(d["trans"][:1],dtype=torch.float32))
# out.joints[:, :24] 应与我们 joints_world 的对应关节一致（差一个 Y-up→Z-up 世界对齐 Rx+90°）
```
> SMPL-H 模型本体受马普所授权,需自行注册下载(mano.is.tue.mpg.de / smpl.is.tue.mpg.de);
> 但**只消费我们导出的数值,不需要该模型**——上面是给你想独立复算/核对时用。
