# CLAUDE.md — SheDance 核心指导（新 AI / 协作者必读）

**最近更新**: 2026-06-03
**一句话**: 看一段动作 → 让 UE MetaHuman **用肌肉物理**把它跳出来。

> 📖 **最高策划文件**: [`docs/SheDance_master_plan.md`](docs/SheDance_master_plan.md) —— 哲学+路线全在那,**先读它**。
> 本文件是它的精炼必读版 + 速查指针。旧 [`docs/design.md`](docs/design.md) 的方法部分(RL/S阶梯)是 5-31 早期框架,**已被 master_plan 取代**。

---

## ⚠️ 灵魂(不可动摇,任何实现都服从于此)

1. **我们仿真的是"一个模仿能力强的人"。** 输出**不是复制**输入动作,是**这具身体(有强处/短处)对意图的再创作**。是这个仿真人跳的舞。
2. **输入神圣,不要动它(连消抖都不做)。** 输入=要模仿的意图;**唯一变换在身体**(肌肉 F_max + 动力学)。改输入会污染输入/输出边界。
3. **肌肉能力 F_max = 物理性过滤器 + 角色个性。** 不可能的抖动需要超人力→肌肉产不出→自然削平(物理去噪);F_max ×2-3=刚猛舞者,×0.7=柔弱跳得软。F_max 就是"舞者内生模型"旋钮。
4. **任意输入→模仿。** 输入可为动捕/大猩猩/木偶/机器;系统用自己的身体尽力模仿。通用形态=跟踪**关键点**而非关节角(支持非人骨架)。这打开"任意动作→生成舞蹈"。

---

## 📍 现状(2026-06-03)

- ✅ **路线已验证**:OpenSim Static Optimization 证明**肌肉能产出整段舞蹈的全部关节力矩**(手搓逆向时"力矩超能力100×"的病理消失)。工具链/模型/运动转换/渲染全通。
- 🔴 **当前焦点 = 实现干净的 B(能力受限前向跟踪)**:让肌肉上限**塑造**输出(做不到处自然偏离)。这要**前向**方法,不是 SO 的精确复现。
  - 难点:干净的能力受限前向跟踪是这条路最硬的一环(力自洽+饱和+稳定)——**正是当初用 RL 的原因**。
  - MocoTrack/Inverse 算力不可行;自建前向 CMC 两版没干净解决;**当前在试 OpenSim CMCTool**。

---

## 🗂 速查

| 找... | 去 |
|------|----|
| **哲学+路线全文** | `docs/SheDance_master_plan.md` |
| 自建逆向(已弃,原理参考) | `docs/biomechanics_inverse_approach.md`、`bio/` |
| **OpenSim 路线代码(在用)** | `bio/osim/`:`convert_qpos_to_mot.py`(运动转换)、`static_opt_osim.py`(SO 基线)、`cmc_tool.py`(CMC)、`cmc_mujoco*.py`(自建前向)、`prune_model.py`、`render_muscle.py`、`analyze_so.py` |
| OpenSim 模型 | `~/shedance/osim/cyclistFullBodyMuscle.osim`(WSL,与 MyoFullBody 同源,171DOF/520肌肉)、`cyclist_pruned.osim`(377肌肉) |
| OpenSim 环境 | WSL `~/shedance/osimenv`(uv venv,`pip opensim 4.6` **含 Moco**) |
| MuJoCo 肌肉模型/重定向 | `~/shedance/musclemimic`(MyoFullBody qpos 缓存、渲染) |
| 渲染产物 | `~/shedance/renders/`、`renders/`(Windows 侧) |
| 算力墙诊断 | `docs/compile_bottleneck_findings.md` |

## 🔑 关键技术事实(别重新踩坑)

- `pip opensim`(4.6)在 **Linux 就带 Moco**(网上说没有=过时)。大肌肉模型上 **SO/CMC(逐帧/前向)>> Moco(轨迹优化,太慢)**。
- 运动:MyoFullBody qpos 按**坐标名直接转 OpenSim `.mot`**(同源,名字对得上)。
- OpenSim Tool 可靠用法:`printToXML` 存 setup 再从 XML 重载 `run()`。Moco 不支持 locked 坐标→`ModOpReplaceJointsWithWelds` 焊掉。
- 储备:pelvis 残差强(扛 GRF 缺口)、关节储备弱/贵(逼肌肉)。详见 memory `reference-opensim-moco-wsl`。

## 🤝 协作

- 用户**主导架构**:先听 trade-off 分析再动手;长期工作要可回退 commit;按**总体路线节点**同步,不必每步汇报。
- **别"逻辑转圈"**:改动不改变结果→停下做受控根因测试,别一直换变量甩锅。
- 中文交流。运行时(Qwen/UE)用户常让常驻,**别自己 kill**。

## 🚦 任何 session 第一件事

1. 读本文件 → 读 `docs/SheDance_master_plan.md`(尤其 §1 哲学)。
2. 看 git status 是否 mid-feature;看 `bio/osim/` 最近改了什么。
3. 任务涉及肌肉求解 → 先想清楚是"精确复现(SO)"还是"能力受限塑造(B/CMC)",二者目标不同。
