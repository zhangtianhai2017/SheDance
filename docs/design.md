# SheDance 设计文档

> ⚠️ **2026-06-03 注**: 本文的**方法部分(RL / S1-S4 阶梯)是 5-31 早期框架,已被
> [`SheDance_master_plan.md`](SheDance_master_plan.md) 取代**(RL 已否决 → 生物力学逆向 → OpenSim;
> 并确立了"会模仿的身体 / 输入神圣 / F_max=角色 / 任意输入→模仿"的核心哲学)。
> 本文的**高层目标(肌肉物理身体、细节自生、双流、身体分层、手指风格层)仍有效**,可作参考。
> 新接手请先读 master_plan。

**最近更新**: 2026-05-31（建仓 + 环境探测）
**阶段**: M0 哑管道（未开始）；S0 已完成环境探测部分

---

## 1. 目标与核心洞察

离线工具：看一段舞蹈视频 → UE 5.6 MetaHuman 跟跳。市面"视频动捕→重定向"已多，本项目
要解决其痛点：**手指、脚掌、重心转移等细节不该全靠识别**。

核心洞察：把角色建成一个**肌肉物理仿真的身体**，"本身就会跳某一类舞"。视频只约束主干 + 给节拍，
看不见的细节由"会跳舞的身体"在物理里跑出来。

## 2. 架构

**A. 双流（横向）**——仿真 ⊕ 识别，在"带置信度的追踪目标"处汇合：

```
识别流：单目视频 → 3D 姿态 + 每关节置信度 + 节拍  ─┐
                                                  ├─► 带置信度追踪目标 ─► RL 肌肉控制器 ─► 动作
仿真流：肌肉骨骼身体 + 该舞种控制策略（"会跳舞")─┘
```

高置信关节硬追，低置信/看不见的交给物理 + 控制器。

**B. 身体分层（纵向，按"质量/接触"决定要不要物理）**

- **物理仿真层**：腿、胯、躯干、重心、落脚——有质量、有接触、会打滑 → 必须物理（脚滑天然消失）。
- **风格化运动层**：手指、腕花、脸——几乎没质量、不承重 → **不交给肌肉物理**，按
  "舞种 + 节拍 + 腕部状态"驱动的程序/学习层。直击手指痛点。

## 3. 现成地基（2026-05-31 检索确认）

| 用途 | 用什么 | 备注 |
|------|--------|------|
| 肌肉身体 + RL 模仿（**主底座**） | **MuscleMimic**（[amathislab/musclemimic](https://github.com/amathislab/musclemimic)，Apache-2.0，JAX/MJWarp）：**MyoFullBody 416 肌肉 / 123 关节 / 72 DOF，全身带手臂+手指**（finger 可禁），自带 SMPL/AMASS 重定向 + GMR-FIT IK | 训练需 Linux+NVIDIA（A6000 ✓）。Demo cache 预重定向、跑 demo 不需 AMASS。**全身肌肉+复杂舞蹈仍无公开成功案例** = 真正研究风险 |
| 肌肉身体（**下肢参考/兜底**） | **KINESIS**（ICRA 2026，[amathislab/Kinesis](https://github.com/amathislab/Kinesis)）：legs(80)/legs_abs(86)/legs_back(290)，**仅下半身、无手臂** | 同实验室。416 全身"coming soon"未释出。conda py3.8+torch cu116 |
| 仿真引擎 | MuJoCo + MyoSuite | WSL2/CUDA |
| 舞蹈数据 | **AIST++**（[主页](https://google.github.io/aistplusplus_dataset/) / [API](https://github.com/google/aistplusplus_api)） | 5.2h / 1408 段 / 10 舞种 / 多视角已知机位 / SMPL+3D。多视角干净 3D → 训练期解耦识别误差 |
| 识别流 | 轨迹优化谱系：Rempe 2020《Contact and Human Dynamics from Monocular Video》、2022《Trajectory Optimization for Physics-Based Reconstruction》 | 已能估接触、消脚滑/穿地 |
| 重定向 | UE 5.6 IK Rig + IK Retargeter | 输出 AnimSequence / FBX |

## 4. 可回退阶梯

> 集成风险前置、研究风险隔离。

- **M0 哑管道**：AIST++ 干净 3D → MetaHuman → UE 直接播放。不碰肌肉/物理/识别。前置最易翻车的重定向。
- **S0 复现肌肉地基**：WSL2 跑通 MuJoCo + MuscleMimic/KINESIS 基础模仿（走/平衡）。
- **S1 肌肉身体跳这支舞**（核心研究风险）：喂 AIST++ 干净参考训控制器 → 物理合理、无脚滑、细节涌现。**v1 主交付**。
- **S2 接入识别**：换成单目视频估计姿态+置信度。（建议 v1.5/v2）
- **S3 手指/风格层**：解耦手势层嫁接。v1 先固定/程序化手型占位。
- **S4 打磨**：微调闭环、批量化、多机位/多舞种。

**v1 = M0 + S0 + S1**（干净参考驱动，单舞种单角色，MetaHuman 在 UE 里跳）。真视频识别(S2)放 v1.5/v2。

## 5. 关键风险

1. **全身肌肉+舞蹈未被公开验证**（S1）。兜底：KINESIS 式混合（腿肌肉/臂力矩）；先挑慢舞句。
2. ~~SMPL → 肌肉骨架重定向~~ → **大幅缓解**：MuscleMimic 自带 SMPL/AMASS 重定向 + GMR-FIT IK。
6. **数据/访问 license 墙**（需用户亲自做，我不能代办）：
   - MuscleMimic demo cache：HuggingFace 上对 `amathislab/demo_dataset` 申请访问 + 建 token（轻，跑 S0 demo 用）。
   - SMPL 身体模型（smpl.is.tue.mpg.de 注册）：重定向**自己的** AIST++ 舞蹈数据时需要（S1）。
   - AMASS（amass.is.tue.mpg.de）：仅用 AMASS 数据时需要；我们用 AIST++，多半可绕开。
3. **训练算力**：~~显存待量~~ → **已解决：A6000 48G 足够全身肌肉**。
4. **WSL2 GPU 直通**：~~待验证~~ → **已解决：直通可用**（见 ENVIRONMENT.md）。
5. **MuJoCo → MetaHuman 骨架**层级对齐，M0 跑通。

## 6. Decision Log

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-05-31 | 底座用**肌肉驱动**而非力矩驱动 | 用户要一步到位；A6000 48G 撑得住 |
| 2026-05-31 | v1 走 **RL 会跳舞的身体**（控制策略），非轨迹优化清洗 | 最贴"身体本身会跳、细节自生"理念 |
| 2026-05-31 | 范围收到**单舞种+单角色+固定机位** | 越窄越快出东西 |
| 2026-05-31 | 阶梯**集成前置**（M0 哑管道先打通 sim→MetaHuman→UE） | v1 必须在 UE 看到 MetaHuman 动；前置最易翻车的重定向 |
| 2026-05-31 | 真视频识别(S2)**移出 v1**，v1 用 AIST++ 干净参考 | 隔离识别误差，先攻最难的肌肉控制 |
| 2026-05-31 | **手指不走肌肉物理**，单独风格层 | 手指无质量/不承重，需要的是风格不是物理；全肌肉手部性价比极低 |
| 2026-05-31 | 训全身 416 肌肉（不退混合方案） | 实测 A6000 48G 显存充裕 |
| 2026-05-31 | **主底座定为 MuscleMimic**（非 KINESIS） | 它有释出的全身 416 肌肉带手臂模型 + 自带 SMPL 重定向；KINESIS 仅下肢、全身未释出。KINESIS 留作下肢参考/兜底 |

## 7. 待确认

- **舞种**：S1 默认挑 AIST++ 较平稳的一支（Pop/Lock），避开 Break 高速地板动作。
- **目标 MetaHuman**：M0 前需指定（用户提供或默认预设）。
- **UE 5.6 安装路径**：M0 前确认。
