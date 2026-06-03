# fullbody 训练编译瓶颈 — 完整诊断与结论

> 2026-06-03。记录本机(WSL2 + RTX A6000 + JAX0.7.2/MJX/warp)上 MyoFullBody 肌肉 PPO
> 训练"编译/首迭代极慢(小时级、常不出 checkpoint)"的穷尽式排查。**给后来者:别重走。**
> 关联:[decision_muscle_training_paths.md](decision_muscle_training_paths.md)、[modular_muscle_dancer.md](modular_muscle_dancer.md)。

## 现象
- 全量 fullbody(354 肌肉)PPO 训练:编译 + 首迭代 **小时级**,常 110 分钟仍无第一个 checkpoint。
- 对照:**bimanual(小骨架、几十肌肉)17 分钟编完并能训**(出过挥手视频)。

## 关键工具数据
- **单步 `mjx.step` 编译 = ~25 秒**,且对 接触关/全关、肌肉 354→140 **几乎不变**(微基准)。
  → 物理"单步"不是小时级瓶颈。
- XLA 自报:`Compiling module jit__lambda for GPU ... The operation took 14m44s`,且 `slow_operation_alarm`
  在每次跑约第 11 分钟触发(**告警只表示"某 op 超阈值",不等于总时长——曾据此误判,勿重蹈**)。
- 系统快照:**无 swap(si/so=0)、无 iowait、磁盘零活动、无网络** → 不是内存颠簸/IO/网络。
- 进程:**1 个 CPU 核 100%(R 态)、其余 31 核闲、GPU ~95%** → **XLA 单线程 CPU 编译**(JAX 固有,
  编译在 CPU、执行在 GPU,搬不到 GPU)。

## 穷尽式对照实测(全部失败)

| 调的变量 | 从→到 | 结果 |
|---------|-------|------|
| 自接触/接触 | 关自接触 / 全关 | ❌ 编译不变(单步 25s 级) |
| 肌肉数 | 354 → 150(按出力保留) | ❌ 110 分钟仍 0 checkpoint |
| 网络规模 | 13.7M/16层 → 8.4M/14×512 | ❌ 110 分钟仍 0 |
| 求解器迭代 | iter 100/ls 50 → 8/4 | ❌ 仍慢 |
| **并行/展开规模** | num_envs 256→32, num_steps 80→8 | ❌ 23 分钟仍 0 |
| 后端 | warp → jax | 编译≈,但 jax 无 FFI → **缓存可生效**(见下) |

> 缓存:**JAX 持久编译缓存机制在本机可用**(小测试 run1 写盘、run2 跨进程命中,确凿)。
> 但 fullbody 那张大图**从没编完过一次**(总被杀/重启),所以缓存从没填上。warp 后端的 FFII
> 可能阻止缓存;**jax 后端 + 编完一次** 理论上可一次性缓存。

## 根因(排除法终点)
唯一**没法减、也最可能主导**的是 **nbody=102(整副骨架)**:它决定"训练步体"图的规模,
而 XLA 单线程编译该图随骨架膨胀而爆炸。bimanual 刚体极少 → 快;fullbody 102 刚体 → 任何配置都卡。
**配置层无解。**

## 为什么作者能训
MuscleMimic 预训练 checkpoint 是真的(我们用它渲染过走路)。配置 `num_envs=8192 / total=2.048e9`
= 数据中心级。**差异在环境/算力**(更快单核 CPU 做编译、原生 Linux、版本组合),**不是方法错**。

## 三条出路
- **A. 换环境训 fullbody**:云 GPU + 原生 Linux(贴近作者),最可能"直接能跑"。→ 追求 fullbody 肌肉跳舞的正路。
- **B. 本机只做能跑的**:bimanual 量级的手/小模块 + 重定向 + 渲染 + UE 侧(见 modular 文档)。fullbody 待更好算力。
- **C. 仅预训练推理 demo**:不训练,渲染预训练策略(如走路)。只为"看到肌肉动"。

## 给后来者的告诫
1. **别再在配置层调** 接触/肌肉/网络/迭代/envs 去救 fullbody 编译——本会话已穷尽证明无效。
2. **别据 `slow_operation_alarm` 判定**"成/败"——它只是 op 超阈值,要看 **checkpoint 是否出现 + 何时**。
3. 想验证某配置是否可行,**让它编完一次**(别中途杀);并确认 `~/.musclemimic/.jax_cache` 是否写盘。
4. 本机可训的 = 小骨架模型(bimanual 已证)。fullbody = 换 A 环境。

## 本机已加的可回退改动(在 WSL musclemimic 仓库,带 .bak)
- `myofullbody.py`:env 变量 `SHEDANCE_KEEP_MUSCLES`(按出力保留前 N 肌肉)、
  `SHEDANCE_SOLVER_ITER`/`SHEDANCE_SOLVER_LS`(降求解器迭代)、jax 分支的减接触 keep-list。
