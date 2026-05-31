# SheDance

视频驱动 · 肌肉仿真 · MetaHuman 舞蹈工具（D1 研究原型）。

看一段舞蹈视频，让 UE 5.6 的 MetaHuman 把动作跟跳出来。与普通"视频动捕→重定向"不同：
角色被建成一个**肌肉物理仿真的身体**，"本身就会跳某一类舞"。视频负责约束主干 + 提供节拍，
手指/脚掌/发力习惯等细节由"会跳舞的身体"在物理里跑出来，而不是硬识别。

- 架构与阶梯：见 [`docs/design.md`](docs/design.md)
- 环境实况：见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)

## 目录

```
data/       AIST++ 子集（选定舞种）、SMPL 模型
sim/        MuJoCo 肌肉模型 + RL 训练（WSL2 侧）
recog/      识别流：单目视频 → 姿态+置信度（S2 起）
retarget/   sim 骨架 → SMPL → MetaHuman 映射 + FBX/AnimSequence 导出
ue/         UE 5.6 测试工程 + IK Rig/Retargeter 资产
docs/       设计文档 + Decision Log
```

## 当前进度

- [x] 仓库骨架 + 设计文档
- [x] 环境探测（A6000 48G / WSL2 Ubuntu-22.04 / GPU 直通 OK）
- [x] 仿真栈验证：MuJoCo + MyoSuite + **MuscleMimic 全身 416 肌肉模型**在 A6000 上加载（JAX/GPU 跑通）
- [x] **S0 imitation demo 跑通**：预训练 416 肌肉策略模仿走路 → `renders/S0_myofullbody_walk_demo.mp4`（548 步，关节位置误差 0.059，EGL 无头渲染）
- [x] **SMPL-H 重定向地基就位**：SMPL-H(300维 neutral，等价替代)+ MANO → `SMPLH_NEUTRAL.pkl`，已验证加载（verts 6890 / joints 73）
- [ ] M0 哑管道：AIST++ 干净动作 → MetaHuman → UE 播放（UE 工程已复制到 `ue/TP_ThirdPerson`，待编译 + 重定向方法）
- [ ] S1 肌肉身体跳 Pop/Lock（待：AIST++ 下载 + 接入 MuscleMimic 重定向）
- [ ] S1 肌肉身体跳这支舞（核心研究风险）
- [ ] S2 接入单目视频识别
- [ ] S3 手指/风格层
- [ ] S4 打磨
