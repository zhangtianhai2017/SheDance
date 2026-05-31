# 环境实况（2026-05-31 探测）

> 这是 S0 "第一关"的探测记录。结论：**计划里两个最大命门当场解决** —— 显存够训全身肌肉，
> WSL2 GPU 直通可用。

## 硬件 / 系统

| 项 | 实况 | 对计划的影响 |
|----|------|------------|
| GPU | **NVIDIA RTX A6000，46068 MiB（≈48 GB）**，driver 596.36 | 显存充裕，**可训全身 416 肌肉**，不必退到"肌肉腿+力矩臂"混合方案 |
| WSL2 | **Ubuntu-22.04**，State=Running，VERSION=2，内核 6.6.114.1-microsoft-standard-WSL2 | Linux/CUDA 训练栈现成 |
| WSL GPU 直通 | **通过**：WSL 内 `nvidia-smi` 正确识别 A6000 48G | 计划"S0 第一关"已过——肌肉 RL 栈可在 WSL 训 |
| Python (WSL) | 3.10.12（`/usr/bin/python3`） | 满足 MuJoCo / MyoSuite / PyTorch |
| conda | 未安装 | 用 venv 即可，不阻塞 |
| nvcc / CUDA toolkit | 未安装 | 不阻塞——PyTorch wheel 自带 CUDA runtime，MuJoCo 不需 nvcc |

## 软件栈

两个 WSL venv：
- `~/shedance-venv`：初始 smoke test（mujoco + myosuite）
- `~/shedance/musclemimic/.venv`（uv 管理）：**主工作环境**，JAX + MuscleMimic + warp

| 组件 | 状态 | 验证 |
|------|------|------|
| **MuJoCo 3.9.0**（smoke env） | ✅ | 自由落体球 100 帧，z→-0.198 |
| **MyoSuite 2.12.2**（smoke env） | ✅ | `myoLegWalk-v0`：80 肌肉 / 34 DOF，随机激励步进 OK |
| **JAX 0.7.2 (CUDA)** | ✅ | `jax.devices()=[CudaDevice(id=0)]`，默认后端 gpu |
| **MuscleMimic** + warp-lang 1.10 | ✅ | `import musclemimic` OK；uv sync --extra cuda ~4.5min |
| **MyoFullBody 模型** | ✅ | **禁手指 354 肌肉 / 88 DOF；含手指 416 肌肉 / 128 DOF；102 刚体**，全 Hill 型 |

> **结论**：完整 GPU 肌肉栈（含**全身 416 肌肉带手臂+手指**）在 A6000 上端到端跑通。
> MuscleMimic 用 **JAX**（非 torch），故无需单独装 PyTorch。

## 待办（被账号/输入阻塞）

- **S0 imitation demo**：需 HuggingFace 对 `amathislab/demo_dataset` 授权 + token（用户）。
- **S1 数据**：AIST++ 下载 + SMPL 注册（smpl.is.tue.mpg.de，用户）用于重定向。
- **M0/UE**：UE 5.6 引擎路径 + 目标 MetaHuman（待用户提供）。

> 文件系统：训练 repo/数据放在 WSL ext4（`~/shedance/`），非 `/mnt/c`（跨文件系统 I/O 慢）。

## Windows 侧（M0 用）

- UE 5.6（待确认安装路径与版本）
- 一个目标 MetaHuman（待用户指定或用默认预设）
