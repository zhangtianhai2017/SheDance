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
| **S0 imitation demo** | ✅ | 预训练 `mm-10m-2` 模仿走路 548 步，关节误差 0.059 → `renders/S0_myofullbody_walk_demo.mp4` |
| **重定向依赖**（torch+smplx+gmr_plus） | ✅ | `uv sync --extra cuda --extra smpl --extra gmr` |
| **SMPL-H + MANO → SMPLH_NEUTRAL.pkl** | ✅ | `smpl_models/SMPLH_NEUTRAL.pkl`，`SMPLH_Parser` 加载 OK（verts 6890 / joints 73）|

> **结论**：完整 GPU 肌肉栈（含**全身 416 肌肉带手臂+手指**）在 A6000 上端到端跑通。
> MuscleMimic 用 **JAX**（非 torch；retarget 子流程才用 torch）。

### SMPL-H 设置要点（坑）

- MuscleMimic 重定向要 **SMPL-H + MANO**（来自 mano.is.tue.mpg.de），**不是**基础 SMPL（smpl.is.tue.mpg.de）。
- 需 2 个下载：**Extended SMPL+H model**（→ `smplh/`）+ **Models & Code**（→ `mano_v1_2/`）。
- 本项目用的是 **300维 neutral** 版（`smplh/SMPLH_NEUTRAL.npz` 重排成 `smplh/neutral/model.npz`）——与 16维官方版**逐值等价**（parser 只用前 10 betas，topology 相同）。
- 本机**无 conda**，官方 `install_smplh.sh` 用不了。改用一次性 `uv venv --python 3.10` + `numpy<1.23` + `chumpy`（**注意：chumpy 0.70 构建需先 `pip install pip` 进该 env**）+ 打 chumpy 的 numpy 导入补丁，再跑 `generate_smplh_model.py --smpl-conf-file ~/.musclemimic/MUSCLEMIMIC_VARIABLES.yaml`。
- 路径已写入 `~/.musclemimic/MUSCLEMIMIC_VARIABLES.yaml`（`musclemimic-set-smpl-model-path`）。

## 待办

- **S1 重定向**：下 AIST++ 舞蹈数据 → 接入 MuscleMimic 重定向（SMPL-H 地基已就位）→ 训练肌肉身体跳舞。
- **M0/UE**：编译已复制的 `ue/TP_ThirdPerson` 工程 → 研究 SMPL/动作 → MetaHuman 重定向方法。

> 文件系统：训练 repo/数据放在 WSL ext4（`~/shedance/`），非 `/mnt/c`（跨文件系统 I/O 慢）。

## Windows 侧（M0 用）

- UE 5.6 Installed 引擎 `C:\Program Files\Epic Games\UE_5.6\`（交接已验证）。
- UE 工程已复制：`SheDance/ue/TP_ThirdPerson`（自 Waysee `.test_project_live` 分支，独立；含 MetaHuman/Maps/角色）。
- ⚠️ 编译注意：共用引擎的 MetaHuman 共享 DLL 雷区——编前查另一项目编辑器是否关闭（见 memory）。
