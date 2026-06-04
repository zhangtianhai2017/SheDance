# Session 2026-06-04 — 性能工程:并行 SO + 渲染瓶颈根治

> 承接上一段(肌肉驱动管线跑通)。本段专做**性能工程**:把"2 分钟舞"从天级压到分钟级。
> 全程坚持**实测**,不靠估计;并记下两条流程教训。

## 一句话
**SO 可无损切块并行(逐帧独立);渲染真凶是 WSL 的 GPU 路径(每帧固定 4.5s),换 osmesa 软渲快 27×,再并行到 6.6min/60s。整体渲染 vs 原 egl 快 68×。**

## 1. 并行 Static Optimization(无损)
- **概念**:SO 逐帧独立(无前向积分、无动量耦合)→ 按时间段切块可并行,**无损**;前向法(MocoTrack/CMC)有动量边界,切了有损。
- **实测**(60s = 6000帧,cyclist_min 149肌肉,cap-16):
  - 墙钟 **960s(16min)**,vs 顺序外推 6840s → **7.1×**;
  - **无损验证**:相邻块重叠帧最大激活差 **2.48e-3 ≈ 0** → 实证逐帧独立。
- 次线性(非 16×)原因:i9-13900KF 混合架构(8 P + 16 E,E 核慢)、超线程不翻倍、重叠开销 ~11%。
- 文件:`make_60s_mot.py` `build_reserves_model.py` `so_chunk.py` `run_parallel_so.sh` `concat_verify.py`

## 2. 渲染瓶颈根治(本段最大发现)
- **Profile 定位**:每帧 mj_forward 0.4ms / update_scene 0.1ms / **render() 4550ms**(占 99.99%)。
- **铁证**:render() 在 128²/256²/512²/720² 下都 ~4s → **与分辨率无关 = WSL GPU 虚拟化栈(egl→D3D12→Mesa→A6000)每次调用的固定提交开销**,不是逐像素。
- **解法:换 osmesa(纯 CPU 软渲)**绕开它 → **171ms/帧,快 27×**;画面与 egl 同帧像素差仅 **1.4/255**(渲对了)。
  - 注:GL 后端经 `glcheck.py` 确认原是 `D3D12 (NVIDIA RTX A6000)` via Mesa——是 GPU 但被那层拖死,不是软件回退。
- **CPU 向量化 + 裸帧管道喂 ffmpeg**:在 egl 下无效(被 4s 淹没),切 osmesa 后才生效。
- **并行渲染**(osmesa,16进程×1线程):**60s = 6.6min(396s)**,5941 帧视频完整。
  - 配比扫描:8×1=11.1 / 12×1=12.8 / **16×1=14.1** / 8×2=13.8 / 4×4=13.0 fps → **16×1 最优**;
  - 全配比挤在 11-14fps = **内存带宽硬墙**;24/32 进程触发 OOM(每进程 ~1GB)。
- **净提速**:60s 渲染 原 egl ~7.5hr → osmesa 并行 6.6min = **68×**。
- 文件:`render_core.py`(向量化 core,osmesa 默认,RENDER_CACHE/RENDER_FRAMES 可配)`render_one.py` `render_parallel.sh` `profile_render.py` `glcheck.py` `slice_sto.py` `make_60s_qpos.py` `sweep_render.sh`

## 3. 温控 / 限载 / 日志
- **GPU 温**:nvidia-smi 精确可读。**CPU 核心温:WSL 读不到**(无 thermal_zone/coretemp/cpufreq;Windows ACPI 区可读但粗且本机实测纹丝不动 = 不可用)。真要核心温需装 HWiNFO。
- 框架:GPU 温门 + 保守限载(cap=核数一半)+ 全程日志 + 过热杀最新 worker。实测负载下温度毫无压力(瓶颈是内存/计算,非热)。

## 4. 现状数字(实测,非估)
| 阶段(60s 舞) | 时间 | 备注 |
|---|---|---|
| 并行 SO @cap16 | 16min | 7.1× |
| 并行渲染 @16×1 | 6.6min | 68× vs 原 egl |

外推 2min ≈ SO 32min + 渲染 13min ≈ **45min**(原方案是天级)。

## 5. 到顶 & 下一步
- 渲染已达 WSL+osmesa 内存带宽 ceiling(~14fps),配比无法再压。
- **治本 = 原生 GPU 渲染(非 WSL)**——绕开软渲内存墙 + WSL 4s,但需换出 WSL 环境(大改,未做)。
- 候选:SO 也推 cap-24/32;回主线(更难动作的"偏离版"愿景);调研原生 GPU 渲染。

## 6. 流程教训(已写入 ~/.claude memory)
1. **长时后台任务必须定期轮巡**——完成通知/脚本自报耗时不可信(本段一个 `wait` 死锁让渲染空转 18min 才发现)。用 Monitor 轮询,看实际产物/进程判断真完成。
2. **bash 有后台监控子进程时,裸 `wait` 会死锁**——必须 `wait "${pids[@]}"` 只等 worker(已犯两次:SO + 渲染)。
3. **交付文件直接 `Invoke-Item` 打开**,别只贴路径(用户那边路径几乎打不开 + 右键不能复制)。
