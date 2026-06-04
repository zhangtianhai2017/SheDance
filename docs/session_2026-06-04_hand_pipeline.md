# Session 2026-06-04 — 手部动作管线 (A / B / C)

> 回主线补手部。用户决定:**手指/手掌暂不走肌肉**(只求动作到位);两分支——输入有手部动作就重定向(B),没有就按舞种生成(C)。

## 背景:简化时手怎么没的(已核实)
- `build_min_model.py` 把 `radius_hand_r/l` **焊死**(腕刚性)+ 删除对保留肢段力臂为 0 的肌肉 → 手部肌肉全删。
- 渲染端 `disable_fingers=True`。
- **关键发现**:OpenSim 源模型 `cyclistFullBodyMuscle`(520 肌肉)**0 条跨腕/手肌肉**(手臂肌肉只到前臂 pro_sup,手是被动末端)→ 手**不可能**在这个模型里肌肉驱动。这正是用户决定"手指走 kinematic"的依据。

## A — 手指驱动 + 渲染端点 (commit 00dc824)
- `MyoFullBody(disable_fingers=False)` = 129 qpos,**40 手指 DOF**(每手 20:拇指4 + 食中环小各4),**穿插在 body 中**,故 89→129 按**关节名**映射。
- 合成 curl 驱动手指 → osmesa 渲染,手指明显开合(`zoom_open` vs `zoom_curl`)。✅
- 文件:`check_finger_layout.py`, `render_fingers_test.py`。

## B — SMPL-H 手姿 → MyoFullBody 手指 重定向 (commit 62ec03c)
- SMPL-H 30 手部关节(`SMPLH_BONE_ORDER`)→ MyoFullBody 手指 DOF(`DOF2MANO` 对应表)。
- flexion ≈ `||axis-angle||`(粗,够"动作到位";真数据可细化为带符号的 flex/abduction 分解)。
- 合成 SMPL-H 输入 **open→fist→point→open**:渲染显示**逐指正确**(point 姿 ≠ 统一 curl)。✅
- 数据现实:**AIST 手部 pose 全 0**(无手 mocap);真外部 SMPL-H/X(AMASS/GRAB)需注册+许可,**我不能建账号/接受许可** → 管路已证,真数据待用户提供。
- 文件:`render_B_retarget.py`。

## C — 按舞种生成手部动作 (本次, 原始愿景)
- 思路:模仿能力强的人缺手指数据时,**按舞风即兴补手**。
- C v1:手指屈曲 = base + **能量耦合**(`qvel` 归一)+ **节拍振荡**,按 genre 参数化(Pop/Lock/Ballet;Pop=锐利/snap)。
- 在 Pop(gPO)上生成 → **完整 Pop 带手**(`C_Pop.mp4`, 1193 帧)。✅
- 文件:`render_C_generate.py`。

### C v2 — 手势库 + 运动学节拍 + 卡拍(本次提升)
- **节拍检测**:关节速度(去根 6 DOF)的**局部极小 = 卡点/停顿 = 运动学拍**(AIST++ 式)。Pop 测得 **~121 BPM**。
- **手势库**:6 keypose(OPEN/FIST/POINT/GUN/SPREAD/RELAX),每手指 (flexion, abduction);每舞种一套序列(Pop/Lock/Ballet);**卡拍切换**(锐利 snap / 柔和 ease,按 genre)。
- **音乐对齐**:⚠️ 磁盘**无 AIST 音频** + **无 librosa** → **不能对真实音轨**;改用运动学拍(因舞本就踩拍,隐含跟音乐)。真音频同步需音频文件 + librosa(用户提供)。
- 在 Pop 上 → `C2_Pop.mp4`(1193 帧,手势随卡点切换;两帧对比可见手势差异)。✅
- 文件:`render_C2_gesture.py`, `check_beat.py`。

## 现状 & 下一步
- A/B/C 三档全部跑通(A、B 合成验证;C 在真实 AIST 上出片)。
- **手指肌肉**:暂不做(用户决定 + 模型也无手肌肉)。若将来要,需带手肌肉的求解模型(MyoFullBody 自身有手肌肉,可走 MuJoCo 侧求解)。
- **B 真数据**:待用户提供/下载带手的 SMPL-H/X 文件。
- **C 提升**:真正的舞种手势库 / 节拍检测 / 与音乐对齐(v1 是"能量+固定节拍"的启发式)。
- 渲染全部走 osmesa(WSL GPU 路径每帧 4.5s 的坑见上一份 session 文档)。
