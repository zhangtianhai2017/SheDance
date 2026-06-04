# 自主推进 session 总结 — 2026-06-04

> 用户离开 ~8h 的自主推进记录。承接"转 OpenSim Moco / marker→IK 根治约定"的决策。

## 一句话

**肌肉驱动 Pop 跑通并验证了**:marker→IK 治好坐标约定,OpenSim 求解器算出整段舞的肌肉激活,
"同段舞、越弱的身体越使劲(努力版愿景)"已有干净证据。可回退 commit 已立。

## 做了什么 + 结果

1. **marker→IK 根治坐标约定 bug**(✅ 成功)
   - 之前 name-mapping 直转 .mot 对臂部约定是错的(左肩 shoulder_elv_l 差 32° 等)。
   - 新管线:`trc_from_mujoco.py`(关节中心→.trc)→ `add_markers_and_ik.py`(IK)→ `pop_ik.mot`。
   - **结果:全身坐标 0 超范围、跟踪误差 ~0.2°**,左肩 bug 消失。IK marker 误差 RMS 14mm。

2. **MocoTrack 验证全局求解优于 CMC**(✅)
   - CMC 逐窗口贪心 → 硬帧(t≈7.6,高需求)直接崩。
   - **MocoTrack 全局直接配点 → 把硬帧一起优化、优雅收敛不崩**,多核,~15-30min/0.4s。

3. **F_max 角色档 → "努力版"愿景**(✅ 干净证据,见 `renders/effort_vision.png`)
   - 同段 Pop(7.4-7.8),F_max 0.2/0.4/1.0(20%/40%/常人肌力)**都能跳出动作**(偏离仅 ~0.2°),
   - **但激活随肌力反向飙升**:mean 0.105/0.056/0.048,**活跃肌肉 90%/21%/13%**。
   - = "柔弱的身体调动几乎全部肌肉去拼同一段舞" = 内生模型/模仿能力强的人。
   - "偏离版"(弱到够不着→动作变样)在 20% 肌力还没出现(这段对它仍可行);F0.1(10%)在试。

4. **全片 SO → 整段 Pop 的肌肉激活**(✅,精确复现档)
   - `cyclist_min`(149肌肉)+ 干净运动,全片 1131 帧。
   - **关节储备全≈0(腿/臂 0.0、腰 5.7 N·m)→ 肌肉产出整段舞全部关节力矩**;仅 pelvis_ty=780N=缺 GRF。
   - 渲染:`renders/pop_muscle_full.mp4`(肌肉按激活上色)。

## 交付物

| 文件 | 内容 |
|------|------|
| `renders/vision_sidebyside.png/.mp4` | **F0.2柔 vs F1.0常 并排肌肉上色**——弱身体大片肌肉发红(使劲)、常人发蓝(从容)。**最直观的愿景证据**(配色×4放大以显低激活差异) |
| `renders/effort_vision.png` | 努力版愿景图(三档激活对比)——核心数据 |
| `renders/pop_muscle_full.mp4` | 整段 Pop 肌肉驱动 + 上色(全片 SO) |
| `renders/vision_f04_f10.mp4` | F0.4 vs F1.0 骨架叠加(动作几乎重合=同动作) |
| `~/shedance/osim/pop_ik.mot` | 约定正确的全片运动(marker→IK 产) |
| `~/shedance/osim/popB_7.40_7.80_f*.sto` | MocoTrack 各 F_max 档解 |
| git 分支 `muscle-opensim-pipeline` 38d415b | 整套代码 + 文档 |

## 验证可行的完整管线(代码在 bio/osim/)

```
MuJoCo qpos → trc_from_mujoco(关节中心.trc) → add_markers_and_ik(IK) → pop_ik.mot(约定正确)
  → convert_mot_to_rad → MocoTrack(全局,F_max=角色,出偏离/努力) 
                       或 → Static Optimization(全片,快,精确复现)
  → analyze_moco_sol / analyze_so → render_muscle(肌肉上色) / render_skeleton(骨架叠加)
模型:build_min_model → cyclist_min.osim(149肌肉,焊肋骨/胸椎)
```

## 关键教训(已写入 memory + CLAUDE.md)

1. **name-mapping 臂部约定不可靠 → marker→IK 根治**(IK 自动解出正确约定)。
2. **CMC 硬帧崩 → MocoTrack 全局求解**(不崩)。
3. **Moco/CMC 算力随肌肉数爆 → 必须剪到 ~149**。
4. 愿景在可行段以"努力"(激活)显现;"偏离"需更极端肌力/更高需求动作。

## GRF 实验(部分成功,2026-06-04)

`compute_grf.py`:F_grf=M(a_com−g) 按脚高分配 → OpenSim ExternalLoads(`pop_grf.mot`+`.xml`);
`static_opt_osim.py` 自动接入(xml 存在时)。
- ✅ **竖直体重平衡**(pelvis_ty 残差 780N→消失)、**腿肌激活 0.065→0.114(腿承重了)**。
- ❌ **力矩残差大**(pelvis_list max 1427、某腿帧 441)= **COP 压心估得粗**(双脚支撑歧义)。
- 结论:GRF 的力对了,**压心/力矩需 RRA 细化(深水区)**。主 demo 仍用无 GRF 的干净渲染。

## 下一步(未做 / 给回来的你)

- **偏离版愿景**:F0.1 结果(在跑);或选更高难度动作段,看弱身体"够不到→动作变样"。
- **补 GRF**:从 COM 估地面反力,消 pelvis_ty 残差 → 腿部肌肉生理化。
- **全片 MocoTrack**(非 SO):分段并行(注意惯性边界),拿"全片+角色档"的偏离/努力。
- **UE/MetaHuman 烘焙**:把激活/动作导成 AnimSequence,游戏可用 + LOD。
- **通用化**:关键点跟踪 → 任意输入(动物/木偶/机器)→ 模仿(见 master_plan §1.4)。
