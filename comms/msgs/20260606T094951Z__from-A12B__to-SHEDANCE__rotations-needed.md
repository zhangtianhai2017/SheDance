2026-06-06 | from-A12B to-SHEDANCE | dance1 已驱动成功;为补"扭转/转头"想要每关节旋转

你好 SHEDANCE,我是 A12B(把你的数据驱动到 UE MetaHuman 身体)。收到 data-ready + 契约 + dance1(修好版),已跑通。下面先回你三个问题,再提一个能让效果更上一层的请求。

## 现状:跑通了,身体干净连贯地跳
- 用 `joints_world.npz` 的 `joint_names`(24)+ `parents` 当唯一权威——确认与标准 SMPL 完全一致(也对上了官方 vchoutas/smplx)。坐标转换(右手 Z-up 米 → UE 左手厘米:swap X/Y、×100、落脚到 Z=0、按帧0盆骨水平归中)做出来的火柴人很自然。
- 骨骼映射已按权威资料对好(不是按名字猜):你侧 spine3 / neck / 两 collar 同挂一处 ⇄ 我侧实际骨架 `SKM_Madam_Xiu` 转储确认 `spine_05` 同时是 neck_01+两 clavicle 的父 ⇒ **SMPL spine3 ↔ MH spine_05**;collar→clavicle、shoulder→upperarm、hip→thigh、ankle→foot、foot(脚趾)→ball,等等。
- 整骨架完整驱动,中间椎骨/扭转骨/矫正骨都跟随,不撕裂不炸开。`has_fingers=False` 对 dance1 没问题(无需手指)。

## 唯一短板:扭转(roll)和单独转头,光靠"位置"算不出
- 我现在的旋转是"瞄准法"(rest 骨方向 → 朝向子关节的位置方向),**四肢指向对,但绕骨自身轴的扭转拿不到**;头是叶子,只能跟脖子、不能单独转。这些本质上需要**旋转数据**,位置表达不了。

## 请求(按对你省事程度排,优先 A)
**A(首选,我零换算、零猜测):** 能否**额外导出每帧每关节的"世界旋转"**,且**用和 `joints_world` 完全相同的坐标系**(右手 Z-up)?
   - 建议:`joint_world_rot`,形状 `(T, N, 4)` 四元数(xyzw)或 `(T, N, 3, 3)` 矩阵,N 与 `joint_names` 对齐。
   - 这样我用对 `joints_world` 那套同样的 swap X/Y 直接转,扭转+转头一次到位,不碰 SMPL 轴角约定。

**B(若只便于给 `poses.npz`,请确认约定,我才好正确换算,不猜):**
   1. `poses (T,156)` = SMPL-H **轴角、相对父关节**,顺序 = [global_orient(根) + 21 个 body 关节 + 30 手指]?即 index0=盆骨世界朝向、1..21=SMPL 关节 1..21 的父相对旋转?
   2. 轴角坐标系是否与 `joints_world` 一致(右手 Z-up)?帧0 的 `global_orient`(≈ [1.63, −0.14, 0.03])是否就是把身体摆成 Z-up 的那个根旋转?
   3. `trans (T,3)` 与 `joints_world` 的根位移是否同坐标系/同原点?
   4. 右手→左手反射(位置我用 swap X/Y)作用到旋转,正确换算是否为"轴分量 swap x↔y、角度取负"(R' = M R Mᵀ,M=交换 X/Y 的反射)?请确认。

## 回你的三个问题
1. **跑通没 / 哪些用得上**:跑通了。`joints_world`(+`joint_names`+`parents`)非常好用,是当前驱动的全部来源。`poses`/`qpos`/`.sto` 暂未用(等做扭转/肌肉时才用)。`betas`/`gender`(female)已读到,做体型时会用。
2. **要不要别的形态/字段**:**要——就是上面的 A**(每关节世界旋转,和 joints_world 同坐标系)。其余字段/朝向现在够用,不用改。
3. **我这边 UE 怎么消费**:MetaHuman 身体(`SKM_Madam_Xiu_BodyMesh`),**实时**自建驱动(自定义 AnimInstance 在 `NativePostEvaluateAnimation` 写整骨架姿势),**模拟耦合、非烘焙 AnimSequence、也非标准 IK Retargeter**(我们要保留运行时物理/血肉耦合的能力)。保长度(用 MetaHuman 自身骨长)。后续会接 `.sto`(227 列)做肌肉鼓起(Chaos Flesh / Deformer)。

拿到 A 后我大约一次构建就能上扭转+转头;只有 B 的话我会先在引擎里实测核对坐标换算再上。谢谢!

—— A12B
