# L2 冻结仪器最终验收 — 预注册协议

依据 [ADR-0004](../adr/0004-l2-final-acceptance-pass-lines.md)、[ADR-0002](../adr/0002-l2-instrument-calibration-envelopes.md)、[ADR-0003](../adr/0003-l2-instrument-frozen-artifacts.md) 与 [Spec #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) decision 10。本协议在任何 L2 终验运行之前冻结；全部通过线数值引用 ADR-0002 冻结表，本文件不产生新测量数值。执行工具：`scripts/nnunet_l2_final_acceptance.py`（判定链）+ `scripts/nnunet_l2_final_acceptance_nifti.py`（NIfTI 执行侧）。

## 1. 输入与侧别

- 终验病例 = 各子挑战 split manifest（`split_id=brats2023-rflow-v1`）的 **20% 最终验收持出集**，全量配额 GLI 250 / SSA 12 / MEN 200 / METS 48 / PED 20。组装时逐病例断言侧别 ∈ holdout（复用 #53 `ManifestSides`）；观测数不足全量配额 → 报告只能标 **provisional**（#51 Further Notes 的可得全量口径语义）。
- 生成侧输入 = 已冻结候选（#53 run contract `status=frozen`）的样本清单：
  - **P1 伪四模态体**：每病例 4 个模态标签独立采样组成一个四模态体；样本清单必须携带**互不相同的 noise seed**（同 seed 同噪声不构成独立采样，拒绝）；跨模态不自洽如实保留。
  - **P2 同掩码四模态体**：每病例同一条件 mask 下的四模态（条件 mask 路径必填）。
  - **P3 四锚轮**：每病例 4 轮，每轮一个真实模态作锚、其余三模态以该锚为 src 生成；4 轮覆盖全部 12 有序模态对（锚不重、方向不缺）。
- 真实参照侧 = 同批病例真实四模态（`<challenge>/<case>/<case>-<suffix>.nii.gz` BraTS 官方布局），**每病例只组装/推理/测量一次，跨 P1/P2/P3 复用**。
- 每个观测分配唯一 `obs_id`：真实侧 `{case}__real`，生成侧 `{case}__gen`（P1/P2）或 `{case}__gen__a{anchor}`（P3 逐锚轮）。predict 与 measure 全程按 `obs_id` 寻址，禁止以裸 case_id 作文件名 stem（同病例多观测会互相覆盖）。

## 2. 仪器输入组装与推理（冻结配置）

- 生成侧 NIfTI 重采样沿用 #38 `InputPreparator` 几何参数（DM 输出 256×256×128 @ (0.94, 0.94, 1.36) mm → 1mm 各向同性 B-spline → 居中裁剪/零填充到 240×240×155），并**修正 #38 的轴向处理**（xyz 切片误用于 zyx 数组；本管线按 zyx 数组轴序正确应用同一几何）。**几何事实如实入报告附录**：1mm 重采样后为 (241, 241, 174)，z 向居中裁掉 ~19 层——生成侧内容被系统性裁剪而真实侧原生 155 层不裁剪，这是 `WT/brain` 分母与质心 z 向上的一支系统性偏差轴（含 DM 训练数据 z 向 pad 对齐方式未单独验证；不粉饰、不补偿，建议 sugon 单病例预检确认量级）。
- P2 条件 mask 对齐到仪器网格用**最近邻**插值（同一几何链，不发明标签值）；无法读取或对齐后形状不符 → 该观测 `input_fail`（undecided 路径），不以任何静默方式计入回切。
- 真实侧输入已满足仪器输入契约（BraTS 原生 NIfTI），**直通不重采样**。
- 推理配置 = ADR-0002 决定 2 / ADR-0003 §4 冻结值：`nnUNetv2_predict` 默认镜像 TTA on（**不传 `--disable_tta`**——该 flag 为 store_true，传 `False` 会被 argparse 当真值）、滑窗 overlap 0.5 默认、fold 0、`nnUNetTrainer250Epochs`；GLI/MEN/METS/PED 用 `nnUNetPlans__3d_fullres`，SSA 用 `nnUNetPlans_SSA_bs16_v1__3d_fullres_bs16`（ADR-0001）。推理单次执行（重复性 p95 已冻结为 0，不重测；可另做 3-rep 抽查诊断，非门槛）。
- 仪器工件核验：终验只接受 ADR-0003 §6 冻结审计 verdict（SHA-256 `9121e8ac73f2bdb3999f83c10666c387ccd4c488b3e71b24cf7ab208353e0f82`，`all_passed: true`）或同脚本复跑通过；核验不过 → 拒绝评估（不是 undecided，是冻结链失效）。

## 3. 逐观测测量（语义母本 = 校准 #36 `measure_case`，裁剪 GT）

- 区域派生：`WT=seg∈{1,2,3}`，`TC=seg∈{1,3}`，`ET=seg∈{3}`；体积 mL = 体素数 × 0.001。
- `input_fail`：四通道文件存在、四模态 shape/spacing/origin 一致、spacing 1mm 各向同性（#36 §2 契约）。
- `run_fail`：推理进程崩溃或预测输出缺失/损坏、pred shape 与输入不符。
- `hier_viol`：pred 值域 ⊄ {0,1,2,3} 或 ET⊆TC⊆WT 违反体素数 > 0。**两侧（真实参照与生成）都做全部三类检查**；真实侧失败同样是仪器/参照链异常 → undecided。
- 测量量（每观测）：三区域体积 mL、质心 (x, y, z) mm（SRI24 物理网格）、`WT/brain`（brain = 组装后仪器输入四通道非零体素并集，规则钉死于此）、`ET/WT`；P2 额外逐区域**条件回切 Dice**（仪器 pred 对条件 mask 按区域派生）。
- 空 pred：是测量结果非失败——体积照记（0 mL）、`WT/brain`/`ET/WT` 按各自分母规则进入或排除、回切 Dice=0 全额进入分布；质心未定义 → 排除并计数。
- 校准语义裁剪（终验无 GT）：去掉 Dice/HD95/sensitivity/precision/GT 侧列；`R_miss`（空 pred 率）作诊断量保留。裁剪不改变失败量语义。

## 4. 统计判定（ADR-0004 通过线）

- **配对与重采样单元**：逐病例配对（生成观测对同病例真实参照测量；P1 为名义配对——伪四模态与真实肿瘤非解剖对应，如实声明）。bootstrap **病例级**：P1/P2 每病例 1 个差值；P3 每病例 4 个锚轮差值整组（cluster）随病例重采样，同一真实测量不作为 4 份独立信息。
- **bootstrap 约定**：B=10,000；seed = 20260821 + 子挑战序号（GLI=1, SSA=2, MEN=3, METS=4, PED=5），逐统计量派生（TOST 量序号 0..13、P2 回切区域序号 100..102，流互不重叠）；分位数 linear 插值（index = q·(n−1)，与校准 `numpy.quantile` 默认一致）；随机源 `random.Random`（Mersenne）——**不试图复现校准侧 PCG64 位流**（#55 是新计算，不复算 ADR-0002 数值）。
- **TOST（每区域）**：体积相对差 `(V_gen−V_real)/V_real` 的 90% CI ⊆ `±E_r,vol[区域]`；质心逐轴位移 90% CI ⊆ `±E_r,centroid[区域]`；`WT/brain` 相对差 90% CI ⊆ `±E_r,vol[WT]`；`ET/WT` 相对差 90% CI ⊆ `±(E_r,vol[ET]+E_r,vol[WT])`。
- **排除规则（预注册，绝不静默）**：某病例某量的真实侧分母为 0 / 空 / 未定义 → 该病例该量排除并计入 `n_excluded`（分侧、分因；TOST 与 P2 回切都计数）；排除仅因分母未定义，生成侧空 pred 从不触发排除（体积相对差为 −1.0 全额进入）。METS 在 ±1.0–1.65 体积 margin 下 TC/ET 全漏仍可过等价是 ADR-0002 已声明的区分力极限，报告必须携带该声明。
- **P2 回切**：逐病例逐区域回切 Dice 分布的单侧 5th percentile bootstrap 95% 下界 ≥ floor（floor = `D_r,low[区域]`，与校准同构统计量）；METS floor=0 恒真 → 显式标 vacuous-pass。
- **undecided 判定链**：`undecided`（任一观测任一侧 input/run/hier 失败）优先于一切；无失败时 TOST ∧（P2 回切）全过 = `pass`，否则 `fail`。五子挑战非补偿 AND 聚合；`undecided` 同样阻塞。Wilson 95% 上界对照真实 R_fail 包络仅诊断报告。
- **包络等值校验（防漂移、防收窄）**：提供受控校准 `summary_<CH>.json` 时，逐值与 ADR-0002 内置 4dp 字面量比对（容差 5e-5，先于判定执行）；任何 floor > `D_r,low` 或 margin < 冻结值 → 拒绝评估。

## 5. 报告与合规

- L2 报告（JSON+MD）聚合级、**零 subject ID**（失败明细只保留 side×failure-type 计数，obs_id 级明细只在受控 CSV）；逐病例测量 CSV 只存受控目录（`/root/private_data`），不入 Git。报告携带：run_id、phase、各挑战观测数与 provisional 标注（**按五挑战冻结配额全集计算，缺席挑战同样计为不足**，并单列 `challenges_missing` 与 `complete_coverage`，防止子集 AND 被读成完整 spec 终验）、冻结 verdict hash、逐量 CI/margin/排除计数、P2 回切下界/floor、R_fail breakdown、undecided/pass/fail、z 向裁剪偏差轴声明、METS/PED 区分力声明。
- 报告经 `scripts/brats_phase_run_contract.py attach --kind l2_report` 锚入 run record（attach 只做指纹，聚合级内容由本协议 §5 保证）。
- 冻结纪律：终验期间仪器 checkpoint、plans、推理配置、校准包络与 ADR-0004 通过线不得更改；禁止以生成样本反向调节任何阈值。
