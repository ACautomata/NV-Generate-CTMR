# L2 仪器校准与误差包络 — 预注册协议

依据 [自训 MONAI nnU-Net L2 测量仪器的构建与校准闸门钉板](https://github.com/ACautomata/NV-Generate-CTMR/issues/32) §3/§5 与 [L2 仪器校准与误差包络](https://github.com/ACautomata/NV-Generate-CTMR/issues/36)。本协议在**任何校准推理或指标计算发生之前**冻结：由 `scripts/nnunet_l2_calibration_prep.py` 将本文件连同校准集清单与其 SHA-256 写入受控目录 `protocol/` 并生成 `SHA256SUMS`。冻结后协议不得修改；数值一经算出写入 ADR，不得回看调整（#32 §5）。

## 1. 校准集

校准集 = **10% 开发集 ∪ 仪器 fold_0 内部验证集**（均为仪器未反向传播病例），逐病例标记来源（`source ∈ {dev, fold_val}`）。病例清单由 split manifest（`split_id=brats2023-rflow-v1`）的 `cases.dev` 与 `splits/fold0_val_cases_<CH>.txt` 唯一确定：

| 子挑战 | dev | fold_0 val | 合计 |
|---|---|---|---|
| GLI | 125 | 175 | 300 |
| SSA | 6 | 8 | 14 |
| MEN | 100 | 140 | 240 |
| METS | 24 | 33 | 57 |
| PED | 10 | 14 | 24 |

20% 终验持出集零接触（不进入输入、GT、清单或任何统计）。

## 2. 输入契约

- 四模态 `t1n/t1c/t2w/t2f → _0000.._0003`（与 #35 训练契约一致）；
- 输入为 BraTS 原始 NIfTI（不预归一化；nnU-Net 预处理管线自理）；
- `fold_val` 病例输入与 GT 取自 `brats2023_nnunet/Dataset50X`（imagesTr/labelsTr）；`dev` 病例输入与 GT 取自 DUA 受控原始数据副本（`<case>-t1n/-t1c/-t2w/-t2f/-seg.nii.gz`），来源与训练侧同源；
- 逐病例输入契约检查：四文件存在、四模态与 seg 的 shape/affine 一致、spacing 为 1mm 各向同性。任一不满足 → 该例该次推理记 `input_fail`。

## 3. 仪器与推理配置（冻结）

- 五子挑战各用其 #35 fold_0 `checkpoint_final.pth`：GLI/MEN/METS/PED = `nnUNetTrainer250Epochs__nnUNetPlans__3d_fullres`；SSA = `nnUNetTrainer250Epochs__nnUNetPlans_SSA_bs16_v1__3d_fullres_bs16`（ADR-0001 例外）；
- 推理入口：`nnunetv2.inference.predict_from_raw_data`（nnunetv2 2.8.1 原生；环境版本锁沿用 `/root/private_data/ENVIRONMENT_LOCK.md`）；
- 推理配置 = 入口默认值，即**镜像 TTA on**（不传 `--disable_tta`）、**滑窗 overlap（step_size）0.5**、`--save_probabilities` off；`nnUNet_compile=f`（DCU 环境开关，非配置改动）；
- **重复性协议**：每病例独立推理 3 次（`rep ∈ {1,2,3}`，三次独立进程、互不共享输出目录）；逐病例记录体积/Dice/centroid/HD95 在 3 次间的最大极差（max−min）。DCU 上若三次 bit 级一致，极差如实记 0。

## 4. 区域派生

对所有子挑战统一：`WT = seg∈{1,2,3}`，`TC = seg∈{1,3}`，`ET = seg∈{3}`（BraTS2023 标签 1/2/3 = NCR-or-NC/ED/ET）。体积（mL）= 体素数 × 0.001（1mm 各向同性）。

## 5. 七类误差逐病例测量

按 [nnU-Net BraTS 肿瘤测量仪器选型与校准调研](https://github.com/ACautomata/NV-Generate-CTMR/issues/22) findings §6.B 全表，逐病例 × 逐区域 `r ∈ {WT, TC, ET}` × 逐 rep：

| 类别 | 测量 |
|---|---|
| 输入/预处理失败 | `input_fail`（§2 契约）、推理进程崩溃或输出缺失（`run_fail`）——失败计分母不剔除 |
| 区域重叠 | Dice、sensitivity（=TP/(TP+FN)）、precision（=TP/(TP+FP)）；pred 空 → Dice=0、sensitivity=0、precision=0 |
| 边界/形状 | HD95 = max(p95(d_gt_surf→pred_surf), p95(d_pred_surf→gt_surf))；表面 = mask XOR binary_erosion(mask) |
| 体积/比例 | V_r（mL，GT 与 pred 各自）、signed bias（V_pred−V_gt）、绝对误差、相对体积误差 `|ΔV|/V_gt`、ET/WT 比（GT 与 pred 各自） |
| 空间与病灶组件 | 区域质心欧氏距离（mm）、GT/pred 连通组件数（26 连通）、假阳性组件数（pred 组件与 GT WT 零重叠）、病灶检出（pred WT 非空） |
| 层级/标签合法性 | `hier_viol`：pred 值域 ⊄ {0,1,2,3} 或 ET⊆TC⊆WT 违反体素数 > 0；GT 空（BraTS 不发生，防御性照测） |
| 重复性 | §3 的 3-rep 最大极差（vol/Dice/centroid/HD95） |

**空 pred 的分层处理（预注册）**：空 pred 是测量结果而非仪器失败——它以 Dice=0、相对体积误差=1.0、sensitivity=0 全额进入可计算量的分布（保守方向）；HD95 与 centroid 距离在数学上不可定义于空表面，其 `E_r,*` 在 **GT∩pred 双侧非空子集**上计算，并同时报告该子集的排除率（排除仅因空 pred）。空 pred 率另列为诊断量 `R_miss`。失败量 `R_fail` 从不因空 pred 剔除分母。

## 6. 预注册统计量（每子挑战独立产出）

| 量 | 定义 |
|---|---|
| `D_r,low` | 区域 Dice 单侧 5th percentile 的 bootstrap 95% 下界：病例级重采样 B=10,000 次，每次算 q5，取 {q5_b} 的 5% 分位 |
| `E_r,vol` | 相对体积误差 p95 的单侧 95% bootstrap 上界：每次重采样算 p95，取 {p95_b} 的 95% 分位 |
| `E_r,centroid` / `E_r,HD95` | 同法（centroid/HD95 在 §5 双侧非空子集上重采样） |
| `R_fail` | `input_fail ∪ run_fail ∪ hier_viol` 按（病例 × rep）计的点估计 + Wilson 95% CI 上界 |
| `R_miss` | 空 pred 率（诊断量，不并入 R_fail） |
| 重复性 | 3-rep 最大极差的 p95（按区域；进 margin 的叠加项） |
| ET 分层 | GT ET < 1 mL 病例分层重算全表（**诊断性**，不设为通过线；主判据用全分布） |

- bootstrap：numpy `Generator(PCG64)`；全局 seed = **20260820**；逐子挑战派生 seed = 20260820 + 序号（GLI=1, SSA=2, MEN=3, METS=4, PED=5）；分位数 `numpy.quantile` 默认 linear 插值；
- 汇总同时报告每区域 median/IQR/p95 与逐病例 CSV 的存放路径；**只按子挑战分层，不跨子挑战合并**；小样本子挑战（SSA 14 例）的宽包络如实保留，不收窄。

## 7. 冻结与纪律

- 冻结顺序：本协议 + 清单 + SHA-256 → 3× 推理 → 指标 → bootstrap → ADR。任何一步不得回改前序产物；
- 逐病例原始指标（含 subject ID）只存服务器受控目录 `l2-instrument-calibration/<repo-commit>/metrics/`，不入库；ADR 只记录元数据、协议与汇总数值表（#32 §4 权利边界）；
- 禁止：用生成样本反向调节任何阈值；基于校准结果回头改仪器（重训仅限 #32 §5 两类触发器）。
