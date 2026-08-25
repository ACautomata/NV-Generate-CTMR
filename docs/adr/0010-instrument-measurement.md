# ADR-0010：仪器读数深模块（InstrumentMeasurement）——src layout 单入口三列族门控与 6 份收编

- **状态**：已接受（2026-08-25；在任何重构执行之前预注册）
- **范围**：「掩码 → WT/TC/ET → 体积/位置/强化比」L2 仪器读数的 6 份散落实现、4 种 hier_viol 语义的统一 interface 钉板（wayfinder 地图 #75 候选 1 / 票 #80）。本 ADR 只钉决策与执行 spec，代码不动；重构与验收在执行期另行开始。

## 背景

「从仪器分割掩码派生 WT/TC/ET 的体积、质心位置、强化比等逐病例测量行」目前散落在 6 个文件、6 份实现，漂移已兑现：

- **REGIONS 字面量 7 份**、形状不一：字面量全同 `{"WT":(1,2,3),"TC":(1,3),"ET":(3,)}`，但 `nnunet_l2_calibration_metrics.py:24` / `nnunet_l2_synthetic_domain_eval.py:524` / `l2_synth_domain_sugon.py:45` / `brats_p1_dev_eval.py:373` 为 dict，终验 judge `nnunet_l2_final_acceptance.py:83-84` 拆为 `REGIONS`（tuple 名）+ `REGION_LABELS`（dict），`brats_p2_dev_eval.py:86` 另名 `INSTRUMENT_REGION_LABELS`。
- **Wilson 上界 5 份、`n==0` 守卫漏 1**：唯一漏守卫的是**终验 judge** `nnunet_l2_final_acceptance.py:686`（函数内直接 `p = k / n`，仅靠调用点 `:680 if n_obs` 兜底）；校准 `:173`、`synthetic :638`、`sugon :353`、`p1_evaluate.py:40` 均有守卫。
- **Dice 4 份、空分母哨兵不一致**：校准 `dice_of:53` 与 `synthetic:603` 用 `math.nan`，终验 `condition_dice:239` 与 `p2:292` 用 `None`。
- **hier_viol 5 份、4 种语义**：终验正确版 = `nnunet_l2_final_acceptance_nifti.py:196-201`（containment + 值域，单表达式、无前置守卫，即 [ADR-0004](0004-l2-final-acceptance-pass-lines.md) 决定 4 的冻结定义）；`synthetic:578` 守卫 vacuous（`else True` 使 TC 空时 ET 检查失效）；`sugon:395/398` 空超集前置守卫（跳过 ET 存在而 TC 空的真违反）；校准 `:154-157` 与 `p1_dev:449` 仅查值域、**无 containment**。
- **两条冻结路径语义本不一致**：校准母版（#1，`calibration_metrics.py`）产 [ADR-0002](0002-l2-instrument-calibration-envelopes.md) 冻结包络（R_fail 的 hier 分量全 0 即按 #1 那套「值域+GT空、无 containment」语义算出）；终验（#2，`final_acceptance_nifti.py` + judge）产 [ADR-0004](0004-l2-final-acceptance-pass-lines.md) 判读（containment）。二者测的是**两件事**——校准那个需 GT（对生成数据无意义）。

硬约束：**不改变终验路径的冻结读数**（[ADR-0002](0002-l2-instrument-calibration-envelopes.md) 校准包络 / [ADR-0004](0004-l2-final-acceptance-pass-lines.md) 通过线；ADR-0004 决定 5「冻结核验不变」）。冻结终验读数 = `nnunet_l2_final_acceptance_nifti.py:187-279` 测量行 + judge 聚合，即本统一的**收敛标准**。校准包络已冻结且**推理不重跑**（ADR-0009），但**测量可在冻结预测上重跑以验证聚合不变**（决定 6）。

## 决定

1. **统一 module 落点与形状（src layout）**：通用「仪器读数」归于 **`src/ctmr/measure/`**（与 [ADR-0008](0008-instrument-grid-geometry.md) 的 `src/ctmr/grid/`、[ADR-0009](0009-frozen-instrument-command.md) 的 `src/ctmr/instrument/` 平级——轴线三环：输入几何 → **仪器读数** → 冻结调用）：
   - `src/ctmr/measure/regions.py` — **REGIONS 单一事实源**（dict 字面量 `{"WT":(1,2,3),"TC":(1,3),"ET":(3,)}`）＋ `RegionMasks` 协作者（掩码 → per-region 布尔提取）。终验 judge 的 tuple 名形式由此 dict **派生**，不另立第二份。
   - `src/ctmr/measure/metrics.py` — 共享度量原语（**类方法，非自由函数**，遵 python.md）：`DiceScore`（空分母 → `None` 单一哨兵，决定 4）、`WilsonUpper`（含 `n==0` 守卫单一版，决定 4）。
   - `src/ctmr/measure/hierarchy.py` — `HierarchyChecker.violates(pred) -> bool`：canonical containment 单表达式（决定 3）。
   - `src/ctmr/measure/measurement.py` — **`CaseMeasurement`** 值对象（canonical 逐病例测量行，6 处并集字段）＋ `to_long_rows()`（校准，1 行/region）/ `to_wide_row()`（终验，1 行/obs）双序列化器。
   - `src/ctmr/measure/measurer.py` — **`InstrumentMeasurer.measure(pred, *, gt=None, condition=None, brain=None) -> CaseMeasurement`** 唯一入口（决定 2），组合上述协作者。
   - 纯变换、numpy 数组进出、**不做文件 IO**、无集群/路径耦合（遵 ADR-0008/0009）；6 个调用脚本改**绝对导入** `from ctmr.measure...`，只留执行入口与受控路径 IO。`src/` 上 `sys.path` 属执行期接线。
2. **统一 interface：mask + 区域定义 → 测量行，三列族显式门控**：
   - 输入：`pred`（仪器网格 240×240×155 上的预测掩码，zyx numpy 数组）必填；`gt` / `condition` / `brain` 为**可选类型化 reference**。
   - **门控**（列族按提供的 reference 显式开启）：给 `gt` → **校准列**（vol_gt/dice/sensitivity/precision/hd95/n_comp/质心距离）；给 `condition` → **回切 Dice 列**（`cond_dice_wt/tc/et`）；给 `brain` → `brain_ml`/`wt_brain`；**生成列**（vol/centroid/`pred_empty`/`hierarchy_violation`）与 `et_wt` 恒定产出。「回切 Dice」即终验 `cond_dice_*`（pred vs condition mask），#2 与 #6 同一概念，统一由 `condition` 门控。
   - canonical `CaseMeasurement` 为**唯一测量逻辑**；long（#1）与 wide（#2）仅序列化差异，化解两冻结路径形状分歧而测量逻辑只写一次。`input_fail`/`run_fail` 为调用方回填的执行标志，非测量逻辑计算。
3. **hier_viol 收敛为单一 containment 语义 + 校准「病例可用性」拆分**：
   - canonical **`hierarchy_violation(pred)`** = containment（ET⊄TC 或 TC⊄WT）∨ 值域 ⊄{0,1,2,3}，单表达式、**无前置守卫**，收敛终验正确版 `final_acceptance_nifti.py:196-201`，即 ADR-0004 决定 4 冻结定义。空超集不豁免（ET 存在而 TC 空 = 真违反，须被捕获）。
   - 校准母版 #1 的「hier_viol」是**不同概念**（需 GT：GT 值域 ∧ pred 值域 ∧ GT-WT 非空），**拆分为独立的「校准病例可用性」检查**（`CalibrationCaseUsability`），仍喂校准 R_fail（保持 ADR-0002 语义）；它**不进入** canonical `CaseMeasurement` 的 `hierarchy_violation`。名称冲突（同叫 hier_viol 却测两件事）即真 bug，拆分消除之。
   - #3（vacuous 守卫）、#4（空超集前置守卫）、#5（仅值域）的 hier_viol **收敛到 containment**（有意变更，见后果）。
4. **REGIONS / Wilson / Dice 唯一事实源**：
   - **REGIONS** 单一字面量（dict），7 份收编；tuple 形式派生。
   - **Wilson** 单一带 `n==0` 守卫版；终验 judge 调用点本已守卫（`final_acceptance.py:680`），收敛后**冻结路径输出不变**。
   - **Dice** 单一版，空分母哨兵统一 **`None`**（对齐终验冻结语义）；校准旧 `nan` 分歧注册（空例本就排除出 Dice 分布，**ADR-0002 聚合不变**）。
5. **6 处收编映射**：

   | # | 文件:行 | 角色 | 收编处置 | 冻结分类 |
   |---|---|---|---|---|
   | 2 | `nnunet_l2_final_acceptance_nifti.py:187-279` ＋ judge `nnunet_l2_final_acceptance.py`（REGIONS `:83-84` / Wilson `:686` / `MEASUREMENT_FIELDS :112-120`） | 终验（ADR-0004） | **纯代码搬移**进 module，读数不得变 | **冻结**（每候选重跑） |
   | 1 | `nnunet_l2_calibration_metrics.py:93-170`（REGIONS `:24` / Wilson `:173` / `dice_of :53` / hier `:154-157` / `CSV_FIELDS :30-38`） | 校准母版（ADR-0002） | 测量逻辑入 module；**hier_viol 改名「病例可用性」（值不变）**；Dice 哨兵 `nan→None`（聚合不变） | **冻结聚合**（包络已记录） |
   | 3 | `nnunet_l2_synthetic_domain_eval.py:515-613`（REGIONS `:524` / Wilson `:638` / Dice `:603` / hier `:578`） | #38 合成域 | 收敛 containment（**有意变更**；修 vacuous 守卫） | 非冻结 |
   | 4 | `l2_synth_domain_sugon.py:363-404`（REGIONS `:45` / Wilson `:353` / hier `:391-402`） | #38 sugon 版 | 收敛 containment（**有意变更**；去空超集前置守卫） | 非冻结 |
   | 5 | `brats_p1_dev_eval.py:369-482`（REGIONS `:373` / hier `:449`） | P1 dev 趋势 | 收敛 containment（**有意变更**；P1 dev 已完成，低风险） | 非冻结 |
   | 6 | `brats_p2_dev_eval.py:271-348`（`INSTRUMENT_REGION_LABELS :86` / `dice :292`） | P2 回切 Dice | 收敛；**cond_dice 已 canonical（`None` 哨兵、借 #2 resampler）→ 读数保持** | 非冻结（在飞 P2 #59） |

6. **数值一致性验收口径（两级门禁，执行期跑）**：
   - **冻结门禁**：① 终验（#2）纯搬移后，在固定持出病例集上重跑测量，测量行 / L2 读数与重构前冻结记录**逐字节一致**（需 sugon）；② 校准（#1）在**冻结校准预测 + GT** 上**重跑测量**（**不重演昂贵推理**——预测为 ADR-0002 冻结产物），ADR-0002 冻结聚合（D_r,low / E_r,vol / E_r,centroid / E_r,HD95 / R_fail 含 breakdown / R_miss / Dice 中位 / 空 pred 排除计数）与已记录值**逐字节一致**（需 sugon 受控目录）。逐病例 CSV 的列改名 / 哨兵差异已注册、不属冻结聚合。
   - **收敛门禁**：#3/#4/#5/#6 收敛后，同一输入下各调用点产出与 canonical 模块输出**逐位一致**——证明落到标准上。unit 级，任意机器可跑。
   - 门禁只保证「冻结路径不变」与「收敛后 = 标准」；#3/#4/#5 的 hier_viol → containment **有意改变**各非冻结路径自身前后读数，不在「不变」承诺内。

## 前置（硬阻塞）

- 本 ADR 只钉决策，**代码不动**；重构落地与两级门禁验证在执行期另行开始（新 effort / 新 issue），且须在 ADR 齐备后启动。
- 依赖 [ADR-0008](0008-instrument-grid-geometry.md)（仪器输入几何）已钉板：测量作用于 `ctmr.grid` 产出的仪器网格掩码，本 module 不重读几何。

## 后果

- **非冻结路径读数有意变更（本 ADR 注册）**：#3/#4/#5 的 hier_viol 收敛为 containment，改变各自 hier_viol 计数；这些是 #38 研究与 dev 侧车路径，非 ADR-0002/0004 冻结对象。冻结终验路径读数不变。
- **校准「hier_viol」概念拆分（本 ADR 注册）**：#1 的 hier_viol 改名「校准病例可用性」（值不变），与 canonical containment 区分；ADR-0002 冻结聚合经冻结门禁②验证不受影响。Dice 哨兵 `nan→None` 同理（空例排除，聚合不变）。
- **#6 读数保持，无新 P2（#59）时机问题**：与 [ADR-0008](0008-instrument-grid-geometry.md):57 的 FID-trend 不同，#6 的 cond_dice 语义已与冻结终验一致，收敛后逐位相同，**不引入 P2 dev 趋势重构点**。落地时机仍受执行期统一调度。
- **src layout 部署形态变化**：与 ADR-0008/0009 一致，`l2_synth_domain_sugon` 等不再自包含，sugon 部署须连同 `src/` 树同步并把 `src/` 加入 `sys.path`。
- 新术语「**仪器读数 / InstrumentMeasurement**」「**层级违反（hierarchy_violation）**」「**校准病例可用性**」入 `CONTEXT.md`。
- 若需改 interface 形状、列族门控、REGIONS/Wilson/Dice 定义、hier_viol 语义、收编映射或验收口径：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
