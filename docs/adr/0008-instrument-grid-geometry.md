# ADR-0008：仪器输入几何统一 module（InstrumentGridGeometry）——src layout 单引擎双 adapter 与 8 份收编

- **状态**：已接受（2026-08-25；在任何重构执行之前预注册）
- **范围**：「1mm 重采样＋居中 crop/pad 到 240×240×155」仪器输入几何的 8 份散落实现、3 种口径的统一 interface 钉板（wayfinder 地图 #75 候选 2 / 票 #76）。本 ADR 只钉决策与执行 spec，代码不动；重构与验收在执行期另行开始。

## 背景

「重采样到 1mm 各向同性＋裁剪到 240×240×155」这一仪器输入几何，目前散落在 7 个脚本共 8 份实现、3 种口径，且漂移已兑现：

- **轴序 bug**：#38 原件 `nnunet_l2_synthetic_domain_eval.py:411-459` 的 `_crop_or_pad` 用 xyz 尺寸 `np.zeros(target_size)` 建数组、xyz 序构建 slices，却直接作用在 zyx 数组上（未 reverse）；`l2_synth_domain_sugon.py:206-236` 同族（`crop_or_pad(arr, NN_SIZE)` 把 xyz target 作用于 zyx 数组）。终验修正版 `nnunet_l2_final_acceptance_nifti.py:52-54` docstring 自述「在副本修正、原件带病保留」。
- **口径自相矛盾**：`p1_prep_inputs.py:4-5` 自述「linear 插值…与 nnUNet 预处理一致」，`nnunet_l2_final_acceptance_nifti.py:8-10` 自述「#38 InputPreparator geometry verbatim — resample to 1mm B-spline」；同一台冻结仪器，prep 脚本用 linear（`p1/p3/brats_p1_dev_eval.py:386`），#38 集成管线与终验却用 B-spline。
- **非居中漂移**：`brats_p1_dev_eval.py:386` 用 `[:limits]` 顶左对齐而非居中（已确认为 bug，非刻意）。

硬约束：**不改变终验路径的冻结读数**（[ADR-0002](0002-l2-instrument-calibration-envelopes.md) 校准包络 / [ADR-0004](0004-l2-final-acceptance-pass-lines.md) 通过线；ADR-0004 决定 5「冻结核验不变」）。冻结终验几何 = `nnunet_l2_final_acceptance_nifti.py:79-117`（B-spline 连续体 ＋ 最近邻 label ＋ 轴序已修正居中 crop/pad），即本统一的**收敛标准**。

**B-spline 的成因（如实记录）**：B-spline 并非刻意选型，系早期某次 spec 探索由代码自动生成、沿用至今。它现在的权威性**不来自「被选为最优」，而来自它被冻结进终验路径、ADR-0002/0004 冻结不可改**；因此它是统一标准的唯一合法选择（linear 若上收编须改终验，构成冻结违约，禁止）。

## 决定

1. **统一 module 形状与落点（src layout）**：新建源码根 `src/`（现仓库无可安装包，`pyproject.toml` 仅 ruff 配置），通用几何以伞包子包落于 **`src/ctmr/grid/`**（包名嵌套无下划线，遵 python.md）：
   - `src/ctmr/grid/geometry.py` — 通用引擎，纯变换、无集群/路径耦合：**内存中 `sitk.Image` 进出、不做文件 IO**（文件读写、受控路径全留调用方）。
     - `TargetGrid`（值对象）：目标网格 = 尺寸 + 间距。仪器网格 `240×240×155 @1mm`、报告侧逐病例网格、趋势特征的 `240×240×160` 均为其实例——「网格是入参」由此得证。
     - `GridResampler`：构造时**注入插值策略**（B-spline / 最近邻 / linear），方法 `resample(img, grid)`；组合 `CenterCropOrPad`。
     - `CenterCropOrPad`：轴序正确的 zyx 居中 crop/pad（即终验修正版 `final_acceptance_nifti.py:96-115` 逻辑：`np.zeros(reversed(target))`、slices 按 xyz 构建、reversed 应用）。
   - `src/ctmr/grid/instrument.py` — **仪器输入几何（InstrumentGridGeometry）** 特例：`INSTRUMENT_GRID = TargetGrid((240,240,155),(1,1,1))` 常量 ＋ 两个钉死 adapter 工厂（连续体 = B-spline、label = 最近邻），即决定 2 的参数表。
   - 8 个调用脚本改为**绝对导入** `from ctmr.grid...`，只留执行入口与集群/受控路径 IO。`src/` 上 `sys.path`（PYTHONPATH 或后续补打包）属执行期接线。
2. **两个显式 adapter（按数据种类，非按插值）的参数表**——即「仪器输入几何」：
   - **连续生成体**：B-spline 插值 · `SetDefaultPixelValue(GetPixelIDValue())` 背景填充 · 居中 crop/pad 到目标网格。
   - **label/条件掩码**：最近邻插值（不发明新 label 值）· 居中 crop/pad 到目标网格。
   - 二者即终验修正版 `GeneratedVolumeResampler.write`(B-spline) / `label_to_grid`(NN) 的既有结构；收敛标准 = `final_acceptance_nifti.py:79-117`。
3. **插值显式策略**：单引擎、插值作构造注入策略。仪器输入 adapter 钉死 B-spline/NN（冻结标准）；非仪器消费方各传其策略（趋势特征用 linear、报告侧 image 用 B-spline / label 用 NN）——消除「同一操作 8 份手写」，但不强制每个消费方绑死 B-spline。
4. **8 份收编映射**：

   | 文件:行 | 现口径 | 处置 |
   |---|---|---|
   | `p1_prep_inputs.py:28` | linear·居中 | linear→**B-spline**，→ 连续体 adapter |
   | `p3_prep_inputs.py:29` | linear·居中 | linear→**B-spline**，→ 连续体 adapter |
   | `brats_p1_dev_eval.py:386` | linear·**非居中** | linear→B-spline ＋ **修非居中→居中**，→ 连续体 adapter |
   | `l2_synth_domain_sugon.py:206` | B-spline·**轴 bug** | **修轴 bug**，→ 连续体 adapter |
   | `nnunet_l2_synthetic_domain_eval.py:411` | B-spline·**轴 bug** | **修轴 bug**，→ 连续体 adapter |
   | `nnunet_l2_final_acceptance_nifti.py:79` | B-spline·居中（收敛标准） | **纯代码搬移**进 module，读数不得变 |
   | `brats_p1_l2_html_nifti.py:102` | B-spline/NN·逐病例网格 | 引擎**客户端**，逐病例 `TargetGrid` 入参 |
   | `brats_p1_dev_eval.py:273`（趋势特征） | numpy 索引抽稀·顶左·网格 160 | **全量收编**：重采样＋居中走引擎、网格 `240×240×160`；百分位强度归一化留在特征提取器（非几何）→ **FID 趋势偏移（有意，见后果）** |

5. **数值一致性验收口径（两级门禁，执行期跑）**：
   - **冻结门禁**：对终验路径（`final_acceptance_nifti.py`）只做**纯代码搬移**（`:79-117` 抽进 `ctmr.grid`、改为调用）；重构后在**固定持出病例集**上重跑 `assemble-execute`＋`measure`，测量 CSV 与重构前冻结记录**逐字节一致**（NIfTI 哈希同）。此跑需 sugon。
   - **收敛门禁**：同一合成输入下，收敛后的 `l2_synth_domain_sugon`/`synthetic_domain_eval`（修轴后）与 `p1/p3/dev_eval:386`（linear→B-spline、居中后）产出须与终验几何输出**逐位一致**——证明落到标准上。sitk 单元级，任意机器可跑。
   - 门禁只保证「冻结路径不变」与「收敛后 = 标准」；linear→B-spline、居中/轴修复**有意改变**各非冻结路径自身前后读数，不在「不变」承诺内。

## 前置（硬阻塞）

- 本 ADR 只钉决策，**代码不动**；重构落地与两级门禁验证在执行期另行开始（新 effort / 新 issue），且须在 ADR 齐备后启动。

## 后果

- **非冻结路径读数有意变更（本 ADR 注册）**：`p1/p3/dev_eval:386` 的 linear→B-spline、`dev_eval:386` 居中修复、`l2_synth_domain_sugon`/`synthetic_domain_eval` 轴 bug 修复，均改变各自输出；这些是 #38 研究与 dev 侧车路径，非 ADR-0002/0004 冻结对象。冻结终验路径读数不变。
- **#3 趋势特征全量收编 → FID 趋势偏移**：`brats_p1_dev_eval.py:273` 经 `from .brats_p1_dev_eval import MrTrendFeatures` 被在飞 P2（#59）dev 侧车复用（`brats_p2_dev_eval.py:66`）。收编使其重采样由 numpy 索引抽稀改为引擎插值、顶左改居中，**RadImageNet FID 特征偏移 → P1/P2 dev FID 趋势在重构点前后不可比**。按预注册文化此变更显式注册于此；**落地时机须在 P2 产生 dev 评估点之前，或对 P2 趋势做重基线**。
- **src layout 部署形态变化**：`p1/p3/l2_synth_domain_sugon` 现以单文件拷至 sugon 跑（docstring 自述 `python3 /root/private_data/l2-synth-eval/p1_prep_inputs.py`）；几何迁入 `src/ctmr/grid/` 后这些脚本不再自包含，**sugon 部署须连同 `src/` 树一起同步**，执行期相应调整 rsync/部署清单，并把 `src/` 加入 `sys.path`。
- 新术语「**仪器输入几何 / InstrumentGridGeometry**」入 `CONTEXT.md`。
- 若需改 interface 形状、adapter 参数表、收编映射或验收口径：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
