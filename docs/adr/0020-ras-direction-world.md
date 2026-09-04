# ADR-0020：全链方向世界声明——RAS 统一、仪器链 flip 退役与 T8 判读修订

- **状态**：已接受（2026-09-04；序列③T4,治理票 #314,父 spec #310）
- **范围**：整链方向语义的单一声明、仪器输入组装 real 侧的 affine 驱动 RAS 统一（copyfile 直通退役）、生成侧 RAS→LPS flip 补偿退役与两侧同世界断言、P2 monitor 同族 flip 处置、`ctmr.domain.orientation` 新 domain 模块、测试 gate、T8 判读修订（另落盘 `deploy/experiments/`）。
- **取代性声明**：
  - **部分取代 ADR-0008**：「Terminal-acceptance-only concerns（the DM RAS->LPS axis flip, file IO）stay with that caller」一句——该 flip 退役,方向世界语义由组装 caller 经 `ctmr.domain.orientation` 承载;**几何本体（adapter 参数表、居中 crop/pad、`CenterCropOrPad` 重置 direction 为 identity、`INSTRUMENT_GRID`）全部维持冻结,逐字不动**。
  - **不改变**：ADR-0002 校准包络数值、ADR-0004 通过线与判定列、ADR-0009 冻结仪器调用、ADR-0010 测量口径、ADR-0012 层注册、#249 写出协议（DM 输出网格与 affine）、仪器权重（nnUNet results v2 树）。

## 背景

- **上游既定约定（initial commit 取证）**：RAS 是 NVIDIA 上游的全链方向世界。initial commit `2cb90dd`（"initial version", 2025-10-23）中 `scripts/diff_model_create_training_data.py`、`scripts/transforms.py`、`scripts/sample.py`、`scripts/utils.py`、`scripts/inference.py` 一致使用 `Orientationd(axcodes="RAS")`;生成写出 affine 为正对角阵（nibabel axcodes = RAS）。本仓后续修复（#249 写出协议、序列③T2 编码链、T3 loader 守卫）均沿用 RAS。
- **错位机制（方向审计 #310 → T1 复核 #318 改判后的真实修复面）**：仪器链组装曾横跨两个方向世界——real 侧 native 直通（BraTS 2023 官方源 ~89.4% LPS / ~10.6% RAS,nibabel axcodes 实证,见 T1 记录 §4.1）,gen 侧经 `DM_GRID_TO_LPS_AXIS_FLIP` 无条件 x/y 翻转到 LPS。对 LPS real 恰好对位;对 RAS real 逐案 x/y 全幅错位（~240 mm）。错位是 **per-case 的方向 class**,不是随机噪声。
- **危害量级（T8 判读修订的依据,详见实验记录）**：T6/T8 监控配额下 METS n=24,期望 ~2-3 例 RAS-coded real 错位,而 METS ET 检出率的判定线粒度恰为 2 例（21/24 fires / 22/24 clears）——污染量级与判定线分辨率同阶;~10.6% 的 gen/real 配对读数（rel_diff、real 体积、real_only 计数）被错位污染。
- **P2 monitor 的同族矛盾（重审结论）**：`RoundTripDice.align_condition` 曾复制终验的 flip（「追踪 final-acceptance resampler path」）,但其 prediction 来自 `L2TrendRunner.prep_inputs`——该路径**从不 flip**（RAS 数组直入仪器）。condition 被 flip 到 LPS 而 prediction 在 RAS,round-trip Dice 趋势跨世界错位。
- **T1 复核（#318）的边界贡献**：labels 树实证已是忠实 RAS 产物（20/20 dice=1.0000）,训练世界（embedding decode 对位 AUC 5/5 as-is 最高）未受方向污染——本票只修仪器读数面,不动训练与数据工件。

## 决定

1. **全链方向世界 = RAS**：训练、编码、采样写出、仪器输入组装、测量对位统一声明 RAS 世界。它不是新发明,是把上游既定约定（initial commit 取证）显式钉板。
2. **domain 新模块 `ctmr.domain.orientation`**：
   - `RasOrientation`：方向世界服务。`to_ras(image)` = affine 驱动的 RAS 统一（轴对齐断言 → `SimpleITK.DICOMOrient("RAS")` → 输出 RAS 断言）;`require_ras(image)` = RAS 世界断言（不修改）。违反 → `NotRasWorldError`。
   - **轴对齐边界**：统一语义覆盖全部 48 种轴对齐编码（permute×flip,物理点守恒由测试逐枚举机器守卫）;斜 direction 是响亮失败而非静默近似——SimpleITK 2.5.6 的 `DICOMOrient` 对斜方向**不抛异常**（实证）,断言必须自持。
   - 与 `ctmr.domain.grid` 同纪律：纯内存 `sitk.Image` 进出,零文件 IO。
3. **仪器输入组装语义（执行侧 `measurement_run`）**：
   - `GeneratedVolumeResampler` 改名 **`InstrumentInputAssembler`**（原名义只覆盖 gen 侧,改后承担两侧）,组合 `InstrumentGridAdapter`（几何,不动）+ `RasOrientation`(方向世界）：
   - **gen 侧 `write`**：`require_ras` 断言 + 冻结几何。生成件的 RAS 世界由 #249 写出协议钉板,非 RAS 声明 = 上游协议破坏,响亮死,**不静默改向**。
   - **real 侧 `write_real`（新）**：`to_ras` affine 驱动统一 + 冻结几何。`copyfile` 直通退役——native 方向混合不再是组装的输入假设,而是被统一的对象。
   - **条件掩码 `label_to_grid`**：`to_ras` 防御统一（labels 树已实证 RAS,幂等）+ 最近邻几何;flip 退役。
4. **flip 补偿退役,两侧同世界断言**：`DM_GRID_TO_LPS_AXIS_FLIP` 与 `_flip_dm_grid_to_lps` 删除。错位 class 由构造消除：两侧在组装点进入同一 RAS 世界,不再需要事后翻转补偿。**文件面不可见性说明**：`CenterCropOrPad` 重置 direction 为 identity（ADR-0008 冻结几何,不动）,故组装产物无法从文件头区分数组世界——同世界断言住在组装内存路径（`require_ras`/`to_ras`+断言）,这是它能「从构造上消除」而非「事后检出」的原因。
5. **P2 monitor 同族处置（重审后同法退役）**：`RoundTripDice.align_condition` 去 flip,`to_ras` + 最近邻几何;同步修复其 prediction（`L2TrendRunner.prep_inputs`,RAS 直入）与 condition 的世界矛盾——该趋势的旧读数跨世界错位,历史断层见后果。parity gate（`test_roundtrip_sidecar.py`）语义从「追踪 flip」改为「两侧同世界逐位一致」。
6. **其余仪器输入组装点同法收口**：`L2TrendRunner.prep_inputs`（gen 侧,补 `require_ras`）、`synthetic_domain.InputPreparator.prepare_case`（gen 侧,补 `require_ras`）——「两侧同世界断言」覆盖全部组装点,不留无断言旁路。
7. **测试 gate**：
   - 新 `tests/domain/test_orientation.py`：RAS 幂等、LPS 统一物理守恒、48 种轴对齐编码全枚举（marker 物理点逐例守恒）、斜 direction 响亮死、`require_ras` 断言。
   - 新 `tests/application/acceptance/distribution/test_real_side_ras_unification.py`（issue #314 验收单测）：同一物理 phantom 以 LPS/RAS 两种编码作 real 件、以 DM 网格 RAS 作 gen 件,经公共 `ObservationInputWriter` 组装后——两 real 编码逐位一致、gen 肿瘤质心与 real 对位 < 3 voxel（旧世界错位 ~180 voxel）;斜 real 件与非 RAS gen 件响亮死。
   - 既有 convergence gate 更新：`FinalAcceptanceReferenceGeometry` 快照去除 flip 行（ADR-0008 标准 + ADR-0020 修订）;fixture 进入 RAS 声明世界。

## 冻结边界（零改动核对声明）

| 对象 | 状态 |
|---|---|
| 仪器权重（nnUNet results v2 树） | 零改动 |
| 校准包络数值（ADR-0002） | 零改动 |
| 判定线/通过线（ADR-0004） | 零改动 |
| `INSTRUMENT_GRID`、adapter 参数表、居中 crop/pad、identity direction 重置（ADR-0008 几何） | 零改动（逐字） |
| 测量口径（ADR-0010 `InstrumentMeasurer`） | 零改动 |
| #249 写出协议（`V1_DM_OUTPUT_GRID`） | 零改动 |
| 训练/数据工件（labels 树、embeddings） | 零改动（T1 复核边界） |
| 改动仅及：输入组装语义（方向世界） | 本票 |

## 后果

- **仪器链读数历史断层（注册）**：`assemble-execute`/`measure` 的 gen/real 配对读数在方向语义上以本票为界——T5/T6/T8 等旧读数是「LPS 直通 + 无条件 flip」污染世界的读数（~10.6% real 观测错位）;新读数不得与旧读数直接对照。dev 监控重跑（T6 配方零改动,语义随代码携带）即得新世界基线。
- **校准重跑与 spec 冻结锚重钉是任何 L2 终验前置**（状态不变,见 20260901 仪器主本丢失记录 §5）：校准集输入组装同受本票语义影响,校准包络须在新世界下重钉——**另票执行,本票不改其状态、不跑校准**。
- **P2 round-trip Dice 趋势断层**：修复前该趋势跨世界错位（系统性偏低）,趋势点以本票为界不可比。
- **旧世界工件不进新仪器链**：非 RAS 声明的 gen 件、斜 direction 件响亮死;若需复用须先显式重排（新语义下的 `to_ras`）或重跑写出。
- **词汇**：新术语「方向世界（RAS direction world）」入 `CONTEXT.md`;`InstrumentInputAssembler` 取代 `GeneratedVolumeResampler` 命名（git 历史即复现锚,不留 shim,沿 #175 惯例）。
- 若需修改方向世界声明、组装语义或轴对齐边界：**新 ADR 取代本条**。
