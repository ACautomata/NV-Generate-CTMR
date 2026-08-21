# ADR-0004：L2 冻结仪器最终验收通过线（五子挑战）

- **状态**：已接受（2026-08-21；通过线在任何 L2 终验运行之前预注册冻结）
- **范围**：生成候选（P1/P2/P3）在五个冻结 L2 测量仪器上的分布对齐、P2 条件回切与 undecided 判定线

## 背景

[Spec：BraTS2023 Rectified Flow 分阶段微调配方](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) decision 10 把 L2 最终验收委托给 [ADR-0002](0002-l2-instrument-calibration-envelopes.md) 的冻结误差包络（"L2/P2 可按 #32 §5 的 margin 公式设定最终通过线"）。[L2 冻结仪器验收](https://github.com/ACautomata/NV-Generate-CTMR/issues/55) 据此定线；本 ADR 只引用 ADR-0002 数值，不产生新测量数值。预注册协议全文见 `docs/calibration/l2-final-acceptance-protocol.md`（先于任何运行冻结）。

## 决定

1. **测量量与 TOST 边界（每子挑战 × 区域 WT/TC/ET）**。以同批病例「真实四模态 → 仪器测量」为参照侧、「仪器使用协议组装的生成四模态 → 仪器测量」为生成侧，逐病例配对：
   - 区域体积相对差 `(V_gen − V_real)/V_real`：病例级（P3 为病例级 cluster，4 锚轮差值整组重采样）bootstrap 90% CI 完整落入 `±E_r,vol[区域]`；
   - SRI24 质心逐轴位移 `(c_gen − c_real)`（mm）：90% CI 完整落入 `±E_r,centroid[区域]`（三轴各自检验）；位移范数 p95 单侧 bootstrap 上界如报告则仅为伴随诊断，不作门槛；
   - `WT/brain` 相对差：margin = `E_r,vol[WT]`；`ET/WT` 相对差：margin = `E_r,vol[ET] + E_r,vol[WT]`（比值相对误差一阶传播为两区域之和；全部为已冻结数，无新数字）。
2. **margin 分解的量纲依据（对 spec 字面公式的解释性预注册）**。`E_r,vol` 无量纲、`E_r,centroid` 单位 mm，二者之和作为单一标量在量纲上不成立；注册规则"margin ≥ E_r,vol + E_r,centroid + 重复性"的忠实实现是**逐轴覆盖仪器误差盒**——体积类量每轴 margin 恰为 `E_r,vol[区域]`、质心类每轴恰为 `E_r,centroid[区域]`，重复性 p95 冻结为 0（ADR-0002）。每轴 margin 等于该轴仪器分辨率，不窄于注册预算。
3. **P2 条件回切 Dice**。逐病例回切 Dice（仪器 pred 对条件 mask 按区域派生）分布的**单侧 5th percentile bootstrap 95% 下界**（与 `D_r,low` 同构的统计量）必须 ≥ floor，floor = `D_r,low[区域]`。METS 三区域 floor=0 → 恒真，报告显式标注 vacuous-pass（区分力极弱是 ADR-0002 已声明的事实，不粉饰）。禁止配置任何 `> D_r,low` 的 floor（工具在载入时拒绝）。
4. **undecided 判定链（非补偿）**。任一待测观测（生成侧或真实参照侧）出现 `input_fail`/`run_fail`/层级违反（ET⊆TC⊆WT 或值域 ⊄ {0,1,2,3}）→ 该子挑战 L2 = `undecided`（仪器不可用，阻塞完整终验；修复方向是仪器/重跑，非生成候选）。未定态优先于 TOST/回切判定。Wilson 95% 上界对照真实校准 R_fail 包络仅作诊断报告。五子挑战按非补偿 AND 聚合。
5. **冻结核验不变**。终验只接受 [ADR-0003](0003-l2-instrument-frozen-artifacts.md) §6 冻结审计（verdict SHA-256 `9121e8ac…`，`all_passed: true`）或其复跑通过；不接受任何更窄的自建核验。仪器 checkpoint、plans、推理配置与校准包络在终验期间不得更改。
6. **空 pred 与未定义量**。空 pred 是测量结果而非失败（照抄校准协议 §5）：体积类相对差在生成侧空 pred 时为 `(0−V_real)/V_real = −1.0` 全额进入分布；分母侧（真实侧体积、WT）为空/未定义时该病例该量**排除并单独计数报告**（双侧非空子集语义），绝不静默丢行。真实侧每病例只测一次，跨 P1/P2/P3 复用同一参照测量。

## 后果

- `scripts/nnunet_l2_final_acceptance.py`（判定链，stdlib-only）与 `scripts/nnunet_l2_final_acceptance_nifti.py`（NIfTI 执行侧，sugon 运行）按本 ADR 与预注册协议实现；L2 报告经 #53 run contract `attach --kind l2_report` 入 run record；
- METS/PED 的宽包络（含 METS ±1.0–1.65 体积 margin 下 TC/ET 全漏仍可过等价的区分力极限）如实携带 ADR-0002 声明，不收窄、不修饰；
- 本 ADR 的解释性决策若需变更，须新 ADR 显式取代，不得原地修改。
