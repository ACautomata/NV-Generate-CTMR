# ADR-0009：冻结仪器调用单一构造点（FrozenInstrumentCommand）——src layout 收编 7 处手写 predict 命令与 3 份 weights_only 白名单

- **状态**：已接受（2026-08-25；在任何重构执行之前预注册）
- **范围**：L2 肿瘤测量仪器「冻结推理配置」的 7 处手写 `nnUNetv2_predict` 调用、3 份逐行复制的 torch `weights_only` 白名单的统一构造点钉板（wayfinder 地图 #75 候选 3 / 票 #77）。本 ADR 只钉决策与执行 spec，代码不动；重构与验收在执行期另行开始。

## 背景

「冻结仪器调用」——以固定配置（fold 0、`nnUNetTrainer250Epochs`、镜像 TTA on；SSA 用 `3d_fullres_bs16`+`nnUNetPlans_SSA_bs16_v1`）驱动 nnU-Net 预测——目前散落多处手写，漂移已兑现：

- **命令构造 7 处、入口名 3 种**：标准 CLI `nnUNetv2_predict` 仅 `nnunet_l2_final_acceptance.py:450-458`（#58 终验，生成 shell；`PredictScriptWriter` 类 433-472）；非标准 `nnUNetv2_predict_from_raw_data`（nnUNetv2 2.x 无此入口名，标准为 `nnUNetv2_predict`）见 `nnunet_l2_synthetic_domain_eval.py:491`、`run_l2_synth_domain_eval.sh:87`；Python in-process 入口 `l2_calibration_predict_entry.py`（→ 原生 `predict_entry_point`）被 `l2_synth_domain_sugon.py:327`、`brats_p1_dev_eval.py:415`、`p1_predict_all.sh`、`l2_calibration_predict.sh` 使用。
- **`--disable_tta False` fatal token 3 处**：`nnunet_l2_synthetic_domain_eval.py:498`、`l2_synth_domain_sugon.py:334`、`run_l2_synth_domain_eval.sh:94`。在 nnUNetv2 2.8.1 argparse（`store_true` + `parse_args()`）下，该 token 触发 `unrecognized arguments: False`、**exit 2**（#78 findings 实测），而非「静默关 TTA」——与「冻结推理配置不可动」相悖。
- **weights_only 白名单 3 份逐字节复制 + import 时改全局状态**：`nnunet_l2_instrument.py:29`、`nnunet_l2_closing_verification.py:26`、`l2_calibration_predict_entry.py:15`，载荷完全相同（`numpy.core.multiarray.scalar`、`numpy.dtype` 及一组数值 dtype 类），均在模块顶层 `add_safe_globals(...)`（import 副作用）。配套 load 点：`nnunet_l2_instrument.py:388`、`nnunet_l2_closing_verification.py:152`（均 `weights_only=True`）。另有 5 处**不同目的**的 `add_safe_globals`（`brats_p1_dev_eval.py:69`、`brats_p2_dev_eval.py:88`、`brats_p1_finetune.py:179`、`brats_p2_finetune.py:170`、`prototype/p3_image_cond_controlnet/p3_common.py:125`；reference-bank / MONAI MetaTensor / TraceKeys），不属本收敛范围。

硬约束：**不改变终验路径的冻结读数**（[ADR-0002](0002-l2-instrument-calibration-envelopes.md) 校准包络 / [ADR-0004](0004-l2-final-acceptance-pass-lines.md) 通过线）。#58 终验判定链（`nnunet_l2_final_acceptance.py`）为冻结路径；#36 校准包络（`l2_calibration_predict.sh` 产生）为冻结读数来源、**不重跑**。#38 历史结论**不重跑、不复核**（#78 判「不可能翻转」，#79 已正式裁决＝维持有效）；本 ADR 只管未来口径正确。

## 决定

1. **统一 module 落点与构造点形状（src layout）**：通用「冻结仪器调用」归于 **`src/ctmr/instrument/`**（与 [ADR-0008](0008-instrument-grid-geometry.md) 的 `src/ctmr/grid/` 平级）：
   - `src/ctmr/instrument/command.py` — **`FrozenInstrumentCommand`**：实例持有该挑战的冻结 spec（`InstrumentSpec` 值对象：`dataset_id`、`config`、`plans`、`trainer`、fold 0；SSA = `3d_fullres_bs16`+`nnUNetPlans_SSA_bs16_v1`）。方法 `build(input_dir, output_dir) -> list[str]` 为**纯函数**：只产出 argv，**不执行、不做文件 IO**（沿用 ADR-0008「纯变换、无集群/路径耦合」）。argv 入口固定为决定 3 的 canonical 入口；执行（subprocess / 写 shell）留调用方。
   - `src/ctmr/instrument/safeglobals.py` — weights_only 白名单单一定义 + `nnunet_safe_globals()` 上下文管理器（决定 4）。
   - `src/ctmr/instrument/predict.py` — 唯一 canonical 执行入口（决定 3）。
   - 调用脚本改为绝对导入 `from ctmr.instrument...`；`src/` 上 `sys.path`（PYTHONPATH 或后续补打包）属执行期接线（同 ADR-0008）。
2. **TTA-on 为不变量，无 TTA 形参**：`FrozenInstrumentCommand.build` **永不产出 `--disable_tta`**——镜像 TTA on 靠省略成立（`store_true` 语义）。interface 不暴露任何 TTA 形参（哪怕锁死），杜绝该 bug 类。「TTA-on 为冻结不变量」写入 module docstring 与 `CONTEXT.md`。
3. **单一 canonical 执行入口 `ctmr.instrument.predict`**（`l2_calibration_predict_entry.py` 提升进 src）：`python -m ctmr.instrument.predict` 在 `nnunet_safe_globals()` 作用域内调 nnUNetv2 原生 `predict_entry_point`。Python 调用方 `subprocess.run(cmd.build(...))`、shell 编排脚本调 `python -m ctmr.instrument.predict`——**共用同一入口**，使「单一构造点」对 shell 消费者同样成立，并顺带消除非标准入口名 `nnUNetv2_predict_from_raw_data`。入口对 torch≥2.6 默认 `weights_only=True` 稳健（无论 nnUNet 内部 load 是否触发）。
4. **weights_only 白名单 scoped 收敛**：单一定义迁 `safeglobals.py`；消费方在 `torch.load` / `predict_entry_point` 处改 `with nnunet_safe_globals():`（torch 2.9 支持 `safe_globals` 上下文管理器），**消除 import 时改全局状态**。收编 3 处复制点；5 处异 payload 白名单**不动**。
5. **调用点收编映射**（7 处命令 + 3 份白名单）：

   | 调用点 | 现口径 | 处置 | 冻结？ |
   |---|---|---|---|
   | `nnunet_l2_final_acceptance.py:450-458` | 标准 CLI·生成 shell·TTA on | 换 `FrozenInstrumentCommand.build`；**纯代码搬移，读数不变** | **冻结**·决定 7 门禁 |
   | `nnunet_l2_synthetic_domain_eval.py:490-500` | 非标准入口 + fatal token | 换 builder；修入口名 + 去 token | 非冻结·有意变更 |
   | `l2_synth_domain_sugon.py:326-335` | subprocess + token | 改写为 `ctmr.instrument` shim（决定 6），去 token | 非冻结·有意变更 |
   | `brats_p1_dev_eval.py:412-424` | subprocess·TTA on | 换 builder（已干净） | 非冻结 |
   | `run_l2_synth_domain_eval.sh:87-98` | 非标准入口 + token | 改调 canonical CLI；修入口名 + 去 token | 非冻结 |
   | `p1_predict_all.sh` | 干净·调入口 | 改调 canonical CLI（未来口径统一） | 非冻结（统一） |
   | `l2_calibration_predict.sh` | 干净·调入口（ADR-0002 来源） | 改调 canonical CLI（历史包络**不重跑**） | 非冻结（统一） |
   | `nnunet_l2_instrument.py:29` + `:388` | import-time 白名单 + `weights_only=True` load | 白名单迁 `safeglobals`；load 处 `with nnunet_safe_globals():` | 非冻结 |
   | `nnunet_l2_closing_verification.py:26` + `:152` | import-time 白名单 + `weights_only=True` load | 白名单迁 `safeglobals`；load 处 `with nnunet_safe_globals():` | 非冻结 |
   | `l2_calibration_predict_entry.py:15` | import-time 白名单 + predict_entry_point | 提升为 `ctmr.instrument.predict` 入口，scoped 包裹 | 非冻结 |

6. **sugon 自包含副本同步策略 = #55+ shim / src-tree**（遵 [ADR-0008](0008-instrument-grid-geometry.md):58）：`l2_synth_domain_sugon.py` 改写为 `sys.path.insert` + `from ctmr.instrument import …` 的薄 shim（范式 `nnunet_l2_final_acceptance_nifti.py:37-44`），sugon 部署连同 `src/` 树一起同步；版本锚用 run contract 现有指纹（`git_commit` + `script_sha256`，`brats_phase_run_contract.py:28,331`），不搞文件头「拷贝来源」声明。
7. **数值一致性验收口径（两级门禁，执行期跑）**：
   - **冻结门禁**：`nnunet_l2_final_acceptance.py` 换用 builder 后，在固定持出病例集上重跑，预测/测量 CSV 与重构前冻结记录**逐字节一致**（同 TTA-on、同 `predict_entry_point`）。需 sugon。
   - **收敛门禁**：非冻结路径收编后，同一输入下各调用点产出的 argv 与 `FrozenInstrumentCommand.build` 输出**逐一相等**——证明落到单一构造点。单元级，任意机器可跑。

## 前置（硬阻塞）

- 本 ADR 只钉决策，**代码不动**；重构落地与两级门禁验证在执行期另行开始（新 effort / 新 issue），且须在 ADR 齐备后启动。

## 后果

- **非冻结路径读数有意变更（本 ADR 注册）**：`synthetic_domain_eval.py` / `l2_synth_domain_sugon.py` / `run_l2_synth_domain_eval.sh` 去除 fatal token、修非标准入口名后，若重跑将产生于 TTA on 下的预测；这些是 #38 研究与 dev 侧车路径，非 ADR-0002/0004 冻结对象，**不改变 #38 历史结论**（#78 判「不可能翻转」）。冻结终验路径读数不变。
- **#38 报告保真勘误（本 ADR 注册）**：#38 报告与 resolution 声明「仪器 = TTA on」，与涉事调用点代码事实不符（实际执行形态为 #78 §2.3 三分支之一：TTA off / fatal 后修正为 on / sugon 版差异）。#38 判定量 R_fail 对 TTA on/off 结构性不敏感，判定不依赖该状态；凡引用 #38 读数者，应以本勘误为准，避免「读数产生于 TTA on」的错误引用扩散。
- **src layout 部署形态变化**：与 [ADR-0008](0008-instrument-grid-geometry.md):58 一致，`l2_synth_domain_sugon` 等不再自包含，sugon 部署须连同 `src/` 树同步并把 `src/` 加入 `sys.path`。
- 新术语「**冻结仪器调用 / FrozenInstrumentCommand**」入 `CONTEXT.md`。
- 若需改 interface 形状、TTA 口径、白名单机制、收编映射、同步策略或验收口径：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
