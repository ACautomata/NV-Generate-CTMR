# ADR-0003：L2 仪器冻结工件清单与分发边界

- **状态**：已接受（2026-08-20；冻结自本 ADR 起生效，终验期间不再更新仪器权重）
- **范围**：五个 L2 肿瘤测量仪器（MONAI nnU-Net fold_0）的全部可复现工件终态冻结与合规审计

## 背景

[自训 MONAI nnU-Net L2 测量仪器的构建与校准闸门钉板](https://github.com/ACautomata/NV-Generate-CTMR/issues/32) §4 要求仪器定线前冻结全部工件；[L2 仪器训练](https://github.com/ACautomata/NV-Generate-CTMR/issues/35) 产出五子挑战 fold_0 checkpoint，[L2 仪器校准与误差包络](https://github.com/ACautomata/NV-Generate-CTMR/issues/36) 完成校准并将数值冻结于 [ADR-0002](0002-l2-instrument-calibration-envelopes.md)。本 ADR 依据 [L2 仪器冻结工件审计打包](https://github.com/ACautomata/NV-Generate-CTMR/issues/37) 记录终态冻结清单、全量 hash 核验结果与分发边界。

## 决定

### 1. 版本锁（训练与校准共同环境）

| 组件 | 版本 / 标识 | 冻结依据 |
|---|---|---|
| MONAI | 1.6.0（commit `eccefc57`） | wheel SHA-256 `8880fb294827448a15299f9313b96c057d87179a4efe7ae0c80085f21ec494ea`，已与 PyPI 官方元数据核对 |
| nnunetv2 | 2.8.1（commit `468cf80`） | wheel SHA-256 `58aeb5719c9ee7bdac4b420106940cf3b21ea08a28a494802b0921ca14fde63f`，已与 PyPI 官方元数据核对 |
| torch（DCU） | 2.9.0+das.opt1.dtk2604 | — |
| triton（DCU） | 3.5.1+das.opt1.dtk2604.torch290 | — |
| DTK | 26.04（`/opt/dtk-26.04`），HIP 6.3.26093 | — |
| 容器 | overlay 镜像实例 `40724b31-8edc-4803-938c-2b4031e56211`（registry digest 容器内不可观测，以实例 UUID 代替） | — |
| 平台 | Ubuntu 22.04.5 / kernel 5.15.0-25 / Python 3.11.9 / 8× DCU gfx936 | — |

版本锁原始记录：受控 `l2-instrument-audit/be683ee…/{version-lock.json, trainer-install.json}`（hash 锚见 §6）。

### 2. 训练工件冻结清单（五子挑战 fold_0）

训练协议：`nnUNetTrainer250Epochs`（250 epochs × 250 iterations，共 62,500 optimizer steps），GLI/MEN/METS/PED 用全默认 `nnUNetPlans__3d_fullres`；SSA 按 [ADR-0001](0001-l2-nnunet-ssa-derived-batch16.md) 用派生 `nnUNetPlans_SSA_bs16_v1__3d_fullres_bs16`（batch-only delta）。

| 挑战 | 数据集 | fold_0 train / val | checkpoint_final SHA-256 |
|---|---|---|---|
| GLI | Dataset501_BraTS2023GLI | 701 / 175 | `664faa638cf94d6667a00635c0bdab0573f3e6c24aa3f07fc3b8a2964bcf21ff` |
| SSA | Dataset502_BraTS2023SSA | 34 / 8 | `d8119534805d0bfc5fd1c50cb0a0262b9ccee0ec9900744a59add669e73de21b` |
| MEN | Dataset503_BraTS2023MEN | 560 / 140 | `310f9af2465e008eb6ee26b969cf47c36502b17d19d8e15e052da1706e0d30a3` |
| METS | Dataset504_BraTS2023METS | 133 / 33 | `aa69db5cd99b31f95454cd1b5e38ae70db9a171cce45ff5b1c11ed5f01961ba2` |
| PED | Dataset505_BraTS2023PED | 54 / 14 | `f967af629331de345937ce8284424d967c36b0b5d32fc83143aa3bc7c08ca6a3` |

| 挑战 | dataset_fingerprint | plans（source / SSA derived） | dataset.json | splits_final.json |
|---|---|---|---|---|
| GLI | `92a9ac2c6e2297391f59fd6046d438cb9114a3b239926b2f6de3fd5edecf4b64` | `e3d98cc94551e44142ad59b76d5cda99e0cac83339a21757cd5e5942e0193e85` | `74b7ae9882c99242c2558688ef5f1149b5096cfc1d7db284f914864424b484ba` | `36967a710549bcd3fcfb63e8ae703664c23e2dffc070255017154abb84b4ebd0` |
| SSA | `c2070fc080b928946b6acd12ca0930354f94c40426c0b5cfdf3ebcb459f04458` | `78fc63e197c9134530e55fb43943a0ac954eda000f847222cc98394bf1a27083` / `be476f7fb6e5773f1d085c5f149c281ffdb58560cd175793d1b8312c91136e63` | `6685db78ad5cb87399f88c1b306d4dd6ba3931afe4572217395bb6a053bec62c` | `29e4b92d397b08342fa2b72743da3b2fec0380bb3a23bd16bdb2ba6d916989ed` |
| MEN | `f1e8b56dc2644a979c8f25d50d12cdca1d6265259086e762dbc6c44efaef2fcf` | `1dcd75da0c17ebd7c2594dd3beeecaa5e5d0fda81a9bd23d15d98353fcdab32` | `5ff33f3d4aaffd9580fe5dedf4b33ec5eef2972238b41f2ddf766fd9c9c22939` | `8208c2d5399112d0efa33bf149f1b2219be9c15b292ad965cf6c9709b81c2813` |
| METS | `ea07114e7caa23d6a9ce419cf229c79103f6339e4013b35daf1ad9d2a9afa875` | `3e4726b76bf9b8b248d80382b85d9db708b8bcb9079edb02a62992f237d40304` | `0b51ff4880f0fbec952eb9b8389faeb62112c4c0466b03a15ab350d0ec7bf6dd` | `451480046fd980a0590419c2253081a8fee1f87f224a0a4d47d724d7edfd689e` |
| PED | `c854bebc26601360cc4e5a1f4a3e1b611cb7969ce2a85ac03859750a8f8abc41` | `44bfe0bdca27562bd261f703431b0f83b0c58b286d6c2996d2d6321e759b5f98` | `45a33dc10d4772cc680749dfe8394899d1731ea1617336ec16407c2334e8a1e3` | `20a271ad847d09d1cd6127e6c786cabdfb2732900f3f44d92788e95c57a949fd` |

（trainer 源码 `nnUNetTrainer250Epochs.py` 五挑战同一 SHA-256 `1a99c254c87a7249ea8b7907b1ada36aa4077ec9ebcc755ffb81236508c596a0`，upstream 基类 `7096efb2040135eb60df3c8bf39cdbcc373e299671fcee365b3aff2f4da4dfbd`。）

### 3. 输入契约

- **通道**：五挑战统一 t1n/t1c/t2w/t2f → `_0000.._0003`；
- **影像**：BraTS2023 原生 NIfTI，1mm、SRI24、240×240×155、skull-stripped，无强度预归一化（仓内管线自动处理）；
- **标签语义（1/2/3，background=0）按瘤种分组**：GLI = NCR/ED/ET；SSA·MEN·METS = NETC/SNFH/ET；PED = NC/ED/ET（各自 dataset.json 随 §2 hash 冻结）；
- **fold_0 划分**：与 [L2 仪器数据准备](https://github.com/ACautomata/NV-Generate-CTMR/issues/34) 的 splits_final.json 一致（§2 hash），20% 终验持出集零接触。

### 4. 推理配置

nnUNetv2 2.8.1 原生 `predict_from_raw_data` 默认值：**镜像 TTA on、滑窗 overlap 0.5**；`nnUNetTrainer250Epochs`；plans 按 §2（SSA 走派生 plans）。详见 [ADR-0002](0002-l2-instrument-calibration-envelopes.md) 决定 2——校准即按此配置执行，配置与校准数值一体冻结。

### 5. 校准产物汇入

预注册误差包络数值表（五子挑战 D_r,low / E_r,vol / E_r,centroid / E_r,HD95 / R_fail / R_miss / 重复性）冻结于 [ADR-0002](0002-l2-instrument-calibration-envelopes.md)；协议 + 病例清单 + 全部输入/GT 的 3,181 条 SHA-256 冻结于受控 `l2-instrument-calibration/252940d0…/protocol/`。逐病例 metrics（含 subject ID）只存受控目录，不入库。

### 6. 终态全量核验（#37 冻结审计）

2026-08-20 独立重算全部冻结工件（脚本 `scripts/nnunet_l2_freeze_audit.py`，repo commit `796b5caa`）：

- **训练侧**：五挑战各 14 项（SSA 15 项，含派生 plans）——raw 契约/计数、channels、fingerprint、plans、trainer 源码、checkpoint_final、训练日志 hash 与 Epoch 249 覆盖——全部通过；
- **校准侧**：`protocol/SHA256SUMS` 3,181 条全量重算，零 mismatch、零缺失；
- **审计锚点**：15 份审计 JSON + version-lock + trainer-install + SSA plans-variant-audit 共 18 个文件的 SHA-256 锚点；
- verdict：受控 `l2-instrument-freeze-audit/796b5caa…/freeze-audit/freeze-audit.json`，SHA-256 `9121e8ac73f2bdb3999f83c10666c387ccd4c488b3e71b24cf7ab208353e0f82`，`all_passed: true`。

### 7. 合规检查

- **公开仓库**：git 跟踪文件无 `.pth`/`.nii(.gz)`/逐病例 CSV 等工件扩展；本地样例目录 `datasets/` 未被跟踪；`docs/`（含 ADR-0002）无 subject ID 明文；
- **存量说明（历史票产物，非仪器工件，如实记录）**：`prototype/` 冒烟脚本含 12 例公开样例数据的 subject ID 引用（[BraTS 样例数据获取](https://github.com/ACautomata/NV-Generate-CTMR/issues/8) 知情决定：样例合规存疑、仅本地冒烟、数据不入库）；`scripts/brats2023_nnunet_prep.py` 含 PED-00024 排除清单（#34 数据质量决定）。二者为样例引用/排除理由记录，非仪器权重或校准逐病例数据，维持原状；
- **Hugging Face**：`ACAutomata` 名下 models / datasets / spaces 全空，无组织——无仪器权重或校准产物泄漏。

### 8. 冻结语义与分发边界

- **冻结**：自本 ADR 起至终验结束，不再更新仪器权重（checkpoint、plans、推理配置、校准包络均不可动）；重训仅限 [ADR-0002](0002-l2-instrument-calibration-envelopes.md) 后果节所引 #32 §5 的两类触发器；
- **分发**：自训仪器权重、预处理产物、校准逐病例数据（含 subject ID）**永不公开再分发**（BraTS CC BY-NC 4.0 + Synapse DUA）；仓库与 ADR 只落元数据（配置、版本、摘要统计、hash）；
- **义务**：对外报告/论文保留 BraTS 挑战的署名与引用义务；
- **真实性**：自训产物如实标注为自行训练的仪器，不冒充独立供应链现成资产。

## 后果

- L2/P2 定线的测量分辨率依据（ADR-0002）与仪器工件（本 ADR）均已冻结，[三层验收标准与持出集钉板](https://github.com/ACautomata/NV-Generate-CTMR/issues/13) 因仪器路线导致的 NO-GO 条件解除路径打通（余 [L2 仪器合成域适用性评估](https://github.com/ACautomata/NV-Generate-CTMR/issues/38)）；
- 任何触发重训的事件将使本 ADR 与 ADR-0002 同时失效，须重走训练→校准→冻结全链并出新 ADR；
- 审计可在任意时刻用同脚本复验（只读、拒绝覆盖 verdict）。
