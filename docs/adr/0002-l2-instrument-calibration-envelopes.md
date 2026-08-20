# ADR-0002：L2 仪器预注册误差包络（五子挑战）

- **状态**：已接受（2026-08-20；数值自校准集算出即冻结，不得回看调整）
- **范围**：五个 L2 肿瘤测量仪器（#35 fold_0 checkpoint）在真实数据上的校准误差包络

## 背景

[自训 MONAI nnU-Net L2 测量仪器的构建与校准闸门钉板](https://github.com/ACautomata/NV-Generate-CTMR/issues/32) §5 预注册了误差包络的统计定义；[L2 仪器训练](https://github.com/ACautomata/NV-Generate-CTMR/issues/35) 已产出五子挑战 fold_0 checkpoint。本 ADR 依据 [L2 仪器校准与误差包络](https://github.com/ACautomata/NV-Generate-CTMR/issues/36) 记录校准执行与数值，作为 L2/P2 定线的测量分辨率依据。

## 决定

1. **协议先行冻结**：校准按 `docs/calibration/l2-instrument-calibration-protocol.md`（预注册协议）执行。协议、校准集清单（635 例 = dev ∪ fold_0 val，逐病例标记来源）与全部输入/GT 的 SHA-256（3,181 条）在**任何校准推理之前**冻结于受控目录 `l2-instrument-calibration/<repo-commit>/protocol/`。
2. **推理配置冻结**（#32 §4）：nnUNetv2 2.8.1 原生 `predict_from_raw_data`，默认镜像 TTA on、滑窗 overlap 0.5；`nnUNetTrainer250Epochs`；GLI/MEN/METS/PED 用 `nnUNetPlans__3d_fullres`，SSA 用 ADR-0001 的 `nnUNetPlans_SSA_bs16_v1__3d_fullres_bs16`；每病例独立推理 3 次（重复性）。
3. **数值表冻结**：下列数值自校准集算出即冻结（bootstrap B=10,000、seed=20260820+挑战序号、percentile 法；协议 §6）。它们是 L2 等价 margin 与 P2 回切下限的测量分辨率依据（margin 公式沿用 [nnU-Net BraTS 肿瘤测量仪器选型与校准调研](https://github.com/ACautomata/NV-Generate-CTMR/issues/22) findings §6.C）。

## 预注册误差包络数值

### GLI（n=300：dev 125 ∪ fold_val 175；观测 = 病例 × rep = 900）

| 区域 | D_r,low (Dice) | E_r,vol | E_r,centroid (mm) | E_r,HD95 (mm) | Dice 中位数 |
|---|---|---|---|---|---|
| WT | 0.8053 | 0.2802 | 5.38 | 21.66 | 0.9606 |
| TC | 0.6819 | 0.4373 | 4.79 | 15.59 | 0.9659 |
| ET | 0.4093 | 0.5702 | 4.41 | 15.16 | 0.9347 |

R_fail = 0（0/900，Wilson 95% 上界 0.0043；input/run/hier 全 0）；R_miss = 3/900（0.33%，诊断量）；空 pred 排除（HD95/centroid 双侧非空子集）：WT 3、TC 9、ET 15；重复性 p95 全 0（三次推理逐病例 bit 级一致）。ET < 1 mL 分层：14 例（诊断性，见受控 summary）。

### SSA（n=14：dev 6 ∪ fold_val 8；观测 42）

| 区域 | D_r,low (Dice) | E_r,vol | E_r,centroid (mm) | E_r,HD95 (mm) | Dice 中位数 |
|---|---|---|---|---|---|
| WT | 0.7046 | 0.7223 | 12.36 | 66.80 | 0.9680 |
| TC | 0.6578 | 0.7767 | 8.64 | 57.25 | 0.9041 |
| ET | 0.7111 | 0.7786 | 8.42 | 51.87 | 0.8965 |

R_fail = 0（0/42，Wilson 95% 上界 0.0838）；R_miss = 0/42；空 pred 排除：全 0；重复性 p95 全 0。ET < 1 mL 分层：0 例。n=14 的宽包络即 SSA 仪器的真实测量分辨率，如实保留不收窄（#32 §2）。

### MEN（n=240：dev 100 ∪ fold_val 140；观测 720）

| 区域 | D_r,low (Dice) | E_r,vol | E_r,centroid (mm) | E_r,HD95 (mm) | Dice 中位数 |
|---|---|---|---|---|---|
| WT | 0.7562 | 0.3235 | 3.92 | 58.17 | 0.9628 |
| TC | 0.7208 | 0.3576 | 6.17 | 58.59 | 0.9693 |
| ET | 0.7501 | 0.3367 | 5.70 | 58.72 | 0.9685 |

R_fail = 0（0/720，Wilson 95% 上界 0.0053）；R_miss = 6/720（0.83%）；空 pred 排除：三区域各 6；重复性 p95 全 0。ET < 1 mL 分层：32 例。

### METS（n=57：dev 24 ∪ fold_val 33；观测 171）

| 区域 | D_r,low (Dice) | E_r,vol | E_r,centroid (mm) | E_r,HD95 (mm) | Dice 中位数 |
|---|---|---|---|---|---|
| WT | 0.0000 | 1.6510 | 28.58 | 93.63 | 0.8632 |
| TC | 0.0000 | 1.0000 | 35.08 | 95.44 | 0.8760 |
| ET | 0.0000 | 1.0000 | 35.08 | 95.22 | 0.8043 |

R_fail = 0（0/171，Wilson 95% 上界 0.0220）；R_miss = 3/171（1.75%）；空 pred 排除：三区域各 3；重复性 p95 全 0。ET < 1 mL 分层：10 例。

**宽包络说明（如实记录，不收窄）**：METS 为多灶小病灶子挑战，≥5% 观测的 Dice≈0（小病灶漏检/极小肿瘤的 Dice 崩塌），把 5th-percentile 的 bootstrap 下界拖至 0；E_r,vol 上界 1.00–1.65（GT 极小时的相对体积误差可超 1）。中位 Dice（0.80–0.88）显示仪器在典型 METS 病例上可用，但按预注册定义，**METS 的 P2 回切 Dice 下限不高于 0（即不可设 Dice 下限线）、L2 等价 margin 至少覆盖 E_r,vol ≈ 1.0–1.65 + E_r,centroid**——等价检验在 METS 上区分力极弱，这是该仪器在 METS 上的真实测量分辨率。

### PED（n=24：dev 10 ∪ fold_val 14；观测 72）

| 区域 | D_r,low (Dice) | E_r,vol | E_r,centroid (mm) | E_r,HD95 (mm) | Dice 中位数 |
|---|---|---|---|---|---|
| WT | 0.0093 | 0.9946 | 17.87 | 28.09 | 0.9094 |
| TC | 0.0105 | 0.9939 | 18.33 | 28.34 | 0.9038 |
| ET | 0.0000 | 1.0000 | 22.65 | 58.02 | 0.6944 |

R_fail = 0（0/72，Wilson 95% 上界 0.0507）；R_miss = 0/72；空 pred 排除：WT/TC 0、ET 18（25%——PED 小 ET/囊变病例的 ET 空 pred，如实进入 Dice=0 与 E_r,vol=1.0 分布）；重复性 p95 全 0。ET < 1 mL 分层：6 例。

### 诊断性分层（ET < 1 mL；不设为通过线）

各子挑战 ET < 1 mL 病例数与分层包络见受控目录 `metrics/summary_<CH>.json` 的 `et_lt_1ml_stratum`；主判据用全分布（协议 §6）。

## 执行审计

- 受控目录：`sugon:/root/private_data/l2-instrument-calibration/252940d0156f4c1258936fa25a1fb28bad61ae22/`（协议+清单+hash 冻结、3× 推理输出、逐病例 CSV、bootstrap 汇总）；
- dev 病例原始数据经 gauss→sugon 受控通道同步（DUA：不入库、不公开再分发；subject ID 只存受控目录）；
- 环境：沿用 `/root/private_data/ENVIRONMENT_LOCK.md` 版本锁（nnunetv2 2.8.1、MONAI 1.6.0、torch-dcu 2.9.0+das.opt1.dtk2604、DTK 26.04）；执行期环境适配仅两处非语义改动（见 issue #36 resolution）。

## 后果

- L2/P2 可按 #32 §5 的 margin 公式设定最终通过线（L2 等价 margin ≥ E_r,vol + E_r,centroid + 重复性；P2 回切 Dice 下限 ≤ D_r,low）；
- 包络按子挑战独立适用，不跨子挑战合并或迁移；
- 禁止用生成样本反向调节任何阈值、禁止基于本表数值回头修改仪器（重训仅限 #32 §5 两类触发器）。
