# ADR-0001：SSA L2 仪器采用派生 Batch-16 Plans

- **状态**：已接受（2026-08-19）
- **范围**：仅 `Dataset502_BraTS2023SSA` 的 L2 肿瘤测量仪器 `fold_0`

## 背景

SSA 的外部 nnunetv2 2.8.1 默认 `3d_fullres` plans 在当前数据上产生 global `batch_size=2`。nnU-Net 的 DDP 要求 global batch 不小于 world size；8 张 DCU 因而不能使用该默认 configuration 直接训练。

本次决定不改变病例级 70/10/20 split、SSA fold_0 的 34 train / 8 val 病例归属、raw NIfTI 输入、四通道顺序、patch、spacing、网络、normalization、preprocessor、learning rate、250 epochs 或每 epoch 250 iterations。它只修订 SSA 的训练 batch 协议。

另：DTK 26.04 / torch-dcu 2.9.0 / Triton 3.5.1 组合在大 patch 的 Conv3d backward 上已复现 `torch.compile` codegen 失败；本仪器训练不得启用 torch.compile。

## 决定

1. 原始 `nnUNetPlans.json` 是不可变基线，并先记录 SHA-256。
2. 为 SSA 生成独立 `nnUNetPlans_SSA_bs16_v1.json`，其顶层 identifier 为 `nnUNetPlans_SSA_bs16_v1`。
3. 派生 plans 仅新增 configuration：

   ```json
   "3d_fullres_bs16": {
     "inherits_from": "3d_fullres",
     "batch_size": 16
   }
   ```

   解析后必须与 `3d_fullres` 共享原 `data_identifier`、patch、spacing、network、normalization 与 preprocessor。
4. 训练固定使用 8 张 DCU、global batch 16、local batch 2；这是 SSA 专属 custom-plan 仪器，不得描述为“全默认 nnU-Net”。
5. 训练入口固定为 MONAI `nnUNetV2Runner.train_single_model`，传入 `config="3d_fullres_bs16"`、`fold=0`、8-card `gpu_id` tuple 和 `p="nnUNetPlans_SSA_bs16_v1"`。
6. 环境固定 `nnUNet_compile=f`。这只关闭已知不稳定的 JIT 包装；不改变模型或训练配方的其余字段。

## 审计与校准纪律

受控持久化目录必须记录原始/派生 plans、fingerprint、dataset.json、splits_final.json、训练命令、环境摘要、DDP preflight、checkpoint 与日志的 hash。逐病例 subject ID、原始数据、权重与校准原始产物不得提交到 Git 或公开上传。

校准开始后，派生 plan、batch、checkpoint 和推理配置均冻结。不得依据校准结果调整 batch、plans、learning rate、patch、preprocessor、epoch 或 iteration。仅执行缺陷或明确版本升级允许重训；每次重训都产生新的版本标识、hash 与校准记录。

## 后果

- SSA 可以在 8-card DDP 上使用每卡 2 个样本训练。
- 这不是其他四个子挑战的自动先例；GLI、MEN、METS、PED 需要各自独立的 plans / batch 决定和审计记录。
- 由于只改 batch，既有 fingerprint 与 preprocessing 可复用；若后续改 spacing、patch 或 preprocessor，则必须生成新 data identifier 并重新预处理。
