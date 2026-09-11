# BraTS 微调采用 1mm 各向同性的目标几何

BraTS 2023 GLI 与 MR-RATE 共用「每轴 round-to-128 + 三线性 resize」的编码管线，但微调与生成统一改用 `dim [256, 256, 128]` + `spacing [1, 1, 1]`：平面 pad 8 层/边（240→256，实测落在空白内，无损），z 向裁掉底部 13 层与顶部 14 层（155→128）。原因是 FID 协议已把评测域定在 1mm 各向同性、256×256×128、中心对齐（`scripts/compute_fid_2-5d_ct.py` 的 `Spacingd(1mm)` → `CenterSpatialCropd(256×256×128)` → `SpatialPadd`），训练域对齐它才能让训练 = 生成 = 评测三处自洽；且目视确认该条件的生成图可用。MR-RATE replay 保持各 volume 的 native 几何不变。

## Status

accepted

## Considered Options

- **维持现状**（240×240×155 直接 resize 到 `[0.9375, 0.9375, 1.2109]`）：FOV 覆盖完整、与 MR-RATE 同管线，但训练域与评测域不一致，且生成图目视不可用。
- **只 pad 平面、z 保持 resize**（`[1.0, 1.0, 1.2109]`）：零解剖损失、FOV_z 保住 155mm，但 z 轴仍与评测域/生成条件不一致。
- **z 向 pad 到 256**（`dim_z=256`，零裁剪零插值）：latent 变 `4×64×64×64`（2× 显存与步时），且 101 层空画布偏离训练分布。

## Consequences

- 每个 case 丢掉底部 13 层与顶部 14 层。实测（1251 例抽 100 例）98 例会切到脑组织——多数 case 的脑组织直接顶到 z=0，被排除的是小脑/脑干下部与顶部区域；此为接受的代价，不以 FOV 换清晰度。
- z 向 FOV 降到 128mm，低于 MR-RATE 的准入下限 140mm，属条件外推；其实际表现以生成样例验收。
- BraTS 已编码的 latent（`[0.9375, 0.9375, 1.2109]` 版本）全部作废，需按新几何重编码。
- T8 冻结的 pretrained 基线生成条件（`[0.94, 0.94, 1.36]`）与 finetuned 臂不再一致，必须按 `[1, 1, 1]` 重新冻结；BRATS real-real 地板（4 条）不受影响。
- 上游 `configs/config_maisi_diff_model_rflow-mr-brain.json` 及所有非 BraTS 链路保持不动。
- MR-RATE replay latent 与 T6 冻结的遗忘参照集不受影响。
