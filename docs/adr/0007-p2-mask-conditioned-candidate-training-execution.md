# ADR-0007：P2 掩码→影像候选训练配方执行决策（ControlNet-only 旁路与回切 Dice）

- **状态**：已接受（2026-08-23；在任何 P2 训练 epoch 启动之前预注册）
- **范围**：#59 P2 掩码→影像候选训练中，spec #51 decision 7 留给执行期的自由度：ControlNet 初始化、开发集轻验收口径、条件回切 Dice 度量与预记录早停规则

## 背景

[Spec #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) decision 7 把 P2 配方钉死在：ControlNet-only、DM/VAE 全冻、从冻结 P1-DM encoder/mid 初始化、纯 BraTS 无 MR-RATE 回放、`lr=1e-5`、batch=1、≤100 epochs、早停、AdamW、PolynomialLR power 2.0、默认 L1、`cache_rate=0`、`weighted_loss=100` 于 `129/130/131`、`use_region_contrasive_loss=off`、checkpoint 必须由可审计的 dev 轻验收选择。这批固定值之外的执行自由度——ControlNet 初始化细节、开发集轻验收口径、回切 Dice 的度量化与早停规则——由本 ADR 在训练启动前注册。

P2 的 DM 来源是 [ADR-0006](docs/adr/0006-p1-final-acceptance-and-dm-freeze.md) 的 `dm_source.json` 账本：只有通过完整终验（L1∧L2∧L3）并被注册的 P1 候选才允许作为 P2 旁路的上游。本 ADR 假定该账本已注册（见「前置」一节）。

## 决定

1. **ControlNet 初始化**：从冻结 P1-DM 的 encoder/mid 拷贝权重（`monai.networks.utils.copy_model_state(controlnet, unet.state_dict())`），后接 8 通道 conditioning embedding（`conditioning_embedding_in_channels=8`，即 `binarize_labels(combined)+bits=8` 产生的 8-bit 掩码条件）。不沿用任何既有 P2/P3 ControlNet 权重（每次候选独立初始化，杜绝掩码语义跨候选择性迁移）。
2. **条件词表**（spec #51 decision 5 / `brats_phase_prep.py`）：`combined` 掩码 = brain=22（四模态 `>0` 并集）叠 1/2/3→129/130/131；`weighted_loss_label=[129,130,131]` 只作用肿瘤亚区；不使用 401/402/403、不加入 CT-body 或脑分区 pseudo-label。条件经 `binarize_labels(bits=8)` 进入 ControlNet。
3. **训练 list 与 fold**：使用 #52 产物 `lists/p2_mask_cond.json`（train 侧 fold=1、dev 侧 fold=0 同存一清单）。`prepare_maisi_controlnet_json_dataloader` 以 `fold=0` 切分：fold==0（dev）→ val、fold!=0（train）→ train。训练只用 train_loader；**val_loader 丢弃**，绝不按训练/验证 loss 选 best（spec 明令禁止「丢弃 val loader 按训练 loss 选 best」旧行为）——候选由 dev 轻验收侧车 `select`。
4. **开发集轻验收**（`scripts/brats_p2_dev_eval.py`，独立 GPU 侧车，每 5 epoch）：
   - 固定 16 例 dev 队列（同 [ADR-0005](docs/adr/0005-p1-candidate-training-execution.md) 配额与排序），固定 per-(case, modality) 种子、cfg=10、30 步、逐病例伴侣 spacing——每评估点 64 个固定四模态样例；以 `ldm_conditional_sample_one_image_from_mask` 采样，条件为 `combined` 掩码；
   - FID 趋势 = 逐目标模态 2.5D RadImageNet FID（三正交面逐面 + 均值，预处理同 L1 字串），真实参照侧 = dev 真实体一次构建缓存——与 P1 一致；
   - L2 趋势 = 冻结仪器（ADR-0003 链、SSA 专属 plans）对生成四模态的 WT/TC/ET 体积中位数与 input/run/hier 失败计数——只记趋势，不做 TOST；
   - **条件回切 Dice 趋势**（P2 专属）：对每个 dev 病例，把仪器从生成四模态推出的肿瘤 mask 与 `combined` 的肿瘤亚区（129/130/131 或按 WT/TC/ET 从条件导出）比对，记每选区 Dice 的中位数与失败计数；该趋势与 FID 一并供候选选择，不做通过线（通过线在 L2 正式报告，`nnunet_l2_final_acceptance` 的 round_trip）。
5. **预记录早停与选择规则**（同 ADR-0005 规则文本）：`m(N)` = 四目标模态平面均值 FID 的均值；`N ≥ 30` 且连续 3 个评估点无新最优即停（写 `<ckpt_dir>/.early_stop`）；候选 = 全程 `argmin m`；硬上限 = 100 epochs。规则文本先于训练写入 run 目录，dev 证据只含 dev 侧 (sub, case)。
6. **scale_factor 与精度**：训练 `scale_factor` 恒取冻结 P1-DM ckpt 值（从 `trained_diffusion_path` 载入），不予重算；DCU 上 bf16 autocast、fp32 兜底保留；DDP 经 torchrun/RCCL（RCCL 映射原 `nccl` 名称）。
7. **DM 冻结边界**：训练仅 `controlnet.requires_grad=True`；`unet.requires_grad=False`，`unet.eval()`。VAE 不在训练图中（P2 训练仅对 latent 做去噪，无解码）。

## 前置（硬阻塞）

- 运行契约 `init`（P2）要求 `--upstream-run` 指向**已冻结且已注册 DM source** 的 P1 候选（`DmSourceLedger.check_upstream` 校验 run_id + checkpoint sha256 双匹配）；`dm_source.json` 未注册时 init 直接 `ContractViolationError("no P1 candidate has passed final acceptance yet; ... conclude a passing P1 run first")`。
- 因此 P2 训练启动前，P1 候选必须已通过完整终验并被 `conclude` 注册（生成 l1/l2/l3 报告、attach、conclude、写 `dm_source.json`）。这不是本 ADR 的执行项，属 #58 的终验落地；缺失则 P2 只能停在代码与准备阶段，不能 `init`。

## 后果

- `scripts/brats_p2_finetune.py`、`scripts/brats_p2_dev_eval.py`、`scripts/brats_p2_launch_train.sh`、`configs/config_brats_p2_train.json` 按本 ADR 实现；selftest 各自可独立复跑；
- P2 ControlNet 权重、生成样本、逐病例回切 Dice 与完整 subject manifest 全部留在受控存储（sugon/gauss private_data），入 Git 的只有脚本、配置与不含 subject/study ID 的规则；
- 若需改 controlnet 初始化、早停规则、回切 Dice 口径或选择规则：新 ADR 取代，既有候选与新候选不得混记为同一实验结论。
