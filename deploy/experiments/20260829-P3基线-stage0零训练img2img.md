# P3 stage-0 零训练 img2img 基线

> **记录性质与采集时间**:本文是 P3 stage-0 零训练 img2img 基线(关联 #60,已 CLOSED)的**读数落盘**,采集于 2026-08-29 午间(约 12:05–12:45 CST,服务器 `crdnotebook-2086462626317500417-wang9691-94660`,全程 SSH 只读)。实验实际运行窗口为 2026-08-25 ~ 2026-08-26,采集时产物已完结且完整(见 §3);本文只收已有读数、未现场扩测。

## 1. 目的与关联 issue

交付可重复、可审计的 P3 阶段 0(stage-0)零训练 img2img baseline:以冻结 P1-DM、src latent 与 tgt 模态标签执行 RF 插值起点的 img2img 推理,作为跨模态候选在 11 个信息充分方向上的统一比较下限。dev 侧 1 病例×12 对、holdout 侧 530 病例全量产出;输出(samples.json/pairs.json)可直接进入 L1 `brats-l1-pairs/1` 三元组的 baseline 侧与 L2 四锚轮参照(docs/evaluation.md「P3 also requires a `brats-l1-pairs/1` manifest」条)。

- 关联 issue:#60(P3 零训练 img2img 基线,CLOSED——本记录为其交付物的读数落盘)
- 上游:#51(P3 分阶段配方父 spec)、#58(P1 完整验收与冻结——提供被钉的 P1-DM checkpoint)
- 下游:#61 / #193(P3 跨模态候选——L1 配对判定的 candidate 侧,训练中断未就绪)
- 父任务:#186(实验记录落盘:曙光 P1/P2/P3 现役实验读数收编);本记录 #192
- 定位约束(stage-0 契约,CONTEXT.md「阶段 0 img2img 基线」词条):零训练、selection 钉 upstream P1-DM checkpoint、**不挂 L1/L2/L3 正式报告、不可 conclude 终验**,只作 P3 候选 L1 paired MAE/SSIM 的 baseline 侧与 L2 四锚轮参照,绝不冒充经 ControlNet 训练或终验通过的候选(`variant=stage0-baseline` 冻结标记)。

## 2. 运行配置与 run_id

| 项 | 值 | 出处 |
|---|---|---|
| run_id | `p3-20260825T151310Z`(dev 与 holdout 两侧一致) | `stage0_dev/pairs.json`、`stage0_holdout/pairs.json` 元数据头 |
| 契约 schema / variant | `brats-p3-stage0-pairs/1` / `stage0-baseline` | 同上 |
| DM checkpoint(冻结 upstream P1-DM) | `/root/private_data/brats2023_rflow_p1/ckpt/epoch_20.pt`(P1 selection = epoch 20,mean dev FID 3.398) | `logs/stage0_*.log` 加载行;P1 selection 见 `/root/private_data/brats2023_rflow_p1/selection.json` |
| dm_checkpoint_sha256 | `9377f8ba2c9d0ad4d2aaaac8874629dd1ba51f5e237231ea9ebed7bfde12946b` | 两侧 `pairs.json` 声明值;采集时对 checkpoint 文件实算 sha256 对拍一致(722,176,447 字节,2026-08-23 03:29 落盘) |
| Autoencoder(冻结) | `checkpoints models/autoencoder_v1.pt loaded`;scale_factor=0.9696779847145081 | `logs/stage0_dev.log`、`logs/stage0_holdout_shard_0.log` 加载行 |
| 推理协议(RF 插值起点 img2img) | `x_t=(1-t)·src_latent+t·noise`;四锚轮——每病例 t1n/t1c/t2w/t2f 各作一次 src 锚、其余三模态为目标,4×3=12 有序模态对;逐对独立 seed 记录于 pairs.json | CONTEXT.md stage-0 词条;docs/evaluation.md #60 条目 |
| 逐对采样参数 | strength=**0.9**、skip first 5 steps、t 888.0 → 0 over **25 steps**(dev 12 对与 holdout 全部 shard 日志逐对一致) | `logs/stage0_dev.log:7` 起逐对 INFO 行;`logs/stage0_holdout_shard_{0..7}.log` 同格式 |
| ControlNet | **无**(日志加载序列仅 VAE + DM;stage-0 定义即不训练、不挂 ControlNet) | `logs/stage0_*.log` 加载行;CONTEXT.md stage-0 词条 |
| 生成脚本 | `brats_p3_stage0_manifest.py` + `brats_p3_stage0_generate.py`(git 历史 PR #95;ADR-0014 记为「#38 img2img 链正式化重指到 P1-DM checkpoint」;服务器现存于 `code_discarded_20260829/scripts/`) | git 历史;服务器目录实测。终态入口为 `ctmr generate cross-modal generate baseline`(docs/evaluation.md) |
| 环境 | `environment_brats_p3_train.json`(P3 目录根,pip freeze 原件) | 目录实测 |
| 运行窗口 | dev:2026-08-25 23:39:53 → 23:40:42 CST(1 病例×12 对);holdout:8 shard 并行 23:44:58 → 08-26 00:53:05(末 shard),约 68 分钟 | `logs/stage0_dev.log`、`logs/stage0_holdout_shard_{0..7}.log` 时间戳(服务器时区 CST=UTC+8) |

run_id 时间戳折合 CST 2026-08-25 23:13:10,早于日志记录的生成开始(23:39:53);其精确语义(run 打开/契约冻结时刻)无独立留痕,此处如实记录。

## 3. 关键读数与产物路径

产物基目录:`/root/private_data/brats2023_rflow_p3/`(2026-08-29 采集,状态:**完整**)。

### 3.1 三层状态

stage-0 契约下不挂 L1/L2/L3 正式报告、不可 conclude(CONTEXT.md):

- **L1:未发生。** stage-0 输出是 P3 候选 L1 `brats-l1-pairs/1` 的 baseline 侧;candidate 侧训练中断于 39/100 epoch(#193 快照),配对判定(MAE ↓≥10%、SSIM ↑≥0.02、双 CI 向好,docs/evaluation.md)未发生。
- **L2:合成域适用性前置证据 PASS。** 「L2 合成域适用性·P3 基线」(关联 #38,转写记录见 `20260829-L2合成域适用性-P3基线.md`)判定五挑战 R_fail_synth 全 0、PASS——其被评的 img2img 生成链与 stage-0 同源(ADR-0014:stage-0 = #38 链正式化重指 P1-DM);stage-0 产物自身无 L2 终验,亦不 conclude。
- **L3:不适用**(基线不进 L3)。

### 3.2 12 有序模态对清单(出处:`stage0_dev/pairs.json` 12 条 records 全覆盖;dev 病例 BraTS-GLI-00267-000,GLI)

| # | src→tgt | seed(dev) |
|---|---------|-----------|
| 1 | t1n→t1c | 435825598 |
| 2 | t1n→t2w | 174791263 |
| 3 | t1n→t2f | 1049770063 |
| 4 | t1c→t1n | 935931288 |
| 5 | t1c→t2w | 1409261790 |
| 6 | t1c→t2f | 1913760240 |
| 7 | t2w→t1n | 103854198 |
| 8 | t2w→t1c | 78475182 |
| 9 | t2w→t2f | 1985389361 |
| 10 | t2f→t1n | 2126542654 |
| 11 | t2f→t1c | 289165370 |
| 12 | t2f→t2w | 1362160256 |

holdout 侧逐病例逐对 seed 见 `stage0_holdout/pairs.json`(6360 条 records);两侧 distinct (src, tgt) 组合均为上表 12 对。

### 3.3 产物完备性(采集时逐项实测)

| 侧 | 计划 | 实测 | 缺失/空文件 | 体积 |
|---|---|---|---|---|
| dev | 1 病例×12 对=12 | 12 个生成 NIfTI + reference_grid 4 个(4 模态) | 0 / 0 | 42 MB |
| holdout | 530 病例×12 对=6360 | 6360 个生成 NIfTI + reference_grid 2120 个(=530×4) | 0 / 0(pairs.json 6360 条逐一存在对拍通过) | 22 GB |

- holdout 病例按 challenge 分布:GLI 250 / MEN 200 / METS 48 / PED 20 / SSA 12(=生成体 3000/2400/576/240/144 ÷ 12;出处:`stage0_holdout/pairs.json`)。
- 8 shard 覆盖:67+67+66×6=530 病例(出处:`samples_shard_{0..7}.json` 各含 1 条 `wrote` 收尾日志)。
- 契约一致性:两侧 run_id、variant、schema、dm_checkpoint_sha256 完全一致。

### 3.4 产物路径(sugon)

- dev:`/root/private_data/brats2023_rflow_p3/stage0_dev/`(`generated/GLI/<case>/at<src>/<tgt>_seed<seed>.nii.gz`、`reference_grid/GLI/<case>/<modality>.nii.gz`、`pairs.json`、`samples.json`)
- holdout:`/root/private_data/brats2023_rflow_p3/stage0_holdout/`(同构,另含逐 shard 的 `pairs_shard_{0..7}.json` / `samples_shard_{0..7}.json`)
- 日志:`/root/private_data/brats2023_rflow_p3/logs/stage0_dev.log`、`logs/stage0_holdout_shard_{0..7}.log`(共 9 份,逐对协议读数留痕)
- 附属:`selftest_contract_606/`、`selftest_manifest_626/` 为 #60 工具链自检 fixture,非实验读数

## 4. 结论与后续动作

**结论(草拟待复核)**:stage-0 零训练 img2img 基线交付完整、可审计——dev+holdout 共 6372 个生成体全部落盘,0 缺失 0 空文件;两侧契约元数据(run_id/schema/variant/checkpoint sha256)一致;12 有序模态对全覆盖;推理协议统一为 strength=0.9、25 步(RF 插值起点),逐对留痕于 9 份日志。#60 三条验收在产物层面可核:官方推理配置与 provenance(§2 表)、四锚轮 12 对清单与阶段运行契约(§3.2/§3.3)、baseline 不冒充候选(`variant=stage0-baseline` 显式标记、无 ControlNet、零训练)。

**与 CLOSED issue 状态的偏差**:#60 已 CLOSED 且其 AC 达成,本记录不翻案;但其下游消费未发生——L1 paired 判定需 candidate 侧就绪,而候选训练中断(#193)。#60 关闭 ≠ P3 L1 验收完成;P3 的 L1/L2/L3 终验整体未跑(与 #193 偏差清单一致)。

**后续动作**:

1. 候选侧就绪后(#193 后续决策),assemble `brats-l1-pairs/1`,以 stage-0 为 baseline 侧执行 L1 paired MAE/SSIM 正式判定(docs/evaluation.md 阈值:MAE ↓≥10%、SSIM ↑≥0.02、双 CI 向好;t1n→t1c 走 known-unobservable 显式例外);草拟待复核。
2. 候选判定不得因相对基线占优而免于终验,候选 selection 钉自己的 ControlNet checkpoint 而非 upstream P1-DM(CONTEXT.md 防混淆条款);草拟待复核。
3. P3 实验若推进,以**新增日期报告**承接,不改写本记录。
4. holdout 侧 22 GB 产物的归档/清理决策留集群运维(ADR-0015 运维面职责边界)。
