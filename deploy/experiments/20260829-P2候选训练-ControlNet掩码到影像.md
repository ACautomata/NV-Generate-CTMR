# P2 候选训练(ControlNet 掩码→影像)

**候选已交付且 run 冻结(status frozen)**——配方逐项符合 #59 验收标准,dev 轻验收选 epoch_15(mean dev FID 4.5712),31 epochs 预登记规则早停,holdout 530 例×4 模态按挑战分目录生成完整(原样呈现,详见「关键读数与产物路径」)。

P2 候选 run `p2-20260824T111958Z` 的训练与候选选择记录。实验窗口:run 记录创建于 2026-08-24T11:19:59Z(run.json `created_utc`),run 冻结于 2026-08-24T22:17:35Z(run.json `frozen_utc`);holdout 生成至 2026-08-25 06:16(服务器本地时间 samples.json mtime);本记录落盘于 2026-08-29,读数经 SSH 只读采集自 sugon 产物。

## 目的与关联 issue

以 #59 的 P2 配方(冻结 P1-DM 与 VAE,ControlNet 从冻结 DM encoder/mid 初始化,纯 BraTS 训练、无 MR-RATE 回放)交付 mask-conditioned ControlNet 候选:同一肿瘤条件掩码(combined labels)生成四模态 BraTS 影像(t1n/t1c/t2w/t2f);dev 轻验收(mean dev FID,argmin)驱动候选选择,预登记早停规则(spec #51 decision 7)控制训练时长。本次落盘属父任务 #186(实验读数收编)。

- 关联 issue:#59(P2 掩码→影像候选,CLOSED)、#51(分阶段配方 spec)、#186(父任务)
- 三层状态:L1 特征已提(1 080 records)、**正式验收未跑(缺口)**;L2 终验已出 FAIL 判定,读数由同批记录另行收编(#191);L3 盲评包未建、无专家目检(父 #186 盘点:此为 CLOSED 状态偏差之一,见「结论与后续动作」)

## 运行配置与 run_id

- run_id:`p2-20260824T111958Z`(run.json `run_id`;created_utc 2026-08-24T11:19:59Z;status=「frozen」@22:17:35Z)
- 产物基目录:`/root/private_data/brats2023_rflow_p2/`;phase 数据根 `/root/private_data/brats2023_rflow_phase/`(manifest sha256 `9893c296d728b1f8…`,与 P1 记录同一 manifest)
- 训练脚本:`/root/nv-phase-59/scripts/brats_p2_finetune.py`(run.json `code_version.script_sha256` = `e303415309e82ddd…`;git_commit 未记录,null)
- 训练配置(`config_brats_p2_train.json`,run.json configs sha256 `d80ae9d6bb6a7385…`;与 `ckpt/train_provenance.json` hyperparameters 一致):`controlnet_train` = batch_size 1 / cache_rate 0 / **lr 1e-05** / n_epochs 100 / **weighted_loss 100 @ weighted_loss_label [129, 130, 131]** / use_region_contrasive_loss **false**(RCL 关闭)/ fold 0;推理 `diffusion_unet_inference` = num_inference_steps 30 / cfg_guidance_scale 10
- 配方守卫(训练日志原文):`P2 recipe guard OK: lr=1e-05 batch=1 weighted_loss=100@[129, 130, 131] RCL=off`(train_20260824T114605Z.log 行 46)
- 训练数据:`lists/p2_mask_cond.json` 8 464 entries,无回放(run.json data_lists sha256 `93a4914b058dde23…`;train_provenance `replay: null`;训练日志「p2 list: 8464 entries … (no replay)」)——纯 BraTS、无 MR-RATE 回放
- 初始化溯源(`records/dm_source.json`,schema `brats-dm-source/1`,registered_utc 2026-08-24T07:47:19Z):
  - 上游候选:P1 epoch_20 `/root/private_data/brats2023_rflow_p1/ckpt/epoch_20.pt`,**sha256 `9377f8ba2c9d0ad4d2aaaac8874629dd1ba51f5e237231ea9ebed7bfde12946b`**,来源 run `p1-20260822T131947Z`(run_record_sha256 `5877b8bb3c0e703d…`);run.json `upstream` 段同值且 status_at_init=「frozen」
  - 基座:`diff_unet_3d_rflow-mr-brain_v1.pt` sha256 `90c4a015879d4f2c…`
  - ControlNet 从冻结 DM encoder/mid 初始化、DM/VAE 全冻:trained_diffusion_path 指向 P1 epoch_20(train_provenance / environment json),VAE `autoencoder_v1.pt` 冻结未动(训练对象仅 ControlNet)
- 精度与规模(train_provenance.json):**amp_dtype bf16**,world_size **7**(7×DCU,torch 2.9.0),NCCL 2.22.3
- 早停规则(`dev_eval/early_stop_rule.json`,原文):metric = t1n/t1c/t2w/t2f 四模态 plane-mean dev 2.5D RadImageNet FID 之均值,固定 **16 例 dev cohort**(dev_cohort.json:GLI 4 / MEN 4 / METS 3 / PED 3 / SSA 2),固定种子、cfg=10、30 步;min_epoch=30、patience=3、hard cap=100(run.json selection.rule 注明出处:spec #51 decision 7)
- 时间线(日志文件名/时间戳为 UTC,括号内为服务器本地 UTC+8):
  - 08-24 11:19:59Z(19:19)run 记录创建
  - 08-24 11:46:05Z(19:46)首次训练启动 → 19:48:10 本地 rank 3 exitcode 1 失败(torch elastic ChildFailedError,`scripts.brats_p2_finetune FAILED`,traceback 未留档/error_file N/A)
  - 08-24 13:34:37Z(21:34)原样重启成功(train+dev_eval 侧车同 ID 日志 20260824T133437Z)
  - 08-24 15:17:14Z(23:17)dev_eval 侧车重启(dev_eval.pid/日志 20260824T151714Z)
  - 08-24 15:29–17:11Z(23:29–08-25 01:11)六个 eval 点(e5…e30)依序产出
  - 08-24 17:14:26Z(08-25 01:14)selection.json 落盘(recorded_utc)
  - 08-25 02:50–02:52 本地 holdout 8 shard 并行生成;06:16 本地 samples.json 合并
  - 08-24 22:17:35Z(08-25 06:17 本地)run.json 冻结(status frozen)

## 关键读数与产物路径

**早停**(ckpt/.early_stop,原文):`{"reason": "no new best for 3 evals (best 4.5712)", "epoch": 30}`——e15 后连续三个 eval 点(e20/e25/e30)无新 best,按预登记规则(epoch≥30 且 3 连无新 best)在 epoch 30 eval 后停止;实际训练 **31 epochs**(ckpt/ 下 epoch_1.pt…epoch_31.pt 共 31 个,latest.json 指向 epoch_31)。

**dev 候选选择 = epoch_15**(`dev_eval/selection.json`,61 939 B):epoch=15,mean_fid=**4.5711745865299**(≈4.5712),rule=「argmin mean dev FID over eval points (pre-recorded)」,checkpoint 指向 `/root/private_data/brats2023_rflow_p2/ckpt/epoch_15.pt`;run.json `selection` 段同值并附证据链:候选 checkpoint sha256 `e24aa838a95fd7e6…`,evidence selection.json sha256 `6adba2bf9c9e3122…`(recorded_utc 2026-08-24T17:21:04Z)。

**dev FID 趋势**(`dev_eval/dev_trend.jsonl`,36 486 B,六 eval 点,原样照抄 `m` 字段,4 位小数;eval_utc 为 UTC):

| eval 点 | e5 | e10 | e15 | e20 | e25 | e30 |
|---|---:|---:|---:|---:|---:|---:|
| mean dev FID(m) | 5.6018 | 4.8524 | **4.5712** | 4.7462 | 4.7988 | 4.8235 |
| eval_utc | 15:29:52Z | 15:41:41Z | 15:53:58Z | 16:08:44Z | 16:41:05Z | 17:11:59Z |

(best=e15;e15→e30 单调回升,触发 patience=3 早停。)

**holdout 生成**(按挑战分目录,完整):530 例 = GLI 250 / MEN 200 / METS 48 / PED 20 / SSA 12(`generated/<挑战>/` 目录逐例计数),每例×4 模态(t1n/t1c/t2w/t2f)= 2 120 卷;manifest `holdout_generated/samples.json`(920 916 B,530 entries,sha256 `2f0d5753fe545ba8…` 见 run.json samples)逐例记录 challenge、condition_mask(指向 combined labels `*-combined.nii.gz`)与各模态生成路径+固定 seed;分片产物 samples_shard_0..7.json 共 **8 shard**(shard_0 写 67 条,shard 日志 logs/shard_0..7.log)。生成用候选 checkpoint 为 epoch_15(shard 日志头原文:`[gen] candidate checkpoint: /root/private_data/brats2023_rflow_p2/ckpt/epoch_15.pt (epoch 15)`),30 步推理、每卷独立 seed;启动脚本 `brats_p2_launch_holdout_gen.sh`(注释原文「8 shards x 1 GPU each」,幂等 skip-if-exists)。

> 注:本 issue(#190)原文写「7 shard manifest」,服务器产物实为 **8 shard**(启动脚本注释、8 个 samples_shard_*.json 与 8 个 shard_*.log 三方一致)——按「以服务器产物为准」纪律记 8。

**L1 特征状态**:`l1_features.log` 原文「wrote 1080 feature records -> …/l1_features/features.json」;l1_features/ 下 features.json(263 164 B)与 1 080 个 .npy 平面特征。正式 L1 验收未跑(无验收报告),缺口如实列入「后续动作」。

**L2 终验产物**:`l2_report/l2_final_acceptance_p2.json`(29 349 B)与同名 .md(6 283 B)已存在;判定与读数不展开,由同批记录 #191 收编。

产物路径(sugon,只读采集;基目录 `/root/private_data/brats2023_rflow_p2/`,mtime 均为 2026-08-24 ~ 08-25 服务器本地时间):

- `records/dm_source.json`(1 469 B;P1 e20 溯源)
- `records/runs/p2-20260824T111958Z/run.json`(3 014 B;frozen;注:records/runs/ 下另含 p3-20260825T151310Z、p3-20260826T051647Z 两个 P3 run 记录,属 P3 快照 #193 范围)
- `ckpt/.early_stop`(65 B)、`ckpt/latest.json`、`ckpt/train_provenance.json`、`ckpt/epoch_1..31.pt`
- `dev_eval/selection.json`(61 939 B)、`dev_eval/dev_trend.jsonl`(36 486 B)、`dev_eval/early_stop_rule.json`(314 B)、`dev_eval/dev_cohort.json`(16 例)
- `holdout_generated/samples.json`(920 916 B)+ samples_shard_0..7.json + generated/{GLI,MEN,METS,PED,SSA}/ + logs/shard_0..7.log
- `l1_features/features.json` + l1_features.log;`l2_report/l2_final_acceptance_p2.{json,md}`
- 训练/评测日志:`logs/train_20260824T{114605Z,133437Z}.log`、`logs/dev_eval_20260824T{133437Z,151714Z}.log`
- 训练配置原件(仓库外):`/root/nv-phase-59/configs/config_brats_p2_train.json`、`config_network_rflow.json`(sha256 `aea761fe7a915bb8…`,与 P1 同一网络定义)
- `tblogs/` 目录为空(无 tfevent 落盘)——如实记录

## 结论与后续动作

**P2 候选训练按 #59 配方完成并冻结**:lr/batch/epochs 上限/bf16/weighted_loss@129-131/RCL 关闭/纯 BraTS 无回放逐项与验收标准一致(配置、train_provenance、配方守卫日志三方印证);候选 epoch_15 由预登记 argmin 规则选出;31 epochs 早停符合预登记 patience 规则;holdout 530 例×4 模态生成完整且 manifest 已入 run.json 证据链。

失败/风险要点(草拟待复核):

1. **验收门未闭合**:#59(CLOSED)验收标准要求候选同时完成 L1/L2/L3 判定,现状为 L1 特征已提但正式验收未跑、L2 终验 FAIL(#191 收编)、L3 盲评包未建——按非补偿判据,P2 候选验收未完成。与 CLOSED issue 的状态偏差如实记录(父 #186 决定,不翻案)。
2. dev FID 在 e15 见底后单调回升(e20 4.75 → e30 4.82),候选质量对训练时长敏感;若重启 P2 训练,eval 点距(5 epochs)内的震荡可能被低估,可考虑加密 eval 或调低 patience 下限。(草拟待复核)
3. 首次启动(08-24 19:46 本地)rank 3 exitcode 1 失败且 traceback 未留档(elastic error_file N/A),21:34 原样重启成功——失败原因不可考;若复现,建议保留完整 traceback。(草拟待复核)
4. `tblogs/` 为空:训练曲线仅有 dev_eval 侧车六点,无逐 step tfevent,损失曲线不可回放。(草拟待复核)

后续动作(草拟待复核):

1. 补跑 P2 正式 L1 验收(特征已提:1 080 records,验收报告缺)。
2. 按 #191 记录的 P2 L2 终验 FAIL 读数组织整改与复验(TOST + 条件回切未过)。
3. 建设 P2 L3 盲评包并认领专家目检(父 #186 user story 7 同款流程)。
4. 三层全过之前,P2 候选按 #59 原义不作为「验收通过」引用;索引页(#195)汇总时沿用本判定。
