# P3 候选训练——跨模态 ControlNet 进行时快照

> **快照性质与采集时间**:本文是 P3 跨模态候选训练(关联 #61)的**进行时快照**,采集于 2026-08-29 12:05 CST(服务器 `crdnotebook-2086462626317500417-wang9691-94660`,全程 SSH 只读)。采集时 dev 侧车**仍活跃**:watch 进程(PID 1782,`ctmr generate cross-modal dev-eval watch`)自 2026-08-29 02:04 起运行,正在评估 epoch_35(样本生成阶段,`dev_eval/epoch_35/samples/` 已写 2464 个文件,最后写入 10:59)。本文只记录采集时刻事实;实验后续推进由新增日期的报告承接,不改写本快照。

## 1. 目的与关联 issue

在冻结的 P1-DM(rflow-mr-brain 微调产物)上训练 image-conditioned ControlNet,使 src 模态影像条件生成 tgt 模态(跨模态),不混入掩码条件;候选完成后须相对 stage-0(零训练 img2img 基线)完成 L1/L2/L3 判定。训练目标 n_epochs=100;本快照记录其进行时状态:**训练于 epoch 40 进行中被外部 SIGTERM 中断,39 个 epoch 完整落盘**。

- 关联 issue:#61(P3 跨模态候选,状态 CLOSED——其 L1/L2/L3 验收项实际未完成,见 §4 偏差清单)
- 父任务:#186(实验记录落盘:曙光 P1/P2/P3 现役实验读数收编)
- 本记录:#193

## 2. 运行配置与 run_id

P3 顶层无 `records/` 目录、无 run.json;训练元数据以 `train_provenance.json` 为准。

| 项 | 值 | 出处 |
|---|---|---|
| 训练脚本 | `code/scripts/brats_p3_finetune.py`(torchrun 启动,world_size=7,amp bf16) | `ckpt/train_provenance.json` |
| 数据列表 | `/root/private_data/brats2023_rflow_phase/lists/p3_pairs.json`(25392 对) | 同上 + 列表文件实测 |
| DM 初始化 | `/root/private_data/brats2023_rflow_p1/ckpt/epoch_20.pt`(P1 候选) | 同上 |
| Autoencoder | `/root/private_data/manifold/models/autoencoder_v1.pt`(冻结) | `environment_brats_p3_train.json` |
| 超参 | batch_size=1,lr=1e-05,n_epochs=100,fold=0,weighted_loss=100(label 129/130/131),use_region_contrasive_loss=false | `ckpt/train_provenance.json` |
| 推理配置 | num_inference_steps=30,**cfg_guidance_scale=0** | `code_refactored/configs/config_brats_p3_train.json` |
| checkpoint 目录 | `/root/private_data/brats2023_rflow_p3/ckpt/`(epoch_1–39 齐全,每个约 287 MB) | 目录实测 |
| dev 侧车评估根 | `/root/private_data/brats2023_rflow_p3/dev_eval/`(dev_list.json 共 3180 条) | 目录实测 |

训练启动时间线:2026-08-26 13:11 首次 torchrun 启动失败(worker exitcode 1,`logs/train_20260826T050928Z.log`);13:17 重启成功(`train_20260826T051701Z.log`,provenance 写于 13:19);13:33 落盘 epoch_1,此后约 16 分钟/epoch。

## 3. 关键读数与产物路径

### 3.1 训练中断事件(出处:`logs/train_20260826T051701Z.log` 尾部、`ckpt/latest.json`)

- **2026-08-26 23:55:28** 最后一条训练日志行:`epoch 40, iter 650/3174, loss: 1.4026`(epoch 40 进行中)。
- **2026-08-26 23:56:54** torchrun elastic agent(PID 10345)抛 `SignalException: Process 10345 got signal: 15`,向 7 个 worker 发送 SIGTERM 后退出——**外部 SIGTERM 触发的中断**,非训练内部错误。
- 最后完整 checkpoint 为 **epoch_39**(`ckpt/epoch_39.pt`,23:53 落盘;`ckpt/latest.json` 指向 epoch 39)。
- `ckpt/` 与产物树中**无任何 `.early_stop` 文件**——非早停退出。
- 训练实际窗口:2026-08-26 13:17 → 23:56 CST(约 10 小时 39 分钟,完成 39/100 epoch + epoch 40 前 650/3174 iter)。

### 3.2 dev 侧车读数(出处:`dev_eval/dev_trend.jsonl`,逐行 eval_utc 为 UTC)

metric=`paired-psnr-ssim`:m 为 t1n/t1c/t2w/t2f 四模态 dev case-mean 3D SSIM 的均值,PSNR 同协议并行记录;早停规则(`dev_eval/early_stop_rule.json`):patience=999,min_epoch=30,max_epoch=100(方向 max,即 N≥30 后连续 999 次评估无新 best 才停)。

| epoch | eval_utc | m(mean SSIM) | mean PSNR (dB) | 备注 |
|---|---|---|---|---|
| 5 | 2026-08-26T15:36:40Z | 0.8511 | 20.06 | |
| 10 | 2026-08-26T19:49:00Z | 0.8686 | 21.37 | |
| 15 | 2026-08-27T00:29:27Z | 0.8726 | 21.53 | |
| 20 | 2026-08-27T05:12:06Z | 0.8792 | 22.03 | |
| 30 | 2026-08-29T01:15:55Z | **0.8829** | **22.10** | **best**(日志 `[eval] epoch 30: ... stop=False`) |

- 每 epoch 分模态明细另存于 `dev_eval/epoch_<N>/trend.json`(如 `epoch_30/trend.json` 与上表一致)。
- `dev_trend.jsonl.fid-legacy`:仅 1 行(epoch 5,FID m=3.92,2026-08-26T13:40Z)——侧车早期以 FID 为指标,后切换为 paired-psnr-ssim。
- **epoch 25 无 trend 行**:`dev_eval/epoch_25/` 目录存在,但前一段侧车日志(`dev_eval_20260826T150949Z.log`)止于 08-27 13:37,该轮评估未完成即中断。
- epoch 35 评估进行时(见首段),尚无 trend 行。

### 3.3 侧车进程时间线(出处:`logs/` 各段日志与 `ps` 实测)

1. 2026-08-26 13:06 起多段 dev_eval 日志(050646Z 空文件、051656Z、071145Z、141028Z、150949Z),逐 epoch 评估与训练并行。
2. 08-27 13:37 前段侧车日志停止(epoch 25 评估中断,未留 trend 行)。
3. 2026-08-29 02:04 侧车以重构后 CLI 重启:`ctmr generate cross-modal dev-eval watch`(`dev_eval_restart_20260829.log`;`early_stop_rule.json` 同时刻写入)。
4. 08-29 09:15 补评 epoch_30 并写入 trend(即 §3.2 的 best 行);09:16 起评估 epoch_35,快照采集时生成阶段仍在推进。

## 4. 结论与后续动作

**结论(草拟待复核)**:跨模态 ControlNet 训练健康推进至 epoch 39后被外部 SIGTERM 中断,非模型或数据故障:dev SSIM 从 e5 0.8511 单调升至 e30 0.8829(PSNR 20.06 → 22.10 dB),e30 为当前 best;39 个 checkpoint 完整在盘,`latest.json` 指向 epoch_39,具备从 epoch_39 续训的条件。dev 侧车独立于 trainer 生命周期,重启后自动衔接评估,当前仍在消耗存量 checkpoint(epoch 35)。

**与 CLOSED issue 状态的偏差**:#61 已 CLOSED,但其验收标准「候选相对阶段 0 完成 L1/L2/L3 判定」未见任何执行记录——L1/L2/L3 三层全部为缺口,且训练本身未完成(39/100)。#61 的关闭状态与本快照记录的事实存在偏差,此处如实记录、不翻案。

**未完成验收(缺口)**:

- L1(候选 vs stage-0 定量):未执行
- L2(分布对齐终验):未执行
- L3(专家目检):未执行
- 训练完成度:39/100 epoch,epoch 40 中断于 iter 650/3174

**后续动作**:

1. **恢复训练/重排决策待定**(由实验负责人决定:自 epoch_39 续训,或以现有 checkpoint 另行安排)。
2. 训练恢复后,由 dev 侧车继续按 early_stop_rule(patience 999 / min_epoch 30 / cap 100)跟踪;若实验推进,以**新增日期报告**记录,不改写本快照。
3. 训练收敛或决策终止后,补跑 L1/L2/L3 三层验收(候选 vs stage-0),补齐 #61 遗留缺口。
