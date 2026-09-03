# 20260903 序列②T8——dev 中间检查点:重训候选监控 + 观察线判读(go/no-go)

**状态: 全链完成(采样 520/520 + 仪器 v2 推理 260/260 + 测量 + 报告);go/no-go 判读 = 停在 dev(不进 L2),后手清单五条状态落盘** | **记录落盘日**: 2026-09-03 | **实际执行**: 2026-09-03,sugon 集群(重拉 03:5x → 报告 05:36:21Z,约 1.7 h)

## 1. 目的与关联 issue

对 T7 整改重训候选(epoch 25,`p1-20260901T141537Z`)以 T6 同款监控链路跑 dev ET/WT 读数,与 T6 现候选基线(同仪器 v2)对照,按 #255 预记录判据做 go/no-go 判读:放行 → 进 T9 终验;停在 dev → 后手清单五条逐条状态落盘,不进 L2。

- 关联 issue: #255(本票)、父 #247(序列②整改重训)、前置 #253(T6 监控链路 + 基线)、#254(T7 重训)
- 判定状态关联: variant=diagnostic,选择面、非验收判定;P1 L2 判定不变(FAIL)
- 观察线(沿 #253 预记录): **METS ET 检出率 < 0.9 或任一挑战 vol_et_rel 中位 > 2 → 黄旗**

## 2. 运行配置与 run_id

- **代码**: 本地 `main@9e7e3d0`(= T6 收官 #307 内容 + dev_monitor.py ruff-format 化 #308)rsync 至 sugon 部署树 `/root/ctmr`;链路配方 `deploy/jobs/run_dev_monitor_etwt.sh`(T6 原配方零改动)
- **run_id**: `p1-20260901T141537Z`——T7 训练启动时刻戳(训练日志 `train_20260901T141537Z.log`,正式拉起 2026-09-01T14:15:37Z;T7 记录未显式登记 run_id 字段,按 T6 命名惯例 `p1-<UTC启动戳>` 构造,特此声明构造依据)
- **checkpoint**: `/root/private_data/ctmr/runs/p1_t7/ckpt/epoch_25.pt`(只读;md5 `0dac81864b68ab35f16dde93039db643` 与 T7 记录逐位一致;selection.json rule=argmin mean dev FID,预记录);VAE `/root/private_data/ctmr/models/autoencoder_v1.pt`(冻结只读,经 env 覆写件)
- **实参**: `P1_ROOT=/root/private_data/ctmr/runs/p1_t7`、`CKPT`、`RUN_ID` 三项覆写,**其余全部配方默认**(DEV_LIST 1060 条 / RAW_ROOT / EMB_ROOT / VAE_DIR / nnUNet 三变量 + `NNUNET_EXT_TRAINER`);`MONITOR_ROOT` 走默认 `$P1_ROOT/dev_monitor` = `/root/private_data/ctmr/runs/p1_t7/dev_monitor`(全新目录,`sampling_provenance.json` 首跑钉本候选 sha256)
- **仪器版本**: **v2**(与 T6 基线同批同树——两读数同仪器可直接对照);第二步b spec 实况覆写五挑战全部按 results 树实况执行(`nnUNetTrainer_250epochs_bf16__nnUNetPlans_v2bs8__3d_fullres_bs8`),与 T6 续跑同语义。偏离声明:监控为选择面 variant=diagnostic,不消费包络判定线,故 T6 记录原列前置「校准重跑与新 ADR」不阻塞本读数——该前置属任何 L2 终验,本次判读停在 dev,前置未触发
- **采样协议**(钉定于 `dev_monitor_sampling.MONITOR_QUOTAS`,与 T6 同款): 分层 GLI 50/MEN 40/METS 24/PED 10/SSA 6 = 130 例 × 4 模态 = 520 体;sha256(case/case|modality) 确定性选择与种子;cfg=10、30 步、RFlowScheduler;写出协议 #249 后世界
- **种子纪律**: 沿 T6——ET 轴作业 B 槽位 200、WT bootstrap CI90 诊断槽位 `dev_monitor_wt_rel_diff = 600`;零 GLOBAL_SEED 判定链接触
- **频率约定**(沿 #247): 候选选择点单次(本候选唯一选择点),训练中不跑
- **population = dev,零 holdout 接触**: dev list 唯一 population 输入;holdout 530 任何工件不进入本链路

### 执行事故与重拉(如实记录)

首轮拉起手动覆写 `RAW_ROOT`/`EMB_ROOT` 时多加了一层 `ASNR-MICCAI-BraTS2023/`(dev_list `image` 字段已自带该前缀,采样臂按 `emb_root / rel` 直接拼接),首例 spacing 读取即 `FileNotFoundError` 响亮死。**零体文件写出、零工件污染**;修正为配方默认值后同 `MONITOR_ROOT` 重拉(provenance 验证同候选指纹通过)。教训:配方默认值是 T6 实跑沉淀的部署事实,覆写前先与 dev_list 字段形态对账。

## 3. 关键读数与产物路径

- 采样产物: `/root/private_data/ctmr/runs/p1_t7/dev_monitor/samples/`(520/520 体 + `sampling_provenance.json`)
- 仪器输入: `/root/private_data/ctmr/runs/p1_t7/dev_monitor/inputs/`(1040 文件 = 260 obs × 4 模态)
- 测量 CSV: `/root/private_data/ctmr/runs/p1_t7/dev_monitor/measurements_dev.csv`(261 行 = 260 obs + header,零 input_fail/run_fail)
- 监控报告: `/root/private_data/ctmr/runs/p1_t7/dev_monitor/report/dev_monitor_diagnostic.{json,md}`(generated 2026-09-03T05:36:21Z)
- 采样协议: `cohort.json` + `plan.json`(population=dev,run_id `p1-20260901T141537Z`);作业日志 `logs/dev_monitor_launch.log`(存 `/root/private_data/ctmr/runs/p1_t7/`)

### 逐挑战 ET/WT 对照(仪器 v2;基线 = T6 `p1-20260822T131947Z`,候选 = T7 `p1-20260901T141537Z`;相对差为 (gen−real)/(gen+real))

| 挑战 | 指标 | 基线(T6) | T7 候选(T8) |
|---|---|---|---|
| **GLI** | ET 检出 k/n | 50/50 | 50/50 |
| | ET vol median gen/real (ml) | 64.63 / 16.18 | 56.18 / 16.18 |
| | ET rel median (CI90) | 2.2424 (−0.799, 276.34) | 2.2438 (−0.930, 142.15) |
| | WT vol median gen/real (ml) | 697.86 / 81.95 | 472.52 / 81.95 |
| | WT rel median (CI90) | 7.6644 (1.928, 117.00) | 4.3805 (0.135, 80.03) |
| **MEN** | ET 检出 k/n | 40/40 | 40/40 |
| | ET vol median gen/real (ml) | 19.28 / 14.47 | 10.42 / 14.47 |
| | ET rel median (CI90) | 0.2317 (−0.999, 118.38) | −0.0628 (−0.986, 147.84) |
| | WT vol median gen/real (ml) | 19.92 / 17.25 | 10.42 / 17.25 |
| | WT rel median (CI90) | 0.2175 (−0.999, 142.45) | −0.2112 (−0.990, 188.58) |
| **METS** | ET 检出 k/n | 16/24(空 pred 4) | **16/24(空 pred 6)** |
| | ET 缺失 real_only | 7 | 7 |
| | ET vol median gen/real (ml) | 0.02 / 4.85 | 0.01 / 4.85 |
| | ET rel median (CI90) | −0.9927 (−1.000, 2.517) | **−0.9987 (−1.000, −0.484)** |
| | WT vol median gen/real (ml) | 1.60 / 31.85 | 0.31 / 31.85 |
| | WT rel median (CI90) | −0.8803 (−1.000, 144.75) | −0.9956 (−1.000, 20.47) |
| **PED** | ET 检出 k/n | 10/10 | 10/10 |
| | ET vol median gen/real (ml) | 8.99 / 4.56 | 20.24 / 4.56 |
| | ET rel median (CI90) | 1.0514 (−0.869, 6.048) | **2.7790 (−0.099, 19.60)** |
| | WT vol median gen/real (ml) | 323.70 / 32.82 | 170.88 / 32.82 |
| | WT rel median (CI90) | 9.6895 (−0.222, 21.23) | 5.1958 (−0.699, 15.03) |
| **SSA** | ET 检出 k/n | 6/6 | 6/6 |
| | ET vol median gen/real (ml) | 494.70 / 21.24 | 247.79 / 21.24 |
| | ET rel median (CI90) | 38.6386 (5.551, 94.26) | **17.3598 (0.191, 34.71)** |
| | WT vol median gen/real (ml) | 1123.24 / 167.11 | 755.58 / 167.11 |
| | WT rel median (CI90) | 6.6617 (3.068, 28.87) | 3.4532 (0.443, 22.65) |

(全轴新增注记: MEN/PED/SSA 的 `n_undefined`、CI 宽度等逐项细节见 `dev_monitor_diagnostic.json`;SSA/PED 样本量 6/10 例,读数方向性参考。)

### 观察线判定

- **基线(T6)**: 黄旗,fired 3 条——METS ET 检出率 0.6667 < 0.9;GLI vol_et_rel 2.2424 > 2.0;SSA vol_et_rel 38.6386 > 2.0
- **候选(T8)**: **黄旗未解除,fired 4 条(净增 1)**——
  1. `METS ET 检出率 0.6667 < 0.9`(主旗,逐位未动)
  2. `GLI vol_et_rel 2.2438 > 2.0`(未动)
  3. `SSA vol_et_rel 17.3598 > 2.0`(读数大降但仍触线)
  4. `PED vol_et_rel 2.7790 > 2.0`(**新越线**,基线 1.0514 未触)

## 4. go/no-go 判读(选择面,预记录判据)

**判据(#255 预记录)**: 放行 = 旗标解除/读数改善,**METS ET 检出率与 vol_et_rel 中位相对基线改善** → 进 T9 终验;无改善 → 停在 dev 不进 L2,后手清单五条逐条状态落盘。

**读数**: METS ET 检出率 16/24 → 16/24(**零变化**,空 pred 反增 4→6);METS vol_et_rel 中位 −0.9927 → −0.9987(**无改善,更贴 −1**);观察线黄旗未解除且 fired 规则净增 1。

### **判读: 停在 dev(no-go),不进 L2**

「协议修正包(clip=True 编码 + token 34 冻结 + 写出 affine)足以恢复空间保真」的假设检验在 dev 选择面**未获支持**:主靶 METS(多灶/小病灶)全轴零改善,体积高估族(GLI/PED/SSA)形态与基线同构。部分轴有方向性改善(SSA ET rel 38.6→17.4;GLI/PED/SSA WT rel 中位 7.7/9.7/6.7 → 4.4/5.2/3.5)但不触判据,且 PED ET 反向越线——综合判读为无改善。

### 后手清单五条逐条状态(#247 预埋,#255 落盘义务)

| # | 假设 | T8 读数证据 | 状态 |
|---|---|---|---|
| ① | 训练侧 z 向几何(155→128) | T7 三改动不含 z 几何;METS 小病灶轴(检出率逐位不动、空 pred 4→6、ET rel 中位贴 −1)零改善,与「小病灶被训练侧 z 向下采样稀释且未被本次处置」相容 | **未被处置;读数相容——与 #247 z-pad 留观触发条件(ET 读数无改善)一并成立,升为下一轮优先消融候选** |
| ② | 生成 FOV vs real 分布残余错配(写出修复只修声明,上缘 431 ml 生成侧学习缺口仍在) | spec 明文锚对账:#247 的 146 例/431 ml 上缘声明域外错配经写出修复(T2)已消——本读数已在真实 spacing 域,声明面错配为零;生成侧同族缺口仍在:大体积高估(GLI WT gen 中位 472 ml vs real 82 ml;SSA 755 vs 167 ml) | **部分支持——声明面已修(431 ml 锚已消),生成侧大体高估学习缺口仍在** |
| ③ | CFG=10 过引导 | 重训未动 cfg;GLI/PED/SSA「检出满 + 尺寸超做」形态与基线同构(T6 §4 已预注采样侧引导方向) | **相容,未被处置——采样侧引导(cfg)消融候选** |
| ④ | 容量/epoch 不足 | epoch 轴被 T7 自身证据反对(e25 best 后 e30/35/40 三连无新 best,patience 3 早停——更多 epoch 无新 best);容量轴无独立读数 | **epoch 轴证据反对;容量轴证据不足** |
| ⑤ | P1 无空间条件架构天花板(战略出口 → P2 掩码条件路线) | METS 是唯一多灶/小病灶挑战;输入域+token 稀释+写出协议三重协议修正后其检出率逐位不动——与「模态标签条件对病灶位置/数目无信息通道」方向一致 | **方向性支持(战略级信号);若 ①③ 消融仍不达,战略出口指向 P2** |

## 5. 结论与后续动作

1. T8 完成:重训候选 dev 监控读数 + 与基线对照 + 观察线判定 + go/no-go 判读全部落盘;AC 三条(读数落盘/判读落盘/零 holdout 零验收判定)达成。工件在受控树,本记录为读数转写主本。
2. **T7 重训候选停在 dev,不送 L2 终验**——终验预算不烧在无改善候选上(#247 user story 9)。
3. 下一轮按 #247 预埋后手次序进入诊断:①z 向几何消融票(#247 z-pad 留观触发条件已成立,与后手①同族,建议并票处理)+ ③CFG 消融票为配方内低成本候选;⑤为战略出口留底。后续票另立,本票不预判其配方。
4. 候选资产:T7 `epoch_25.pt` 与全套监控工件保留于 `p1_t7` 树;`sampling_provenance.json` 钉死候选指纹,该 MONITOR_ROOT 不可复用于其他候选。
5. 仪器 v2 状态不变:校准重跑 + spec 冻结锚重钉 + 新 ADR 仍是任何 L2 终验的前置(见 20260901 仪器主本丢失记录 §5,不在本票范围)。

## 验收对照(#255)

- [x] 重训候选 dev 监控读数落盘(逐挑战 + 旗标状态 + 与现候选基线对照表)——本记录 §3
- [x] go/no-go 判读落盘:停在 dev + 后手清单五条逐条状态——本记录 §4
- [x] 零 holdout 接触;零验收判定——dev list 唯一 population(plan.json population=dev),holdout 530 工件零进入;variant=diagnostic,全程无验收判定产生
