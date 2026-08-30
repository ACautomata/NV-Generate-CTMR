# P1 根因甄别·诊断作业 D(同 seed 换 token 采样亮核甄别,模态条件稀释)

**总体结论:token 34 的增强语义在同 seed 对照下显著存活——t1c(34) 臂亮核顶部统计为其余四臂的 2 倍以上,配对差 CI90 全部不跨 0;34 对对照泛 MR(8) 的增益份额中位数 0.6041(CI90 0.3447, 0.6682)落 mixed 带——增广稀释的实测损伤存在(≈40% 亮核势被拉平到泛 MR)但 34 语义主体未被磨平。RC-2(模态标签增广稀释)从 ET 缺失主嫌疑降级为叠加因素;t1n/t2w/t2f(29/30/31) 臂亮核与泛 MR 8 臂同档(非 t1c 臂的模态特异亮度语义未立起来),但 ET 条件线索唯一载体是 34 且其语义显著——replay 29-31 语义锚方向对 ET 整改目标证据不足。**

诊断作业 D(#209,父 #205 序列①服务器段)的同 seed 换 token 采样甄别。`variant=diagnostic`:不产生任何验收判定,checkpoint 与 VAE 只读,训练产物零改动;bootstrap 种子独立于正式判定链(诊断基 900,000,000,作业 D 无挑战带,臂 CI 槽 300..304、对比槽 310..313、份额槽 320)。服务器报告生成于 2026-08-30T08:16:53Z,本记录落盘于 2026-08-30,读数经 SSH 只读采集自 sugon 工件。

## 目的与关联 issue

本地取证(`20260829-P1根因分析-本地取证阶段.md`)把 RC-2/L3 嫌疑(~19% 模态标签增广稀释 token 34,CFG=10 放大 (cond−uncond) 差)列为 ET 缺失主嫌疑之一。本作业在固定 16 例 dev cohort(GLI 4/MEN 4/METS 3/PED 3/SSA 2,DevCohortBuilder)上以冻结 P1 候选 checkpoint 在冻结采样配方(cfg=10、30 步、RFlowScheduler)下做五臂采样——t1n(29)/t1c(34)/t2w(30)/t2f(31) + 对照「泛 MR」(8,增广把 34 扰到的目标);每例一 seed(sha256(case|t1c),冻结采样规则)五臂共用,噪声逐位一致,输出差异全部归因 token 条件,定量增广稀释对亮核的实际损伤。

- 关联 issue:#209(本作业)、#205(根因甄别父票)、#206/#207/#208(作业 A/B/C,归因合流)、#217(复核作业 A)
- 判定状态关联:不改任何验收判定;P1 L2 仍为 FAIL,L1/L3 缺口不变

## 运行配置与 run_id

- run_id:`p1-20260822T131947Z`(从 `l2_final_acceptance_p1.json` binding 读取)
- 执行:sugon GPU 作业(`deploy/jobs/run_token_dilution_d.sh`,GPU 采样 + CPU 统计两步),执行窗口 2026-08-30 16:10–16:17(拉起至读数采集,全程约 7 分钟:80 臂次采样 fp16 + bootstrap B=10000 统计)
- 代码版本:`token_swap_sampling.py` md5 `26830e27…`、`token_dilution.py` md5 `45efb1b0…`、配方 md5 `6f8f5f0a…`,与 main head b8b1ba8 一致
- 部署:现役部署树 `/root/nv-phase-60`(完整仓库形态;`src/ctmr` 全量 112 文件 md5 清单比对,108 个共同文件与本地逐字节一致);新增 3 个文件(两模块 + 配方)零覆盖。**服务器资产重组事件**:执行当日 `/root/private_data` keep 集迁入 `/root/private_data/ctmr/`(负责人执行的重组),前置路径全部显式指向新布局:`P1_ROOT=/root/private_data/ctmr/brats2023_rflow_p1`、`DEV_LIST=/root/private_data/ctmr/brats2023_rflow_phase/lists/p1_image_only_dev.json`(1060 条,配方默认路径已不存在)、`EMB_ROOT=/root/private_data/ctmr/brats2023_rflow_phase/embeddings`(dev cohort 16 例 sidecar 逐一验证 16/16 命中,spacing [0.9375, 0.9375, 1.2109375] 全可读)、`NV_CONFIGS=/root/nv-phase-60/configs`(三件套与本地 md5 一致)
- VAE:`environment json` 的 `trained_autoencoder_path`(相对路径 `models/autoencoder_v1.pt`)经部署树 `/root/nv-phase-60/models/autoencoder_v1.pt` 解析,md5 `917cfb1e49631c8a713e3bb7c758fbca` 与冻结 canonical 全同(重组后幸存副本,源自 `brats2023_rflow_p3/code_discarded_20260829/models/`);冻结只读
- 环境:torch-dcu 2.9.0、numpy 1.26.4、monai 1.6.0、nibabel 5.4.2;`token_dilution.py` 依赖 SimpleITK,执行前新装 SimpleITK 2.5.6;DCU 单卡(cuda:0),hy-smi 确认负载
- 首次拉起(15:49)因重组路径迁移即时 FATAL 于配方前置检查,零 GPU 工作;其遗留物旧路径 `/root/private_data/brats2023_rflow_p1/token_dilution/{run_d.log,run_d.pid}`(仅一行 FATAL)保留未动,如实登记
- 种子:seed 锚 = sha256(case|t1c) 五臂共用(非 t1c 臂的 seed 刻意不取冻结规则在该 modality 的正式 seed,同 seed 对照使然;统计端从工件文件名复核 seed 锚,不一致即拒绝);bootstrap 诊断基 900,000,000,槽位分配见记录头部声明,与正式判定链 GLOBAL_SEED=20260821 及作业 A(0/1/100/101)、B(200)严格分离(#217 复核占挑战带 ×1000+300,与作业 D 基带 300..320 数值域不重叠)

## 关键读数与产物路径

产物(sugon 工件区,不入 git):`/root/private_data/ctmr/brats2023_rflow_p1/token_dilution/token_dilution_diagnostic.{json,md}`,schema `token-dilution-diagnostic/1`;采样产物 `token_dilution/samples/`(80 体积,16 例 × 5 臂,0 excluded)。

### 逐臂亮核统计(非零体素基底,前 0.5% 均值 median (CI90))

| 臂 | token | n | 前 0.5% 均值 median (CI90) | P99 median | P99.9 median | max median |
|---|---|---:|---:|---:|---:|---:|
| t1n(29) | 29 | 16 | 1348.7277 (1264.1476, 1484.7316) | 1208.0000 | 1296.0000 | 1434.0000 |
| **t1c(34)** | 34 | 16 | **3036.6269 (2434.5934, 3514.3916)** | 1257.0000 | **2578.0000** | **4013.5000** |
| t2w(30) | 30 | 16 | 1507.5200 (1365.6099, 1658.7782) | 1092.0000 | 1419.5000 | 1806.5000 |
| t2f(31) | 31 | 16 | 1187.9719 (1140.2830, 1533.8953) | 1040.5000 | 1129.0000 | 1381.5000 |
| 泛 MR(8) | 8 | 16 | 1161.3460 (1077.8351, 2167.5330) | 1047.5000 | 1118.0000 | 1435.5000 |

### token 34 配对差(前 0.5% 均值,同 seed,n=16)

| 对照 | 差 median (CI90) | 判读 |
|---|---:|---|
| 34 − 29(t1n) | 1682.3149 (1156.2868, 2100.3183) | CI90 下界为正,显著 |
| 34 − 30(t2w) | 1500.1636 (920.6211, 2070.5565) | CI90 下界为正,显著 |
| 34 − 31(t2f) | 1824.5154 (1283.7692, 2334.7656) | CI90 下界为正,显著 |
| 34 − 8(泛 MR) | 1764.4881 (1024.9087, 2335.6797) | CI90 下界为正,显著 |

### 甄别主读数:34 对 8 增益份额

增益份额 = clamp((亮核(34) − 亮核(8)) / 亮核(34), 0, 1),主指标前 0.5% 均值;分带沿作业 A 的 2/3–1/3 归因先例(≤1/3 dilution_dominant,≥2/3 semantics_intact,之间 mixed)。

- 跨 case 份额中位数 **0.6041(CI90 0.3447, 0.6682)→ mixed**
- 逐 case 分带:mixed ×15、semantics_intact ×1(GLI-00547 0.6682,贴 2/3 线)、**dilution_dominant ×0**
- METS cohort 3 例份额 0.6060/0.6187/0.6148——与总体一致,**无特异恶化**

## 结论与后续动作

**结论:**

1. **「生成 t1c 出不了亮核」在条件轴上证伪**:同 seed 下 token 34 的亮核顶部统计为其余四臂 2 倍以上,四个配对差 CI90 全部不跨 0——t1c 条件下增强语义的幅度势充分。
2. **RC-2(增广稀释)降级为叠加因素**:稀释的实测损伤存在但中等——34 对 8 份额 0.6041 落 mixed 带(≈40% 亮核势被拉平到泛 MR,未达 semantics_intact 的 ≥2/3,也未落 dilution_dominant 的 ≤1/3);增广没有把 token 34 的语义磨平。
3. **ET 缺失不能由 token 稀释单独解释**:作业 B 的 ET 缺失全部集中 METS(10/48),而 D 的 METS cohort 份额无特异恶化,且 16 例 cohort 五臂全部产出亮核(D 测亮核幅度,不测 ET 检出)——METS 缺失的机理指向空间/尺寸保真侧,与作业 A/复核的质心放置低精度、WT 体积系统性过估同向。
4. **29/30/31 臂亮核与泛 MR 8 臂同档**(1188–1508 vs 1161)——非 t1c 臂的模态特异亮度语义未立起来,与本地取证「29-31/34 只收 BraTS 带瘤梯度、语义拉向带瘤脑先验」的互渗预测相容;但 ET 条件线索唯一载体是 34 且其语义显著,replay 29-31 语义锚方向对 ET 整改目标证据不足(对泛 MR 多模态生成质量的潜在影响另议,不在本甄别范围)。

后续动作:

- [ ] 实验负责人复核本记录,解除「草拟待复核」标注
- [ ] 读数交收编票(#210)与作业 A/复核 A/B/C 合流,产出「读数→结论→整改方向」裁决表
- [ ] 服务器重组后的新路径布局(P1_ROOT/DEV_LIST/EMB_ROOT/部署树)已在本记录登记,后续作业配方沿用;旧路径首轮遗留 log/pid 待负责人定夺是否清除
