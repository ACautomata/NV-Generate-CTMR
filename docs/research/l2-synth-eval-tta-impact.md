# #38 TTA 影响量级纸面量化:判定 margin vs 镜像 TTA 差异

> 对应 GitHub issue [#78](https://github.com/ACautomata/NV-Generate-CTMR/issues/78)「纸面量化 #38 TTA 影响量级:判定 margin vs TTA 差异」(地图 [#75](https://github.com/ACautomata/NV-Generate-CTMR/issues/75)),评估对象为 [#38](https://github.com/ACautomata/NV-Generate-CTMR/issues/38)「L2 仪器合成域适用性评估」(关票 PR [#50](https://github.com/ACautomata/NV-Generate-CTMR/pull/50),commit `797bdf5`)。
>
> 研究截点:2026-08-24。纯纸面研究:**没有跑任何新推理、没有触碰集群、没有修改任何现有代码或文档**(本 findings 文件除外)。
>
> 范围澄清:#38 的判定量是 R_fail(input_fail / run_fail / hier_viol)与真实校准 R_fail 的对照,以及作为观察项(非判定量)的空 pred 计数。ADR-0004 的 TOST / 分布对齐统计属于 #58 终验链,本文仅在 §5 说明其不受本 bug 影响。

## TL;DR:不可能翻转

**结论档位:不可能翻转。** 且结论比「TTA off 是否伤害判定」更强:#38 的判定量 R_fail 的三个组件对镜像 TTA 的开/关在**结构上不敏感**——无论 #38 执行时 TTA 实际处于 on 还是 off 状态,五子挑战 × 两模式的 PASS 判定都不可能因此翻转。理由(逐组件论证见 §4):

1. `input_fail` 检查发生在推理**之前**(输入 NIfTI 的几何契约),TTA 不触及输入文件;
2. `run_fail` 是进程级失败,TTA off 是 TTA on 的严格计算子集(少 7 次镜像前向 + 平均),不引入新失败路径——反而 DCU 上 TTA on 才是更重的一方;
3. `hier_viol` 从 nnUNetv2_predict 的 **argmax 单标签体**按标签值派生 WT={1,2,3} / TC={1,3} / ET={3},ET⊆TC⊆WT 对任何合法标签体恒成立,值域检查只可能因非 {0,1,2,3} 标签触发,而 argmax 输出不可能产生——TTA on/off 同样如此。

唯一被 TTA 触碰的量是**空 pred 计数**(METS P1 2/20、P3 42/80;MEN P3 2/80),但它在 #38 的判定语义里是「测量结果而非失败」的**观察项**,不构成 gate。文献量级(nnU-Net 官方定性「slightly lower」、Wang et al. 2019 脑肿瘤 TTA 增益 ≈ +0.5–1.0 Dice 点)也只够在个位数病例上翻动极小病灶的检出倾向,远不足以构成任何判定翻转。

附:对 bug 前提本身的一个重要修正——在 nnUNetv2 2.8.1 官方 argparse 定义(`store_true` + `parse_args()`)下,`--disable_tta False` 的标准行为是 **fatal error(`unrecognized arguments: False`,exit 2)**,而非静默禁用 TTA。#38 产出了完整预测且 run_fail=0,故实际执行形态存在三种可能(§2.3),但结论对三种分支全部稳健。

## 1. 已有受控产物盘点

### 1.1 本地(本仓库)实际有什么

| 产物 | 位置 | 内容 |
|---|---|---|
| #38 P1 判定报告 | 原 `eval/l2-synth-domain/report_p1.md`(随 #145 出清;读数沉淀于 §3 与文末附录,全文见 git 历史) | 五挑战 R_fail 全 0(0/20×3、0/14×2),细分 input/run/hier 全 0,METS 空 pred 2;总体 PASS |
| #38 P3 判定报告 | 原 `eval/l2-synth-domain/report_p3.md`（随 #145 出清；读数沉淀于 §3 与文末附录,全文见 git 历史） | 五挑战 R_fail 全 0(0/80×3、0/56×2),细分全 0,空 pred METS 42、MEN 2;总体 PASS |
| #38 关票记录 | [issue #38 resolution](https://github.com/ACautomata/NV-Generate-CTMR/issues/38)(2026-08-20) | 判定口径「逐挑战 point ≤ 真实包络(=0)」、样本构成、Wilson 上界、undecided 判据落盘 |
| 真实校准包络(对照基准) | `docs/adr/0002-l2-instrument-calibration-envelopes.md` | 五挑战 R_fail 全 0(0/900、0/42、0/720、0/171、0/72),E_r,vol / E_r,centroid / D_r,low 数值表 |
| 校准协议(R_fail 定义与推理配置) | `docs/calibration/l2-instrument-calibration-protocol.md` | TTA on(不传 flag)为冻结配置之一 |
| 终验通过线(TOST 边界) | `docs/adr/0004-l2-final-acceptance-pass-lines.md`、`docs/calibration/l2-final-acceptance-protocol.md` | TOST 边界 = ADR-0002 冻结包络;明确警示 store_true 语义 |
| 涉事调用点源码 | 见 §2.1 | 4 处手写调用,其中 3 处带 `--disable_tta False` |

**判定统计在本地是完整的**:两份 report MD 覆盖了全部十个子挑战(P1/P3 × 五挑战)的 k/n、细分与空 pred 计数,issue #38 resolution 补充了判定规则与 Wilson 上界。#38 的 margin 量化(§3)不需要任何 sugon 侧产物。

### 1.2 本地没有、只在 sugon 的(如需取证再拉取)

以下均非本结论所需,列出供后续会话按需拉取(路径依据 issue #38 resolution 与 `scripts/l2_synth_domain_sugon.py:24` 的 `EVAL_ROOT`):

- `sugon:/root/private_data/l2-synth-eval/report_{p1,p3}/` — 逐病例 case_results(含病例 ID,DUA 约束不入库);
- `sugon:/root/private_data/l2-synth-eval/{p1,p3}_predictions/` — 原始预测 NIfTI 与 `predict.log`(后者是分辨 §2.3 三分支的最直接证据:实际执行的命令行与其 stderr);
- `sugon:/root/private_data/l2-synth-eval/run_eval.py` — 当时刻的自包含副本(可对照仓库版 `scripts/l2_synth_domain_sugon.py` 是否一致);
- #58 终验链产物(L2 报告 JSON、measurements.csv、final_acceptance verdict、L2 HTML 报告)也在 sugon 受控存储(PR #72 明言「只落受控存储,不入 git」)——与 #38 bug 无关(§5),仅在需要终验数字时拉取。

## 2. Bug 事实核查

### 2.1 调用点盘点(2026-08-24,当前 main `53cd97a`)

| 调用点 | 形式 | TTA 状态 |
|---|---|---|
| `scripts/nnunet_l2_synthetic_domain_eval.py:498` | 生成的 predict 脚本字符串 `--disable_tta False` | 若原样执行 → 见 §2.2 |
| `scripts/run_l2_synth_domain_eval.sh:94` | 编排脚本直接调 `nnUNetv2_predict_from_raw_data … --disable_tta False` | 同上(注:`nnUNetv2_predict_from_raw_data` 并非 nnUNetv2 2.x 标准入口名,标准为 `nnUNetv2_predict`) |
| `scripts/l2_synth_domain_sugon.py:326-335` | sugon 自包含副本,subprocess 传 `["--disable_tta", "False"]` 给 `scripts.l2_calibration_predict_entry`(argv 原样透传给 nnUNetv2 原生 `predict_entry_point`,见 `scripts/l2_calibration_predict_entry.py:21-24`) | 若原样执行 → 见 §2.2 |
| `scripts/p1_predict_all.sh`(P1 回放,#57/#58)| 显式不传 flag,注释「TTA 保持默认开启」 | **TTA on** |
| `scripts/nnunet_l2_final_acceptance.py:445-471`(终验判定链,#55/#58)| 生成的脚本省略该 flag,docstring 436-438 明言 store_true 语义 | **TTA on** |
| `scripts/l2_calibration_predict.sh`(#36 校准,ADR-0002 的数值来源)| 不传 flag,注释冻结配置「镜像 TTA on」 | **TTA on** |

### 2.2 argparse 语义实测(关键修正)

nnUNetv2 v2.8.1 官方源码(`nnunetv2/inference/predict_from_raw_data.py`,[v2.8.1 tag](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/inference/predict_from_raw_data.py))定义:

```python
parser.add_argument('--disable_tta', action='store_true', required=False, default=False, ...)
…
args = parser.parse_args()
```

本地实测(标准库 argparse,与推理无关的纯语义验证):`store_true` 不消费参数值,`--disable_tta False` 中 `False` 成为多余 token,`parse_args()` 报 `error: unrecognized arguments: False`,**exit 2,进程不产出任何预测**——而非「把字符串 False 当真值、静默禁用 TTA」。(终验脚本 docstring「passing any value, including False, would turn it off」与 memory 里「argparse 会当 True」的表述,是对意图的正确保守、对机制的不精确描述。)

### 2.3 #38 实际执行形态的三个分支与结论稳健性

Issue #38 有完整预测输出且 run_fail=0(附录),与「该 token 原样执行必 fatal」并存,只有三种解释:

- **分支 A**:sugon 环境下该 token 组合以某种途径被解析为 `disable_tta=True`(如 DCU 侧打包差异),预测产出且 TTA **off**——即 ticket 前设的状态;
- **分支 B**:原样命令 fatal(exit 2)后被执行者发现并修正重跑(去掉 token 或不传 flag),最终读数产生于 TTA **on**,与报告声明一致;
- **分支 C**:sugon 上实际执行的 `run_eval.py` 与仓库版在关票整理时有差异(仓库版随 PR #50 一次进入,`git log --follow` 仅 `797bdf5` 一提交,无从本地分辨)。

分辨证据在 sugon 的 `predict.log` / run_eval 输出(§1.2)。**但 §4 的论证对三个分支全部成立**:无论 TTA 实际 on 还是 off,R_fail 判定不可翻转。本文不需要分辨分支即可给出结论。

### 2.4 对照基准未受污染

ADR-0002 的全部包络数值(含 R_fail_real=0)产生于 #36 校准,其编排脚本 `l2_calibration_predict.sh` 不传该 flag(TTA on),与协议声明一致——**#38 判定所对照的基准是干净的**,不存在「两侧都错位」的复合问题。

## 3. Margin 量化

### 3.1 #38 的判定结构:零预算阶跃,不是连续 margin

判定规则(issue #38 resolution):每子挑战 R_fail_synth.point ≤ R_fail_real.point,而真实校准 R_fail.point = **0**(ADR-0002 五挑战全部 0/900、0/42、0/720、0/171、0/72)。即通过线是 **k=0 的离散零失败预算**:

- 实测:P1 各挑战 0/20(GLI/MEN/METS)、0/14(PED/SSA);P3 各挑战 0/80(GLI/MEN/METS)、0/56(PED/SSA);细分 input_fail=run_fail=hier_viol=0(逐挑战数字全文见文末附录)。
- 「margin」的准确刻画:**当前余量 = 0 例失败;任何 1 例新增 fail 都会把该子挑战翻成不 PASS(#38 语境)/ undecided(终验语境)**。不存在「差多少百分点」的连续余量——翻转问题因此化归为「TTA 差异能否制造哪怕 1 例这三类失败」(§4:不能)。
- Wilson 95% 上界(P1 ≤ 0.2153、P3 ≤ 0.0642;逐挑战值见文末附录)只是伴随诊断量,不作 gate(issue #38 resolution、`scripts/nnunet_l2_synthetic_domain_eval.py:615-636`)。

### 3.2 观察项(非判定量):空 pred 计数

| 模式 | GLI | MEN | METS | PED | SSA |
|---|---|---|---|---|---|
| P1 | 0 | 0 | **2/20** | 0 | 0 |
| P3 | 0 | **2/80** | **42/80** | 0 | 0 |

(来源:#38 判定报告原表,逐挑战转录于文末附录。)空 pred 在 #38 与 ADR-0004 §6 的语义里均为「测量结果而非失败」;issue #38 resolution 明言「不影响输入/层级契约判定,留作终验伴随监控的重点观察对象」。这是 TTA 唯一可能触碰的读数(§4.4)。

### 3.3 范围澄清:TOST / 分布对齐不属 #38

Issue #38 报告不含 GT 对照、不含 TOST(报告只有 R_fail 与空 pred)。TOST 边界(E_r,vol / E_r,centroid)是 ADR-0004 为 #58 终验预注册的判定量,其执行侧(`nnunet_l2_final_acceptance.py:445-471` 生成的 predict 脚本与 `p1_predict_all.sh`)**均不带该 flag、TTA on**,与 ADR-0002 校准同状态——**终验 TOST 判定不受本 bug 影响**。#58 的 L2 报告数字在 sugon 受控存储,本地不可达,但与本 bug 无关,无需为它拉取。

## 4. 翻转分析:判定量对 TTA 的结构性不敏感

### 4.1 input_fail — 不可能受影响

`scripts/nnunet_l2_synthetic_domain_eval.py:545-561`:检查的是**输入文件**的几何契约(四模态 size/spacing/origin 一致 + 1mm isotropic;读不了文件则 input_fail=run_fail=1)。这发生在任何推理之前,与 TTA 零耦合。#38 的输入文件由生成侧固定,不随仪器推理配置变化。

### 4.2 run_fail — 不可能因「TTA off」新增

`scripts/nnunet_l2_synthetic_domain_eval.py:563-570`:run_fail 仅在预测 NIfTI 缺失/不可读时触发。镜像 TTA 是推理内部对同一输入多做 7 次翻转前向再平均(Isensee et al. 2019:「Test time augmentation by mirroring along all axes is applied」,[arXiv:1904.08128](https://arxiv.org/abs/1904.08128));TTA off 是其严格计算子集,不引入任何新的失败路径(崩溃/OOM 风险单调上升于计算量,而 bug 方向是「off」=更轻)。已观测 0 失败,不可能因「本应 off 却 on」或反之而冒出新失败。

### 4.3 hier_viol — 结构上恒不触发

`scripts/nnunet_l2_synthetic_domain_eval.py:572-591`:层级检查把预测标签体按值派生 WT=isin(1,2,3)、TC=isin(1,3)、ET=(==3) 后检查 ET⊆TC⊆WT;值域检查标签 ⊆ {0,1,2,3}。nnUNetv2_predict 的输出是逐体素 argmax 的整数标签(softmax 聚合后取最大类,TTA on/off 只改变聚合的 logits/概率,不改变输出格式与候选类)。对任何 {0,1,2,3} 标签体:标签 3 的体素必在 TC(含 3)与 WT(含 3)内,标签 1 必在 WT(含 1)内——**层级恒成立**;值域恒合法。故 hier_viol 对该仪器输出是恒假事件,与 TTA 无关。(脚本 578-582 行的索引写法本身有方向瑕疵,但因结构恒真,无论哪条分支都不可触发。)

### 4.4 唯一可触面:空 pred 计数 — 不足以翻转,且不是判定量

空 pred = 仪器 argmax 全 0(整例未检出)。TTA 的镜像平均对边界体素概率有小幅系统性平滑,对**极小病灶**(METS 多灶小转移)的检出可能在个位数病例上翻转。量级旁证:

- nnU-Net 官方对 `--disable_tta` 的定性:「at the expense of a **slightly lower** segmentation quality … (8x speedup)」([documentation/inference_example_Prostate.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/inference_example_Prostate.md),另见 [issue #1326](https://github.com/MIC-DKFZ/nnUNet/issues/1326) 引述);
- Wang et al. 2019(3D 脑肿瘤分割,TTA vs 单预测基线):Dice 92.03→92.79、92.00→92.88、91.98→93.05,即 **≈ +0.5–1.0 个 Dice 点**,ASSD 同步小幅改善([PMC6783308](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783308/) Table 1);
- 本仓 ADR-0002 旁证:同一仪器在**真实**数据上 TTA on 的测量误差包络已达 E_r,vol ≈ 0.28–1.65、E_r,centroid ≈ 3.9–35.1 mm(五挑战全表),远宽于 TTA 量级——TTA 引入的读数差异淹没在仪器自身的测量分辨率内,这正是 ADR-0002 §2 把 TTA on 冻结为配置、而 margin 用包络而非逐体素一致来定的原因。

即便 TTA on 重跑使 METS 空 pred 从 42/80 移动若干例,#38 的判定输出(R_fail 全 0 → 双模式五挑战 PASS)不变;该计数在报告中的角色是终验伴随监控的观察对象,而终验本身在 TTA on 下执行(§3.3),口径自洽。

### 4.5 img2img 基线对比(P3 模式)

P3 的判定量同样是 R_fail(0/352 全过)+ 空 pred 观察项,不含任何 TTA 敏感的连续判据。P3 的存在意义(img2img 零训练基线的合成域适用性前置证据,issue #38)不依赖具体读数点位。

## 5. 结论与处置建议

**档位:不可能翻转。**

- #38 的全部十个子挑战判定(P1/P3 × 五挑战 PASS)建立在 R_fail=0 上,该量的三个组件对 TTA on/off 结构性不敏感(§4.1–4.3),在 TTA 实际状态未知(§2.3 三分支)的前提下依然全部成立;
- TTA 量级(文献 ≈ +0.5–1.0 Dice 点;本仓包络旁证其远小于仪器测量分辨率)只够触碰到非判定量的空 pred 计数(§4.4);
- 对照基准(#36 校准 → ADR-0002 包络)与下游 TOST 判定链(#58 终验)分别在 TTA on 下产生,均不受本 bug 影响(§2.4、§3.3);
- 因此**不需要因 TTA 问题重跑 #38**,也不需要为结论去拉取 §1.2 的 sugon 产物。

两条不影响上述结论的记录性建议:

1. **报告保真勘误**:#38 报告与 resolution 声明「仪器 = TTA on」,与涉事调用点的代码事实不符(实际形态为 §2.3 三分支之一)。若后续有人引用 #38 读数,宜在相关 issue 上补一条勘误注记,避免「读数产生于 TTA on」的错误引用扩散。改代码本身(修掉 3 处 token)是独立决策,不在本票范围。
2. **可分辨性**:如需把 §2.3 三分支收敛为单一事实,sugon `predict.log`(`/root/private_data/l2-synth-eval/{p1,p3}_predictions/<CH>/predict.log`)的命令行回显即可一锤定音;纯取证,非结论必需。

## 附录:#38 判定报告出清存档摘录

两份判定报告(随 PR [#50](https://github.com/ACautomata/NV-Generate-CTMR/pull/50)、commit `797bdf5` 入库)已于 #145 遗留目录出清时从仓库删除;判定概要与空 pred 观察项此前已分别沉淀于 §1.1 与 §3.2,此处补齐笔记未曾覆盖的部分:**逐挑战 Wilson 上界全表**(此前仅存最大值摘要)与**两份报告的「方向说明」段**。原文全文锚在 git 历史:`git log --diff-filter=D --oneline -- eval/l2-synth-domain/` 定位出清提交,`git show <出清提交>^:eval/l2-synth-domain/report_<p>.md` 取回。

### 十个子挑战判定数字(报告原表逐列转录)

| 模式 | 挑战 | 样本数 | R_fail_synth (k/n) | Wilson 95% 上界 | R_fail_real | 空 pred | 判定 |
|---|---|---|---|---|---|---|---|
| P1 直出 | GLI | 20 | 0/20 | 0.1611 | 0/900 | 0 | PASS |
| P1 直出 | MEN | 20 | 0/20 | 0.1611 | 0/720 | 0 | PASS |
| P1 直出 | METS | 20 | 0/20 | 0.1611 | 0/171 | 2 | PASS |
| P1 直出 | PED | 14 | 0/14 | 0.2153 | 0/72 | 0 | PASS |
| P1 直出 | SSA | 14 | 0/14 | 0.2153 | 0/42 | 0 | PASS |
| P3 img2img | GLI | 80 | 0/80 | 0.0458 | 0/900 | 0 | PASS |
| P3 img2img | MEN | 80 | 0/80 | 0.0458 | 0/720 | 2 | PASS |
| P3 img2img | METS | 80 | 0/80 | 0.0458 | 0/171 | 42 | PASS |
| P3 img2img | PED | 56 | 0/56 | 0.0642 | 0/72 | 0 | PASS |
| P3 img2img | SSA | 56 | 0/56 | 0.0642 | 0/42 | 0 | PASS |

R_fail 细分 input_fail=run_fail=hier_viol 全 0(十个子挑战逐一);两份报告总体判定均为 **PASS**。

### 方向说明摘录

- **P1 报告**:P2 方向前置证据缺位已知情接受——掩码 ControlNet 训练前不存在 v1 可产样本,P2 依赖终验伴随监控兜底;P1 直出样本保留跨模态不一致性(独立采样),其 R_fail 只覆盖仪器对合成输入的运行/层级契约,不构成对 P2 配方产出的预测。
- **P3 报告**:img2img 零训练基线(RF 插值 strength=0.9,无 ControlNet),每轮一个真实模态作锚、其余三模态以该锚为 src 生成,12 有序模态对全覆盖;真实锚通道直接用原始数据(重采样对齐),生成通道为 v1 DM img2img 输出。跨模态自洽性强于 P1 但弱于待训 P3 ControlNet,仅作合成域适用性前置证据。

## 来源索引

| 事实 | 来源 |
|---|---|
| #38 P1/P3 判定数字 | 原 `eval/l2-synth-domain/report_{p1,p3}.md`,随 #145 出清;现存读数与原文取回方式见本文附录 |
| #38 判定规则 / Wilson / 空 pred 语义 | [issue #38 resolution](https://github.com/ACautomata/NV-Generate-CTMR/issues/38)(2026-08-20T11:56:01Z) |
| R_fail 三组件定义 | `scripts/nnunet_l2_synthetic_domain_eval.py:536-636` |
| 真实校准 R_fail=0 与包络数值 | `docs/adr/0002-l2-instrument-calibration-envelopes.md:16-68` |
| 校准推理配置 TTA on | `docs/calibration/l2-instrument-calibration-protocol.md:30`、`scripts/l2_calibration_predict.sh`(头注释与命令) |
| store_true 语义警示(终验侧) | `scripts/nnunet_l2_final_acceptance.py:433-471`、`docs/calibration/l2-final-acceptance-protocol.md:20` |
| nnUNetv2 2.8.1 argparse 定义 | [v2.8.1 predict_from_raw_data.py](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/inference/predict_from_raw_data.py)(`store_true` + `parse_args()`);本地 argparse 实测 exit 2 |
| TTA「slightly lower」定性 | [nnUNet inference_example_Prostate.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/inference_example_Prostate.md)、[issue #1326](https://github.com/MIC-DKFZ/nnUNet/issues/1326) |
| 镜像 TTA 机制 | Isensee et al. 2019,[arXiv:1904.08128](https://arxiv.org/abs/1904.08128)(「Test time augmentation by mirroring along all axes is applied」) |
| TTA 增益 ≈ +0.5–1.0 Dice 点 | Wang et al. 2019,[PMC6783308](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783308/) Table 1(3D 脑肿瘤,92.03→92.79 等) |
| #58 终验 L2 产物仅在受控存储 | PR [#72](https://github.com/ACautomata/NV-Generate-CTMR/pull/72) 描述;`scripts/brats_p1_l2_html_nifti.py` 执行侧 |
