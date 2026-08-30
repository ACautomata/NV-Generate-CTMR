# ADR-0018:孤儿库退役——仪器供给与数据获取管线归 git 历史

- **状态**:已接受(2026-08-30;架构评审 grilling 会话产出,先于实现预注册)
- **范围**:`infrastructure/provisioning/` 五件与 `infrastructure/dataio/` 零调用方七件的退役处置、`ctmr data` CLI 门面的撤销、零调用方白名单守卫的设立。
- **来源**:2026-08-29 架构评审候选 8 的 grilling 会话;孤儿判定事实(全仓零调用方 grep、docstring 承诺链、仪器供给冻结状态)随退役 PR 落盘。

## 背景

- 退役清单内十一个模块的 docstring 均承诺「a later CLI slice takes over」——即 ADR-0015 §3 钉过的 `ctmr data prep/encode/download` 动词面。M0–M7 迁移收官后该动词面从未落地,十一件全部零生产调用方;但每件出生即带绿色测试,收敛门禁全绿给出**假活性信号**——没有任何 guard 区分「活库」与「带测试的退役件」。
- 仪器供给已完成并冻结:SSA 派生 `nnUNetPlans_SSA_bs16_v1` 与 `nnUNetTrainer250Epochs` 已安装入 nnunetv2,冻结审计 verdict `all_passed`(sha 9121e8ac…,受控存储);`ddp_preflight` 硬编码 `/root/private_data` 与 8-DCU 契约,属一次性 DCU 时代产物,本地永久不可执行。
- 数据获取走 `deploy/data/` 运维配方(shell),不经 Python 包;`downloads.py`(204 行)的服务对象不存在。

## 决定

1. **退役清单**:`provisioning/{dataset_prep,ddp_preflight,install_trainer,plan_variant,trainer_250_epochs}.py`;`dataio/{downloads,find_masks,quality_check,plotting,transforms,sample_mask,mask_postprocess}.py`;随件测试同撤。`sample_mask` 内的 `# noqa: F401` 纯转发 shim 随文件消亡。
2. **保留活件**:`dataio/{list_assembly,augmentation,morphology}.py`(morphology 经 augmentation 存活)。
3. **复现锚**:git 历史 + 受控 audit 记录——与 #38 轻复制族、scripts/ 退役同一教义(CONTEXT.md「历史运行器」词条)。本 ADR 显式声明:**仪器供给流程与数据获取管线不可原样重放**;未来仪器重训或增量数据属新裁决,届时按需从 git 历史复活所需件,不从退役件再抄。
4. **CLI 门面**:`ctmr data` 门自 `NOT_MIGRATED_FAMILIES` 移除(动词面与活代码重新一致,`test_cli_entry` 相应改钉);`experiment` 门保留(ADR-0015 决定 11 明示推迟 ExperimentRecord 接口,豁免)。
5. **守卫**:`test_endstate_guards` 旁增「零调用方模块白名单」断言——清账后白名单为空;未来任何零调用方模块必须显式登记方可入仓,孤儿状态从静默变声明。

## 取代关系

- 撤销 ADR-0015 §3 所钉 `ctmr data` 动词面的未兑现部分(不再规划该动词面);ADR-0015 其余全部内容(installable package、统一 CLI、CheckpointRepository、deploy 运维面、测试面)不动。
- 不触 ADR-0013(测试范式)、ADR-0016(生成实体)。

## 实现附记(#230,2026-08-31)

决定 1 清单内的十二个文件(决定 1 列名口径;标题「11 件」以 shim 样本不单独计件)全部退役删除,随件测试同撤。退役 PR 落盘的孤儿判定事实:

- **零调用方 grep**:决定 1 清单内各件在 src 全树(import 语句与 VERBS 注册表、console/`python -m` 入口、deploy shell 配方)零生产引用;`mask_postprocess` 仅被同清单的 `sample_mask` shim 引用,随件消亡。唯一跨清单引用是两个 L2 测试借道 provisioning 件取常量/安装源(`DIR_SUFFIX`、trainer shadow 源),分别改接活代码 `OFFICIAL_TREE_SUFFIX` 与自持合成 shadow 源。
- **docstring 承诺链**:清单内各件 docstring 所指「a later CLI slice takes over」动词面随 M0–M7 收官从未落地,与背景节判定一致。
- **仪器供给冻结状态**:维持(已安装入 nnunetv2、受控审计 verdict `all_passed`);`install_trainer` 等安装器退役后,重训需求按决定 3 从 git 历史复活。

守卫首扫偏离申报:正探针同时发现三个 ADR 未列名的既有零调用方模块——`distribution/calibration_prep`(#36 校准集组装,协议已冻结)、`distribution/freeze_audit`(#37 冻结工件审计,verdict 已入受控存储)、`quantitative/fid_2d5`(一次性 2.5D FID 计算器,dev-trend 实际引用 `quantitative.fid`)。三者同为历史运行件,但不在决定 1 的封闭清单内,本票无权扩大退役范围;按决定 5「孤儿状态从静默变声明」显式登记白名单并逐件注明身份,处置(退役或复活挂门)留待独立裁决。验收标准「清账后为空」据此修订为:**清单内件清账后,白名单仅余显式声明的既有历史运行件**;白名单设反向断言(条目必须是当前孤儿),清单内账清空后白名单不得再增长,只可随裁决清偿。
