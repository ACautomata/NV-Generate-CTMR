# ADR-0005：P1 带肿瘤候选训练配方执行决策（回放队列与开发集选择）

- **状态**：已接受（2026-08-22；在任何 P1 训练 epoch 启动之前预注册）
- **范围**：#57 P1 image-only 带肿瘤候选训练中，spec #51 decision 6 留给执行期的自由度：MR-RATE 回放队列构成、网格上界、开发集轻验收口径与预记录早停规则

## 背景

[Spec #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) decision 6 与 [issue #10 钉板](https://github.com/ACautomata/NV-Generate-CTMR/issues/10) 把 P1 配方钉死在：全参续训、VAE 冻结、复用 ckpt `scale_factor`、`lr=2e-6`、batch=1、≤100 epochs、RF uniform（scale 1.4）+ PolynomialLR(2.0)、L1、`augment_modality_label=0.1`、BraTS : MR-RATE = 1:1 数据 list 级回放。回放队列的具体构成（MR-RATE 8.1 TB 主仓的哪个变体、哪些模态、多少研究）、训练网格上界与开发集轻验收的具体口径，spec 留给执行期——本 ADR 在训练启动前注册这些决策。

## 决定

1. **回放变体 = 主仓 native-space defaced whole-brain**。v1 基模训练数据（`data/README.md` §3.4）是 MR-RATE batches 0–27 的 whole-brain 与 skull-stripped 两半；回放取 whole-brain 半（原生体数据本身，免掩码处理），沿用其原始模态 token `mri_t1=9 / mri_t2=10 / mri_flair=11`——正是 #10 §2 预期的"批内约一半样本仍带 MR-RATE 原标签（9/10/11…）"。不使用 gauss 已缓存的 MR-RATE-**atlas** 衍生仓（MNI 配准变体不在 v1 训练分布内）。
2. **队列构成**（`scripts/brats_p1_replay_prep.py select`，确定性、可审计）：
   - 脑研究过滤：研究在 MR-RATE-atlas 注册集中（atlas 集覆盖脑队列；v1 73,516 受试者与 atlas 研究数同量级）；
   - 病人级 MR-RATE `train` split（其 val/test 不触碰）；
   - 序列 = T1w/T2w/FLAIR、`is_derived/is_localizer/is_subtraction` 全 false；
   - 原生逐轴尺寸 ∈ [32, 320] → 最近 128 倍数网格逐轴 ≤256（latent ≤64），与 BraTS 64×64×32 同驻 batch=1 训练；
   - 排序 `sha256('<study>/<series>')` 升序取前 7404（与 BraTS p1_train 7404 严格 1:1）。SWI/MRA 不入回放：其 token 行在 P1 内收不到梯度、不发生遗忘，共享主干的防遗忘锚定由纳入的三模态承担。
3. **契约接线**：run contract 的 data-list side 集合扩为 `train/dev/replay`；`replay` 仅 P1 可用，其 `(sub, case)` 必须**不在** BraTS split manifest 内（按对偶与裸 case id 双重校验，防 BraTS 病例借 replay 标签回流训练侧）；P2/P3 init 带 replay 即违约（spec：回放仅全参续训）。
4. **开发集轻验收**（`scripts/brats_p1_dev_eval.py`，独立 GPU 侧车，每 5 epoch）：
   - 固定 16 例 dev 队列（GLI 4 / SSA 2 / MEN 4 / METS 3 / PED 3，`sha256('<sub>/<case>')` 升序取配额），固定 per-(case, modality) 种子、cfg=10、30 步、逐病例伴侣 spacing——每评估点 64 个固定四模态样例；
   - FID 趋势 = 逐目标模态 2.5D RadImageNet FID（三正交面逐面 + 均值），预处理 = percentile 0–99.5→[0,1]、RAS、1 mm、zero-pad 240×240×160（与 L1 协议字串同族），真实参照侧 = 全部 dev 侧真实体（一次构建缓存）；
   - L2 趋势 = 冻结仪器（ADR-0003 链、SSA 专属 plans）对生成伪四模态的 WT/TC/ET 体积中位数与 input/run/hier 失败计数——只记趋势，不做 TOST。
5. **预记录早停与选择规则**：`m(N)` = 四目标模态平面均值 FID 的均值；`N ≥ 30` 且连续 3 个评估点无新最优即停（写 `<ckpt_dir>/.early_stop`，训练进程 epoch 边界轮询）；候选 = 全程 `argmin m` 的 epoch；硬上限 = 100 epochs。规则文本先于训练写入 run 目录（`early_stop_rule.json`），dev 证据只含 dev 侧 (sub, case)。
6. **scale_factor 与精度**：训练 `scale_factor` 恒取基模 ckpt 值，首 batch 重算 `1/std(z)` 仅作 sanity（相对差 >0.5 拒绝启动）；DCU 上 bf16 autocast 默认、fp32 兜底保留；DDP 经 torchrun/RCCL。

## 后果

- `scripts/brats_p1_finetune.py`、`scripts/brats_p1_dev_eval.py`、`scripts/brats_p1_replay_prep.py`、`scripts/brats_p1_launch_train.sh`、`configs/config_brats_p1_train.json` 按本 ADR 实现；selftest 各自可独立复跑；
- 回放队列文件、MR-RATE 体数据与 embedding 全部留在受控存储（gauss/sugon private_data），入 Git 的只有脚本与不含 subject/study ID 的规则；
- 若需改回放比例、队列构成或早停规则：新 ADR 取代，既有候选与新候选不得混记为同一实验结论。
