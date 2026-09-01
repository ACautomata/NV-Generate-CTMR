# 序列② T4:clip=True 编码配方合入 + 训练 embedding 重编码(sugon 作业)

**总体结论**:训练编码管线 mri 归一化旗标 clip=False→True 合入并单测钉板(>1.0 外推输入经工厂后有界);sugon 侧全量 7404 条训练 embedding 以新旗标重编码至 `embeddings_cliptrue/`(7404/7404、missing 0),md5 清单随工件落盘;域内自评 MAE 抽检(200 例)双对照臂读数与作业 C 双链锚同档(直通臂 0.0057≈0.0062、工件臂 0.0782≈0.0823)——配方落地、工件同构、环境无漂移。冻结 VAE 只读,评估链与判定线零改动。 | **记录落盘日**: 2026-09-01

## 1. 目的与关联 issue

作业 C(#208)实测:clip=False 归一化把训练 t1c 顶部 ~0.5% 体素外推到 >1.0,落在冻结 autoencoder_v1 重建域之外(外推层自评 MAE 0.8673、域内 0.0062,外推伴生瘤内负值伪影)。父票 #247 收编裁决 #1 采纳项:训练编码归一化 clip=False→True,消除实测失真源;随之全量重编码训练 embedding 供 T7 重训使用(clip=True 世界里旧 embedding 不可复用)。定性:「消除已实测失真源;对 L2 主指标疗效不单独归因」。

- 关联 issue:#251(本票)、#247(序列②父票)、#208(作业 C,读数锚来源)、#250/#249(T3/T2,同序列先例)
- 判定状态关联:variant=diagnostic,不产生任何验收判定;P1 L2 仍为 FAIL

## 2. 代码改动(本地)

- `ctmr/infrastructure/maisi_engine/instance_definition.py`:mri 臂 `ScaleIntensityRangePercentilesd(clip=False→True)`——vendored freeze-side 模块的唯一一处记录偏差(模块头注释写明 #251 依据);ct 臂不动
- 单测 `tests/infrastructure/maisi_engine/test_intensity_transform_factory.py`(4 枚):合成体积(95% 0.0 / 4.8% 1.0 / 0.2% 10.0,99.5p 锚点落主体段)→ mri 工厂输出有界(max=1.0,尾部截到 1.0 而非重缩放);旗标 `.scaler.clip is True`;ct 臂 clip=True 不回归
- 重编码作业模块 `ctmr/application/generation/modality_label/reencode.py`(8 枚单测):`EmbeddingManifest`(遍历新树→md5/bytes/shape 登记 `manifest.jsonl` + 与训练 list 对账 missing)、`ReencodeSelfCheck`(均匀步长抽样 + 双对照臂读数 + 聚合)、`ReencodeReport`(variant=diagnostic 报告);wiring `modality_label_reencode_runtime()` 注入 vendored 编码链与引擎(ADR-0019 §2)
- 作业脚本 `deploy/jobs/run_embedding_reencode_t4.sh`:沿作业 D 登记路径布局的覆写 env json + 前置检查
- 全量 pytest 907 passed / 2 skipped(gpu 预期);ruff check+format 全绿

## 3. sugon 执行环境(2026-09-01 实例,作业 D 的部署树已随系统盘易失)

实例重建事实与应对(全部如实登记):

- **实例**:`crdnotebook-2094451859199361026-wang9691-93101`(新分配,作业 D 的 `/root/nv-phase-60` 部署树与 nv-dcu-smoke/manifold 目录已消失;系统盘易失属已知限制)
- **部署树重建**:`/root/nv-phase-t4/`(易失盘)src+configs 由本地 worktree tar 上传;`instance_definition` clip=True 版 md5 与本地一致
- **数据树**(`/root/private_data/ctmr/` 持久盘,重组后布局):训练 list `/root/private_data/ctmr/data/phase/lists/p1_image_only.json` **7404 条**(4 模态 skull-stripped);旧 embedding 树 `data/phase/embeddings/`(8465 个 _emb.nii.gz,真实文件,不动)
- **raw 断链修复**:`data/phase/raw/` 下 10585 个符号链接指向已消失的 `/root/private_data/brats2023_nnunet/`(重组前布局)。按训练 list 全量 7404 条重建链接镜像树 `data/phase/raw_relinked/`(→ `ctmr/data/nnunet_raw/Dataset50X/imagesTr/<case>_<channel>.nii.gz`,通道映射 t1n/t1c/t2w/t2f=0000..0003,挑战映射 GLI..PED=501..505;linked 7404、missing_target 0)。旧 raw 树不动
- **VAE 补回**:当前实例唯一 VAE 是断链;从本地副本(`~/Documents/hope/models/`,与作业 D 登记冻结 canonical 逐字节一致)上传 `/root/private_data/ctmr/models/autoencoder_v1.pt`,md5 `917cfb1e49631c8a713e3bb7c758fbca` 复验一致。冻结只读
- **python 环境重建**(系统盘易失,按 ENVIRONMENT_LOCK + 作业 C 事实重建):torch-dcu 2.9.0(实例自带)、numpy 1.26.4、monai 1.6.0、nibabel 5.4.2、scipy 1.13.1——全部 `--no-deps` 安装,装后 `import torch` 复验;与作业 C 环境逐项一致

### 执行中实例二次重分配(如实登记)

全量编码进行中(单卡 558 + 首轮 4 卡至 2030/7404),平台把实例再次重分配(`…93101`→`…59970`,2026-09-01 ~01:12):系统盘二次易失(部署树、pip 依赖、bashrc 注入全丢),在跑 worker 全部被杀。**持久盘数据无损**——已编码 2030 条 embedding、raw_relinked、lists、VAE 全部保留。应对:按同法重建 python 环境(依赖逐项同上表)与部署树(clip=True 文件 md5 `5cd8c3c7…` 与本地复验一致),以 `SHARDS=4` + `setsid nohup`(脱离 ssh 会话,防连接中断误杀)续跑;编码链跳过已存在文件,已编码 2030 条零浪费。

## 4. 关键读数与产物路径

产物(sugon 工件区,不入 git):冒烟 `/root/private_data/ctmr/runs/p1/reencode_t4_smoke/embedding_reencode_report.{json,md}`,schema `embedding-reencode/1`;全量同目录 `reencode_t4/`。

### 冒烟(3 例:t1c/t1n/t2w 各一,全链路)

编码成功 3/3(z=[1,4,64,64,32] fp32,affine 0.9375/0.9375/1.2109375 与作业 D 登记 spacing 一致);manifest 3 entries、missing 0。

**双对照臂条件 MAE(median;锚为作业 C 第五轮读数,参照值非判定线)**:

| 臂 | 层(目标) | 冒烟读数 | 作业 C 锚 | 判读 |
|---|---:|---:|---|
| 直通臂(现场 encode,无滑窗) | [0,1] 域内 → clip 后输入 | **0.0072** | 0.0062(clip=True 臂) | 同档——VAE/归一化/resize 环境与作业 C 一致,漂移排除 |
| 直通臂 | >1.0 外推 → clip 后输入 | **0.0558** | 0.0559(clip=True 臂) | 几乎一致 |
| 工件臂(滑窗链,落盘工件 decode) | [0,1] 域内 → clip 后输入 | **0.0759** | 0.0823(noclip 臂对旧工件的同链自评) | 同档——新工件与旧工件同构,仅 clip 差异 |
| 工件臂 | >1.0 外推 → clip 后输入 | **0.6074** | 0.8673(noclip 臂外推层自评) | 显著下降——截尾把外推层重建失真压掉,方向与作业 C 裁决一致 |

分层轴读数:noclip 世界外推高度 median 1.7562(q05 1.3855, q95 2.4827);raw 99.5p 锚点 median 640;n_over≈0.4%(与作业 C「top ~0.5% 外推」一致)。

**口径注记(重要)**:作业 C 的 0.0062 是直通链(现场 `encode_stage_2_inputs`,无滑窗)读数;训练工件走 `create_training_data` 的 SlidingWindowInferer(gaussian,roi 320³>grid 256³)链,两条链数值不同——旧工件同链自评即 0.0823。故抽检设双对照臂:直通臂复现 0.006 量级证明环境一致,工件臂对标 0.0823 证明工件同构。

### 全量重编码(7404 条,已完成)

- 执行:4 卡分片(`SHARDS=4`,每片单卡 worker `--encode-only`,cuda:0..3)+ 收尾单卡(`--skip-encode`:manifest 对账 + 抽检 200 + 报告);跨两次实例执行(单卡 → 首轮 4 卡 → 续跑 4 卡),编码链跳过已存在文件,全程零浪费
- 条目数 / md5 清单:`embeddings_cliptrue/manifest.jsonl` **7404 行 = 训练 list 全量,missing 0**;manifest md5 `a4356adcdd14804c218796684cb4a550`(1.6 MB,含逐文件 path/bytes/shape/md5);新树 14G
- 抽检报告:`reencode_t4/embedding_reencode_report.json` md5 `5f90ac93129c57c086f6cc21b973ba59`,schema `embedding-reencode/1`,run_id `p1-20260822T131947Z`(读自 P1 终验 binding)

**抽检 200 例(均匀步长)双对照臂条件 MAE(median;锚为作业 C 第五轮读数,参照值非判定线)**:

| 臂 | 层(目标) | 全量读数(median,q05–q95) | 作业 C 锚 | 判读 |
|---|---:|---:|---|---|
| 直通臂(现场 encode,无滑窗) | [0,1] 域内 → clip 后输入 | **0.0057**(0.0044–0.0082) | 0.0062 | 同档——跨实例重建后环境仍与作业 C 一致 |
| 直通臂 | >1.0 外推 → clip 后输入 | **0.0457**(0.0314–0.0680) | 0.0559 | 同档 |
| 工件臂(滑窗链,落盘工件 decode) | [0,1] 域内 → clip 后输入 | **0.0782**(q05 0.0052–q95 0.1007) | 0.0823(旧工件同链自评) | 同档——新工件与旧工件同构,仅 clip 差异 |
| 工件臂 | >1.0 外推 → clip 后输入 | **0.5318**(0.0388–0.7587) | 0.8673(noclip 外推层自评) | 显著下降——截尾把外推层重建失真压掉,方向与作业 C 裁决一致 |

分层轴读数(200 例):noclip 世界外推高度 median 1.7152(q05 1.2189, q95 3.0298);raw 99.5p 锚点 median 644——与冒烟(1.7562 / 640)一致。

## 5. 结论与后续动作

- [x] clip=True 配方合入 + transform 工厂单测(>1.0 输入输出有界)
- [x] 全量训练 embedding 重编码完成(新落盘路径与 md5 清单登记)——见 §4 全量段
- [x] 域内自评 MAE 抽检读数与作业 C 双链口径同档(直通 0.006 量级、工件 0.08 量级)
- [x] 实验记录落盘(本文件)
- [x] 冻结 VAE 只读;评估链与判定线零改动(代码零触碰 L1/L2/L3 面;AD-0002/0004)

后续动作:

- T7 重训消费 `embeddings_cliptrue/`(env json `embedding_base_dir` 改指新根即可;训练循环读 `<case>_emb.nii.gz` 相对路径不变),审计面 = manifest.jsonl 的 md5
- 旧 embedding 树与旧 raw 树保留为回退锚,暂不清理
- KNOWN DEBT:重编码抽检诊断种子占槽 base+400..407(沿作业 C/D 先例的 bandless 块),待 challenge_registry follow-up 一并登记
- 作业 D 登记的 `brats2023_rflow_phase` 布局在当前实例不存在;本作业以当前实例实测布局执行并登记(§3)
