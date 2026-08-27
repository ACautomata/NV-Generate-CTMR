"""Issue #38 L2 仪器合成域适用性评估——v1 DM 直出样本生成与仪器评估管线（自 scripts/nnunet_l2_synthetic_domain_eval 平移，ticket #140；旧 sugon 版运行器 scripts/l2_synth_domain_sugon.py 已由 deploy/jobs/run_l2_synth_domain_eval.sh 配方取代退役）。

在 v1 DM 基模直出样本（不依赖任何微调产物、零循环）上跑冻结的五子挑战
L2 测量仪器，测输入契约失败率与层级违反率（ET⊆TC⊆WT），对照真实校准
R_fail 给出合成域适用性判定。

两种生成策略：
  P1 式（独立采样）：按模态标签独立采样 ×4 组成伪四模态体，跨模态不自洽如实保留；
  P3 式（img2img）：每轮一个真实模态作锚，v1 DM 从该锚生成其余三模态，4 轮覆盖
  全部 12 有序模态对。

所有输出约束于 ``/root/private_data``；不向 Git 写入患者 ID 或 NIfTI 数据。

用法示例（在 sugon DCU 集群上执行）::

  # Step 1: 生成 P1 式独立采样样本
  python -m ctmr.application.acceptance.distribution.synthetic_domain generate \\
      --mode p1 \\
      --case-list /root/private_data/l2-synth-eval/case_lists/p1_cases.json \\
      --v1-model-dir /root/private_data/models \\
      --output-dir /root/private_data/l2-synth-eval/p1_samples

  # Step 2: 生成 P3 式 img2img 样本
  python -m ctmr.application.acceptance.distribution.synthetic_domain generate \\
      --mode p3 \\
      --case-list /root/private_data/l2-synth-eval/case_lists/p3_cases.json \\
      --v1-model-dir /root/private_data/models \\
      --output-dir /root/private_data/l2-synth-eval/p3_samples

  # Step 3: 组装 nnU-Net 输入（把 DM 输出重采样并格式化为仪器输入契约）
  python -m ctmr.application.acceptance.distribution.synthetic_domain prep-inputs \\
      --sample-dir /root/private_data/l2-synth-eval/p1_samples \\
      --nnunet-root /root/private_data/brats2023_nnunet \\
      --output-dir /root/private_data/l2-synth-eval/p1_nnunet_inputs

  # Step 4: 跑冻结仪器推理
  python -m ctmr.application.acceptance.distribution.synthetic_domain predict \\
      --input-dir /root/private_data/l2-synth-eval/p1_nnunet_inputs \\
      --results-root /root/private_data/nnUNet_results \\
      --output-dir /root/private_data/l2-synth-eval/p1_predictions

  # Step 5: 计算指标 + 与真实 R_fail 对照
  python -m ctmr.application.acceptance.distribution.synthetic_domain evaluate \\
      --sample-dir /root/private_data/l2-synth-eval/p1_samples \\
      --input-dir /root/private_data/l2-synth-eval/p1_nnunet_inputs \\
      --pred-dir /root/private_data/l2-synth-eval/p1_predictions \\
      --calibration-summary /root/private_data/l2-instrument-calibration/.../metrics \\
      --output-dir /root/private_data/l2-synth-eval/report_p1

冻结语义沿用 ADR-0002/ADR-0003：仪器权重与推理配置不可动。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ctmr.domain.grid import InstrumentGridAdapter
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand

# ── 常量 ──────────────────────────────────────────────────────────────────

PERSISTENT_ROOT = Path("/root/private_data")

# BraTS 模态 → v1 DM modality label（modality_mapping.json）
BRATS_MODALITY_LABELS = {
    "t1n": 9,  # mri_t1 (skull-stripped)
    "t1c": 17,  # mri_t1c (基模原始索引)
    "t2w": 10,  # mri_t2
    "t2f": 11,  # mri_flair
}

# nnU-Net 通道后缀 → BraTS 模态
NNUNET_CHANNELS = {
    "0000": "t1n",
    "0001": "t1c",
    "0002": "t2w",
    "0003": "t2f",
}

# BraTS 标签语义
BRATS_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

# v1 DM 输出参数（#10 P1 配方钉板）
V1_DM_OUTPUT_SIZE = (256, 256, 128)
V1_DM_SPACING = (0.94, 0.94, 1.36)  # mm

# 每个子挑战用于合成域评估的样本数
SAMPLES_PER_CHALLENGE = {
    "GLI": 20,
    "SSA": 14,  # SSA 全量仅 42 例，14 例已是 1/3
    "MEN": 20,
    "METS": 20,
    "PED": 14,  # PED 全量 68 例
}


# ── 数据类 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseEntry:
    """一个用于合成域评估的病例条目。"""

    case_id: str
    challenge: str
    source: str  # "dev" | "fold_val" | "holdout"
    images_dir: Path  # 含 <case>/{case}-{t1n,t1c,t2w,t2f,seg}.nii.gz


@dataclass(frozen=True)
class GenerationConfig:
    """v1 DM 生成配置。"""

    model_dir: Path
    output_dir: Path
    mode: str  # "p1" | "p3"
    seed: int = 42
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 2.0


@dataclass
class SynthCaseResult:
    """一个合成病例的生成结果。"""

    case_id: str
    challenge: str
    mode: str
    modalities_generated: dict[str, Path]  # modality_name → nifti path
    anchor_modality: str | None = None  # P3 模式下使用的锚模态


# ── 病例选择 ────────────────────────────────────────────────────────────


class CaseSelector:
    """从 nnU-Net 数据集中选择用于合成域评估的病例。

    优先从 dev 集选取（已在校准中使用、可直接比对），不足时从 fold_val 补充。
    不使用 holdout 集（#32 要求 20% 零接触）。
    """

    def __init__(self, nnunet_root: Path):
        self.nnunet_root = nnunet_root

    def select(self, challenge: str, n: int, seed: int = 42) -> list[CaseEntry]:
        manifest_path = self.nnunet_root / "splits" / "split_manifest_brats2023-rflow-v1.json"
        manifest = json.loads(manifest_path.read_text())
        challenge_data = manifest["challenges"][challenge]

        dev_cases = sorted(challenge_data["cases"]["dev"])
        fold_val_path = self.nnunet_root / "splits" / f"fold0_val_cases_{challenge}.txt"
        fold_val_cases = sorted(fold_val_path.read_text().split())

        # 优先 dev，不足补 fold_val
        selected_dev = dev_cases[:n]
        remaining = n - len(selected_dev)
        selected_fv = fold_val_cases[:remaining] if remaining > 0 else []

        raw_dir = self.nnunet_root / f"Dataset{self._dataset_id(challenge):03d}_BraTS2023{challenge}"
        entries = []
        for case in selected_dev:
            entries.append(CaseEntry(case, challenge, "dev", raw_dir))
        for case in selected_fv:
            entries.append(CaseEntry(case, challenge, "fold_val", raw_dir))
        return entries

    @staticmethod
    def _dataset_id(challenge: str) -> int:
        return {"GLI": 501, "SSA": 502, "MEN": 503, "METS": 504, "PED": 505}[challenge]


# ── v1 DM 样本生成 ──────────────────────────────────────────────────────


class V1DMSampler:
    """v1 DM 基模采样器。

    P1 式：按模态标签独立采样 ×4 组成伪四模态体。
    P3 式：每轮一个真实模态作锚，v1 DM 从锚生成其余三模态。

    依赖 ``scripts.diff_model_infer`` 的模型加载与推理逻辑。
    实际执行时需在 sugon DCU 环境中运行。
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate_p1(self, case: CaseEntry) -> list[SynthCaseResult]:
        """P1 式独立采样：每个模态标签独立生成，组成伪四模态体。

        返回 1 个 SynthCaseResult（含 4 个模态的生成路径）。
        跨模态不自洽如实保留——这正是 P1 评估的目的。
        """
        case_dir = self.config.output_dir / case.challenge / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        modality_paths = {}
        for mod_name, mod_label in BRATS_MODALITY_LABELS.items():
            output_path = case_dir / f"{mod_name}.nii.gz"
            if output_path.exists():
                modality_paths[mod_name] = output_path
                continue
            self._run_v1_dm_inference(
                modality_label=mod_label,
                output_path=output_path,
                seed=self.config.seed,
            )
            modality_paths[mod_name] = output_path

        return [
            SynthCaseResult(
                case_id=case.case_id,
                challenge=case.challenge,
                mode="p1",
                modalities_generated=modality_paths,
            )
        ]

    def generate_p3(self, case: CaseEntry) -> list[SynthCaseResult]:
        """P3 式 img2img 锚轮：每轮一个真实模态作锚，生成其余三模态。

        返回 4 个 SynthCaseResult（每个锚一轮），每轮含 3 个生成模态 + 1 个锚模态。
        """
        results = []
        real_modalities = self._load_real_modalities(case)

        for anchor_name in BRATS_MODALITY_LABELS:
            anchor_dir = self.config.output_dir / case.challenge / case.case_id / f"anchor_{anchor_name}"
            anchor_dir.mkdir(parents=True, exist_ok=True)

            modality_paths = {anchor_name: real_modalities[anchor_name]}
            target_modalities = [m for m in BRATS_MODALITY_LABELS if m != anchor_name]

            for target_name in target_modalities:
                output_path = anchor_dir / f"{target_name}_from_{anchor_name}.nii.gz"
                if output_path.exists():
                    modality_paths[target_name] = output_path
                    continue
                self._run_v1_dm_img2img(
                    anchor_image=real_modalities[anchor_name],
                    target_modality_label=BRATS_MODALITY_LABELS[target_name],
                    anchor_modality_label=BRATS_MODALITY_LABELS[anchor_name],
                    output_path=output_path,
                    seed=self.config.seed,
                )
                modality_paths[target_name] = output_path

            results.append(
                SynthCaseResult(
                    case_id=case.case_id,
                    challenge=case.challenge,
                    mode="p3",
                    modalities_generated=modality_paths,
                    anchor_modality=anchor_name,
                )
            )
        return results

    def _load_real_modalities(self, case: CaseEntry) -> dict[str, Path]:
        """加载真实 BraTS 四模态 NIfTI 路径。"""
        paths = {}
        for suffix, mod_name in NNUNET_CHANNELS.items():
            # BraTS 文件命名：<case>-<mod>.nii.gz
            candidate = case.images_dir / case.case_id / f"{case.case_id}-{mod_name}.nii.gz"
            if not candidate.exists():
                # 备选：nnU-Net 格式 <case>_<suffix>.nii.gz
                candidate = case.images_dir / f"{case.case_id}_{suffix}.nii.gz"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"real modality not found for {case.case_id}/{mod_name}: "
                    f"tried {case.images_dir / case.case_id / f'{case.case_id}-{mod_name}.nii.gz'}"
                )
            paths[mod_name] = candidate
        return paths

    def _run_v1_dm_inference(
        self,
        modality_label: int,
        output_path: Path,
        seed: int,
    ) -> None:
        """调用 v1 DM 无条件生成指定模态。

        实际执行时，本方法构造 config JSON 并调用 ``scripts.diff_model_infer``。
        在非 DCU 环境中，本方法写入占位脚本供手动执行。
        """
        # 构造推理配置 JSON
        infer_config = {
            "diffusion_unet_inference": {
                "dim": list(V1_DM_OUTPUT_SIZE),
                "spacing": list(V1_DM_SPACING),
                "top_region_index": [0, 1, 0, 0],
                "bottom_region_index": [0, 0, 1, 0],
                "random_seed": seed,
                "num_inference_steps": self.config.num_inference_steps,
                "modality": modality_label,
                "cfg_guidance_scale": self.config.cfg_guidance_scale,
            }
        }
        config_path = output_path.parent / f"config_mod{modality_label}.json"
        config_path.write_text(json.dumps(infer_config, indent=2) + "\n")

        # 调用 diff_model_infer
        env_config = str(self.config.model_dir / "environment_maisi_diff_model_rflow-mr-brain.json")
        model_config = str(config_path)
        model_def = str(self.config.model_dir / "config_maisi_diff_model_rflow-mr-brain.json")

        cmd_parts = [
            "python -m scripts.diff_model_infer",
            f"-e {env_config}",
            f"-c {model_config}",
            f"-t {model_def}",
            "-g 1",
        ]
        script_path = output_path.parent / f"run_mod{modality_label}.sh"
        script_path.write_text("#!/bin/bash\nset -euo pipefail\n" + " ".join(cmd_parts) + "\n")
        script_path.chmod(0o755)

        # 注意：实际执行需要在 sugon DCU 环境中手动运行这些脚本
        # 或通过 orchestrate 子命令批量提交
        print(f"[INFO] Generated inference script: {script_path}")

    def _run_v1_dm_img2img(
        self,
        anchor_image: Path,
        target_modality_label: int,
        anchor_modality_label: int,
        output_path: Path,
        seed: int,
    ) -> None:
        """调用 v1 DM img2img：从锚模态生成目标模态。

        使用 RF 插值起点（v1 DM 可先验非 t1c 方向），
        将真实锚模态编码到 latent space 后加噪→去噪生成目标模态。

        实际执行需在 sugon DCU 环境中运行。
        """
        script_content = f"""#!/bin/bash
set -euo pipefail
# P3 img2img: anchor={anchor_image.stem}, target_modality={target_modality_label}
# 通过 RF 插值从锚 latent 生成目标模态
# TODO: 需要实现 img2img 推理管线（当前 v1 DM 基模不直接支持 img2img）
# 建议方案：对锚模态加噪到特定 timestep，然后以目标模态标签为条件去噪
echo "[STUB] img2img inference not yet implemented for v1 DM base model"
echo "anchor: {anchor_image}"
echo "target_modality_label: {target_modality_label}"
echo "anchor_modality_label: {anchor_modality_label}"
echo "output: {output_path}"
"""
        script_path = output_path.parent / f"run_img2img_{target_modality_label}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        print(f"[INFO] Generated img2img script: {script_path}")


# ── nnU-Net 输入组装 ───────────────────────────────────────────────────


class InputPreparator:
    """把 v1 DM 输出重采样并格式化为 nnU-Net 仪器输入契约。

    v1 DM 输出：256×256×128 @ 0.94mm
    nnU-Net 期望：BraTS 原生 240×240×155 @ 1mm isotropic

    处理流程：
    1. 读取 DM 输出 NIfTI
    2. 重采样到 1mm isotropic
    3. 裁剪/填充到 240×240×155
    4. 保存为 nnU-Net 输入格式：<case>_0000.nii.gz .. <case>_0003.nii.gz
    """

    def prepare_case(
        self,
        case_id: str,
        challenge: str,
        modality_paths: dict[str, Path],
        output_dir: Path,
    ) -> Path:
        """组装一个病例的 nnU-Net 输入目录。

        ADR-0008 收编：几何（B-spline 重采样到 1mm + 居中 crop/pad 到 240×240×155）
        由 ctmr.domain.grid 的 continuum 适配器执行（修复了原 ``_crop_or_pad`` 把 xyz target
        直接作用于 zyx 数组的轴序 bug）。
        """
        case_input_dir = output_dir / challenge
        case_input_dir.mkdir(parents=True, exist_ok=True)

        for suffix, mod_name in NNUNET_CHANNELS.items():
            src_path = modality_paths[mod_name]
            dst_path = case_input_dir / f"{case_id}_{suffix}.nii.gz"
            if dst_path.exists():
                continue

            img = sitk.ReadImage(str(src_path))
            aligned = InstrumentGridAdapter.continuum().align(img)
            sitk.WriteImage(aligned, str(dst_path))

        return case_input_dir


# ── 仪器推理 ────────────────────────────────────────────────────────────


class InstrumentRunner:
    """运行冻结的 nnU-Net 仪器推理。

    命令构造已收编 ADR-0009:canonical 入口 ``python -m ctmr.instrument.predict``
    (镜像 TTA on 靠省略;SSA 派生 plans/config 在 spec 内),argv 与
    ``FrozenInstrumentCommand.build`` 输出逐一相等。
    """

    def __init__(self, results_root: Path):
        self.results_root = results_root

    def predict_challenge(
        self,
        challenge: str,
        input_dir: Path,
        output_dir: Path,
    ) -> Path:
        """对指定子挑战运行全部病例的仪器推理。"""
        pred_dir = output_dir / challenge
        pred_dir.mkdir(parents=True, exist_ok=True)

        # nnU-Net predict 期望目录结构：
        # input_dir/<case_id>_0000.nii.gz .. _0003.nii.gz
        # 输出到 pred_dir/<case_id>.nii.gz

        # 冻结推理配置由单一构造点给出(ADR-0009);日志经 shell 重定向落
        # pred_dir/predict.log(旧 `--verbose <path>` 是 store_true 后带值,
        # 与 #78 的 TTA token 同族的 fatal argparse 形式)。
        cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(input_dir, pred_dir)

        # 生成脚本在独立 shell 里跑 canonical 入口:src 树自举到 PYTHONPATH
        # (与 nnunet_l2_final_acceptance 的 writer 同族,ADR-0009 决定 6)。
        # 生成脚本在独立 shell 里跑 canonical 入口:本检出的 src 树自举到 PYTHONPATH
        # (#140 迁家后 repo 与 flat 部署两种形态同源——包根即 <checkout>/src)。
        package_src = Path(__file__).resolve().parents[4]
        script = (
            "#!/bin/bash\nset -euo pipefail\n"
            + f'export PYTHONPATH="{package_src}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
            + " ".join(cmd)
            + f" 2>&1 | tee {pred_dir / 'predict.log'}\n"
        )

        script_path = output_dir / f"predict_{challenge}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"[INFO] Generated prediction script: {script_path}")
        return pred_dir


# ── 指标计算与判定 ──────────────────────────────────────────────────────


class MetricsCalculator:
    """计算合成域评估指标并与真实校准 R_fail 对照。

    核心指标：
    - R_fail_synth：合成样本上的输入失败率 + 层级违反率
    - R_fail_real：真实校准 R_fail（从 ADR-0002 载入）
    - 适用性判定：R_fail_synth > R_fail_real + margin → undecided
    """

    REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
    Z95 = 1.959963984540054

    def evaluate_case(
        self,
        case_id: str,
        challenge: str,
        inputs_dir: Path,
        pred_dir: Path,
        gt_dir: Path | None = None,
    ) -> dict:
        """评估一个合成病例的仪器输出。"""
        result = {
            "case": case_id,
            "challenge": challenge,
            "input_fail": False,
            "run_fail": False,
            "hier_viol": False,
            "per_region": {},
        }

        # 1. 输入契约检查
        try:
            inputs = [sitk.ReadImage(str(inputs_dir / f"{case_id}_{s}.nii.gz")) for s in ("0000", "0001", "0002", "0003")]
            reference = (inputs[0].GetSize(), inputs[0].GetSpacing(), inputs[0].GetOrigin())
            consistent = all((img.GetSize(), img.GetSpacing(), img.GetOrigin()) == reference for img in inputs[1:])
            isotropic = all(abs(s - 1.0) < 1e-3 for s in inputs[0].GetSpacing())
            result["input_fail"] = not (consistent and isotropic)
        except (RuntimeError, OSError):
            result["input_fail"] = True
            result["run_fail"] = True
            return result

        # 2. 读取预测
        pred_path = pred_dir / f"{case_id}.nii.gz"
        try:
            pred_img = sitk.ReadImage(str(pred_path))
            pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
        except (RuntimeError, OSError):
            result["run_fail"] = True
            return result

        # 3. 层级违反检查（ET⊆TC⊆WT）
        wt_pred = np.isin(pred_arr, (1, 2, 3))
        tc_pred = np.isin(pred_arr, (1, 3))
        et_pred = pred_arr == 3

        # 层级约束：ET ⊆ TC ⊆ WT
        if et_pred.sum() > 0 and not np.all(et_pred[tc_pred] if tc_pred.sum() > 0 else True):
            # ET 中有像素不在 TC 中
            et_outside_tc = et_pred & ~tc_pred
            if et_outside_tc.sum() > 0:
                result["hier_viol"] = True
        if tc_pred.sum() > 0 and not np.all(tc_pred[wt_pred]):
            # TC 中有像素不在 WT 中
            tc_outside_wt = tc_pred & ~wt_pred
            if tc_outside_wt.sum() > 0:
                result["hier_viol"] = True

        # 值域检查
        if not np.isin(pred_arr, (0, 1, 2, 3)).all():
            result["hier_viol"] = True

        # 4. 逐区域测量（如果有 GT）
        if gt_dir is not None:
            gt_path = gt_dir / f"{case_id}.nii.gz"
            if gt_path.exists():
                gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.uint8)
                for region, labels in self.REGIONS.items():
                    gt_mask = np.isin(gt_arr, labels)
                    pred_mask = np.isin(pred_arr, labels)
                    vol_gt = float(gt_mask.sum()) * 0.001
                    vol_pred = float(pred_mask.sum()) * 0.001
                    denom = int(gt_mask.sum()) + int(pred_mask.sum())
                    dice = 2 * np.logical_and(gt_mask, pred_mask).sum() / denom if denom > 0 else math.nan
                    result["per_region"][region] = {
                        "dice": dice,
                        "vol_gt_ml": vol_gt,
                        "vol_pred_ml": vol_pred,
                        "detected": bool(pred_mask.sum() > 0),
                    }

        return result

    def compute_r_fail(self, case_results: list[dict]) -> dict:
        """计算合成样本的 R_fail。"""
        n_obs = len(case_results)
        k_input = sum(r["input_fail"] for r in case_results)
        k_run = sum(r["run_fail"] for r in case_results)
        k_hier = sum(r["hier_viol"] for r in case_results)
        k_fail = sum(r["input_fail"] or r["run_fail"] or r["hier_viol"] for r in case_results)

        return {
            "k": k_fail,
            "n": n_obs,
            "point": k_fail / n_obs if n_obs else math.nan,
            "wilson_95_upper": self._wilson_upper(k_fail, n_obs),
            "breakdown": {
                "input_fail": k_input,
                "run_fail": k_run,
                "hier_viol": k_hier,
            },
        }

    def _wilson_upper(self, k: int, n: int) -> float:
        if n == 0:
            return math.nan
        p = k / n
        denom = 1 + self.Z95**2 / n
        center = (p + self.Z95**2 / (2 * n)) / denom
        half = (self.Z95 / denom) * math.sqrt(p * (1 - p) / n + self.Z95**2 / (4 * n**2))
        return min(1.0, center + half)

    def load_real_r_fail(self, calibration_metrics_dir: Path, challenge: str) -> dict:
        """从 ADR-0002 校准结果载入真实 R_fail。"""
        summary_path = calibration_metrics_dir / f"summary_{challenge}.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"calibration summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text())
        return summary["R_fail"]

    def determine_verdict(
        self,
        r_fail_synth: dict,
        r_fail_real: dict,
        challenge: str,
    ) -> str:
        """合成域适用性判定。

        判定逻辑（#32 §5 合成域适用性）：
        - R_fail_synth ≤ R_fail_real → PASS（合成域适用）
        - R_fail_synth > R_fail_real → UNDECIDED（仪器在合成域上行为不确定）
        - UNDECIDED 阻塞完整 spec 终验，修复方向是仪器或重跑

        注意：R_fail_real 全为 0（ADR-0002），所以任何非零 R_fail_synth
        都会导致 UNDECIDED。这是预期行为——v1 DM 直出样本与真实 BraTS 的
        分布差异可能导致仪器输入契约违规。
        """
        synth_rate = r_fail_synth.get("point", 0)
        real_rate = r_fail_real.get("point", 0)

        if synth_rate <= real_rate:
            return "PASS"
        return "UNDECIDED"


# ── 报告生成 ────────────────────────────────────────────────────────────


class ReportGenerator:
    """生成合成域适用性评估报告。"""

    def generate(
        self,
        all_results: dict[str, dict],  # challenge → evaluation result
        output_dir: Path,
        mode: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "title": "L2 仪器合成域适用性评估报告",
            "issue": 38,
            "mode": mode,
            "description": ("在 v1 DM 直出样本上测试 L2 测量仪器的输入失败率与层级违反率，建立合成域适用性前置证据"),
            "per_challenge": {},
            "overall_verdict": "PASS",
        }

        for challenge, result in all_results.items():
            verdict = result["verdict"]
            report["per_challenge"][challenge] = {
                "n_samples": result["r_fail_synth"]["n"],
                "r_fail_synth": result["r_fail_synth"],
                "r_fail_real": result["r_fail_real"],
                "verdict": verdict,
                "details": result.get("details", {}),
            }
            if verdict == "UNDECIDED":
                report["overall_verdict"] = "UNDECIDED"

        # P2 方向证据缺位记录
        if mode == "p1":
            report["p2_evidence_gap"] = (
                "P2 方向前置证据缺位已知情接受（掩码 ControlNet 训练前不存在 v1 可产样本），P2 依赖终验伴随监控（undecided 语义）兜底"
            )

        report_path = output_dir / f"synthetic_domain_report_{mode}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        # 也生成可读 markdown
        md_path = output_dir / f"synthetic_domain_report_{mode}.md"
        md_content = self._to_markdown(report)
        md_path.write_text(md_content)

        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report_path

    def _to_markdown(self, report: dict) -> str:
        lines = [
            f"# {report['title']}",
            "",
            f"**Issue**: [#{report['issue']}](https://github.com/ACautomata/NV-Generate-CTMR/issues/{report['issue']})",
            f"**模式**: {report['mode']}",
            f"**总体判定**: **{report['overall_verdict']}**",
            "",
            "## 各子挑战结果",
            "",
            "| 挑战 | 样本数 | R_fail_synth | R_fail_real | 判定 |",
            "|------|--------|-------------|-------------|------|",
        ]

        for challenge, details in report["per_challenge"].items():
            synth = details["r_fail_synth"]
            real = details["r_fail_real"]
            lines.append(
                f"| {challenge} | {details['n_samples']} | "
                f"{synth['point']:.4f} ({synth['k']}/{synth['n']}) | "
                f"{real['point']:.4f} ({real['k']}/{real['n']}) | "
                f"**{details['verdict']}** |"
            )

        lines.extend(["", "## R_fail 细分", ""])

        for challenge, details in report["per_challenge"].items():
            breakdown = details["r_fail_synth"]["breakdown"]
            lines.append(f"### {challenge}")
            lines.append(f"- 输入失败: {breakdown['input_fail']}/{details['n_samples']}")
            lines.append(f"- 运行失败: {breakdown['run_fail']}/{details['n_samples']}")
            lines.append(f"- 层级违反: {breakdown['hier_viol']}/{details['n_samples']}")
            lines.append("")

        if "p2_evidence_gap" in report:
            lines.extend(
                [
                    "## P2 方向说明",
                    "",
                    report["p2_evidence_gap"],
                    "",
                ]
            )

        lines.extend(
            [
                "## 冻结语义",
                "",
                "仪器权重与推理配置沿用 ADR-0002/ADR-0003 冻结状态。",
                "本评估不修改仪器，不反向调节任何阈值。",
                "",
            ]
        )

        return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────────────────────────


def cmd_generate(args: argparse.Namespace) -> None:
    """Step 1: 生成 v1 DM 直出样本。"""
    config = GenerationConfig(
        model_dir=Path(args.v1_model_dir),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        seed=args.seed,
    )
    sampler = V1DMSampler(config)

    case_list_path = Path(args.case_list)
    case_entries = json.loads(case_list_path.read_text())

    all_results = []
    for entry in case_entries:
        case = CaseEntry(
            case_id=entry["case_id"],
            challenge=entry["challenge"],
            source=entry.get("source", "dev"),
            images_dir=Path(entry["images_dir"]),
        )
        if args.mode == "p1":
            results = sampler.generate_p1(case)
        elif args.mode == "p3":
            results = sampler.generate_p3(case)
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        all_results.extend(results)

    manifest_path = config.output_dir / "generation_manifest.json"
    manifest = [
        {
            "case_id": r.case_id,
            "challenge": r.challenge,
            "mode": r.mode,
            "anchor": r.anchor_modality,
            "modalities": {k: str(v) for k, v in r.modalities_generated.items()},
        }
        for r in all_results
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[OK] Generated {len(all_results)} sample sets, manifest: {manifest_path}")


def cmd_prep_inputs(args: argparse.Namespace) -> None:
    """Step 2: 组装 nnU-Net 输入。"""
    preparator = InputPreparator()
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    manifest = json.loads((sample_dir / "generation_manifest.json").read_text())

    for entry in manifest:
        modality_paths = {k: Path(v) for k, v in entry["modalities"].items()}
        preparator.prepare_case(
            case_id=entry["case_id"],
            challenge=entry["challenge"],
            modality_paths=modality_paths,
            output_dir=output_dir,
        )
    print(f"[OK] Prepared nnU-Net inputs for {len(manifest)} cases")


def cmd_predict(args: argparse.Namespace) -> None:
    """Step 3: 运行冻结仪器推理。"""
    runner = InstrumentRunner(Path(args.results_root))
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    challenges = os.listdir(input_dir) if input_dir.is_dir() else []
    for challenge in challenges:
        if challenge.startswith("Dataset") or not (input_dir / challenge).is_dir():
            continue
        runner.predict_challenge(challenge, input_dir / challenge, output_dir)
    print(f"[OK] Generated prediction scripts for {len(challenges)} challenges")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Step 4: 计算指标并生成报告。"""
    calculator = MetricsCalculator()
    reporter = ReportGenerator()

    sample_dir = Path(args.sample_dir)
    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    cal_metrics_dir = Path(args.calibration_summary)
    output_dir = Path(args.output_dir)

    manifest = json.loads((sample_dir / "generation_manifest.json").read_text())

    # 按挑战分组
    by_challenge: dict[str, list[dict]] = {}
    for entry in manifest:
        by_challenge.setdefault(entry["challenge"], []).append(entry)

    all_results = {}
    for challenge, entries in by_challenge.items():
        case_results = []
        for entry in entries:
            case_id = entry["case_id"]
            result = calculator.evaluate_case(
                case_id=case_id,
                challenge=challenge,
                inputs_dir=input_dir / challenge,
                pred_dir=pred_dir / challenge,
            )
            case_results.append(result)

        r_fail_synth = calculator.compute_r_fail(case_results)
        r_fail_real = calculator.load_real_r_fail(cal_metrics_dir, challenge)
        verdict = calculator.determine_verdict(r_fail_synth, r_fail_real, challenge)

        all_results[challenge] = {
            "r_fail_synth": r_fail_synth,
            "r_fail_real": r_fail_real,
            "verdict": verdict,
            "details": {
                "case_results": case_results,
            },
        }

    mode = "p1" if "p1" in str(sample_dir) else "p3"
    reporter.generate(all_results, output_dir, mode)


def cmd_create_case_lists(args: argparse.Namespace) -> None:
    """辅助命令：为每个子挑战创建病例列表。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nnunet_root = Path(args.nnunet_root)
    selector = CaseSelector(nnunet_root)

    for mode in ("p1", "p3"):
        cases = []
        for challenge, n in SAMPLES_PER_CHALLENGE.items():
            entries = selector.select(challenge, n, seed=args.seed)
            for entry in entries:
                cases.append(
                    {
                        "case_id": entry.case_id,
                        "challenge": entry.challenge,
                        "source": entry.source,
                        "images_dir": str(entry.images_dir),
                    }
                )

        case_list_path = output_dir / f"{mode}_cases.json"
        case_list_path.write_text(json.dumps(cases, indent=2) + "\n")
        print(f"[OK] {mode}: {len(cases)} cases → {case_list_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="生成 v1 DM 直出样本")
    gen.add_argument("--mode", choices=("p1", "p3"), required=True)
    gen.add_argument("--case-list", type=Path, required=True)
    gen.add_argument("--v1-model-dir", type=Path, required=True)
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=42)

    # prep-inputs
    prep = sub.add_parser("prep-inputs", help="组装 nnU-Net 输入")
    prep.add_argument("--sample-dir", type=Path, required=True)
    prep.add_argument("--nnunet-root", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)

    # predict
    pred = sub.add_parser("predict", help="运行冻结仪器推理")
    pred.add_argument("--input-dir", type=Path, required=True)
    pred.add_argument("--results-root", type=Path, required=True)
    pred.add_argument("--output-dir", type=Path, required=True)

    # evaluate
    ev = sub.add_parser("evaluate", help="计算指标并生成报告")
    ev.add_argument("--sample-dir", type=Path, required=True)
    ev.add_argument("--input-dir", type=Path, required=True)
    ev.add_argument("--pred-dir", type=Path, required=True)
    ev.add_argument("--calibration-summary", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)

    # create-case-lists
    cl = sub.add_parser("create-case-lists", help="创建评估病例列表")
    cl.add_argument("--nnunet-root", type=Path, required=True)
    cl.add_argument("--output-dir", type=Path, required=True)
    cl.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "prep-inputs":
        cmd_prep_inputs(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "create-case-lists":
        cmd_create_case_lists(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
