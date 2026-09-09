"""BRATS dataset.json 阶段①生成器（spec #13 §②.3 / §②.4，ticket T1 #17）.

阶段①（latent 之前）：枚举 BraTS 2023 GLI 训练集 cases × 4 模态（seg 保留在盘，
不进 dataset.json），按 subject 级切分（固定 seed，同 subject 多 timepoint 不跨集），
产出单个 dataset.json：

- ``"training"``:   [{"image": <相对 data-base-dir>, "modality": <label 字符串>}, ...]
  val cases 不在 training（§②.4：val 只作 BRATS FID 图像域参照，不建 latent）。
- ``"validation"``: [case 目录名, ...] —— FID 参照切分名单（训练代码只读 training）。

modality 由文件后缀映射：t1n→``mri_t1n``、t1c→``mri_t1ce``、t2w→``mri_t2w``、t2f→``mri_t2f``。
val subject 数 = floor(subject 总数 × val_fraction)；固定 seed ⇒ 重跑逐行复现。
默认对字典序第一个 case 跑 §②.1 抽查断言（t1c shape (240, 240, 155)、seg ∈ {0,1,2,3}）。

用法（gauss，数据经共享池 symlink 就位）::

    uv run python -m scripts.create_brats_dataset_json \\
        --training-data-dir $RUN_ROOT/datasets/brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \\
        --data-base-dir $RUN_ROOT/datasets \\
        --output $RUN_ROOT/datasets/brats2023-gli/dataset.json

阶段②（latent 之后扫描 embedding header 写 sidecar）在本脚本之外另行扩展。
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

SCAN_NAME_PATTERN = re.compile(r"BraTS-GLI-\d{5}-\d{3}")
NIFTI_EXTENSION = ".nii.gz"
SEG_SUFFIX = "seg"
SUFFIX_TO_MODALITY = {"t1n": "mri_t1n", "t1c": "mri_t1ce", "t2w": "mri_t2w", "t2f": "mri_t2f"}
REQUIRED_SUFFIXES = (SEG_SUFFIX, *SUFFIX_TO_MODALITY)
EXPECTED_SHAPE = (240, 240, 155)
SEG_LABELS = {0, 1, 2, 3}


@dataclass(frozen=True)
class BraTSScan:
    """一个 BraTS case（单 timepoint scan），目录名形如 ``BraTS-GLI-00000-000``."""

    directory: str

    @property
    def subject(self) -> str:
        """Longitudinal subject（``BraTS-GLI-XXXXX`` 段）；同 subject 多 timepoint 不跨 train/val。"""
        return self.directory.rsplit("-", 1)[0]

    def path(self, root: Path, suffix: str) -> Path:
        """case 文件在 ``root`` 下的完整路径；``<directory>-<suffix>.nii.gz`` 约定只此一处。"""
        return root / self.directory / f"{self.directory}-{suffix}{NIFTI_EXTENSION}"


@dataclass(frozen=True)
class SubjectSplit:
    """subject 级 holdout 切分结果."""

    train: frozenset[str]
    val: frozenset[str]


class BraTSScanIndex:
    """训练数据目录的 case 索引：扫描目录名并校验每 case 五文件齐全."""

    def __init__(self, training_data_dir: Path) -> None:
        self._training_data_dir = training_data_dir
        if not training_data_dir.is_dir():
            raise FileNotFoundError(f"training data dir not found: {training_data_dir}")
        self._scans = self._scan_directories()

    @property
    def scans(self) -> list[BraTSScan]:
        """全部 case，按目录名排序."""
        return list(self._scans)

    @property
    def subjects(self) -> list[str]:
        """去重后的 subject 段，排序."""
        return sorted({scan.subject for scan in self._scans})

    def _scan_directories(self) -> list[BraTSScan]:
        scans = []
        for case_dir in sorted(path for path in self._training_data_dir.iterdir() if path.is_dir()):
            if not SCAN_NAME_PATTERN.fullmatch(case_dir.name):
                raise ValueError(f"unexpected case directory name: {case_dir.name}")
            scan = BraTSScan(directory=case_dir.name)
            missing = sorted(suffix for suffix in REQUIRED_SUFFIXES if not scan.path(self._training_data_dir, suffix).is_file())
            if missing:
                raise ValueError(f"case {scan.directory} is missing suffixes: {missing}")
            scans.append(scan)
        return scans


class HoldoutSplitter:
    """subject 级 95/5 切分：固定 seed 洗牌后取前 floor(n × val_fraction) 个 subject 进 val."""

    def __init__(self, val_fraction: float = 0.05, seed: int = 42) -> None:
        self._val_fraction = val_fraction
        self._seed = seed

    def split(self, subjects: list[str]) -> SubjectSplit:
        shuffled = sorted(subjects)
        random.Random(self._seed).shuffle(shuffled)
        n_val = int(len(shuffled) * self._val_fraction)
        return SubjectSplit(train=frozenset(shuffled[n_val:]), val=frozenset(shuffled[:n_val]))


class BraTSDatasetList:
    """阶段① dataset.json 组装：``training`` 条目（val cases 排除）+ ``validation`` 切分名单."""

    def __init__(
        self,
        training_data_dir: Path,
        data_base_dir: Path,
        scans: list[BraTSScan],
        split: SubjectSplit,
    ) -> None:
        self._training_data_dir = training_data_dir
        self._data_base_dir = data_base_dir
        self._scans = scans
        self._split = split

    def to_dict(self) -> dict:
        return {"training": self._training_entries(), "validation": self._validation_roster()}

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.write("\n")

    def _training_entries(self) -> list[dict]:
        entries = []
        for scan in sorted(self._scans, key=lambda item: item.directory):
            if scan.subject in self._split.val:
                continue
            for suffix in sorted(SUFFIX_TO_MODALITY):
                entries.append({"image": self._image_path(scan, suffix), "modality": SUFFIX_TO_MODALITY[suffix]})
        return entries

    def _validation_roster(self) -> list[str]:
        return sorted(scan.directory for scan in self._scans if scan.subject in self._split.val)

    def _image_path(self, scan: BraTSScan, suffix: str) -> str:
        return scan.path(self._training_data_dir, suffix).relative_to(self._data_base_dir).as_posix()


class NiftiSpotCheck:
    """§②.1 抽查断言：t1c shape == (240, 240, 155)、seg labels ⊆ {0, 1, 2, 3}."""

    def __init__(self, training_data_dir: Path) -> None:
        self._training_data_dir = training_data_dir

    def run(self, scan: BraTSScan) -> None:
        """断言不过即抛 ValueError（不用 assert：`python -O` 会剥离 assert，抽查静默失效）."""
        t1c = nib.load(str(scan.path(self._training_data_dir, "t1c")))
        if t1c.shape != EXPECTED_SHAPE:
            raise ValueError(f"{scan.directory}: t1c shape {t1c.shape} != {EXPECTED_SHAPE}")
        seg = np.asarray(nib.load(str(scan.path(self._training_data_dir, SEG_SUFFIX))).dataobj)
        labels = set(np.unique(seg).tolist())
        if not labels <= SEG_LABELS:
            raise ValueError(f"{scan.directory}: seg labels {sorted(labels)} not within {sorted(SEG_LABELS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        required=True,
        help="ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData 目录",
    )
    parser.add_argument(
        "--data-base-dir",
        type=Path,
        required=True,
        help="image 相对路径的基准目录（训练 env 的 data_base_dir）",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 dataset.json 路径")
    parser.add_argument("--seed", type=int, default=42, help="切分随机 seed（固定 ⇒ 可复现）")
    parser.add_argument("--skip-spot-check", action="store_true", help="跳过 §②.1 nibabel 抽查断言")
    args = parser.parse_args()

    index = BraTSScanIndex(args.training_data_dir)
    if not args.skip_spot_check:
        NiftiSpotCheck(args.training_data_dir).run(index.scans[0])
        print(f"spot check passed for {index.scans[0].directory}")
    split = HoldoutSplitter(seed=args.seed).split(index.subjects)
    dataset_list = BraTSDatasetList(
        training_data_dir=args.training_data_dir,
        data_base_dir=args.data_base_dir,
        scans=index.scans,
        split=split,
    )
    dataset_list.save(args.output)
    payload = dataset_list.to_dict()
    print(
        f"scans={len(index.scans)} subjects={len(index.subjects)} "
        f"training={len(payload['training'])} validation_cases={len(payload['validation'])} "
        f"validation_subjects={len(split.val)} seed={args.seed}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
