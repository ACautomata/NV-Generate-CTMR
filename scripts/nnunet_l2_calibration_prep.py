#!/usr/bin/env python3
"""组装 Issue #36 L2 仪器校准集并冻结预注册协议（校准前置步骤）。

校准集 = 10% 开发集 ∪ 仪器 fold_0 内部验证集（均为仪器未训练病例，逐病例标记
来源）。本程序把四模态推理输入与 GT 以符号链接组入受控目录 ``inputs/``、
``gt/``，并将预注册协议与病例清单连同 SHA-256 冻结进 ``protocol/`` —— 顺序
纪律：冻结完成后才允许任何推理与指标计算（docs/calibration/
l2-instrument-calibration-protocol.md §7）。

所有输出均约束于 ``/root/private_data``；本程序绝不向 Git 写入患者 ID 或
NIfTI 数据。
"""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SPLITS_DIRNAME = "splits"
MANIFEST_NAME = "split_manifest_brats2023-rflow-v1.json"
DATASET_IDS = {"GLI": 501, "SSA": 502, "MEN": 503, "METS": 504, "PED": 505}
CHANNEL_SUFFIXES = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}

# docs/calibration/l2-instrument-calibration-protocol.md §1 的钉板计数；
# 组装时逐挑战校验，不符即失败退出（不静默继续）。
EXPECTED_COUNTS = {
    "GLI": {"dev": 125, "fold_val": 175},
    "SSA": {"dev": 6, "fold_val": 8},
    "MEN": {"dev": 100, "fold_val": 140},
    "METS": {"dev": 24, "fold_val": 33},
    "PED": {"dev": 10, "fold_val": 14},
}


@dataclass(frozen=True)
class ChallengeLayout:
    """一个子挑战校准输入的来源位置。"""

    code: str
    images_tr: Path
    labels_tr: Path
    dev_raw: Path  # dev 病例原始目录（含 <case>/<case>-{t1n,t1c,t2w,t2f,seg}.nii.gz）


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_lists(nnunet_root: Path) -> dict[str, dict[str, list[str]]]:
    manifest = json.loads((nnunet_root / SPLITS_DIRNAME / MANIFEST_NAME).read_text())
    cases: dict[str, dict[str, list[str]]] = {}
    for code in DATASET_IDS:
        dev = sorted(manifest["challenges"][code]["cases"]["dev"])
        fold_val = sorted((nnunet_root / SPLITS_DIRNAME / f"fold0_val_cases_{code}.txt").read_text().split())
        expected = EXPECTED_COUNTS[code]
        if len(dev) != expected["dev"] or len(fold_val) != expected["fold_val"]:
            sys.exit(f"{code}: 计数不符协议钉板 dev={len(dev)}/{expected['dev']} " f"fold_val={len(fold_val)}/{expected['fold_val']}")
        overlap = set(dev) & set(fold_val)
        if overlap:
            sys.exit(f"{code}: dev 与 fold_val 存在交集 {sorted(overlap)}")
        cases[code] = {"dev": dev, "fold_val": fold_val}
    return cases


def link_case_inputs(case: str, source: str, layout: ChallengeLayout, inputs_dir: Path, gt_dir: Path) -> None:
    """把一个病例的四模态 + GT 组入推理布局（全部符号链接，不复制数据）。"""
    if source == "fold_val":
        for suffix in CHANNEL_SUFFIXES:
            src = layout.images_tr / f"{case}_{suffix}.nii.gz"
            if not src.is_file():
                sys.exit(f"缺少 fold_val 输入 {src}")
            (inputs_dir / f"{case}_{suffix}.nii.gz").symlink_to(src.resolve())
        gt_src = layout.labels_tr / f"{case}.nii.gz"
    else:
        case_dir = layout.dev_raw / case
        for suffix, modality in CHANNEL_SUFFIXES.items():
            src = case_dir / f"{case}-{modality}.nii.gz"
            if not src.is_file():
                sys.exit(f"缺少 dev 输入 {src}")
            (inputs_dir / f"{case}_{suffix}.nii.gz").symlink_to(src.resolve())
        gt_src = case_dir / f"{case}-seg.nii.gz"
    if not gt_src.is_file():
        sys.exit(f"缺少 GT {gt_src}")
    (gt_dir / f"{case}.nii.gz").symlink_to(gt_src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--nnunet-root", type=Path, required=True, help="brats2023_nnunet 根（含 Dataset50X 与 splits/）")
    parser.add_argument("--protocol-src", type=Path, required=True, help="入库协议文档 l2-instrument-calibration-protocol.md")
    parser.add_argument(
        "--dev-raw", action="append", required=True, metavar="CH=DIR", help="dev 病例原始数据目录（<case>/<case>-t1n.nii.gz 布局），每挑战一次"
    )
    args = parser.parse_args()

    dev_raw_map: dict[str, Path] = {}
    for item in args.dev_raw:
        code, _, directory = item.partition("=")
        if code not in DATASET_IDS or not directory:
            sys.exit(f"--dev-raw 格式应为 CH=DIR，收到 {item!r}")
        dev_raw_map[code] = Path(directory)
    missing = set(DATASET_IDS) - set(dev_raw_map)
    if missing:
        sys.exit(f"缺少 dev 数据目录: {sorted(missing)}")

    cases = load_case_lists(args.nnunet_root)
    protocol_dir = args.calibration_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    frozen: list[str] = []  # "sha256  相对路径" 行，最终写入 SHA256SUMS
    protocol_copy = protocol_dir / "l2-instrument-calibration-protocol.md"
    shutil.copyfile(args.protocol_src, protocol_copy)
    frozen.append(f"{sha256_file(protocol_copy)}  protocol/l2-instrument-calibration-protocol.md")

    total = 0
    for code, dataset_id in DATASET_IDS.items():
        layout = ChallengeLayout(
            code=code,
            images_tr=args.nnunet_root / f"Dataset{dataset_id}_BraTS2023{code}" / "imagesTr",
            labels_tr=args.nnunet_root / f"Dataset{dataset_id}_BraTS2023{code}" / "labelsTr",
            dev_raw=dev_raw_map[code],
        )
        inputs_dir = args.calibration_root / "inputs" / code
        gt_dir = args.calibration_root / "gt" / code
        inputs_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for source in ("fold_val", "dev"):
            for case in cases[code][source]:
                link_case_inputs(case, source, layout, inputs_dir, gt_dir)
                manifest_rows.append({"case": case, "source": source})
        total += len(manifest_rows)

        manifest_path = protocol_dir / f"calibration_cases_{code}.json"
        manifest_path.write_text(
            json.dumps(
                {"challenge": code, "cases": manifest_rows, "counts": {k: len(v) for k, v in cases[code].items()}}, indent=2, ensure_ascii=False
            )
            + "\n"
        )
        frozen.append(f"{sha256_file(manifest_path)}  protocol/{manifest_path.name}")

        # 输入与 GT 的逐文件 hash：防组装后篡改（符号链接解析到实体文件）。
        for entry in sorted(inputs_dir.iterdir()) + sorted(gt_dir.iterdir()):
            relative = entry.relative_to(args.calibration_root)
            frozen.append(f"{sha256_file(entry)}  {relative}")
        print(f"{code}: {len(manifest_rows)} 例组装完成（dev {len(cases[code]['dev'])} + fold_val {len(cases[code]['fold_val'])}）")

    (protocol_dir / "SHA256SUMS").write_text("\n".join(frozen) + "\n")
    print(f"共 {total} 例；协议与清单已冻结于 {protocol_dir}（{len(frozen)} 条 hash）")


if __name__ == "__main__":
    main()
