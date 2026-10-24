# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visual-kit preparation for the T10 HITL verdict (ticket T10, issue #28).

Spec #13 §⑤.3 assigns the visual checks (anatomy, contrast, skull absence,
label discrimination, decode artifacts) to the user; the agent's job is to
have everything ready: 40 triplanar montages for the BRATS side (4 new labels
x 10 volumes, central slices in all three planes) plus a 4-volume quickscan
sheet for every old label whose FID landed in the §⑤.2 quickscan band (the
only place a visual verdict overrides a borderline FID).

Montages render the generator's native output domain (MR mapped to (0, 1000))
in grayscale, one row per plane — sagittal, coronal, axial — from the volume's
center slice. The manifest json lists every produced file with its source
volume so the GitHub Release upload (publish_experiment convention) is a
single directory.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scripts.acceptance_protocol import BRATS_LABELS

PLANES = ("sagittal", "coronal", "axial")
MR_DISPLAY_RANGE = (0.0, 1000.0)


class CentralSliceExtractor:
    """Reads a NIfTI volume and returns its three orthogonal center slices."""

    @staticmethod
    def slices(volume_path: Path) -> dict[str, np.ndarray]:
        data = np.asarray(nib.load(str(volume_path)).dataobj, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {data.shape}: {volume_path}")
        center = tuple(axis // 2 for axis in data.shape)
        return {
            "sagittal": data[center[0], :, :],
            "coronal": data[:, center[1], :],
            "axial": data[:, :, center[2]],
        }


class TriplanarMontageBuilder:
    """One volume -> one PNG with its three center slices side by side."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._renderer = MontageRenderer()

    def build_label_set(self, volume_dir: Path, prefix: str, label: int, seed: int, n_volumes: int, seed_choice: int = 42) -> list[Path]:
        paths = VolumeSelector.select(volume_dir, prefix, label, seed, n_volumes, seed_choice)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for path in paths:
            index_token = path.name.replace(".nii.gz", "").split("_")[-1]
            output = self._output_dir / f"montage_{prefix}_label{label}_seed{seed}_{index_token}.png"
            self._renderer.render_triplanar(CentralSliceExtractor.slices(path), output, title=path.name.replace(".nii.gz", ""))
            written.append(output)
        return written


class QuickScanSheetBuilder:
    """The quickscan verdict sheet: n volumes of one label stacked as a grid."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._renderer = MontageRenderer()

    def build_label_set(self, volume_dir: Path, prefix: str, label: int, seed: int, n_volumes: int, seed_choice: int = 42) -> list[Path]:
        paths = VolumeSelector.select(volume_dir, prefix, label, seed, n_volumes, seed_choice)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        output = self._output_dir / f"quickscan_{prefix}_label{label}_seed{seed}.png"
        per_volume = [CentralSliceExtractor.slices(path) for path in paths]
        self._renderer.render_grid(per_volume, output, row_titles=[path.name.replace(".nii.gz", "") for path in paths])
        return [output]


class VolumeSelector:
    """Deterministic volume picking: sorted by filename, then a seeded head choice."""

    @staticmethod
    def select(volume_dir: Path, prefix: str, label: int, seed: int, n_volumes: int, seed_choice: int) -> list[Path]:
        pattern = f"{prefix}_label{label}_seed{seed}_*.nii.gz"
        available = sorted(Path(volume_dir).glob(pattern))
        if len(available) < n_volumes:
            raise FileNotFoundError(f"need {n_volumes} volumes matching {pattern}, found only {len(available)} in {volume_dir}")
        rng = np.random.default_rng(seed_choice)
        indices = sorted(rng.choice(len(available), size=n_volumes, replace=False))
        return [available[i] for i in indices]


class MontageRenderer:
    """Grayscale matplotlib rendering of center slices in the generator's (0, 1000) output domain."""

    def render_triplanar(self, planes: dict[str, np.ndarray], output: Path, title: str) -> None:
        figure, axes = plt.subplots(1, 3, figsize=(9, 3.4))
        for axis, plane in zip(axes, PLANES):
            axis.imshow(planes[plane].T, cmap="gray", vmin=MR_DISPLAY_RANGE[0], vmax=MR_DISPLAY_RANGE[1], origin="lower")
            axis.set_title(plane, fontsize=8)
            axis.axis("off")
        figure.suptitle(title, fontsize=7)
        figure.tight_layout()
        figure.savefig(output, dpi=130)
        plt.close(figure)

    def render_grid(self, per_volume: list[dict[str, np.ndarray]], output: Path, row_titles: list[str]) -> None:
        figure, axes = plt.subplots(len(per_volume), 3, figsize=(9, 3.0 * len(per_volume)))
        axes = np.atleast_2d(axes)
        for row, (planes, row_title) in enumerate(zip(per_volume, row_titles)):
            for column, plane in enumerate(PLANES):
                axis = axes[row][column]
                axis.imshow(planes[plane].T, cmap="gray", vmin=MR_DISPLAY_RANGE[0], vmax=MR_DISPLAY_RANGE[1], origin="lower")
                axis.axis("off")
                if column == 0:
                    axis.set_title(row_title, fontsize=6, loc="left")
        figure.tight_layout()
        figure.savefig(output, dpi=110)
        plt.close(figure)


class VisualKitAssembler:
    """Assembles the full visual kit for one snapshot: BRATS montages + quickscan sheets + manifest."""

    def __init__(self, outputs_forgetting: Path, outputs_brats: Path, forgetting_prefix: str, brats_prefix: str, output_root: Path, adjudication_json: Path, volumes_per_label: int = 10, quickscan_volumes: int = 4) -> None:
        self._outputs_forgetting = Path(outputs_forgetting)
        self._outputs_brats = Path(outputs_brats)
        self._forgetting_prefix = forgetting_prefix
        self._brats_prefix = brats_prefix
        self._output_root = Path(output_root)
        self._adjudication_json = Path(adjudication_json)
        self._volumes_per_label = volumes_per_label
        self._quickscan_volumes = quickscan_volumes

    def assemble(self) -> dict:
        adjudication = json.loads(self._adjudication_json.read_text())

        brats_builder = TriplanarMontageBuilder(output_dir=self._output_root / "brats")
        brats_count = 0
        for label in BRATS_LABELS:
            # §⑤.3 fixes one seed for the visual check: 4 labels x 10 volumes = 40 montages.
            brats_count += len(
                brats_builder.build_label_set(volume_dir=self._outputs_brats, prefix=self._brats_prefix, label=label, seed=42, n_volumes=self._volumes_per_label)
            )

        quickscan_builder = QuickScanSheetBuilder(output_dir=self._output_root / "quickscan")
        quickscan_labels = adjudication.get("quickscan_labels", [])
        quickscan_sheets = []
        for label in quickscan_labels:
            for seed in (42, 1337):
                quickscan_sheets.extend(
                    quickscan_builder.build_label_set(volume_dir=self._outputs_forgetting, prefix=self._forgetting_prefix, label=int(label), seed=seed, n_volumes=self._quickscan_volumes)
                )

        manifest = {
            "adjudication_json": str(self._adjudication_json),
            "brats_pngs": brats_count,
            "brats_labels": list(BRATS_LABELS),
            "volumes_per_label": self._volumes_per_label,
            "quickscan_labels": quickscan_labels,
            "quickscan_sheets": [str(path) for path in quickscan_sheets],
        }
        (self._output_root / "montage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_table", type=Path, required=True)
    parser.add_argument("--run_id", required=True, help="snapshot run_id, e.g. N300e300")
    parser.add_argument("--adjudication_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    with open(args.task_table) as file:
        task_table = json.load(file)
    snapshot = next(s for s in task_table["snapshots"] if s["run_id"] == args.run_id)

    assembler = VisualKitAssembler(
        outputs_forgetting=snapshot["outputs_forgetting"],
        outputs_brats=snapshot["outputs_brats"],
        forgetting_prefix=snapshot["generation_prefix"],
        brats_prefix=snapshot["brats_generation_prefix"],
        output_root=args.output_root,
        adjudication_json=args.adjudication_json,
    )
    manifest = assembler.assemble()
    print(f"visual kit ready: {manifest['brats_pngs']} BRATS montages, quickscan labels {manifest['quickscan_labels']} -> {args.output_root}")


if __name__ == "__main__":
    main()
