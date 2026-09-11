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

"""BRATS real-real FID floor split: validation cases -> two disjoint halves (ticket T8, issue #24).

The BRATS acceptance floor (spec #13 section 5.3) is a real-real FID: the validation
cases are split into two halves and the two halves are compared against each other,
anchoring the FID magnitude this sample size produces when **both** sides are real
data. The number is frozen once (section 5.5) and reused as the sanity reference for
every acceptance point -- never recomputed.

The split is at **case** level (``BraTS-GLI-XXXXX-XXX`` names from the stage-1
dataset.json ``validation`` key), so all four modality volumes of one subject stay in
the same half. Per-label FID filelists are expanded from the case halves with the
dataset's file-suffix mapping (label ``mri_t1ce`` -> suffix ``t1c``, etc.).

Usage::

    python -m scripts.split_real_real_halves \
        --brats-json data/brats2023-gli/dataset.json \
        --seed 42 \
        --output-dir runs/floor-split-20260911

writes ``split_record.json`` (both halves + seed, the frozen artifact),
``filelist_half_a.txt`` / ``filelist_half_b.txt`` (per-case volume lists) and
``filelist_half_<a|b>_<label>.txt`` per BRATS label.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

BRATS_LABEL_SUFFIX = {"mri_t1n": "t1n", "mri_t1ce": "t1c", "mri_t2w": "t2w", "mri_t2f": "t2f"}
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitFreezeRecord:
    """The frozen half assignment: seed + both halves + where the cases came from."""

    seed: int
    half_a: list[str]
    half_b: list[str]
    validation_source: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SplitFreezeRecord":
        with path.open() as file:
            payload = json.load(file)
        return cls(
            seed=payload["seed"],
            half_a=payload["half_a"],
            half_b=payload["half_b"],
            validation_source=payload["validation_source"],
        )


class HalvesSplitter:
    """Seeded case-level half split: deterministic per seed, disjoint, covering every case."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def split(self, cases: list[str]) -> tuple[list[str], list[str]]:
        """First half = first ceil(n/2) of the seeded shuffle; duplicates are refused."""
        if len(set(cases)) != len(cases):
            raise ValueError("duplicate validation cases in the dataset.json validation key")
        shuffled = sorted(cases)
        random.Random(self._seed).shuffle(shuffled)
        midpoint = (len(shuffled) + 1) // 2
        return shuffled[:midpoint], shuffled[midpoint:]


class CaseFilelistExpander:
    """Expands case names into per-label volume paths under the BRATS training-data root."""

    def __init__(self, data_prefix: str) -> None:
        self._data_prefix = data_prefix.rstrip("/")

    def expand(self, cases: list[str], label: str) -> list[str]:
        """Volume paths of one label for the given cases, in the given case order."""
        if label not in BRATS_LABEL_SUFFIX:
            raise ValueError(f"unknown BRATS label {label!r}; expected one of {sorted(BRATS_LABEL_SUFFIX)}")
        suffix = BRATS_LABEL_SUFFIX[label]
        return [f"{self._data_prefix}/{case}/{case}-{suffix}.nii.gz" for case in cases]


class ValidationCaseSource:
    """Reads the validation case names from the stage-1 BRATS dataset.json."""

    def __init__(self, brats_json: Path) -> None:
        self._brats_json = brats_json

    def cases(self) -> list[str]:
        with self._brats_json.open() as file:
            payload = json.load(file)
        cases = payload["validation"]
        if not cases:
            raise ValueError(f"no validation cases in {self._brats_json}")
        return list(cases)


class RealRealFloorFilelists:
    """Writes the frozen filelists: per-case volumes and per-label volumes for each half."""

    def __init__(self, output_dir: Path, expander: CaseFilelistExpander) -> None:
        self._output_dir = output_dir
        self._expander = expander

    def write(self, half_a: list[str], half_b: list[str]) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for name, cases in (("a", half_a), ("b", half_b)):
            self._write_lines(f"filelist_half_{name}.txt", cases)
            for label in BRATS_LABEL_SUFFIX:
                self._write_lines(f"filelist_half_{name}_{label}.txt", self._expander.expand(cases, label=label))

    def _write_lines(self, filename: str, lines: list[str]) -> None:
        with (self._output_dir / filename).open("w") as file:
            file.writelines(f"{line}\n" for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brats-json", type=Path, required=True, help="stage-1 BRATS dataset.json holding the validation key")
    parser.add_argument(
        "--data-prefix", default="brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", help="volume path prefix inside the FID filelists"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split seed (fixed => reproducible halves)")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the frozen record + filelists")
    args = parser.parse_args()

    cases = ValidationCaseSource(args.brats_json).cases()
    splitter = HalvesSplitter(seed=args.seed)
    half_a, half_b = splitter.split(cases)

    record = SplitFreezeRecord(seed=args.seed, half_a=half_a, half_b=half_b, validation_source=str(args.brats_json))
    record.save(args.output_dir / "split_record.json")
    RealRealFloorFilelists(args.output_dir, CaseFilelistExpander(args.data_prefix)).write(half_a, half_b)

    print(f"validation cases: {len(cases)} -> half_a={len(half_a)}, half_b={len(half_b)} (seed={args.seed})")
    print(f"frozen record + filelists written to {args.output_dir}")


if __name__ == "__main__":
    main()
