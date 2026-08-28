# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controlled L1 evidence readers and the report writer.

Migrated verbatim from ``scripts/brats_l1_quantitative.py`` (#141): auditable
JSON/NIfTI/feature-manifest reads from controlled storage and the strict-JSON
report write that refuses any path inside a git work tree (DUA rule). The
pinned MR [0, 1] intensity protocol comes from ``ctmr.domain.intensity_protocol``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.quantitative.fid import FeatureRecord, L1QuantitativeError
from ctmr.application.acceptance.quantitative.paired import P3PairObservation

FEATURE_SCHEMA = "brats-l1-features/1"
PAIR_SCHEMA = "brats-l1-pairs/1"
FEATURE_EXTRACTOR = "radimagenet_resnet50"
MR_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad"


@dataclass(frozen=True)
class FeatureManifest:
    """Controlled FID feature records and their extractor provenance."""

    records: tuple[FeatureRecord, ...]
    protocol: dict


class ControlledJsonReader:
    """Reads an auditable JSON document from controlled storage."""

    def read(self, path, label):
        resolved = Path(path)
        if not resolved.is_file():
            raise L1QuantitativeError(f"{label} not found: {resolved}")
        try:
            return json.loads(resolved.read_text())
        except json.JSONDecodeError as error:
            raise L1QuantitativeError(f"{label} is not valid JSON: {resolved} ({error})") from error


class FrozenRunRecordReader:
    """Loads a frozen phase run and its pinned challenge manifest."""

    def __init__(self, documents):
        self._documents = documents

    def read(self, path):
        record = self._documents.read(path, "run record")
        try:
            FrozenRunBinding.from_record(record)  # freeze gate: binding extract validates the run state
        except FrozenRunBindingError as error:
            raise L1QuantitativeError(str(error)) from error
        return record

    def challenges(self, record):
        try:
            manifest_path = record["manifest"]["path"]
            manifest = self._documents.read(manifest_path, "phase manifest")
            challenges = tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"run record has no readable phase manifest: {error}") from error
        if not challenges:
            raise L1QuantitativeError("phase manifest has no challenges")
        return challenges


class FeatureManifestReader:
    """Loads case-level three-plane feature arrays without accessing an extractor or network."""

    def __init__(self, documents):
        self._documents = documents

    def read(self, path):
        manifest_path = Path(path)
        payload = self._documents.read(manifest_path, "L1 feature manifest")
        if payload.get("schema") != FEATURE_SCHEMA:
            raise L1QuantitativeError(f"feature manifest schema must be {FEATURE_SCHEMA!r}")
        protocol = payload.get("protocol")
        extractor = protocol.get("feature_extractor") if isinstance(protocol, dict) else None
        if not isinstance(extractor, dict) or extractor.get("name") != FEATURE_EXTRACTOR or not self._sha256(extractor.get("weights_sha256")):
            raise L1QuantitativeError(f"feature manifest must record {FEATURE_EXTRACTOR} and a SHA-256 weights hash")
        if protocol.get("mr_preprocessing") != MR_PREPROCESSING:
            raise L1QuantitativeError(f"feature manifest mr_preprocessing must be {MR_PREPROCESSING}")
        records = tuple(self._record(manifest_path.parent, row) for row in payload.get("records", []))
        if not records:
            raise L1QuantitativeError("feature manifest has no records")
        return FeatureManifest(records=records, protocol=protocol)

    def _sha256(self, value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    def _record(self, root, row):
        try:
            relative_path = Path(row["path"])
            feature_path = relative_path if relative_path.is_absolute() else root / relative_path
            if not feature_path.is_file():
                raise L1QuantitativeError(f"feature array not found: {feature_path}")
            return FeatureRecord(
                cohort=row["cohort"],
                challenge=row["challenge"],
                case=row["case"],
                target_modality=row["target_modality"],
                plane=row["plane"],
                features=np.load(feature_path, allow_pickle=False),
                src_modality=row.get("src_modality"),
            )
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"feature manifest record is incomplete: {error}") from error


@dataclass(frozen=True)
class NiftiVolume:
    """A loaded MR volume and its spatial transform."""

    data: np.ndarray
    affine: np.ndarray


class NiftiVolumeReader:
    """Loads a NIfTI image without resampling its established evaluation grid."""

    def read(self, path, label):
        image_path = Path(path)
        if not image_path.is_file():
            raise L1QuantitativeError(f"{label} NIfTI not found: {image_path}")
        try:
            image = nib.load(image_path)
            return NiftiVolume(data=image.get_fdata(dtype=np.float64), affine=image.affine)
        except (OSError, ValueError) as error:
            raise L1QuantitativeError(f"{label} NIfTI cannot be read: {image_path} ({error})") from error


class P3PairManifestReader:
    """Loads same-case P3 target, stage-0 baseline, and candidate NIfTI triplets."""

    def __init__(self, documents, volumes, normalizer):
        self._documents = documents
        self._volumes = volumes
        self._normalizer = normalizer

    def read(self, path):
        manifest_path = Path(path)
        payload = self._documents.read(manifest_path, "P3 pair manifest")
        if payload.get("schema") != PAIR_SCHEMA:
            raise L1QuantitativeError(f"P3 pair manifest schema must be {PAIR_SCHEMA!r}")
        records = tuple(self._record(manifest_path.parent, row) for row in payload.get("records", []))
        if not records:
            raise L1QuantitativeError("P3 pair manifest has no records")
        return records

    def _record(self, root, row):
        try:
            reference = self._load(root, row["reference"], "reference")
            baseline = self._load(root, row["baseline"], "stage-0 baseline")
            candidate = self._load(root, row["candidate"], "candidate")
            self._same_geometry(reference, baseline, candidate, row["case"])
            return P3PairObservation(
                challenge=row["challenge"],
                case=row["case"],
                src_modality=row["src_modality"],
                target_modality=row["target_modality"],
                reference=self._normalizer.normalize(reference.data, "P3 reference"),
                baseline=self._normalizer.normalize(baseline.data, "P3 stage-0 baseline"),
                candidate=self._normalizer.normalize(candidate.data, "P3 candidate"),
            )
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"P3 pair record is incomplete: {error}") from error

    def _load(self, root, text_path, label):
        path = Path(text_path)
        return self._volumes.read(path if path.is_absolute() else root / path, label)

    def _same_geometry(self, reference, baseline, candidate, case):
        volumes = (reference, baseline, candidate)
        if any(volume.data.shape != reference.data.shape for volume in volumes[1:]):
            raise L1QuantitativeError(f"P3 pair {case} has mismatched NIfTI shapes")
        if any(not np.allclose(volume.affine, reference.affine) for volume in volumes[1:]):
            raise L1QuantitativeError(f"P3 pair {case} has mismatched NIfTI affines")


class L1ReportWriter:
    """Persists the final machine-readable report without allowing NaN JSON values."""

    def write(self, report, path):
        output = Path(path)
        self._assert_controlled(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        except (TypeError, ValueError) as error:
            raise L1QuantitativeError(f"L1 report cannot be represented as strict JSON: {error}") from error
        return output

    def _assert_controlled(self, output):
        for parent in output.resolve().parents:
            if (parent / ".git").exists():
                raise L1QuantitativeError(f"L1 report output lives inside a git work tree ({parent}); controlled reports must stay outside the repo")
