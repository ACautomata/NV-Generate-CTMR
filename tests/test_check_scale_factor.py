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

"""Tests for scripts.check_scale_factor — the pre-finetune scale_factor sanity gate (ticket T7 #23, spec #13 section 4.5)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from scripts.check_scale_factor import ScaleFactorReference, ScaleFactorSanityCheck, StratifiedLatentSample, main
from scripts.latent_sidecars import LatentEntry

REPO_ROOT = Path(__file__).resolve().parents[1]

V1_SCALE_FACTOR = 0.9697
LABELS = ("mri_t1", "mri_t2", "mri_flair", "mri_swi", "mri_mra")
VOLUMES_PER_LABEL = 2
LATENT_SHAPE = (2, 4, 4, 4)


def constant_std_latent(path: Path, scale: float) -> None:
    """A latent whose Bessel-corrected (ddof=1) whole-volume std is exactly ``scale``.

    Half the voxels sit at -a, half at +a, so the population std is ``a``; writing at
    ``a = scale x sqrt((N-1)/N)`` makes the sample std (what the check measures, and what
    ``torch.std`` measures in training) come out at exactly ``scale``.
    """
    total = int(np.prod(LATENT_SHAPE))
    amplitude = scale * math.sqrt((total - 1) / total)
    values = np.full(LATENT_SHAPE, -amplitude, dtype=np.float32)
    values.reshape(-1)[: total // 2] = amplitude
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def dataset_entries(scale_by_label: dict[str, float], tmp_path: Path) -> tuple[list[LatentEntry], Path]:
    """Latents on disk at the requested per-label std, plus the dataset.json that lists them."""
    entries = []
    payload = {"training": []}
    for label in LABELS:
        for volume in range(VOLUMES_PER_LABEL):
            image = f"mri/batch00/STUDY{volume}/img/STUDY{volume}_{label}-raw-axi.nii.gz"
            constant_std_latent(tmp_path / "embeddings" / image.replace(".nii.gz", "_emb.nii.gz"), scale_by_label[label])
            entry = LatentEntry(image=image, modality=label)
            entries.append(entry)
            payload["training"].append(entry.to_record())
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps(payload, indent=2) + "\n")
    return entries, dataset_json


@pytest.fixture
def v1_ckpt(tmp_path: Path) -> Path:
    """A stand-in for the v1 checkpoint: the keys the sanity check reads, at the published value."""
    path = tmp_path / "diff_unet_3d_rflow-mr-brain_v1.pt"
    torch.save({"scale_factor": torch.tensor(V1_SCALE_FACTOR)}, path)
    return path


@pytest.fixture
def replay_dataset_on_distribution(tmp_path: Path) -> Path:
    """Latents whose std matches the v1 reference exactly: 1/std == scale_factor."""
    _entries, dataset_json = dataset_entries({label: 1.0 / V1_SCALE_FACTOR for label in LABELS}, tmp_path)
    return dataset_json


@pytest.fixture
def replay_dataset_off_distribution(tmp_path: Path) -> Path:
    """Latents at twice the reference std -- the gross shift preprocessing OOD would produce."""
    _entries, dataset_json = dataset_entries({label: 2.0 / V1_SCALE_FACTOR for label in LABELS}, tmp_path)
    return dataset_json


class TestScaleFactorReference:
    def test_reads_the_value_from_the_checkpoint(self, v1_ckpt: Path) -> None:
        reference = ScaleFactorReference.from_checkpoint(v1_ckpt)
        assert reference.value == pytest.approx(V1_SCALE_FACTOR)
        assert reference.source == str(v1_ckpt)

    def test_implied_training_std_is_the_reciprocal(self, v1_ckpt: Path) -> None:
        assert ScaleFactorReference.from_checkpoint(v1_ckpt).implied_training_std == pytest.approx(1.0 / V1_SCALE_FACTOR)

    def test_explicit_value_names_its_source(self) -> None:
        reference = ScaleFactorReference.from_value(1.0056)
        assert reference.value == pytest.approx(1.0056)
        assert reference.source == "--reference-scale-factor"


class TestStratifiedLatentSample:
    def test_draws_the_requested_number_of_latents(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.0 for label in LABELS}, tmp_path)
        sample = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=7, seed=42)
        assert len(sample.draw()) == 7

    def test_a_round_robin_draw_spreads_across_labels(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.0 for label in LABELS}, tmp_path)
        sample = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=5, seed=42)
        drawn = [entry.modality for entry, _std in sample.draw()]
        assert sorted(drawn) == sorted(LABELS)

    def test_the_draw_is_deterministic_for_a_seed(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.0 for label in LABELS}, tmp_path)
        first = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=7, seed=42).draw()
        second = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=7, seed=42).draw()
        assert [entry.image for entry, _ in first] == [entry.image for entry, _ in second]

    def test_per_latent_std_matches_the_file(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.5 for label in LABELS}, tmp_path)
        sample = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=10, seed=42)
        assert all(std == pytest.approx(1.5) for _entry, std in sample.draw())

    def test_a_label_filter_restricts_the_pool(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.0 for label in LABELS}, tmp_path)
        sample = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=10, seed=42, labels=("mri_t1",))
        drawn = sample.draw()
        assert {entry.modality for entry, _std in drawn} == {"mri_t1"}

    def test_an_empty_filtered_pool_is_refused(self, tmp_path: Path) -> None:
        entries, _ = dataset_entries({label: 1.0 for label in LABELS}, tmp_path)
        with pytest.raises(ValueError, match="mri_ct"):
            StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=10, seed=42, labels=("mri_ct",))


class TestScaleFactorSanityCheck:
    def test_on_distribution_data_passes_and_records_the_report(self, v1_ckpt: Path, replay_dataset_on_distribution: Path, tmp_path: Path) -> None:
        report_path = tmp_path / "scale_factor_report.json"
        verdict = ScaleFactorSanityCheck(
            reference=ScaleFactorReference.from_checkpoint(v1_ckpt),
            dataset_json=replay_dataset_on_distribution,
            embedding_base_dir=tmp_path / "embeddings",
            n_samples=10,
            seed=42,
            threshold=0.2,
            report_path=report_path,
        ).run()
        assert verdict["verdict"] == "PASS"
        report = json.loads(report_path.read_text())
        assert report["verdict"] == "PASS"
        assert report["reference"]["scale_factor"] == pytest.approx(V1_SCALE_FACTOR)
        assert report["estimate"]["scale_factor_estimate"] == pytest.approx(V1_SCALE_FACTOR, rel=1e-5)
        assert report["comparison"]["domain"] == "scale_factor"
        assert report["comparison"]["deviation_relative"] == pytest.approx(0.0, abs=1e-5)
        assert report["comparison"]["std_domain_deviation_relative"] == pytest.approx(0.0, abs=1e-5)
        assert len(report["estimate"]["samples"]) == 10

    def test_off_distribution_data_verdicts_block_and_records_both_domains(
        self, v1_ckpt: Path, replay_dataset_off_distribution: Path, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "scale_factor_report.json"
        report = ScaleFactorSanityCheck(
            reference=ScaleFactorReference.from_checkpoint(v1_ckpt),
            dataset_json=replay_dataset_off_distribution,
            embedding_base_dir=tmp_path / "embeddings",
            n_samples=10,
            seed=42,
            threshold=0.2,
            report_path=report_path,
        ).run()
        assert report["verdict"] == "BLOCK"
        # Latents at twice the reference std: the std-domain deviation is exactly +100%, the
        # reciprocal scale_factor-domain deviation 50% -- both recorded, the gate decides on
        # the scale_factor domain.
        assert report["comparison"]["deviation_relative"] == pytest.approx(0.5, rel=1e-3)
        assert report["comparison"]["std_domain_deviation_relative"] == pytest.approx(1.0, rel=1e-3)

    def test_per_label_table_lands_in_the_report(self, v1_ckpt: Path, replay_dataset_on_distribution: Path, tmp_path: Path) -> None:
        report_path = tmp_path / "scale_factor_report.json"
        ScaleFactorSanityCheck(
            reference=ScaleFactorReference.from_checkpoint(v1_ckpt),
            dataset_json=replay_dataset_on_distribution,
            embedding_base_dir=tmp_path / "embeddings",
            n_samples=10,
            seed=42,
            threshold=0.2,
            report_path=report_path,
        ).run()
        per_label = json.loads(report_path.read_text())["estimate"]["per_label"]
        assert set(per_label) == set(LABELS)
        assert all(per_label[label]["std_mean"] == pytest.approx(1.0 / V1_SCALE_FACTOR, rel=1e-5) for label in LABELS)

    def test_estimated_scale_factor_is_the_reciprocal_mean_std(self, tmp_path: Path) -> None:
        """The estimator mirrors training's 1/torch.std(first batch): batch_size=1 -> one volume's std."""
        entries, _ = dataset_entries({label: 1.25 for label in LABELS}, tmp_path)
        sample = StratifiedLatentSample(entries=entries, embedding_base_dir=tmp_path / "embeddings", n_samples=10, seed=42)
        drawn = sample.draw()
        mean_std = sum(std for _entry, std in drawn) / len(drawn)
        assert 1.0 / mean_std == pytest.approx(0.8, rel=1e-6)
        assert math.isclose(1.0 / 1.25, 0.8)


class TestMain:
    def test_end_to_end_pass_against_a_checkpoint(
        self, v1_ckpt: Path, replay_dataset_on_distribution: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        report_path = tmp_path / "report.json"
        sys.argv = [
            "check_scale_factor",
            "--dataset-json",
            str(replay_dataset_on_distribution),
            "--embedding-base-dir",
            str(tmp_path / "embeddings"),
            "--v1-ckpt",
            str(v1_ckpt),
            "--n-samples",
            "10",
            "--report",
            str(report_path),
        ]
        main()
        assert json.loads(report_path.read_text())["verdict"] == "PASS"
        assert "PASS" in capsys.readouterr().out

    def test_end_to_end_block_exits_nonzero(self, v1_ckpt: Path, replay_dataset_off_distribution: Path, tmp_path: Path) -> None:
        sys.argv = [
            "check_scale_factor",
            "--dataset-json",
            str(replay_dataset_off_distribution),
            "--embedding-base-dir",
            str(tmp_path / "embeddings"),
            "--v1-ckpt",
            str(v1_ckpt),
            "--n-samples",
            "10",
            "--report",
            str(tmp_path / "report.json"),
        ]
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 1

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_scale_factor", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "--v1-ckpt" in result.stdout
