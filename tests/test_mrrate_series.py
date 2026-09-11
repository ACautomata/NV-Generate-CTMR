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

"""Tests for scripts.mrrate_series — the one owner of the MR-RATE on-disk layout (ticket T6 #22).

The layout is pinned against literals taken from the real manifests and the real study
zip structure, so the download, filter, skull-strip and encode stages cannot drift apart.
"""

import pytest

from scripts.create_replay_manifest import MODALITIES, MrRateSeries
from scripts.mrrate_series import SKULL_STRIPPED_LABEL, WHOLE_BRAIN_LABEL, MrRateVariant


@pytest.fixture
def variant() -> MrRateVariant:
    """A real manifest row: patient 101, study 5KNUAYLYTN, series t1w-raw-sag in batch00."""
    return MrRateVariant.for_series("batch00", "5KNUAYLYTN", "t1w-raw-sag")


@pytest.fixture
def from_image_path() -> MrRateVariant:
    """The same acquisition, built the way the download stage sees it: from the zip member path."""
    return MrRateVariant("mri/batch07/AAA111AAA/img/AAA111AAA_t2w-raw-axi.nii.gz")


class TestMrRateVariantPaths:
    def test_whole_brain_is_the_published_image(self, variant: MrRateVariant) -> None:
        assert variant.whole_brain_path == "mri/batch00/5KNUAYLYTN/img/5KNUAYLYTN_t1w-raw-sag.nii.gz"

    def test_mask_mirrors_the_image_into_seg_with_the_brain_mask_suffix(self, variant: MrRateVariant) -> None:
        assert variant.mask_path == "mri/batch00/5KNUAYLYTN/seg/5KNUAYLYTN_t1w-raw-sag_brain-mask.nii.gz"

    def test_skull_stripped_twin_sits_next_to_the_image(self, variant: MrRateVariant) -> None:
        # Same directory as the image (not seg/), so one data_base_dir reaches both entries.
        assert variant.skull_stripped_path == "mri/batch00/5KNUAYLYTN/img/5KNUAYLYTN_t1w-raw-sag_skull_stripped.nii.gz"

    def test_the_three_paths_never_collide(self, variant: MrRateVariant) -> None:
        assert len({variant.whole_brain_path, variant.mask_path, variant.skull_stripped_path}) == 3

    def test_identifiers_survive_the_round_trip(self, variant: MrRateVariant) -> None:
        assert variant.study_uid == "5KNUAYLYTN"
        assert variant.series_id == "t1w-raw-sag"
        assert variant.modality == "t1w"

    def test_parses_a_suffixed_series_id_without_eating_the_study_uid(self, from_image_path: MrRateVariant) -> None:
        # A leading-uid remnant would make "AAA111AAA_t2w-raw-axi" parse back as modality "aaa111aaa".
        assert from_image_path.series_id == "t2w-raw-axi"
        assert from_image_path.modality == "t2w"
        assert from_image_path.study_uid == "AAA111AAA"

    def test_parses_a_numbered_duplicate_series(self) -> None:
        variant = MrRateVariant("mri/batch18/2CKPHAULPF/img/2CKPHAULPF_t1w-raw-axi-3.nii.gz")
        assert variant.series_id == "t1w-raw-axi-3"
        assert variant.modality == "t1w"

    def test_every_modality_keeps_its_directory_layout(self) -> None:
        for modality in MODALITIES:
            built = MrRateVariant.for_series("batch03", "STUDY123", f"{modality}-raw-axi")
            directory = built.whole_brain_path.rsplit("/", 1)[0]
            assert built.skull_stripped_path.startswith(f"{directory}/")
            assert built.mask_path.startswith("mri/batch03/STUDY123/seg/")


class TestMrRateVariantLabels:
    def test_whole_brain_label_table_follows_the_spec(self) -> None:
        """Spec section 3.2's table, pinned against literals: these strings are the contract with v1."""
        assert dict(WHOLE_BRAIN_LABEL) == {
            "t1w": "mri_t1",
            "t2w": "mri_t2",
            "flair": "mri_flair",
            "swi": "mri_swi",
            "mra": "mri_mra",
        }

    def test_a_series_id_resolves_to_its_modalitys_label(self) -> None:
        for modality, label in WHOLE_BRAIN_LABEL.items():
            variant = MrRateVariant.for_series("batch00", "STUDY123", f"{modality}-raw-axi")
            assert variant.label == label

    def test_skull_stripped_label_is_the_dual_twin(self) -> None:
        variant = MrRateVariant.for_series("batch00", "STUDY123", "flair-raw-sag")
        assert variant.label == "mri_flair"
        assert variant.skull_stripped_label == "mri_flair_skull_stripped"

    def test_skull_stripped_labels_cover_all_five_modalities(self) -> None:
        assert SKULL_STRIPPED_LABEL == {
            "t1w": "mri_t1_skull_stripped",
            "t2w": "mri_t2_skull_stripped",
            "flair": "mri_flair_skull_stripped",
            "swi": "mri_swi_skull_stripped",
            "mra": "mri_mra_skull_stripped",
        }

    def test_training_entries_are_the_dual_derivation_whole_brain_first(self) -> None:
        """The two records every consumer (T6 finalize, T7 merge) states once via this factory."""
        variant = MrRateVariant.for_series("batch00", "STUDY123", "t1w-raw-axi")
        entries = variant.training_entries()
        assert [(entry.image, entry.modality) for entry in entries] == [
            ("mri/batch00/STUDY123/img/STUDY123_t1w-raw-axi.nii.gz", "mri_t1"),
            ("mri/batch00/STUDY123/img/STUDY123_t1w-raw-axi_skull_stripped.nii.gz", "mri_t1_skull_stripped"),
        ]


class TestManifestPathAgreement:
    """The manifest generator and the download stage must name the same files."""

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_variant_reads_back_the_exact_paths_the_manifest_writes(self, modality: str) -> None:
        series = MrRateSeries(
            batch="batch11", patient_uid="7", study_uid="STUDY456", series_id=f"{modality}-raw-axi-2", modality=modality, series_number=1.0
        )
        variant = MrRateVariant(series.image_path)
        assert variant.whole_brain_path == series.image_path
        assert variant.mask_path == series.mask_path
        assert variant.skull_stripped_path == series.image_path.replace(".nii.gz", "_skull_stripped.nii.gz")

    def test_variant_reproduces_the_manifest_identifiers(self) -> None:
        series = MrRateSeries(
            batch="batch02", patient_uid="10156", study_uid="G23BXKXD5U", series_id="t1w-raw-axi", modality="t1w", series_number=2.0
        )
        variant = MrRateVariant(series.image_path)
        assert variant.study_uid == series.study_uid
        assert variant.series_id == series.series_id
        assert variant.label == series.label

    def test_for_series_and_the_manifest_builder_agree(self) -> None:
        built = MrRateVariant.for_series("batch07", "AAA111AAA", "t2w-raw-axi")
        from_manifest = MrRateVariant(
            MrRateSeries(
                batch="batch07", patient_uid="1", study_uid="AAA111AAA", series_id="t2w-raw-axi", modality="t2w", series_number=2.0
            ).image_path
        )
        assert built.whole_brain_path == from_manifest.whole_brain_path
        assert built.mask_path == from_manifest.mask_path
        assert built.skull_stripped_path == from_manifest.skull_stripped_path
