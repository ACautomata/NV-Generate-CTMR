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

"""Static validation of the BRATS finetune training configs (ticket T3, issue #19).

Covers spec #13 §②.2 (new modality labels 40–43), §④.3 (snapshot interval +
derived training config) and §④.4 (per-N env configs with the v1 checkpoint
reachable only through the read-only ``existing_ckpt_filepath``).
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

V1_CKPT_FILENAME = "diff_unet_3d_rflow-mr-brain_v1.pt"
BRATS_LABELS = {"mri_t1n": 40, "mri_t1ce": 41, "mri_t2w": 42, "mri_t2f": 43}
REPLAY_SIZES = (300, 500, 1000)


@pytest.fixture(scope="module")
def read_config():
    """Factory: parse one config JSON from the repo's configs/ directory."""

    def _read(name: str) -> dict:
        with open(CONFIGS_DIR / name) as f:
            return json.load(f)

    return _read


@pytest.fixture(scope="module")
def modality_mapping(read_config) -> dict:
    return read_config("modality_mapping.json")


@pytest.fixture(scope="module")
def brats_train_config(read_config) -> dict:
    return read_config("config_maisi_diff_model_rflow-mr-brain-brats.json")


@pytest.fixture(scope="module")
def upstream_train_config(read_config) -> dict:
    return read_config("config_maisi_diff_model_rflow-mr-brain.json")


@pytest.fixture(scope="module")
def upstream_env_config(read_config) -> dict:
    return read_config("environment_maisi_diff_model_rflow-mr-brain.json")


class TestModalityMapping:
    def test_brats_labels_present_with_specified_ints(self, modality_mapping: dict) -> None:
        for label, expected in BRATS_LABELS.items():
            assert modality_mapping[label] == expected

    def test_brats_labels_contain_mri_substring(self, modality_mapping: dict) -> None:
        # The preprocessing intensity normalization dispatches on the "mri" substring
        # (scripts/diff_model_create_training_data.py::create_transforms); a missing
        # substring would silently skip normalization for BraTS volumes.
        for label in BRATS_LABELS:
            assert "mri" in label

    def test_brats_label_ints_do_not_collide_with_existing_labels(self, modality_mapping: dict) -> None:
        values = list(modality_mapping.values())
        assert len(values) == len(set(values)), "duplicate modality ints in mapping"

    def test_unknown_still_maps_to_zero_for_cfg(self, modality_mapping: dict) -> None:
        assert modality_mapping["unknown"] == 0


class TestTrainingConfig:
    def test_n_epochs_lowered_to_300(self, brats_train_config: dict) -> None:
        assert brats_train_config["diffusion_unet_train"]["n_epochs"] == 300

    def test_save_interval_enabled_at_50(self, brats_train_config: dict) -> None:
        assert brats_train_config["diffusion_unet_train"]["save_interval"] == 50

    def test_lr_and_batch_size_unchanged(self, brats_train_config: dict, upstream_train_config: dict) -> None:
        # spec §④.2: Adam lr=1e-5 + batch=1/GPU are locked to v1 values.
        for key in ("lr", "batch_size", "cache_rate"):
            assert brats_train_config["diffusion_unet_train"][key] == upstream_train_config["diffusion_unet_train"][key]

    def test_inference_section_kept_verbatim(self, brats_train_config: dict, upstream_train_config: dict) -> None:
        # spec §⑤.1: forgetting checks run the native v1 inference conditions
        # (dim/spacing/cfg/steps), so the section must stay byte-identical in meaning.
        assert brats_train_config["diffusion_unet_inference"] == upstream_train_config["diffusion_unet_inference"]

    def test_snapshot_grid_matches_spec(self, brats_train_config: dict) -> None:
        # Epochs {50, 100, 150, 200, 250, 300} = 6 snapshots per replay size.
        n_epochs = brats_train_config["diffusion_unet_train"]["n_epochs"]
        save_interval = brats_train_config["diffusion_unet_train"]["save_interval"]
        snapshots = [e for e in range(1, n_epochs + 1) if e % save_interval == 0]
        assert snapshots == [50, 100, 150, 200, 250, 300]


class TestEnvConfigs:
    @pytest.fixture(scope="class")
    def env_configs(self, read_config) -> dict[int, dict]:
        return {n: read_config(f"environment_maisi_diff_model_rflow-mr-brain_N{n}.json") for n in REPLAY_SIZES}

    def test_model_dir_isolated_per_replay_size(self, env_configs: dict[int, dict]) -> None:
        for n, config in env_configs.items():
            assert config["model_dir"] == f"./models/brats_finetune_N{n}"

    def test_model_filename_named_by_replay_size(self, env_configs: dict[int, dict]) -> None:
        for n, config in env_configs.items():
            assert config["model_filename"] == f"diff_unet_3d_rflow-mr-brain_N{n}.pt"

    def test_existing_ckpt_readonly_points_to_v1(self, env_configs: dict[int, dict], upstream_env_config: dict) -> None:
        for config in env_configs.values():
            assert config["existing_ckpt_filepath"] == upstream_env_config["existing_ckpt_filepath"]
            assert V1_CKPT_FILENAME in config["existing_ckpt_filepath"]

    def test_v1_weights_absent_from_every_write_path(self, env_configs: dict[int, dict]) -> None:
        # spec §④.4: v1 must be physically untouchable — the write path is
        # model_dir + model_filename (every-epoch overwrite + snapshots); the v1
        # filename may appear only inside existing_ckpt_filepath.
        for n, payload in env_configs.items():
            write_path = f"{payload['model_dir']}/{payload['model_filename']}"
            assert V1_CKPT_FILENAME not in write_path
            for key, value in payload.items():
                if key != "existing_ckpt_filepath":
                    assert V1_CKPT_FILENAME not in str(value), f"v1 leak in env N{n} field {key!r}"

    def test_data_paths_follow_merged_dataset_per_replay_size(self, env_configs: dict[int, dict]) -> None:
        for n, config in env_configs.items():
            assert config["json_data_list"] == f"./dataset_rflow-mr-brain_N{n}.json"

    def test_shared_infra_matches_upstream_env(self, env_configs: dict[int, dict], upstream_env_config: dict) -> None:
        shared = ("data_base_dir", "embedding_base_dir", "output_dir", "output_prefix", "trained_autoencoder_path", "modality_mapping_path")
        for config in env_configs.values():
            for key in shared:
                assert config[key] == upstream_env_config[key], f"{key} drifted from upstream env"

    def test_replay_sizes_have_distinct_write_paths(self, env_configs: dict[int, dict]) -> None:
        write_paths = {f"{c['model_dir']}/{c['model_filename']}" for c in env_configs.values()}
        assert len(write_paths) == len(REPLAY_SIZES)
