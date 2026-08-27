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

"""Entry-point gates for the VAE training command (scripts.train_vae).

The end-to-end training loop itself is pinned for real in
tests/application/test_vae_train.py (alternating G/D updates, validation
scores); this module covers the caller-side seams the notebook used to own:
the three-config merge, the data-list contract, and the argparse surface.
Torch-level: runs on CPU in the CI torch-stack job.
"""

import json

import pytest

from scripts.train_vae import load_data_lists, load_settings, main


@pytest.mark.torch
def test_train_vae_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.torch
def test_load_settings_merges_three_config_layers(tmp_path):
    network = tmp_path / "network.json"
    network.write_text(json.dumps({"spatial_dims": 3, "image_channels": 1, "autoencoder_def": {"num_channels": [64]}}))
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"model_dir": "models/", "finetune": True, "trained_autoencoder_path": "models/v1.pt"}))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_option": {"random_aug": False, "spacing_type": "original", "spacing": None, "select_channel": 0},
                "autoencoder_train": {
                    "batch_size": 2,
                    "patch_size": [8, 8, 8],
                    "lr": 1e-4,
                    "amp": False,
                    "n_epochs": 1,
                    "val_interval": 10,
                },
            }
        )
    )

    settings = load_settings(str(network), str(config), str(environment))

    assert settings.spatial_dims == 3  # network layer
    assert settings.model_dir == "models/" and settings.finetune is True  # environment layer
    assert settings.batch_size == 2 and settings.patch_size == [8, 8, 8]  # training layer
    assert settings.amp is False and settings.select_channel == 0
    assert settings.autoencoder_def == {"num_channels": [64]}


def test_load_data_lists_requires_a_valid_class(tmp_path):
    good = tmp_path / "train_ct.json"
    good.write_text(json.dumps([{"image": "ct_1.nii.gz", "class": "ct"}, {"image": "ct_2.nii.gz", "class": "ct"}]))

    assert load_data_lists([str(good)]) == [
        {"image": "ct_1.nii.gz", "class": "ct"},
        {"image": "ct_2.nii.gz", "class": "ct"},
    ]

    bad = tmp_path / "train_broken.json"
    bad.write_text(json.dumps([{"image": "x.nii.gz", "class": "pet"}]))
    with pytest.raises(ValueError, match="class"):
        load_data_lists([str(bad)])
