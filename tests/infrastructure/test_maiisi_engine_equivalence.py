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


"""Import resolution + equivalence smoke at the new home (issue #134)."""

import importlib
import inspect
import json
from pathlib import Path

import pytest

# Heavy-dep guards: whole module skips when deep deps are absent (torch marker).
pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("tqdm")
pytest.importorskip("nibabel")

pytestmark = pytest.mark.torch

ENGINE_PKG = "ctmr.infrastructure.maiisi_engine"

VENDORED_MODULES = [
    "diff_model_setting",
    "diff_model_train",
    "diff_model_infer",
    "diff_model_create_training_data",
    "sample",
    "utils_infer",
]


def _pair(name):
    new_mod = importlib.import_module(ENGINE_PKG + "." + name)
    old_mod = importlib.import_module("scripts." + name)
    assert new_mod.__name__ == ENGINE_PKG + "." + name
    return old_mod, new_mod


def test_engine_modules_importable_at_new_home():
    repo_root = Path(__file__).resolve().parents[2]
    engine_root = repo_root / "src" / "ctmr" / "infrastructure" / "maiisi_engine"
    for name in VENDORED_MODULES:
        module = importlib.import_module(ENGINE_PKG + "." + name)
        assert Path(module.__file__).is_relative_to(engine_root), name


def test_utils_bridge_forwards_object_identity():
    bridge = importlib.import_module(ENGINE_PKG + ".utils")
    origin = importlib.import_module("scripts.utils")
    assert bridge.define_instance is origin.define_instance
    assert bridge.dynamic_infer is origin.dynamic_infer
    assert bridge.get_body_region_index_from_mask is origin.get_body_region_index_from_mask


def test_load_config_equivalent_on_synthetic_configs(tmp_path):
    env_config = {
        "model_dir": str(tmp_path / "ckpt"),
        "embedding_base_dir": str(tmp_path / "emb"),
        "json_data_list": str(tmp_path / "datalist.json"),
        "output_dir": str(tmp_path / "out"),
        "output_prefix": "smoke",
    }

    model_config = {
        "diffusion_unet_inference": {
            "dim": [128, 128, 128],
            "spacing": [1.0, 1.0, 1.0],
            "modality": 1,
            "cfg_guidance_scale": 0.0,
            "num_inference_steps": 5,
            "random_seed": 42,
        },
        "diffusion_unet_train": {"lr": 0.0001, "n_epochs": 2, "batch_size": 1, "cache_rate": 0.0},
        "noise_scheduler": {"num_train_timesteps": 1000},
    }

    model_def = {
        "diffusion_unet_def": {
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": [8, 8],
            "attention_levels": [False, False],
        },
        "autoencoder_def": {"latent_channels": 4},
    }

    paths = []
    for filename, payload in (("env.json", env_config), ("model.json", model_config), ("def.json", model_def)):
        path = tmp_path / filename
        path.write_text(json.dumps(payload))
        paths.append(str(path))

    old_mod = importlib.import_module("scripts.diff_model_setting")
    new_mod = importlib.import_module(ENGINE_PKG + ".diff_model_setting")
    ns_old = old_mod.load_config(*paths)
    ns_new = new_mod.load_config(*paths)

    assert vars(ns_old) == vars(ns_new)


def test_public_symbol_sources_parity_with_scripts_home():
    compared = 0
    for name in VENDORED_MODULES:
        old_mod, new_mod = _pair(name)
        for attr in dir(new_mod):
            if attr.startswith("_"):
                continue
            obj = getattr(new_mod, attr)
            if getattr(obj, "__module__", "") != new_mod.__name__:
                continue  # imported symbol (bridge/re-export), not defined here
            twin = getattr(old_mod, attr, None)
            assert twin is not None, name + "." + attr + " missing from scripts home"
            src_new = inspect.getsource(obj)
            src_old = inspect.getsource(twin)
            assert src_new == src_old, "runtime source drift at " + name + "." + attr
            compared += 1

    # Sanity floor: the loop must have compared substantive definitions.
    assert compared >= 20, "only " + str(compared) + " symbols compared -- inventory drift?"
