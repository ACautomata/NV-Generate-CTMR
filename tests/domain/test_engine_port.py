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

"""Contract tests for the generation-engine port (ADR-0019 §3, #269).

The port spells the engine face the generation families drive -- config
parsing (``load_config``), model loading (``define_instance``/``load_models``/
``load_image_models``) and the inference primitives (``dynamic_infer``/
``recon_model``). A fake adapter drives the port contract end to end; the real
``ctmr.infrastructure.engine.MaisiEngine`` adapter must satisfy the same port
and delegate to the frozen maisi_engine functions (whose behavior the engine
smoke suite pins).
"""

import argparse
import json
import logging

import pytest
import torch

from ctmr.domain.engine import GenerationEngine

pytestmark = pytest.mark.torch


class FakeGenerationEngine:
    """In-memory engine stand-in: canned instances, a probe model, calls counted."""

    def __init__(self):
        self.calls = []
        self.builds = {"diffusion_unet_def": IdentityModel()}

    def load_config(self, env_config_path, model_config_path, model_def_path):
        self.calls.append("load_config")
        return argparse.Namespace(env=env_config_path, model=model_config_path, defs=model_def_path)

    def define_instance(self, args, instance_def_key):
        self.calls.append(f"define_instance:{instance_def_key}")
        return self.builds.get(instance_def_key, f"instance<{instance_def_key}>")

    def load_models(self, args, device, logger):
        self.calls.append("load_models")
        return "autoencoder", "unet", 0.3

    def load_image_models(self, args, device):
        self.calls.append("load_image_models")
        return "autoencoder", "unet", "controlnet", torch.tensor(0.3), "scheduler"

    def dynamic_infer(self, inferer, model, images):
        self.calls.append("dynamic_infer")
        return model(images)

    def recon_model(self, autoencoder, scale_factor):
        self.calls.append("recon_model")
        return ("recon", autoencoder, scale_factor)


class IdentityModel(torch.nn.Module):
    def forward(self, images):
        return images


def probe_through_the_port(engine):
    """One consumer flow the families share, driven purely through the port."""
    args = engine.load_config("env.json", "model.json", "def.json")
    unet = engine.define_instance(args, "diffusion_unet_def")
    synthetic = engine.dynamic_infer(None, unet, torch.zeros(1, 1, 2, 2, 2))
    recon = engine.recon_model("autoencoder", 0.3)
    return args, unet, synthetic, recon


def test_the_fake_adapter_drives_the_port_contract():
    engine = FakeGenerationEngine()
    assert isinstance(engine, GenerationEngine)

    args, unet, synthetic, recon = probe_through_the_port(engine)
    assert args.env == "env.json"
    assert isinstance(unet, IdentityModel)
    assert synthetic.shape == (1, 1, 2, 2, 2)
    assert recon == ("recon", "autoencoder", 0.3)
    assert engine.calls == ["load_config", "define_instance:diffusion_unet_def", "dynamic_infer", "recon_model"]


def test_the_real_adapter_satisfies_the_port():
    from ctmr.infrastructure.engine import MaisiEngine

    assert isinstance(MaisiEngine(), GenerationEngine)


def test_the_real_adapter_load_config_merges_the_three_json_files(tmp_path):
    from ctmr.infrastructure.engine import MaisiEngine

    env, model, defs = {"env_only": True}, {"overlap": "model"}, {"overlap": "def"}
    paths = []
    for name, payload in [("env.json", env), ("model.json", model), ("def.json", defs)]:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        paths.append(str(path))

    args = MaisiEngine().load_config(*paths)
    assert vars(args)["env_only"] is True
    assert vars(args)["overlap"] == "def"  # env -> model -> def merge order, as the frozen loader does


def test_the_real_adapter_define_instance_builds_the_configured_object():
    from ctmr.infrastructure.engine import MaisiEngine

    args = argparse.Namespace(noise_scheduler={"_target_": "monai.networks.schedulers.DDPMScheduler", "num_train_timesteps": 5})
    scheduler = MaisiEngine().define_instance(args, "noise_scheduler")
    assert scheduler.num_train_timesteps == 5


def test_the_real_adapter_dynamic_infer_delegates_the_primitive():
    from monai.inferers.inferer import SlidingWindowInferer

    from ctmr.infrastructure.engine import MaisiEngine

    image = torch.zeros(1, 1, 4, 4, 4)
    inferer = SlidingWindowInferer(roi_size=(2, 2, 2))
    out = MaisiEngine().dynamic_infer(inferer, IdentityModel(), image)
    assert torch.equal(out, image)  # sliding-window over an identity network still returns the volume


def test_the_real_adapter_recon_model_wraps_the_autoencoder():
    from ctmr.infrastructure.engine import MaisiEngine
    from ctmr.infrastructure.maisi_engine.utils_infer import ReconModel

    recon = MaisiEngine().recon_model(autoencoder="ae", scale_factor=0.3)
    assert isinstance(recon, ReconModel)
    assert recon.autoencoder == "ae" and recon.scale_factor == 0.3


def test_the_real_adapter_checkpoint_loaders_delegate_to_the_frozen_functions(monkeypatch):
    from ctmr.infrastructure.engine import MaisiEngine
    from ctmr.infrastructure.maisi_engine import diff_model_infer, utils_infer

    seen = {}

    def fake_load_models(args, device, logger):
        seen["load_models"] = (args, device, logger)
        return "autoencoder", "unet", 0.3

    def fake_load_image_models(args, device):
        seen["load_image_models"] = (args, device)
        return "autoencoder", "unet", "controlnet", 0.3, "scheduler"

    monkeypatch.setattr(diff_model_infer, "load_models", fake_load_models)
    monkeypatch.setattr(utils_infer, "load_image_models", fake_load_image_models)

    args = argparse.Namespace(trained_autoencoder_path="ae.pt")
    device = torch.device("cpu")
    logger = logging.getLogger("engine-port")  # the Logger port's stdlib realization; passed verbatim

    assert MaisiEngine().load_models(args, device, logger) == ("autoencoder", "unet", 0.3)
    assert MaisiEngine().load_image_models(args, device) == ("autoencoder", "unet", "controlnet", 0.3, "scheduler")
    assert seen["load_models"] == (args, device, logger)
    assert seen["load_image_models"] == (args, device)
