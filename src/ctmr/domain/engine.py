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

"""GenerationEngine: the engine loading and inference port (ADR-0019 §3, #269).

The engine face the generation families drive, spelled as a port: config
parsing (``load_config``), model loading (``define_instance`` builds a network
from the parsed defs; ``load_models``/``load_image_models`` assemble the
checkpoint-backed network sets) and the inference primitives
(``dynamic_infer`` -- the size-dispatched sliding-window-or-direct call --
and ``recon_model``, the latent decode wrapper factory). Protocol only -- the
concrete adapter mounting the frozen maisi_engine functions lives in
``ctmr.infrastructure.engine``.
"""

from argparse import Namespace
from typing import Protocol, runtime_checkable

import torch

from ctmr.domain.logging import Logger


@runtime_checkable
class GenerationEngine(Protocol):
    """模型加载、配置解析与推理原语的适配面 (ADR-0019 §3)."""

    def load_config(self, env_config_path: str, model_config_path: str, model_def_path: str) -> Namespace:
        """Parse the env/model/def json triple into one config namespace (later files win)."""
        ...

    def define_instance(self, args: Namespace, instance_def_key: str):
        """Instantiate the network (or scheduler) named by the instance-def key."""
        ...

    def load_models(self, args: Namespace, device: torch.device, logger: Logger) -> tuple:
        """Load the img2img pair: autoencoder + diffusion UNet + scale factor."""
        ...

    def load_image_models(self, args: Namespace, device: torch.device) -> tuple:
        """Load the image-side assembly: autoencoder + UNet + ControlNet + scale + scheduler."""
        ...

    def dynamic_infer(self, inferer, model, images) -> torch.Tensor:
        """Run one inference call: the model directly on a small input, the inferer otherwise."""
        ...

    def recon_model(self, autoencoder, scale_factor) -> torch.nn.Module:
        """Build the latent decode wrapper (z -> autoencoder decode, scale-divided)."""
        ...
