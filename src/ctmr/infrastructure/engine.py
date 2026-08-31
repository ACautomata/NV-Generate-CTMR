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

"""MaisiEngine -- the frozen maisi_engine functions mounted as the GenerationEngine port adapter (ADR-0019 §3, #269).

Every method delegates one frozen function of ``ctmr.infrastructure.maisi_engine``
(the vendored snapshot, whose behavior tests/infrastructure/maisi_engine/
test_engine_smoke.py pins); this module adds the delegation and nothing else --
the engine functions stay byte-for-byte frozen, the families reach them only
through this port adapter. Distributed session bootstrap
(``initialize_distributed``) is not engine loading/inference and stays outside
the port (its home is ruled by the B2 topology work, #278).
"""

import torch

from ctmr.domain.logging import Logger
from ctmr.infrastructure.maisi_engine import diff_model_infer, diff_model_setting, inference_primitives, instance_definition, utils_infer


class MaisiEngine:
    """The GenerationEngine realization over the frozen maisi_engine functions."""

    def load_config(self, env_config_path, model_config_path, model_def_path):
        return diff_model_setting.load_config(env_config_path, model_config_path, model_def_path)

    def define_instance(self, args, instance_def_key):
        return instance_definition.define_instance(args, instance_def_key)

    def load_models(self, args, device: torch.device, logger: Logger):
        return diff_model_infer.load_models(args, device, logger)

    def load_image_models(self, args, device: torch.device):
        return utils_infer.load_image_models(args, device)

    def dynamic_infer(self, inferer, model, images) -> torch.Tensor:
        return inference_primitives.dynamic_infer(inferer, model, images)

    def recon_model(self, autoencoder, scale_factor) -> torch.nn.Module:
        return utils_infer.ReconModel(autoencoder=autoencoder, scale_factor=scale_factor)
