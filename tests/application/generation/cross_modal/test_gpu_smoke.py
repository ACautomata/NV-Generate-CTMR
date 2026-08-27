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

"""GPU-tier smoke of the cross_modal training kernel with the real MAISI networks.

Acceptance criterion 6 (ticket 08): the family's GPU-level self-check is a
pytest gpu-marked test, executed only where a CUDA device and the opt-in flag
are present (server runbook) -- locally and in CI it auto-skips via
``tests/conftest.py`` while the torch-marked CPU gates carry the merge decision.

The value over the CPU fakes in ``test_kernels.py`` is the real network face:
the ``define_instance``-built MAISI ``ControlNetMaisi`` / ``DiffusionModelUNetMaisi``
must accept the exact keyword contract ``TrainKernel.train_batch`` issues and
their down/mid residuals must flow into the UNet (tiny fakes cannot catch a
MONAI forward-signature drift). The definitions below mirror the pinned
production ``config_network_p3.json`` topology at toy width so the step fits any
server GPU.

Known observation from the local tier (recorded for the first server run):
monai 1.6.0's ``apps.generation.maisi`` fork merges the ControlNet residuals
with an in-place ``+=`` inside ``DiffusionModelUNetMaisi._apply_down_blocks``
(the stock ``nets`` implementation is out-of-place there), which on some
torch builds trips an autograd version check during backward. The server's
pinned environment trained this very path before, so its verdict on this gate
is the authoritative one -- if it reproduces here, that is a genuine finding
of this self-check, not a false alarm.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

import torch  # noqa: E402  (importorskip must precede the torch-dependent imports)
from monai.networks.schedulers import RFlowScheduler  # noqa: E402

from ctmr.application.generation.cross_modal.train import TrainKernel  # noqa: E402
from ctmr.infrastructure.maiisi_engine.instance_definition import define_instance  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.gpu]


def _network_defs():
    """The production config_network_p3.json topology at toy width."""
    return SimpleNamespace(
        latent_channels=4,
        controlnet_def={
            "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
            "num_channels": [32, 64],
            "attention_levels": [False, False],
            "num_head_channels": [0, 32],
            "num_res_blocks": 1,
            "use_flash_attention": False,
            "conditioning_embedding_in_channels": 4,
            "conditioning_embedding_num_channels": [8],
            "num_class_embeds": 128,
            "resblock_updown": True,
            "include_fc": True,
        },
        diffusion_unet_def={
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_channels": [32, 64],
            "attention_levels": [False, False],
            "num_head_channels": [0, 32],
            "num_res_blocks": 1,
            "use_flash_attention": False,
            "include_top_region_index_input": False,
            "include_bottom_region_index_input": False,
            "include_spacing_input": True,
            "num_class_embeds": 128,
            "resblock_updown": True,
            "include_fc": True,
        },
    )


def _kernel_on(device):
    args = SimpleNamespace(
        controlnet_train={"weighted_loss": 100.0, "weighted_loss_label": [129]},
        noise_scheduler={"num_train_timesteps": 1000},
    )
    kernel = TrainKernel(args, device=device, logger=logging.getLogger("gpu-smoke"), local_rank=0)
    kernel._scale_factor = torch.tensor(1.05, device=device.type)
    # The production scheduler shape (pinned recipe values from config_network_p3.json).
    kernel._noise_scheduler = RFlowScheduler(
        num_train_timesteps=1000, use_discrete_timesteps=False, use_timestep_transform=True, sample_method="uniform", scale=1.4
    )
    kernel._controlnet = define_instance(_network_defs(), "controlnet_def").to(device)
    kernel._unet = define_instance(_network_defs(), "diffusion_unet_def").to(device)
    return kernel


def test_train_batch_closes_with_the_real_maisi_networks_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    kernel = _kernel_on(torch.device("cuda"))

    image = torch.randn(1, 4, 16, 16, 8, device="cuda")
    src_image = torch.randn(1, 4, 16, 16, 8, device="cuda")
    label = torch.zeros(1, 1, 32, 32, 16, device="cuda")
    label[..., :16, :, :] = 129  # exercise the weighted-tumour-subregion branch
    batch = {
        "image": image,
        "src_image": src_image,
        "label": label,
        "spacing": torch.ones(1, 3, device="cuda"),
        "modality": torch.tensor([29], device="cuda"),
    }

    loss = kernel.train_batch(batch)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    cond_grad = kernel._controlnet.controlnet_cond_embedding.conv_in.weight.grad  # type: ignore[index]
    assert cond_grad is not None and torch.isfinite(cond_grad).all()
    unet_grad = kernel._unet.out.weight.grad  # type: ignore[index]
    assert unet_grad is not None and torch.isfinite(unet_grad).all()
