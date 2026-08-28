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

"""Convergence gate: per-stage checkpoint payload schemas (ADR-0011, #111).

Pins the pre-#111 payload key sets verbatim: P1 carries ``unet_state_dict``,
P2/P3 carry ``controlnet_state_dict``; epoch/loss/num_train_timesteps/
scale_factor are the shared skeleton. The stage kernels stay in the thin script
entries, so this test imports them (torch/monai level, installed in the CI
full-dependency tier per ADR-0015 §6).
"""

from types import SimpleNamespace

from ctmr.application.generation.cross_modal.train import TrainKernel as CrossModalTrainKernel
from ctmr.application.generation.mask.train import TrainKernel as MaskTrainKernel
from ctmr.application.generation.modality_label.train import TrainKernel as ModalityLabelTrainKernel

# The pre-#111 checkpoint payload key sets, verbatim (do not edit).
P1_PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict"]
P2_PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "controlnet_state_dict"]
P3_PAYLOAD_KEYS = P2_PAYLOAD_KEYS
FAKE_STATE = {"fake": "weights"}


class _FakeModule:
    def state_dict(self):
        return dict(FAKE_STATE)


def _kernel_args(noise_timesteps=1000):
    return SimpleNamespace(noise_scheduler={"num_train_timesteps": noise_timesteps})


def test_p1_payload_key_set_is_kept():
    kernel = ModalityLabelTrainKernel(_kernel_args(), device=None, logger=None, local_rank=0)
    kernel._unet = _FakeModule()
    payload = kernel.checkpoint_payload(3, 0.25, 1.0)
    assert list(payload) == P1_PAYLOAD_KEYS
    assert payload["unet_state_dict"] == FAKE_STATE
    assert payload["num_train_timesteps"] == 1000
    assert payload["scale_factor"] == 1.0


def test_p2_payload_key_set_is_kept():
    args = SimpleNamespace(
        noise_scheduler={"num_train_timesteps": 1000},
        controlnet_train={"weighted_loss": 100, "weighted_loss_label": [129, 130, 131]},
    )
    kernel = MaskTrainKernel(args, device=None, logger=None, local_rank=0)
    kernel._controlnet = _FakeModule()
    payload = kernel.checkpoint_payload(5, 0.5, 1.0)
    assert list(payload) == P2_PAYLOAD_KEYS
    assert payload["controlnet_state_dict"] == FAKE_STATE


def test_p3_payload_key_set_is_kept():
    args = SimpleNamespace(
        noise_scheduler={"num_train_timesteps": 1000},
        controlnet_train={"weighted_loss": 100, "weighted_loss_label": [129, 130, 131]},
    )
    kernel = CrossModalTrainKernel(args, device=None, logger=None, local_rank=0)
    kernel._controlnet = _FakeModule()
    payload = kernel.checkpoint_payload(7, 0.75, 1.0)
    assert list(payload) == P3_PAYLOAD_KEYS
    assert payload["controlnet_state_dict"] == FAKE_STATE
