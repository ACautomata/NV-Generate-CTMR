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

"""ControlNetBypass: the P2/P3 ControlNet-only bypass entity (ADR-0016, issue #172).

``ControlNetBypass`` is the independent bypass entity hung off the frozen DM:
it carries the trainable ControlNet and owns the conditioned forward paths the
P2/P3 families drive -- one conditioned residuals forward for training, and
the CFG-composed double forward (conditioned vs the unconditional counterpart
the application prepared) for sampling.  It adds no state beyond its member
and carries no checkpoint identity: the trainable lineage stays expressed by
``WeightsRef`` and the per-epoch checkpoint payload (``controlnet_state_dict``);
the runtime object is rebuilt by the application from checkpoint payloads.
"""

from __future__ import annotations

import torch


class ControlNetBypass:
    """The trainable ControlNet bypass as a runtime collaborator, not an identity.

    One member (the MONAI ControlNetMaisi) plus the conditioned forward
    behaviour: ``residuals`` runs the single-condition forward (training, and
    the cfg=0 sampling branch) and the CFG-composed batch=2 double forward
    when the unconditional counterpart travels with the call.
    """

    def __init__(self, controlnet: torch.nn.Module):
        self._controlnet = controlnet

    @property
    def controlnet(self) -> torch.nn.Module:
        return self._controlnet

    def residuals(self, noisy_latent, timesteps, controlnet_cond, modality, uncond_cond=None):
        """The conditioned (down, mid) residuals for one forward.

        With ``uncond_cond`` (the CFG>0 sampling branch) this is the legacy
        batch=2 double forward: ``x``/``timesteps`` duplicated, the condition
        pair (conditioned | unconditional), and ``class_labels = (modality |
        zeros)`` -- the ``run_controlnet_conditioned_image_dm`` CFG
        composition bit for bit.
        """
        if uncond_cond is None:
            return self._controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond, class_labels=modality)
        return self._controlnet(
            x=torch.cat([noisy_latent, noisy_latent]),
            timesteps=torch.cat([timesteps, timesteps]),
            controlnet_cond=torch.cat([controlnet_cond, uncond_cond]),
            class_labels=torch.cat([modality, torch.zeros_like(modality)]),
        )
