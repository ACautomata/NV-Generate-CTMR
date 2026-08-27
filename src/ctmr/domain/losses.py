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

"""Pure VAE loss math (ADR-0015 §2).

``kl_loss`` is extracted verbatim from ``scripts/utils.KL_loss`` (issue #142;
from->to mapping declared in the PR -- the ``scripts`` copy stays until M5
retires it).
"""

import torch


def kl_loss(z_mu, z_sigma):
    """Kullback-Leibler divergence between N(z_mu, z_sigma^2) and N(0, 1),
    summed over all non-batch dims and averaged over the batch.

    Keeps the learned latent distribution close to a standard normal -- the
    regulariser term of the VAE objective.

    Args:
        z_mu: mean of the latent variable distribution, [N,C,H,W,D] or [N,C,H,W].
        z_sigma: standard deviation, same shape as ``z_mu``.

    Returns:
        torch.Tensor: scalar KL divergence averaged over the batch.
    """
    eps = 1e-10
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl) / kl.shape[0]
