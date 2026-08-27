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

"""Pure loss mathematics for the generation study (no I/O — see ADR-0015 §2)."""

import torch


def KL_loss(z_mu, z_sigma):  # noqa: N802
    """
    Compute the Kullback-Leibler (KL) divergence loss for a variational autoencoder (VAE).

    The KL divergence measures how one probability distribution diverges from a second, expected probability distribution.
    In the context of VAEs, this loss term ensures that the learned latent space distribution is close to a standard normal distribution.

    Args:
        z_mu (torch.Tensor): Mean of the latent variable distribution, shape [N,C,H,W,D] or [N,C,H,W].
        z_sigma (torch.Tensor): Standard deviation of the latent variable distribution, same shape as 'z_mu'.

    Returns:
        torch.Tensor: The computed KL divergence loss, averaged over the batch.
    """
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]
