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

"""
Mask generation module.

Generates a 3D body-region label mask from scratch using a DDPM-based latent
diffusion model conditioned on a 10-d ``anatomy_size`` vector. See
``skills/mask-generation.md`` for the algorithm walkthrough.

Also hosts the shared helper ``filter_mask_with_organs``.

Engine-side primitives are imported from their canonical homes (issue #134)
rather than duplicated: ``ReconModel`` / ``initialize_noise_latents`` from
``maiisi_engine.utils_infer``, ``dynamic_infer`` / ``check_input_ct`` /
``check_input_mr`` from ``maiisi_engine.inference_primitives``. They stay
bound to this module's namespace so ``from ctmr.infrastructure.dataio.sample_mask
import ...`` keeps resolving.
"""

import logging
import warnings

import torch
from monai.inferers.inferer import DiffusionInferer, SlidingWindowInferer
from monai.networks.schedulers import DDPMScheduler

from ctmr.infrastructure.dataio.mask_postprocess import (
    general_mask_generation_post_process,
    remap_labels,
)
from ctmr.infrastructure.maiisi_engine.inference_primitives import check_input_ct, check_input_mr, dynamic_infer  # noqa: F401
from ctmr.infrastructure.maiisi_engine.utils_infer import ReconModel, initialize_noise_latents  # noqa: F401


def ldm_conditional_sample_one_mask(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    anatomy_size,
    device,
    latent_shape,
    label_dict_remap_json,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic mask using a latent diffusion model.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        anatomy_size (torch.Tensor): Tensor specifying the desired anatomy sizes.
        device (torch.device): The device to run the computation on.
        latent_shape (tuple): The shape of the latent space.
        label_dict_remap_json (str): Path to the JSON file for label remapping.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        torch.Tensor: The generated synthetic mask.
    """
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device)
        anatomy_size = torch.FloatTensor(anatomy_size).unsqueeze(0).unsqueeze(0).half().to(device)
        # synthesize latents
        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Mask noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        # mask generator is DDPM
        inferer_ddpm = DiffusionInferer(noise_scheduler)
        latents = inferer_ddpm.sample(
            input_noise=latents,
            diffusion_model=diffusion_unet,
            scheduler=noise_scheduler,
            verbose=True,
            conditioning=anatomy_size.to(device),
        )

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_mask = dynamic_infer(inferer, recon_model, latents)
        synthetic_mask = torch.softmax(synthetic_mask, dim=1)
        synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
        # mapping raw index to 132 labels
        synthetic_mask = remap_labels(synthetic_mask, label_dict_remap_json)

        ###### post process #####
        data = synthetic_mask.squeeze().cpu().detach().numpy()

        labels = [23, 24, 26, 27, 128]
        target_tumor_label = None
        for index, size in enumerate(anatomy_size[0, 0, 5:10]):
            if size.item() != -1.0:
                target_tumor_label = labels[index]

        logging.info(f"target_tumor_label for postprocess:{target_tumor_label}")
        data = general_mask_generation_post_process(data, target_tumor_label=target_tumor_label, device=device)
        synthetic_mask = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    return synthetic_mask


def filter_mask_with_organs(combine_label, anatomy_list):
    """
    Filter a mask to only include specified organs.

    Args:
        combine_label (torch.Tensor): The input mask.
        anatomy_list (list): List of organ labels to keep.

    Returns:
        torch.Tensor: The filtered mask.
    """
    # final output mask file has shape of output_size, contains labels in anatomy_list
    # it is already interpolated to target size
    combine_label = combine_label.long()
    # filter out the organs that are not in anatomy_list
    for i in range(len(anatomy_list)):
        organ = anatomy_list[i]
        # replace it with a negative value so it will get mixed
        combine_label[combine_label == organ] = -(i + 1)
    # zero-out voxels with value not in anatomy_list
    combine_label[combine_label > 0] = 0
    # output positive values
    combine_label = -combine_label
    return combine_label
