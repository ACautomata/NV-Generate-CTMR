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

"""The P3 anchor-image → src-latent adapter (ADR-0016, issue #175).

``AnchorLatentEncoder`` builds the source-latent condition tensor of the
cross-modal families: real anchor volume → the training-matching preprocessing
(percentile 99.5 normalisation + RAS + resize to the generation grid) → VAE
encode → float.  The baseline and candidate writers share it; the retired
vendored img2img entry owned this logic as ``load_anchor_latent`` (deleted
with issue #175, git history is the provenance anchor).

Layering (ADR-0019 §1-§3, issue #274): the sliding-window inference call rides
the injected ``GenerationEngine`` port -- the encoder holds no infrastructure
address.
"""

from __future__ import annotations

import monai.transforms as monai_t
import torch
from monai.inferers.inferer import SlidingWindowInferer


class AnchorLatentEncoder:
    """The anchor-image condition adapter: preprocess → VAE encode → normalized latent (1,C,H,W,D)."""

    def __init__(self, autoencoder, device, output_size, logger, engine):
        self._autoencoder = autoencoder
        self._device = device
        self._output_size = output_size
        self._logger = logger
        self._engine = engine

    def encode(self, anchor_path: str) -> torch.Tensor:
        """锚影像 → 训练同款预处理 → encode → 归一化 latent (1,C,H,W,D),float32。"""
        transforms = monai_t.Compose(
            [
                monai_t.LoadImage(image_only=True),
                monai_t.EnsureChannelFirst(),
                monai_t.Orientation(axcodes="RAS"),
                monai_t.EnsureType(dtype=torch.float32),
                # clip=True (issue #313, series-③ T3): the T4 factory moved the
                # training encoding arm to clip=True (job C's measured verdict --
                # extrapolated >1.0 inputs leave the frozen VAE's reconstruction
                # domain); the inference anchor must encode into the same bounded
                # domain or its condition falls outside the training distribution.
                monai_t.ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
                monai_t.Resize(spatial_size=self._output_size, mode="trilinear"),
            ]
        )
        x = transforms(anchor_path).unsqueeze(0).to(self._device)  # [1,1,X,Y,Z]
        self._logger.info(f"anchor {anchor_path} -> {tuple(x.shape)}")

        # encode 走训练数据构建同款 sliding-window(roi 大于输入时整体前向)
        encode_inferer = SlidingWindowInferer(
            roi_size=[320, 320, 160],
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=0.4,
            sw_device=self._device,
            device=self._device,
        )
        with torch.amp.autocast("cuda", enabled=True):
            z = self._engine.dynamic_infer(encode_inferer, self._autoencoder.encode_stage_2_inputs, x)
        # encode 的中间激活占用大,释放缓存后再进入去噪循环(否则叠出 OOM)
        torch.cuda.empty_cache()
        # encode 在 autocast 下输出 half;去噪循环的输入统一 float32(与 P1 的 randn 一致)
        return z.float()
