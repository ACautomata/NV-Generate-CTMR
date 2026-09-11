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

"""Tests for the initial-noise seam of ``scripts.diff_model_infer`` (ticket T8, issue #24).

``run_inference`` used to draw its own noise, which left the only randomness in a
generation invisible to whoever called it. The batch generator needs it visible: the
frozen baseline's gen-gen pairing (spec #13 section 5.1) is the guarantee that two
models start from the same (label, seed, index) noise, and that only holds if the
stream position belongs to the caller. ``LatentNoise`` is where it lives.
"""

import torch
from monai.utils import set_determinism

from scripts.diff_model_infer import LatentNoise

OUTPUT_SIZE = (256, 256, 128)


class TestLatentNoise:
    def test_shape_divides_the_output_size_by_the_latent_divisor(self) -> None:
        noise = LatentNoise(latent_channels=4, divisor=64, device=torch.device("cpu"))

        assert noise.shape(OUTPUT_SIZE) == (1, 4, 4, 4, 2)

    def test_the_same_seed_yields_the_same_noise(self) -> None:
        noise = LatentNoise(latent_channels=4, divisor=64, device=torch.device("cpu"))

        set_determinism(42)
        first = noise.draw(OUTPUT_SIZE)
        set_determinism(42)
        second = noise.draw(OUTPUT_SIZE)

        assert torch.equal(first, second)

    def test_a_different_seed_yields_different_noise(self) -> None:
        """The other direction: a draw that ignored the seed would sail through the test above."""
        noise = LatentNoise(latent_channels=4, divisor=64, device=torch.device("cpu"))

        set_determinism(42)
        first = noise.draw(OUTPUT_SIZE)
        set_determinism(1337)
        second = noise.draw(OUTPUT_SIZE)

        assert not torch.equal(first, second)

    def test_a_draw_lands_on_the_configured_device(self) -> None:
        """CUDA keeps one default generator per device, so a CPU draw would not be this noise."""
        noise = LatentNoise(latent_channels=4, divisor=64, device=torch.device("cpu"))

        assert noise.draw(OUTPUT_SIZE).device.type == "cpu"
