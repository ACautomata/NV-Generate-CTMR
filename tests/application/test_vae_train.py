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

"""Pins for the extracted VAE/GAN training orchestration (ADR-0015 §8, batch M6).

The settings lifted from ``train_vae_tutorial.ipynb`` cells 24/26/28/30 are
the subject here: the Adam eps switch on AMP, the three-segment warmup
ladder, the GradScaler init/growth constants, the INSTANCE patch
discriminator construction, and the alternating G/D loop mechanics
(detach order, scheduler stepping, per-epoch checkpoint + best-val save).
Torch-level: skips itself on light stacks via ``pytest.importorskip``
(ADR-0013 §4); runs on CPU with no weight downloads -- every loss is
injected as a lightweight fake, so ``PerceptualLoss`` is never constructed.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

import torch  # noqa: E402  (importorskip must precede the torch-dependent import)
import torch.utils.data  # noqa: E402

from ctmr.application.vae_train import (  # noqa: E402
    VaeGanTrainer,
    build_discriminator,
    build_intensity_loss,
    build_optimizers,
    build_scalers,
    build_schedulers,
    loss_weighted_sum,
    warmup_rule,
)

pytestmark = pytest.mark.torch


class TinyVaeAutoencoder(torch.nn.Module):
    """(recon, z_mu, z_sigma) stand-in: conv encode -> deterministic conv decode."""

    def __init__(self):
        super().__init__()
        self.enc_mu = torch.nn.Conv3d(1, 1, 3, padding=1)
        self.enc_sigma = torch.nn.Conv3d(1, 1, 3, padding=1)
        self.dec = torch.nn.Conv3d(1, 1, 3, padding=1)

    def forward(self, image):
        z_mu = self.enc_mu(image)
        z_sigma = self.enc_sigma(image)
        return self.dec(z_mu), z_mu, z_sigma


class TinyDiscriminator(torch.nn.Module):
    """Feature-list stand-in whose last entry acts as the differentiable logit map."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 3, padding=1)

    def forward(self, image):
        feat = self.conv(image)
        return [feat.mean(), feat]


class RecordingWriter:
    """TensorBoard stand-in recording every scalar tag/value/step tuple."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        value_float = value.item() if isinstance(value, torch.Tensor) else float(value)
        self.scalars.append((tag, value_float, step))


def fake_adv_loss(logits, target_is_real, for_discriminator):
    return ((logits - (1.0 if target_is_real else 0.0)) ** 2).mean()


def fake_perceptual_loss(reconstruction, image):
    return ((reconstruction - image) ** 2).mean()


class OneImageDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"image": torch.rand(1, 4, 4, 4)}


def make_trainer(writer, visualize=None):
    return VaeGanTrainer(
        TinyVaeAutoencoder(),
        TinyDiscriminator(),
        torch.device("cpu"),
        writer,
        kl_weight=1.0,
        adv_weight=1.0,
        perceptual_weight=1.0,
        lr=3e-4,
        intensity_loss=torch.nn.MSELoss(),
        adv_loss=fake_adv_loss,
        perceptual_loss=fake_perceptual_loss,
        visualize=visualize,
    )


def make_loader():
    return torch.utils.data.DataLoader(OneImageDataset(), batch_size=1)


def test_warmup_ladder_boundaries():
    # cell 26 verbatim: <10 -> 0.01, <20 -> 0.1, else -> 1.0
    assert warmup_rule(0) == 0.01
    assert warmup_rule(9) == 0.01
    assert warmup_rule(10) == 0.1
    assert warmup_rule(19) == 0.1
    assert warmup_rule(20) == 1.0
    assert warmup_rule(999) == 1.0


@pytest.mark.parametrize("amp,expected_eps", [(True, 1e-06), (False, 1e-08)])
def test_optimizer_eps_switches_on_amp(amp, expected_eps):
    optimizer_g, optimizer_d = build_optimizers(TinyVaeAutoencoder(), TinyDiscriminator(), lr=3e-4, amp=amp)
    assert optimizer_g.param_groups[0]["eps"] == expected_eps
    assert optimizer_d.param_groups[0]["eps"] == expected_eps
    assert optimizer_g.param_groups[0]["lr"] == 3e-4


def test_schedulers_share_three_segment_warmup_lambda():
    optimizer_g, optimizer_d = build_optimizers(TinyVaeAutoencoder(), TinyDiscriminator(), lr=3e-4, amp=False)
    scheduler_g, scheduler_d = build_schedulers(optimizer_g, optimizer_d)
    assert scheduler_g.lr_lambdas[0] is warmup_rule
    assert scheduler_d.lr_lambdas[0] is warmup_rule
    assert scheduler_g.get_last_lr() == [3e-4 * warmup_rule(0)]
    assert scheduler_d.get_last_lr() == [3e-4 * warmup_rule(0)]


def test_amp_scalers_pinned_init_and_growth():
    scaler_g, scaler_d = build_scalers(True)
    # the 'cuda' scaler disables itself with no CUDA present and drops the ctor
    # constants from its state -- the pinned values are asserted where they live.
    if not torch.cuda.is_available():
        assert not scaler_g.is_enabled()
        return
    assert scaler_g.get_scale() == 2.0**8
    assert scaler_g.get_growth_factor() == 1.5
    assert scaler_d.get_scale() == 2.0**8
    assert scaler_d.get_growth_factor() == 1.5


def test_no_amp_means_no_scalers():
    assert build_scalers(False) == (None, None)


def test_intensity_loss_mapping():
    assert type(build_intensity_loss("l2")) is torch.nn.MSELoss
    assert type(build_intensity_loss("l1")) is torch.nn.L1Loss
    assert type(build_intensity_loss("anything-else")) is torch.nn.L1Loss


def test_discriminator_construction_matches_cell24():
    discriminator = build_discriminator(3)
    outputs = discriminator(torch.randn(1, 1, 32, 32, 32))  # 16^3 already degenerates to single-point spatial
    # notebook indexes [-1], so the net must hand back a non-empty feature sequence
    assert isinstance(outputs, list | tuple)
    assert len(outputs) >= 1
    assert outputs[-1].shape[0] == 1


def test_loss_weighted_sum_formula():
    out = loss_weighted_sum({"recons_loss": 2.0, "kl_loss": 3.0, "p_loss": 10.0}, kl_weight=0.5, perceptual_weight=0.1)
    assert out == 2.0 + 0.5 * 3.0 + 0.1 * 10.0


def test_load_finetune_weights_accepts_raw_and_payload_layouts(tmp_path):
    reference_state = TinyVaeAutoencoder().state_dict()
    raw_path = tmp_path / "raw.pt"
    payload_path = tmp_path / "payload.pt"
    torch.save(reference_state, raw_path)
    torch.save({"unet_state_dict": reference_state}, payload_path)

    trainer_raw = make_trainer(RecordingWriter())
    trainer_raw.load_finetune_weights(str(raw_path))
    trainer_payload = make_trainer(RecordingWriter())
    trainer_payload.load_finetune_weights(str(payload_path))

    for key, expected in reference_state.items():
        assert torch.equal(trainer_raw.autoencoder.state_dict()[key], expected)
        assert torch.equal(trainer_payload.autoencoder.state_dict()[key], expected)


def test_forward_with_inferer_small_image_skips_window():
    probe = SimpleNamespace(forward_calls=0)

    class ProbeModel:
        def __call__(self, image):
            probe.forward_calls += 1
            return image * 2.0

    model = ProbeModel()
    inferer = SimpleNamespace(roi_size=[8, 8, 8])  # numel(images[0]) == 64 <= prod(roi)
    out = VaeGanTrainer._forward_with_inferer(inferer, model, torch.ones(1, 1, 4, 4, 4))
    assert probe.forward_calls == 1
    assert torch.equal(out, torch.full((1, 1, 4, 4, 4), 2.0))


def test_forward_with_inferer_adjusts_roi_to_image_and_restores():
    window_shapes = []

    class FakeSlidingWindow:
        def __init__(self):
            self.roi_size = [8, 4, 2]

        def __call__(self, network, inputs):
            window_shapes.append(tuple(self.roi_size))
            return network(inputs) * 3.0

    inferer = FakeSlidingWindow()
    images = torch.randn(1, 1, 4, 6, 8)  # numel 192 > prod(roi)==64 -> sliding path
    out = VaeGanTrainer._forward_with_inferer(inferer, lambda x: x, images)
    assert window_shapes == [(4, 4, 2)]  # per-dimension min adjustment pinned
    assert inferer.roi_size == [8, 4, 2]  # original roi restored afterwards
    assert torch.equal(out, images * 3.0)


def test_fit_loop_runs_full_gan_alternation_on_cpu(tmp_path):
    writer = RecordingWriter()
    visualizer_args = []
    trainer = make_trainer(writer, visualize=lambda images, recon, w, epoch: visualizer_args.append((epoch,)))
    ckpt_g = str(tmp_path / "autoencoder.pt")
    ckpt_d = str(tmp_path / "discriminator.pt")

    result = trainer.fit(
        make_loader(),
        make_loader(),
        n_epochs=2,
        val_interval=1,
        checkpoint_path_g=ckpt_g,
        checkpoint_path_d=ckpt_d,
        val_sliding_window_patch_size=[4, 4, 4],
    )

    # two batches per epoch x two epochs; one G step and one D step per batch
    assert trainer.total_step == 4
    assert trainer.scheduler_g.last_epoch == 2  # one scheduler.step() per epoch
    assert trainer.scheduler_d.last_epoch == 2
    tags = [tag for tag, _, _ in writer.scalars]
    assert tags.count("train_recons_loss_iter") == 4
    assert tags.count("train_adv_loss_iter") == 4
    assert tags.count("train_fake_loss_iter") == 4
    assert tags.count("train_real_loss_iter") == 4
    assert tags.count("train_recons_loss_epoch") == 2
    assert "val_one_sample_scale_factor" in tags
    assert visualizer_args == [(0,), (1,)]  # hook fires once per validated epoch, in order
    assert math.isfinite(result["best_val_loss"])
    assert result["total_steps"] == 4

    assert Path(ckpt_g).exists()
    assert Path(ckpt_d).exists()
    best_files = sorted(tmp_path.glob("autoencoder_epoch*.pt"))
    assert "autoencoder_epoch0.pt" in {p.name for p in best_files}  # first validation beats the sentinel
