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

"""Convergence-gate tests for the pinned recipe specs (ADR-0011 decision 4, #111).

``MaskRecipeSpec`` reproduces the pre-#111 ``P2RecipeGuard`` behaviour verbatim
(ADR-0007 values, raise messages, the ``is not False`` identity check on RCL and
the missing-``n_epochs`` default); ``P1RecipeSpec`` is the runtime guard
ADR-0005 pinned but never had (validation only, no recipe value change);
``CrossModalRecipeSpec`` pins the mask-equivalent recipe plus CFG=0 and the no-warm-start
clause. The specs read plain config dicts -- stdlib-only, any machine (ADR-0013 §4).
"""

import json
import re
from pathlib import Path

import pytest

from ctmr.domain.recipe import CrossModalRecipeSpec, MaskRecipeSpec, P1RecipeSpec

REPO = Path(__file__).resolve().parents[2]
P2_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p2_train.json").read_text())
P3_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p3_train.json").read_text())
P1_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p1_train.json").read_text())
RFLOW_NETWORK = json.loads((REPO / "configs/config_network_rflow.json").read_text())


class _QuietLogger:
    def info(self, message):
        self.messages = getattr(self, "messages", []) + [message]


# ── mask: the pre-#111 P2RecipeGuard behaviour, verbatim (ADR-0007 values) ──


def test_mask_accepts_the_pinned_config():
    logger = _QuietLogger()
    assert MaskRecipeSpec(dict(P2_TRAIN_CONFIG["controlnet_train"]), logger).check() is True
    assert "mask recipe guard OK: lr=1e-05 batch=1 weighted_loss=100@[129, 130, 131] RCL=off" in logger.messages


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lr", 2e-05, "pinned mask lr is 1e-05, got 2e-05 (ADR-0007)"),
        ("batch_size", 2, "pinned mask batch_size is 1, got 2 (ADR-0007)"),
        ("weighted_loss", 50, "pinned mask weighted_loss is 100, got 50 (ADR-0007)"),
        ("weighted_loss_label", [129, 130], "pinned mask weighted_loss_label is [129, 130, 131], got [129, 130] (ADR-0007)"),
        ("cache_rate", 1, "pinned mask cache_rate is 0, got 1 (ADR-0007)"),
        ("n_epochs", 101, "pinned mask max n_epochs is 100, got 101 (ADR-0007)"),
    ],
)
def test_mask_deviation_raises_with_the_verbatim_message(field, value, message):
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    config[field] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        MaskRecipeSpec(config, _QuietLogger()).check()


@pytest.mark.parametrize("rcl", [True, None, 0], ids=["true", "null", "zero"])
def test_mask_rcl_identity_check_is_not_truthiness(rcl):
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    config["use_region_contrasive_loss"] = rcl
    with pytest.raises(ValueError, match="mask recipe forbids use_region_contrasive_loss \\(must be OFF, ADR-0007\\)"):
        MaskRecipeSpec(config, _QuietLogger()).check()


def test_mask_missing_n_epochs_defaults_to_the_max_cap():
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    del config["n_epochs"]
    assert MaskRecipeSpec(config, _QuietLogger()).check() is True


# ── P1: the ADR-0005 runtime guard (added, never a recipe change) ──


def test_p1_accepts_the_pinned_config():
    logger = _QuietLogger()
    spec = P1RecipeSpec(dict(P1_TRAIN_CONFIG["diffusion_unet_train"]), dict(RFLOW_NETWORK["noise_scheduler"]), logger)
    assert spec.check() is True
    assert "P1 recipe guard OK: lr=2e-06 batch=1 epochs<=100 rflow uniform scale=1.4" in logger.messages[0]
    assert "frozen_modality_tokens=[34]" in logger.messages[0]  # the committed recipe freezes t1c (issue #250)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lr", 2e-05, "pinned P1 lr is 2e-06, got 2e-05 (ADR-0005)"),
        ("batch_size", 2, "pinned P1 batch_size is 1, got 2 (ADR-0005)"),
        ("n_epochs", 101, "pinned P1 max n_epochs is 100, got 101 (ADR-0005)"),
    ],
)
def test_p1_deviation_raises(field, value, message):
    config = dict(P1_TRAIN_CONFIG["diffusion_unet_train"])
    config[field] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        P1RecipeSpec(config, dict(RFLOW_NETWORK["noise_scheduler"]), _QuietLogger()).check()


@pytest.mark.parametrize("freeze", ["34", [34.0], [True], 34, [34, "30"]], ids=["str", "float", "bool", "bare-int", "mixed"])
def test_p1_malformed_freeze_raises(freeze):
    """The freeze key is recipe-reachable: a malformed value fails the launch, not the training (issue #250)."""
    config = dict(P1_TRAIN_CONFIG["diffusion_unet_train"])
    config["frozen_modality_tokens"] = freeze
    with pytest.raises(ValueError, match=re.escape("P1 frozen_modality_tokens must be a list of ints")):
        P1RecipeSpec(config, dict(RFLOW_NETWORK["noise_scheduler"]), _QuietLogger()).check()


def test_p1_freeze_absent_keeps_the_historical_guard_line():
    """A config without the freeze key guards identically to the pre-#250 line (no suffix)."""
    config = dict(P1_TRAIN_CONFIG["diffusion_unet_train"])
    del config["frozen_modality_tokens"]
    logger = _QuietLogger()
    assert P1RecipeSpec(config, dict(RFLOW_NETWORK["noise_scheduler"]), logger).check() is True
    assert logger.messages[0].endswith("scale=1.4")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("sample_method", "ddpm", "pinned P1 noise_scheduler.sample_method is uniform, got ddpm (ADR-0005)"),
        ("scale", 1.0, "pinned P1 noise_scheduler.scale is 1.4, got 1.0 (ADR-0005)"),
    ],
)
def test_p1_rflow_scheduler_deviation_raises(field, value, message):
    scheduler = dict(RFLOW_NETWORK["noise_scheduler"])
    scheduler[field] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        P1RecipeSpec(dict(P1_TRAIN_CONFIG["diffusion_unet_train"]), scheduler, _QuietLogger()).check()


def test_p1_missing_n_epochs_defaults_to_the_max_cap():
    config = dict(P1_TRAIN_CONFIG["diffusion_unet_train"])
    del config["n_epochs"]
    assert P1RecipeSpec(config, dict(RFLOW_NETWORK["noise_scheduler"]), _QuietLogger()).check() is True


# ── cross-modal: the mask-equivalent recipe plus CFG=0 and no warm-start ──


def test_p3_accepts_the_pinned_config():
    logger = _QuietLogger()
    spec = CrossModalRecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        dict(P3_TRAIN_CONFIG["diffusion_unet_inference"]),
        logger,
        trained_controlnet_path=None,
    )
    assert spec.check() is True
    assert "cross-modal recipe guard OK:" in logger.messages[0]


def test_p3_cfg_off_is_pinned():
    spec = CrossModalRecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        {"cfg_guidance_scale": 5.0},
        _QuietLogger(),
        trained_controlnet_path=None,
    )
    with pytest.raises(ValueError, match=r"cross-modal candidate is evaluated/selected with CFG OFF \(cfg_guidance_scale=0\); got 5.0"):
        spec.check()


def test_p3_warm_start_from_a_controlnet_is_forbidden():
    spec = CrossModalRecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        dict(P3_TRAIN_CONFIG["diffusion_unet_inference"]),
        _QuietLogger(),
        trained_controlnet_path="/some/controlnet.pt",
    )
    with pytest.raises(ValueError, match="cross-modal recipe forbids warm-starting from a ControlNet checkpoint \\(P1-DM init only\\)"):
        spec.check()
