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

``P2RecipeSpec`` reproduces the pre-#111 ``P2RecipeGuard`` behaviour verbatim
(ADR-0007 values, raise messages, the ``is not False`` identity check on RCL and
the missing-``n_epochs`` default); ``P1RecipeSpec`` is the runtime guard
ADR-0005 pinned but never had (validation only, no recipe value change);
``P3RecipeSpec`` pins the P2-equivalent recipe plus CFG=0 and the no-warm-start
clause. The specs read plain config dicts -- stdlib-only, any machine (ADR-0013 §4).
"""

import json
import re
from pathlib import Path

import pytest

from ctmr.domain.recipe import P1RecipeSpec, P2RecipeSpec, P3RecipeSpec

REPO = Path(__file__).resolve().parents[2]
P2_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p2_train.json").read_text())
P3_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p3_train.json").read_text())
P1_TRAIN_CONFIG = json.loads((REPO / "configs/config_brats_p1_train.json").read_text())
RFLOW_NETWORK = json.loads((REPO / "configs/config_network_rflow.json").read_text())


class _QuietLogger:
    def info(self, message):
        self.messages = getattr(self, "messages", []) + [message]


# ── P2: the pre-#111 P2RecipeGuard behaviour, verbatim (ADR-0007 values) ──


def test_p2_accepts_the_pinned_config():
    logger = _QuietLogger()
    assert P2RecipeSpec(dict(P2_TRAIN_CONFIG["controlnet_train"]), logger).check() is True
    assert "P2 recipe guard OK: lr=1e-05 batch=1 weighted_loss=100@[129, 130, 131] RCL=off" in logger.messages


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lr", 2e-05, "pinned P2 lr is 1e-05, got 2e-05 (ADR-0007)"),
        ("batch_size", 2, "pinned P2 batch_size is 1, got 2 (ADR-0007)"),
        ("weighted_loss", 50, "pinned P2 weighted_loss is 100, got 50 (ADR-0007)"),
        ("weighted_loss_label", [129, 130], "pinned P2 weighted_loss_label is [129, 130, 131], got [129, 130] (ADR-0007)"),
        ("cache_rate", 1, "pinned P2 cache_rate is 0, got 1 (ADR-0007)"),
        ("n_epochs", 101, "pinned P2 max n_epochs is 100, got 101 (ADR-0007)"),
    ],
)
def test_p2_deviation_raises_with_the_verbatim_message(field, value, message):
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    config[field] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        P2RecipeSpec(config, _QuietLogger()).check()


@pytest.mark.parametrize("rcl", [True, None, 0], ids=["true", "null", "zero"])
def test_p2_rcl_identity_check_is_not_truthiness(rcl):
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    config["use_region_contrasive_loss"] = rcl
    with pytest.raises(ValueError, match="P2 recipe forbids use_region_contrasive_loss \\(must be OFF, ADR-0007\\)"):
        P2RecipeSpec(config, _QuietLogger()).check()


def test_p2_missing_n_epochs_defaults_to_the_max_cap():
    config = dict(P2_TRAIN_CONFIG["controlnet_train"])
    del config["n_epochs"]
    assert P2RecipeSpec(config, _QuietLogger()).check() is True


# ── P1: the ADR-0005 runtime guard (added, never a recipe change) ──


def test_p1_accepts_the_pinned_config():
    logger = _QuietLogger()
    spec = P1RecipeSpec(dict(P1_TRAIN_CONFIG["diffusion_unet_train"]), dict(RFLOW_NETWORK["noise_scheduler"]), logger)
    assert spec.check() is True
    assert "P1 recipe guard OK: lr=2e-06 batch=1 epochs<=100 rflow uniform scale=1.4" in logger.messages


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


# ── P3: the P2-equivalent recipe plus CFG=0 and no warm-start ──


def test_p3_accepts_the_pinned_config():
    logger = _QuietLogger()
    spec = P3RecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        dict(P3_TRAIN_CONFIG["diffusion_unet_inference"]),
        logger,
        trained_controlnet_path=None,
    )
    assert spec.check() is True
    assert "P3 recipe guard OK:" in logger.messages[0]


def test_p3_cfg_off_is_pinned():
    spec = P3RecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        {"cfg_guidance_scale": 5.0},
        _QuietLogger(),
        trained_controlnet_path=None,
    )
    with pytest.raises(ValueError, match=r"P3 candidate is evaluated/selected with CFG OFF \(cfg_guidance_scale=0\); got 5.0"):
        spec.check()


def test_p3_warm_start_from_a_controlnet_is_forbidden():
    spec = P3RecipeSpec(
        dict(P3_TRAIN_CONFIG["controlnet_train"]),
        dict(P3_TRAIN_CONFIG["diffusion_unet_inference"]),
        _QuietLogger(),
        trained_controlnet_path="/some/controlnet.pt",
    )
    with pytest.raises(ValueError, match="P3 recipe forbids warm-starting from a ControlNet checkpoint \\(P1-DM init only\\)"):
        spec.check()
