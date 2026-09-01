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

"""Pinned-recipe guards, first-class PhaseHarness hooks (ADR-0011 decision 4).

Each spec pins the frozen recipe values of one stage (ADR-0005 P1 / ADR-0007
mask / the mask-equivalent-plus-CFG=0 cross-modal) and raises on any deviation. P1's guard is
the runtime validation ADR-0005 pinned but never had -- it changes no recipe
value. Guards are value objects over plain config dicts: stdlib-only, so the
convergence gate runs on any machine (ADR-0013 §4).
"""

from __future__ import annotations

# P1's pre-registered dev-side early-stop rule values (ADR-0005; the identical
# defaults the retired dev-eval sidecar's CLI carried): the embedded periodic
# validation (issue #278, ADR-0019 §5) evaluates the trend against these -- the
# max cap is the trainer's own n_epochs, supplied at assembly time.
P1_DEV_EARLY_STOP = {"patience": 3, "min_epoch": 30}


class MaskRecipeSpec:
    """Pinned-recipe guard for the frozen mask-conditioned recipe (ADR-0007).

    Behaviour reproduces the pre-#111 ``P2RecipeGuard`` verbatim, including the
    identity check (``is not False``) on ``use_region_contrasive_loss`` and the
    missing-``n_epochs`` default to the max cap.
    """

    PINNED_LR = 1e-5
    PINNED_BATCH = 1
    PINNED_WEIGHTED_LOSS = 100
    PINNED_WEIGHTED_LABELS = [129, 130, 131]
    PINNED_CACHE_RATE = 0
    MAX_EPOCHS = 100

    def __init__(self, train_config, logger):
        self._cfg = train_config
        self._logger = logger

    def check(self):
        cfg = self._cfg
        if cfg.get("lr") != self.PINNED_LR:
            raise ValueError(f"pinned mask lr is {self.PINNED_LR}, got {cfg.get('lr')} (ADR-0007)")
        if cfg.get("batch_size") != self.PINNED_BATCH:
            raise ValueError(f"pinned mask batch_size is {self.PINNED_BATCH}, got {cfg.get('batch_size')} (ADR-0007)")
        if cfg.get("weighted_loss") != self.PINNED_WEIGHTED_LOSS:
            raise ValueError(f"pinned mask weighted_loss is {self.PINNED_WEIGHTED_LOSS}, got {cfg.get('weighted_loss')} (ADR-0007)")
        if cfg.get("weighted_loss_label") != self.PINNED_WEIGHTED_LABELS:
            raise ValueError(f"pinned mask weighted_loss_label is {self.PINNED_WEIGHTED_LABELS}, got {cfg.get('weighted_loss_label')} (ADR-0007)")
        if cfg.get("use_region_contrasive_loss", False) is not False:
            raise ValueError("mask recipe forbids use_region_contrasive_loss (must be OFF, ADR-0007)")
        if cfg.get("cache_rate") != self.PINNED_CACHE_RATE:
            raise ValueError(f"pinned mask cache_rate is {self.PINNED_CACHE_RATE}, got {cfg.get('cache_rate')} (ADR-0007)")
        if cfg.get("n_epochs", self.MAX_EPOCHS) > self.MAX_EPOCHS:
            raise ValueError(f"pinned mask max n_epochs is {self.MAX_EPOCHS}, got {cfg.get('n_epochs')} (ADR-0007)")
        self._logger.info(
            f"mask recipe guard OK: lr={self.PINNED_LR} batch={self.PINNED_BATCH} "
            f"weighted_loss={self.PINNED_WEIGHTED_LOSS}@{self.PINNED_WEIGHTED_LABELS} RCL=off"
        )
        return True


class P1RecipeSpec:
    """Pinned-recipe guard for the P1 continuation (ADR-0005), validation only.

    ADR-0005 pinned lr=2e-6, batch=1, <=100 epochs and RF uniform timestep
    sampling (scale 1.4); this guard enforces the config-reachable values at
    launch without changing any recipe value (the code-literal deltas --
    PolynomialLR power 2.0, L1, augment prob 0.1, 1:1 mix -- live in the kernel).
    """

    PINNED_LR = 2e-06
    PINNED_BATCH = 1
    PINNED_SAMPLE_METHOD = "uniform"
    PINNED_RF_SCALE = 1.4
    MAX_EPOCHS = 100

    def __init__(self, train_config, scheduler_config, logger):
        self._cfg = train_config
        self._scheduler = scheduler_config
        self._logger = logger

    def check(self):
        cfg = self._cfg
        if cfg.get("lr") != self.PINNED_LR:
            raise ValueError(f"pinned P1 lr is {self.PINNED_LR}, got {cfg.get('lr')} (ADR-0005)")
        if cfg.get("batch_size") != self.PINNED_BATCH:
            raise ValueError(f"pinned P1 batch_size is {self.PINNED_BATCH}, got {cfg.get('batch_size')} (ADR-0005)")
        if cfg.get("n_epochs", self.MAX_EPOCHS) > self.MAX_EPOCHS:
            raise ValueError(f"pinned P1 max n_epochs is {self.MAX_EPOCHS}, got {cfg.get('n_epochs')} (ADR-0005)")
        if self._scheduler.get("sample_method") != self.PINNED_SAMPLE_METHOD:
            raise ValueError(
                f"pinned P1 noise_scheduler.sample_method is {self.PINNED_SAMPLE_METHOD}, got {self._scheduler.get('sample_method')} (ADR-0005)"
            )
        if self._scheduler.get("scale") != self.PINNED_RF_SCALE:
            raise ValueError(f"pinned P1 noise_scheduler.scale is {self.PINNED_RF_SCALE}, got {self._scheduler.get('scale')} (ADR-0005)")
        freeze = cfg.get("frozen_modality_tokens")
        if freeze is not None and (
            not isinstance(freeze, list) or not all(isinstance(token, int) and not isinstance(token, bool) for token in freeze)
        ):
            raise ValueError(f"P1 frozen_modality_tokens must be a list of ints, got {freeze!r} (issue #250)")
        self._logger.info(
            f"P1 recipe guard OK: lr={self.PINNED_LR} batch={self.PINNED_BATCH} "
            f"epochs<={self.MAX_EPOCHS} rflow {self.PINNED_SAMPLE_METHOD} scale={self.PINNED_RF_SCALE}"
            + (f" frozen_modality_tokens={freeze}" if freeze is not None else "")
        )
        return True


class CrossModalRecipeSpec:
    """Pinned-recipe guard for the frozen cross-modal recipe (mask recipe verbatim + CFG=0).

    ``trained_controlnet_path`` rides in so the no-warm-start-from-mask clause is
    checked at the same rank-0 point the pre-#111 entry guarded it.
    """

    PINNED_LR = 1e-5
    PINNED_BATCH = 1
    PINNED_WEIGHTED_LOSS = 100
    PINNED_WEIGHTED_LABELS = [129, 130, 131]
    PINNED_CACHE_RATE = 0
    MAX_EPOCHS = 100
    PINNED_CFG = 0.0

    def __init__(self, train_config, inference_config, logger, trained_controlnet_path=None):
        self._cfg = train_config
        self._infer = inference_config or {}
        self._logger = logger
        self._controlnet_path = trained_controlnet_path

    def check(self):
        if self._controlnet_path is not None:
            raise ValueError("cross-modal recipe forbids warm-starting from a ControlNet checkpoint (P1-DM init only)")
        cfg = self._cfg
        if cfg.get("lr") != self.PINNED_LR:
            raise ValueError(f"pinned cross-modal lr is {self.PINNED_LR}, got {cfg.get('lr')} (mask-equivalent recipe)")
        if cfg.get("batch_size") != self.PINNED_BATCH:
            raise ValueError(f"pinned cross-modal batch_size is {self.PINNED_BATCH}, got {cfg.get('batch_size')}")
        if cfg.get("weighted_loss") != self.PINNED_WEIGHTED_LOSS:
            raise ValueError(f"pinned cross-modal weighted_loss is {self.PINNED_WEIGHTED_LOSS}, got {cfg.get('weighted_loss')}")
        if cfg.get("weighted_loss_label") != self.PINNED_WEIGHTED_LABELS:
            raise ValueError(f"pinned cross-modal weighted_loss_label is {self.PINNED_WEIGHTED_LABELS}, got {cfg.get('weighted_loss_label')}")
        if cfg.get("use_region_contrasive_loss", False) is not False:
            raise ValueError("cross-modal recipe forbids use_region_contrasive_loss (must be OFF)")
        if cfg.get("cache_rate") != self.PINNED_CACHE_RATE:
            raise ValueError(f"pinned cross-modal cache_rate is {self.PINNED_CACHE_RATE}, got {cfg.get('cache_rate')}")
        if cfg.get("n_epochs", self.MAX_EPOCHS) > self.MAX_EPOCHS:
            raise ValueError(f"pinned cross-modal max n_epochs is {self.MAX_EPOCHS}, got {cfg.get('n_epochs')}")
        if self._infer.get("cfg_guidance_scale", 0.0) != self.PINNED_CFG:
            raise ValueError(
                f"cross-modal candidate is evaluated/selected with CFG OFF (cfg_guidance_scale=0); got {self._infer.get('cfg_guidance_scale')}"
            )
        self._logger.info(
            f"cross-modal recipe guard OK: lr={self.PINNED_LR} batch={self.PINNED_BATCH} weighted_loss={self.PINNED_WEIGHTED_LOSS}"
            f"@{self.PINNED_WEIGHTED_LABELS} RCL=off cfg={self.PINNED_CFG}"
        )
        return True
