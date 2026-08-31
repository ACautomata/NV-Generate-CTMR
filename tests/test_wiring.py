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

"""Composition-root gates (issue #270 / ADR-0019 §2; mask assembly issue #273).

``ctmr.wiring`` is the terminal composition root -- the one home of concrete
implementation knowledge, one module per subcommand family, outside the three
layers and parallel to ``cli.py``. Its family modules compose lazily
(importlib on dispatch, the ``cli.py`` discipline): importing the package
must pull no third-party dependency, so the light sci-stack CI job can always
import the composition root, and a family module that starts eagerly pulling
its adapters turns this probe red (``ctmr.wiring.measure`` is only reached
through the dispatch registry, a gap the cli purity gate cannot see). The
behavioral face of the train dispatch (spawn vs in-process worker entry) is
pinned through the CLI seam in
``tests/application/generation/modality_label/test_spawn.py`` -- these gates
pin the structure only.

The mask train assembly (#273) is gated here with fake adapters seeded into
``sys.modules``: ``mask_train_runtime`` must hoist every concrete
construction the entry used to make (config merge + flag patching, the
distributed session, the logger, the amp-selected precision executor, the
bypass mounting) and hand the entry one runtime record; the delegation, not
the adapters, is what this gate pins (the adapters' own gates live under
tests/infrastructure).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from ctmr.wiring import generate as wiring_generate


def test_wiring_imports_pull_no_third_party_dependency(light_import_probe):
    assert light_import_probe("ctmr.wiring, ctmr.wiring.generate, ctmr.wiring.measure") == "[]"


# ------------------------------------------------------- mask train assembly (#273)


class _RecordingEngine:
    """The fake GenerationEngine adapter: records the config triple it merges."""

    def __init__(self, calls):
        self._calls = calls

    def load_config(self, env_config_path, model_config_path, model_def_path):
        self._calls["config"] = (env_config_path, model_config_path, model_def_path)
        return types.SimpleNamespace(modality_mapping_path=self._calls["mapping_path"], model_dir="RUN_DIR")


def _seed_fake_adapters(monkeypatch, tmp_path):
    """Seed fake infrastructure adapters behind the assembly's lazy imports."""
    calls = {"mapping_path": None}
    mapping = tmp_path / "modality_mapping.json"
    mapping.write_text(json.dumps({"t1n": 29}) + "\n")
    calls["mapping_path"] = str(mapping)

    engine_mod = types.ModuleType("ctmr.infrastructure.engine")
    engine_mod.MaisiEngine = lambda: _RecordingEngine(calls)
    setting_mod = types.ModuleType("ctmr.infrastructure.maisi_engine.diff_model_setting")

    def _fake_initialize(num_gpus):
        calls["dist"] = num_gpus
        return 3, 8, "DEVICE"

    def _fake_logging(name):
        calls["log"] = name
        return "LOGGER"

    setting_mod.initialize_distributed = _fake_initialize
    setting_mod.setup_logging = _fake_logging
    executors_mod = types.ModuleType("ctmr.infrastructure.gradient_executors")
    for name in ("Fp16GradientExecutor", "Bf16GradientExecutor", "PlainGradientExecutor"):
        setattr(executors_mod, name, type(name, (), {}))
    mounting_mod = types.ModuleType("ctmr.infrastructure.bypass_mounting")

    class _FakeBypassMounting:
        def __init__(self, args, device, logger):
            calls["mounting"] = (args, device, logger)

    mounting_mod.BypassMounting = _FakeBypassMounting
    for name, module in (
        ("ctmr.infrastructure.engine", engine_mod),
        ("ctmr.infrastructure.maisi_engine.diff_model_setting", setting_mod),
        ("ctmr.infrastructure.gradient_executors", executors_mod),
        ("ctmr.infrastructure.bypass_mounting", mounting_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls, executors_mod, mounting_mod


def _mask_train_args(amp=True, amp_dtype="bf16"):
    return types.SimpleNamespace(
        env_config_path="env.json",
        model_config_path="config.json",
        model_def_path="network.json",
        num_gpus=8,
        amp=amp,
        amp_dtype=amp_dtype,
    )


@pytest.mark.parametrize(
    ("amp", "amp_dtype", "executor_attr"),
    [(True, "fp16", "Fp16GradientExecutor"), (True, "bf16", "Bf16GradientExecutor"), (False, "bf16", "PlainGradientExecutor")],
)
def test_mask_train_runtime_assembles_the_ports(monkeypatch, tmp_path, amp, amp_dtype, executor_attr):
    """The assembly hoists every construction the entry used to make: the
    engine-merged config with the CLI flags patched in, the modality mapping
    read, the session bootstrap receiving ``-g``, the logger, the
    amp-selected precision executor and the mounted bypass mounting."""
    calls, executors_mod, mounting_mod = _seed_fake_adapters(monkeypatch, tmp_path)

    runtime = wiring_generate.mask_train_runtime(_mask_train_args(amp, amp_dtype))

    assert calls["config"] == ("env.json", "config.json", "network.json")
    assert runtime.merged.amp is amp and runtime.merged.amp_dtype == amp_dtype
    assert runtime.merged.env_config_path == "env.json"
    assert runtime.merged.model_config_path == "config.json"
    assert runtime.merged.model_def_path == "network.json"
    assert runtime.merged.modality_mapping == {"t1n": 29}
    assert calls["dist"] == 8  # the -g value reaches the session bootstrap
    assert runtime.local_rank == 3 and runtime.device == "DEVICE"
    assert runtime.logger == "LOGGER" and calls["log"] == "mask-finetune"
    assert isinstance(runtime.gradient_executor, getattr(executors_mod, executor_attr))
    assert isinstance(runtime.mounting, mounting_mod.BypassMounting)
    assert calls["mounting"] == (runtime.merged, "DEVICE", "LOGGER")


def test_generation_engine_mounts_lazily(monkeypatch):
    """The sampling/monitoring engine face is one lazy adapter lookup."""
    engine_mod = types.ModuleType("ctmr.infrastructure.engine")
    sentinel = object()
    engine_mod.MaisiEngine = lambda: sentinel
    monkeypatch.setitem(sys.modules, "ctmr.infrastructure.engine", engine_mod)
    assert wiring_generate.generation_engine() is sentinel
