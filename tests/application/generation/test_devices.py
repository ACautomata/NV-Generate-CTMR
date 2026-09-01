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

"""The unified device-injection contract for the diagnostic/inference entries (issue #280, ADR-0019 §8).

The scattered entries used to self-select ``torch.device("cuda" if
torch.cuda.is_available() else "cpu")`` inside their ``main``; the unified
injection replaces that with one flag definition point (``add_device_flag``)
and one resolution point (``resolve_device``). The resolution fallback must
stay byte-for-byte the hardcoded-era behavior -- an absent flag resolves to
cuda when available, cpu otherwise -- so the default invocation path is
unchanged; an explicit ``--device cpu`` (or ``cuda:N``) overrides it. The
existing ``intensity_domain``/``reencode`` arms keep their own
``default="cpu"`` surfaces: those arms are CPU-diagnostic by design and are
not this cleanup's scope. Torch-level: runs without a GPU (the fallback is
exercised through monkeypatched availability, the explicit path is pure
device construction) -- never skipped around the torch mark.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from ctmr.application.generation.devices import DEVICE_FLAG_HELP, add_device_flag, resolve_device

pytestmark = pytest.mark.torch


def _parser():
    parser = argparse.ArgumentParser()
    add_device_flag(parser)
    return parser


# --------------------------------------------------------------- the flag face


def test_flag_defaults_to_none():
    assert _parser().parse_args([]).device is None


def test_flag_accepts_an_explicit_device_string():
    assert _parser().parse_args(["--device", "cpu"]).device == "cpu"
    assert _parser().parse_args(["--device", "cuda:1"]).device == "cuda:1"


def test_flag_help_text_is_the_single_definition():
    parser = _parser()
    help_text = next(action.help for action in parser._actions if action.dest == "device")
    assert help_text == DEVICE_FLAG_HELP


# ------------------------------------------------------------ the resolution face


def test_absent_flag_falls_back_to_the_hardcoded_era_behavior(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")


def test_explicit_flag_bypasses_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:1") == torch.device("cuda", 1)
    assert resolve_device("cuda:0").type == "cuda"
