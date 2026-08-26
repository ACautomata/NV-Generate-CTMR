"""Torch-level convergence-gate tests for the frozen weights_only allowlist (ADR-0009, #107).

Proves the four properties of decision 4: the payload is the verbatim
collapsed copy, importing the module never mutates global torch state,
activation is scoped (restores the exact prior allowlist), and torch>=2.6
default ``weights_only=True`` loads of the numpy payload work inside the
scope only. Auto-skipped when torch / numpy are absent (``pytest.importorskip``,
ADR-0013 §4); no nnunetv2, no cluster, no external data.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

import numpy  # noqa: E402  (importorskip must precede the torch-dependent import)
import torch  # noqa: E402

from ctmr.instrument.safeglobals import NNUNET_SAFE_GLOBALS, nnunet_safe_globals  # noqa: E402

pytestmark = pytest.mark.torch


def test_allowlist_payload_matches_the_frozen_copies():
    assert numpy.core.multiarray.scalar in NNUNET_SAFE_GLOBALS
    assert numpy.dtype in NNUNET_SAFE_GLOBALS
    assert len(NNUNET_SAFE_GLOBALS) == 2 + 11  # the verbatim payload of the three collapsed copies


def test_import_does_not_mutate_global_torch_state():
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    # comparing before/after the module import: torch itself registers some safe
    # globals at import time (NestedTensor & friends) -- that baseline is not ours to assert on
    check = (
        "import torch; before = list(torch.serialization.get_safe_globals()); "
        "import ctmr.instrument.safeglobals; "
        "assert torch.serialization.get_safe_globals() == before"
    )
    result = subprocess.run([sys.executable, "-c", check], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_activation_is_scoped_and_restores_the_prior_state():
    before = list(torch.serialization.get_safe_globals())  # copy: get_safe_globals may return the live list
    with nnunet_safe_globals():
        assert set(NNUNET_SAFE_GLOBALS) <= set(torch.serialization.get_safe_globals())
    assert set(torch.serialization.get_safe_globals()) == set(
        before
    )  # same allowlist content back -- no residue, no clobbering (order is not part of the semantics)


def test_weights_only_load_is_robust_inside_the_scope(tmp_path):
    # prelude: this scenario needs a clean allowlist -- no other test may have
    # added the nnU-Net payload globally
    assert numpy.core.multiarray.scalar not in torch.serialization.get_safe_globals()
    artifact = tmp_path / "numpy_scalar.pt"
    torch.save(numpy.float64(3.14), artifact)
    with pytest.raises(Exception):  # any weights_only rejection; the wording varies across torch versions
        torch.load(artifact, weights_only=True)
    with nnunet_safe_globals():
        assert float(torch.load(artifact, weights_only=True)) == 3.14
    with pytest.raises(Exception):  # the scope restored the prior state -- no residue
        torch.load(artifact, weights_only=True)
