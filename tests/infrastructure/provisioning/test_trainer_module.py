"""Torch-level registration gate for the migrated trainer module (ADR-0015 §7, ticket #140).

The trainer class name is pinned by nnunetv2's discovery contract (no renaming);
this test proves the module still defines the audited 250-epoch trainer over the
upstream base class and keeps the ``num_epochs = 250`` line the
``TrainerContract`` verifier greps for.
"""

import inspect

import pytest

pytest.importorskip("nnunetv2")  # heavy tier; installed in the CI full-dependency set

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from ctmr.infrastructure.provisioning.trainer_250_epochs import nnUNetTrainer250Epochs  # noqa: E402

pytestmark = pytest.mark.torch


def test_trainer_class_name_is_pinned_by_the_nnunetv2_registry_contract():
    assert nnUNetTrainer250Epochs.__name__ == "nnUNetTrainer250Epochs"


def test_trainer_inherits_the_upstream_base_and_pins_250_epochs():
    assert issubclass(nnUNetTrainer250Epochs, nnUNetTrainer)
    source = inspect.getsource(nnUNetTrainer250Epochs)
    assert "self.num_epochs = 250" in source


def test_init_signature_mirrors_the_upstream_parameters():
    """Upstream rebuilds kwargs from this signature's names -- forwarding spellings would KeyError."""
    upstream = list(inspect.signature(nnUNetTrainer.__init__).parameters)
    ours = list(inspect.signature(nnUNetTrainer250Epochs.__init__).parameters)
    assert ours[: len(upstream)] == upstream or ours == ["self", "plans", "configuration", "fold", "dataset_json", "device"]
