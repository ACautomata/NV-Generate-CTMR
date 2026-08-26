"""Convergence-gate tests for the frozen instrument command (ADR-0009, #107).

Pins the canonical argv per challenge (the embedded snapshot is the drift
anchor) and proves parameter-level equality with the frozen
terminal-acceptance invocation (``PredictScriptWriter``, pre-#107), plus the
frozen invariants: ``build`` is pure (no execution, no file IO) and mirror TTA
stays on (no TTA parameter anywhere, never ``--disable_tta``). Stdlib-only:
any machine, no torch / nnunetv2 / cluster (ADR-0013). The sugon
byte-identical rerun stays with the frozen gate (ADR-0009 decision 7).
"""

import sys
from dataclasses import fields

import pytest

from ctmr.instrument.command import INSTRUMENT_SPECS, FrozenInstrumentCommand, InstrumentSpec
from scripts.nnunet_l2_final_acceptance import PredictScriptWriter

# The canonical argv snapshot per challenge, exactly as ADR-0009 decisions 1+3
# pin it. Do not edit -- drift here is exactly what this gate exists to catch.
CANONICAL_SPEC_OPTIONS = [
    ("GLI", ["-d", "Dataset501_BraTS2023GLI", "-c", "3d_fullres", "-p", "nnUNetPlans", "-tr", "nnUNetTrainer250Epochs", "-f", "0"]),
    ("SSA", ["-d", "Dataset502_BraTS2023SSA", "-c", "3d_fullres_bs16", "-p", "nnUNetPlans_SSA_bs16_v1", "-tr", "nnUNetTrainer250Epochs", "-f", "0"]),
    ("MEN", ["-d", "Dataset503_BraTS2023MEN", "-c", "3d_fullres", "-p", "nnUNetPlans", "-tr", "nnUNetTrainer250Epochs", "-f", "0"]),
    ("METS", ["-d", "Dataset504_BraTS2023METS", "-c", "3d_fullres", "-p", "nnUNetPlans", "-tr", "nnUNetTrainer250Epochs", "-f", "0"]),
    ("PED", ["-d", "Dataset505_BraTS2023PED", "-c", "3d_fullres", "-p", "nnUNetPlans", "-tr", "nnUNetTrainer250Epochs", "-f", "0"]),
]

FIVE_CHALLENGES = [challenge for challenge, _ in CANONICAL_SPEC_OPTIONS]


@pytest.mark.parametrize("challenge,expected_options", CANONICAL_SPEC_OPTIONS)
def test_build_produces_the_canonical_argv(challenge, expected_options):
    argv = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build("/raw/in", "/pred/out")
    assert argv[:7] == [sys.executable, "-m", "ctmr.instrument.predict", "-i", "/raw/in", "-o", "/pred/out"]
    assert argv[7:] == expected_options


def test_specs_cover_exactly_the_five_challenges():
    assert sorted(INSTRUMENT_SPECS) == sorted(FIVE_CHALLENGES)


def test_ssa_pins_the_derived_batch16_configuration():
    spec = INSTRUMENT_SPECS["SSA"]  # the one deliberate deviation (ADR-0001)
    assert spec.config == "3d_fullres_bs16"
    assert spec.plans == "nnUNetPlans_SSA_bs16_v1"


def test_build_never_touches_the_filesystem():
    command = FrozenInstrumentCommand(InstrumentSpec(dataset_id="Dataset501_BraTS2023GLI", config="3d_fullres", plans="nnUNetPlans"))
    argv = command.build("/nonexistent-instrument-input", "/nonexistent-instrument-output")
    assert argv[4] == "/nonexistent-instrument-input"  # -i passed through: no directory had to exist
    assert argv == command.build("/nonexistent-instrument-input", "/nonexistent-instrument-output")  # same input -> same argv


@pytest.mark.parametrize("challenge", FIVE_CHALLENGES)
def test_mirror_tta_stays_on_by_omission(challenge):
    argv = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build("/raw/in", "/pred/out")
    assert "--disable_tta" not in argv  # the frozen invariant: the token is a fatal argparse bug (#78), never a TTA switch
    assert not [field for field in fields(InstrumentSpec) if "tta" in field.name]  # the interface exposes no TTA parameter


def test_terminal_acceptance_invocation_is_the_builder_argv(tmp_path):
    """The #108 adoption: PredictScriptWriter emits exactly ``build(...)`` -- the
    frozen call site lands on the single construction point, so the generated
    script line is verbatim the builder argv (entry ``\\ -m ctmr.instrument.predict``,
    the canonical spec flags, no TTA token)."""
    challenges = {challenge: {} for challenge in FIVE_CHALLENGES}  # the writer only reads the keys
    PredictScriptWriter({"challenges": challenges}, tmp_path).write()

    for challenge in FIVE_CHALLENGES:
        cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(tmp_path / "inputs" / challenge, tmp_path / "predictions" / challenge)
        line = (tmp_path / f"predict_{challenge}.sh").read_text().splitlines()[-1]
        assert line == " ".join(cmd)
