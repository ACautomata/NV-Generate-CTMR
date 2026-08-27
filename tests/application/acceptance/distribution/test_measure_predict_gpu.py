"""The gpu-tier measure-predict inference chain, promoted to pytest (issue #140; ADR-0015 §6).

Executed on the DCU/cluster hosts only (``pytest --run-gpu``): it drives the
canonical verb ``ctmr measure predict`` end to end against the frozen SSA
instrument inside the controlled persistent tree, then leaves the artefacts in
place for the operator. Nothing here runs locally or in CI -- the markers keep
those environments skipping.
"""

import os

import pytest

pytest.importorskip("nnunetv2")  # heavy tier; installed in the CI full-dependency set

from ctmr.application.acceptance.distribution.instrument_training import PERSISTENT_ROOT  # noqa: E402
from ctmr.domain.grid import INSTRUMENT_GRID  # noqa: E402
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand  # noqa: E402
from ctmr.infrastructure.nnunet_runner import MeasurePredictVerb  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.gpu]


def _server_prerequisites() -> str | None:
    """Names the first missing server-side prerequisite, or None when all hold."""
    ssa_results = PERSISTENT_ROOT / "nnUNet_results" / INSTRUMENT_SPECS["SSA"].dataset_id
    if not ssa_results.is_dir():
        return f"frozen instrument results missing: {ssa_results}"
    if os.environ.get("nnUNet_compile", "").lower() not in {"f", "false", "0"}:
        return "formal inference requires nnUNet_compile=f on this DCU stack"
    return None


@pytest.fixture()
def synthetic_case(tmp_path):
    """One synthetic observation-shaped case on the instrument grid contract."""
    try:
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover - the tier installs SimpleITK
        pytest.skip("SimpleITK unavailable")
    shape = tuple(reversed(INSTRUMENT_GRID.size))  # array layout is zyx
    image = sitk.Image(shape[::-1], sitk.sitkUInt8)  # xyz size for sitk
    image.SetSpacing((1.0, 1.0, 1.0))
    case_dir = tmp_path / "inputs" / "SSA"
    case_dir.mkdir(parents=True)
    for suffix in ("0000", "0001", "0002", "0003"):
        sitk.WriteImage(image, str(case_dir / f"SYNSRV-0001_{suffix}.nii.gz"))
    return {"input_dir": case_dir, "output_dir": tmp_path / "predictions"}


def test_measure_predict_runs_the_frozen_ssa_instrument_end_to_end(synthetic_case):
    """Server gate: the canonical verb completes a real nnUNetv2 prediction round."""
    missing = _server_prerequisites()
    if missing is not None:
        pytest.skip(f"DCU host prerequisite not met: {missing}")
    spec = INSTRUMENT_SPECS["SSA"]
    argv = FrozenInstrumentCommand(spec).build(synthetic_case["input_dir"], synthetic_case["output_dir"] / "SSA")[5:]  # strip `<python> -m ctmr measure predict`
    code = MeasurePredictVerb().run(argv)
    assert code == 0
