"""The stdlib + Pillow html renderer, observed as pytest (issue #140).

The resident ``RendererSelfTest`` became these test functions when the report
pair moved into the distribution package (ADR-0015 §6). Synthetic arrays only
(no numpy / no NIfTI): palette mapping, windowing geometry, PNG encoding,
overlay compositing, measurement presentation, case sampling, the index
summary and HTML assembly with id escaping. Torch-marked tier per ticket #140.
"""

import base64
import io

import pytest
from PIL import Image

from ctmr.application.acceptance.distribution.final_acceptance import MODALITIES
from ctmr.application.acceptance.distribution.html_report import (
    BraTSRegionPalette,
    CaseSampler,
    GrayscaleWindowing,
    IndexSummarizer,
    L2HtmlReport,
    MeasurementPresenter,
    SliceRenderer,
)

pytestmark = pytest.mark.torch


def test_palette_separates_labels_and_maps_background_to_none():
    palette = BraTSRegionPalette()
    assert palette.color_for_label(2) != palette.color_for_label(3)  # WT and ET must not share a colour
    assert palette.color_for_label(0) is None


def test_windowing_preserves_grid_geometry():
    gray = [[0, 10, 100, 200], [30, 5000, 1420, 60]]
    windowed = GrayscaleWindowing().apply(gray)
    assert len(windowed) == 2 and len(windowed[0]) == 4


def test_render_gray_emits_a_base64_png():
    png = SliceRenderer().render_gray([[0, 10, 100, 200], [30, 5000, 1420, 60]])
    assert png.startswith("iVBOR")


_OVERLAY_GRAY = [[0, 10, 100, 200], [30, 5000, 1420, 60]]
_OVERLAY_LABEL = [[0, 0, 3, 0], [0, 0, 0, 0]]


def test_overlay_blends_et_red_dominant_and_keeps_background_grayscale():
    png = SliceRenderer().render_overlay(_OVERLAY_GRAY, _OVERLAY_LABEL)
    assert png.startswith("iVBOR")
    img = Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")
    r, g, b = img.getpixel((2, 0))  # label 3 (ET) -> red-dominant
    assert r > g and r > b
    r0, g0, b0 = img.getpixel((0, 0))  # background label 0 -> grayscale
    assert abs(r0 - g0) <= 12 and abs(g0 - b0) <= 12
    img.close()


_ROW = {
    "vol_wt_ml": "50.0",
    "vol_tc_ml": "30.0",
    "vol_et_ml": "10.0",
    "wt_brain": "0.04",
    "et_wt": "0.10",
    "cond_dice_wt": "0.95",
    "cond_dice_tc": "0.93",
    "cond_dice_et": "",
}


def test_measurement_presentation_formats_volumes_ratios_and_missing_dice():
    presenter = MeasurementPresenter()
    assert presenter.volume_fields(_ROW)["WT"] == "50.0"
    assert presenter.ratio_fields(_ROW)["wt_brain"] == "4.0%"
    assert presenter.dice_fields(_ROW)["ET"] == "—"  # missing dice -> em dash, never a fake number
    assert presenter.vol_present(_ROW)


def test_case_sampler_keeps_per_challenge_count_and_deduplicates_real_gen_rows():
    rows = [{"challenge": "GLI", "case": f"GLI-{i:03d}"} for i in range(10)] + [{"challenge": "GLI", "case": f"GLI-{i:03d}"} for i in range(10)]
    sampled = CaseSampler().sample(rows, 3)
    assert len(sampled) == 3
    assert len(set(sampled)) == 3


_INDEX_ROWS = [
    {"challenge": "GLI", "case": "A", "side": "real"},
    {"challenge": "GLI", "case": "A", "side": "gen"},
    {"challenge": "MEN", "case": "B", "side": "real"},
]

_REPORT_JSON = {
    "overall_verdict": "fail",
    "per_challenge": {
        "GLI": {
            "verdict": "fail",
            "tost": [
                {"quantity": "vol_wt_rel", "passed": False},
                {"quantity": "vol_tc_rel", "passed": False},
            ],
        }
    },
}


def test_index_summarizer_reads_the_report_json_shape():
    index = IndexSummarizer().summarize(_INDEX_ROWS, _REPORT_JSON)
    assert index["overall"] == "fail"
    assert index["challenges"]["GLI"]["n_cases"] == 1
    assert (index["challenges"]["GLI"]["tost_passed"], index["challenges"]["GLI"]["tost_total"]) == (0, 2)
    assert index["challenges"]["MEN"]["verdict"] == "unknown"  # default verdict for a challenge with no rows


def test_report_escapes_case_ids_and_carries_exactly_one_card_per_case():
    case = {
        "case_id": 'BraTS-GLI-00119-000"onload="alert(1)',
        "challenge": "GLI",
        "slice_label": "axial z=64",
        "modality_slices": {m: {"real": _OVERLAY_GRAY, "gen": _OVERLAY_GRAY} for m in MODALITIES},
        "overlays": {"gen": _OVERLAY_LABEL, "real": _OVERLAY_LABEL},
        "measurements": {"real": _ROW, "gen": _ROW},
    }
    report = L2HtmlReport().build([case], index={"overall": "fail"}, note="note")
    assert report.startswith("<!DOCTYPE html>")
    assert report.count('class="card"') == 1
    assert "&quot;" in report  # the injected quote must be escaped, never executed
    assert "fail" in report  # the index overall verdict is shown
