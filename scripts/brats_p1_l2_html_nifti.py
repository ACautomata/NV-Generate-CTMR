#!/usr/bin/env python3
"""Issue #58 sugon execution side for the L2 generated-case HTML report.

Reads the L2-acceptance data (generated four modalities, real four modalities,
L2 instrument predictions, measurements.csv) on the sugon host, resamples
everything onto the generated image grid (the unified display space), picks a
representative slice per case, and hands the sliced arrays to
``brats_p1_l2_html`` (stdlib + Pillow) to produce a single self-contained HTML
report.  Runs where numpy + SimpleITK + Pillow are available; the rendered HTML
is written to a controlled path -- subject ids and per-case measurements never
land in git.

Two commands:

  discover   scan ``--real-root`` once and write a case -> real-image directory
             index (JSON) so `render` does not glob per case.  The raw data root
             is ``<real-root>/raw/ASNR-MICCAI-BraTS2023/<challenge-dir>/<case>/``;
             the challenge-dir stem is resolved here, not hard-coded.
  render     measurements.csv -> sampled cases -> slices -> HTML report.
             Reuses ``brats_p1_l2_html.CaseSampler`` (sampling),
             ``MeasurementPresenter`` (volumes) and ``L2HtmlReport`` (page).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813  (standard medical-imaging alias)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brats_p1_l2_html import CaseSampler, IndexSummarizer, L2HtmlReport  # noqa: E402
from nnunet_l2_final_acceptance import MODALITIES  # noqa: E402

REAL_CANDIDATE_ROOTS = ("raw/ASNR-MICCAI-BraTS2023", "ASNR-MICCAI-BraTS2023", ".")
VIEW_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}  # sitk array layout is zyx

DEFAULT_NOTE = (
    "P1 is an unconditional image-only generation (per-modality sampling, no tumour "
    "conditioning); L2 TOST tests generated-vs-real tumour-volume equivalence, so "
    "cases that do not preserve the tumour structure are expected to fall short. "
    "The wall of panels is for visual assessment of generated image quality and the "
    "L2 instrument predictions (WT/TC/ET), not a substitute for the TOST verdict."
)


class RealImageIndex:
    """Resolve each case to its real four-modality directory on disk.

    A single recursive glob over ``--real-root`` builds ``{case: challenge_dir}``
    once, so the challenge-directory naming does not have to be hard-coded.
    """

    def __init__(self, real_root):
        self._real_root = Path(real_root)

    def scan(self):
        """Return {case: {"dir": Path, "challenge_dir": str}} for every real case."""
        base = self._resolve_base()
        index = {}
        for image_path in base.glob("**/*-t1c.nii.gz"):
            case = image_path.name[: -len("-t1c.nii.gz")]
            index.setdefault(case, {"dir": str(image_path.parent), "challenge_dir": image_path.parent.parent.name})
        return index

    def write_index(self, out):
        """Scan once and persist the case index JSON to ``out``."""
        index = self.scan()
        Path(out).write_text(json.dumps(index, indent=2))
        print(f"[OK] real image index -> {out} ({len(index)} cases)")

    def _resolve_base(self):
        for rel in REAL_CANDIDATE_ROOTS:
            base = self._real_root / rel
            if any(base.glob("**/*-t1c.nii.gz")):
                return base
        raise SystemExit(f"no BraTS real image tree found under {self._real_root} (tried {REAL_CANDIDATE_ROOTS})")


class SliceScene:
    """Produce 2D slices at a chosen axis for one case on the unified grid.

    The reference grid is the generated t1n volume; every other image (real
    modalities, gen/real predictions) is resampled onto it.  The representative
    slice is chosen at the axis whose label centroid is the generated WT centre
    (falling back to the middle slice when the prediction is empty).
    """

    def __init__(self, gen_root, pred_root, case_index):
        self._gen_root = Path(gen_root)
        self._pred_root = Path(pred_root)
        self._case_index = case_index
        self._resampler_linear = self._make_resampler(sitk.sitkBSpline)
        self._resampler_label = self._make_resampler(sitk.sitkNearestNeighbor)

    @staticmethod
    def _make_resampler(interpolator):
        filter = sitk.ResampleImageFilter()
        filter.SetInterpolator(interpolator)
        return filter

    def _find_gen(self, challenge, case, modality):
        case_dir = self._gen_root / challenge / case
        matches = list(case_dir.glob(f"{case}_{modality}_seed*.nii.gz"))
        return matches[0] if matches else None

    def _load_generated(self, challenge, case, modalities):
        reference = None
        volumes = {}
        for modality in modalities:
            path = self._find_gen(challenge, case, modality)
            if path is None:
                volumes[modality] = None
                continue
            image = sitk.ReadImage(str(path))
            if reference is None:
                reference = image
            volumes[modality] = image
        return reference, volumes

    def _load_real(self, case, modalities):
        info = self._case_index.get(case)
        image = {}
        if info is None:
            return None, image
        case_dir = Path(info["dir"])
        for modality in modalities:
            path = case_dir / f"{case}-{modality}.nii.gz"
            image[modality] = sitk.ReadImage(str(path)) if path.exists() else None
        return info["challenge_dir"], image

    def _load_prediction(self, challenge, case, side):
        path = self._pred_root / challenge / f"{case}__{side}.nii.gz"
        return sitk.ReadImage(str(path)) if path.exists() else None

    def _to_grid(self, reference, image, label):
        if image is None:
            return None
        self._resampler_linear.SetReferenceImage(reference)
        self._resampler_label.SetReferenceImage(reference)
        if label:
            return sitk.GetArrayFromImage(self._resampler_label.Execute(image)).astype(np.int32)
        return sitk.GetArrayFromImage(self._resampler_linear.Execute(image))

    def center_index(self, label_array, view, fallback_center):
        """Return the slice index along ``view`` of the tumour label centroid."""
        loc = np.argwhere(label_array > 0) if label_array is not None else None
        if loc is None or loc.size == 0:
            return fallback_center
        return int(np.mean(loc[:, VIEW_AXIS[view]]))

    def build_case(self, challenge, case, view):
        """Return the CaseCard dict (sliced arrays + measurements) for one case.

        The caller attaches measurements.  ``modality_slices`` keeps raw 2D numpy
        arrays (windowed by the renderer) and ``overlays`` keeps raw label arrays.
        """
        reference, gen_volumes = self._load_generated(challenge, case, MODALITIES)
        if reference is None:
            return None
        _, real_volumes = self._load_real(case, MODALITIES)
        gen_label = self._to_grid(reference, self._load_prediction(challenge, case, "gen"), label=True)
        real_label = self._to_grid(reference, self._load_prediction(challenge, case, "real"), label=True)
        axis = VIEW_AXIS[view]
        ref_array = sitk.GetArrayFromImage(reference)
        center = self.center_index(gen_label if gen_label is not None else real_label, view, ref_array.shape[axis] // 2)

        def _slice_at(arr):
            if arr is None:
                return None
            return np.take(arr, center, axis=axis)

        modality_slices = {}
        for modality in MODALITIES:
            gen_arr = sitk.GetArrayFromImage(gen_volumes[modality]) if gen_volumes[modality] is not None else None
            real_arr = self._to_grid(reference, real_volumes.get(modality), label=False)
            modality_slices[modality] = {"gen": _slice_at(gen_arr), "real": _slice_at(real_arr)}

        overlays = {"gen": _slice_at(gen_label), "real": _slice_at(real_label)}
        return {"slice_label": f"{view}, centre={center}", "modality_slices": modality_slices, "overlays": overlays}


class RenderRunner:
    """Run the ``render`` command: rows -> sampled cases -> sliced scenes -> HTML.

    Mirrors the class-based structure of the rest of the pipeline.  The directory
    scan, slice selection and NIfTI resampling happen in the collaborators; this
    object orchestrates them and writes the report to ``--out`` (controlled
    storage; subject ids and per-case measurements never land in git).
    """

    def __init__(self, args):
        self._args = args

    def run(self):
        rows, by_key = self._read_rows(self._args.measurements)
        report_json = self._load_l2_report(self._args.eval_root)
        index = IndexSummarizer().summarize(rows, report_json)
        case_index = self._load_case_index()
        scene = SliceScene(self._args.gen_root, self._args.pred_root, case_index)
        selected = CaseSampler().sample(rows, self._args.samples_per_challenge)

        cases = []
        missing = []
        for challenge, case in selected:
            data = scene.build_case(challenge, case, self._args.views)
            if data is None:
                missing.append((challenge, case))
                continue
            data.update(
                {
                    "case_id": case,
                    "challenge": challenge,
                    "measurements": {"real": by_key.get((case, "real"), {}), "gen": by_key.get((case, "gen"), {})},
                }
            )
            cases.append(data)

        report = L2HtmlReport().build(cases, index=index, note=self._args.note)
        Path(self._args.out).write_text(report)
        print(f"[OK] L2 case report -> {self._args.out} ({len(cases)} cases; {len(missing)} missing)")
        return 0

    def _read_rows(self, path):
        """Return a list of row dicts plus a (case, side) -> row index."""
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows, {(row["case"], row["side"]): row for row in rows}

    def _load_l2_report(self, eval_root):
        """Return the parsed L2 report JSON (or None when unavailable)."""
        eval_root = Path(eval_root) if eval_root else None
        if eval_root is None:
            return None
        for name in ("l2_final_acceptance_p1.json", "l2_json.json"):
            path = eval_root / name
            if path.exists():
                return json.loads(path.read_text())
        return None

    def _load_case_index(self):
        """Return the case -> real-image index, honouring ``--case-index`` if given."""
        if not self._args.case_index:
            return RealImageIndex(self._args.real_root).scan()
        index = json.loads(Path(self._args.case_index).read_text())
        return {case: {"dir": Path(entry["dir"]), "challenge_dir": entry["challenge_dir"]} for case, entry in index.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("discover", help="write a case -> real-image directory index")
    p.add_argument("--real-root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(handler="discover")

    p = sub.add_parser("render", help="measurements -> sampled cases -> HTML report")
    p.add_argument("--measurements", required=True)
    p.add_argument("--gen-root", required=True)
    p.add_argument("--real-root", required=True)
    p.add_argument("--pred-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eval-root", default=None, help="dir containing l2_final_acceptance_p1.json (optional)")
    p.add_argument("--case-index", default=None, help="pre-generated real image index (optional)")
    p.add_argument("--samples-per-challenge", type=int, default=6, help="cases per challenge (0 = all)")
    p.add_argument("--views", default="axial", choices=VIEW_AXIS)
    p.add_argument("--note", default=DEFAULT_NOTE)
    p.set_defaults(handler="render")

    args = parser.parse_args()

    if args.handler == "discover":
        RealImageIndex(args.real_root).write_index(args.out)
        return 0
    return RenderRunner(args).run()


if __name__ == "__main__":
    sys.exit(main())
