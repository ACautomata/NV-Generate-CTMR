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

"""L2 generated-case HTML report renderer (issue #58, P1 final acceptance tail).

Renders a browsable, self-contained HTML report that shows the P1-candidate
generated cases: each case card pairs the generated four modalities against the
real ones and overlays the L2 instrument predictions (WT/TC/ET) on a common
grid, alongside the L2 measurement table. It is the visual/assessment surface
for the L2 acceptance result -- nominally ``overall = fail`` because P1 is an
unconditional image-only generation (per-modality sampling, no tumour conditioning)
while L2 TOST tests generated-vs-real tumour-volume equivalence.

This module is the stdlib + Pillow half of the two-file pipeline (mirroring
``final_acceptance`` / ``measurement_run``, this package):

  html_report.py        this renderer -- pixel synthesis + HTML assembly
                        (no NIfTI / numpy)
  html_report_nifti.py  execution side -- NIfTI read, unified-space resample,
                        slice selection, then calls this renderer

The renderer consumes already-sliced 2D grayscale / label arrays; it never opens
a NIfTI file, so it runs anywhere with Pillow (stdlib + Pillow). Output is a
single self-contained HTML file with base64-embedded PNGs, opened directly in a
browser -- subject ids and per-case measurements stay in controlled storage and
never land in git.
"""

import base64
import html
import io

from PIL import Image

from ctmr.application.acceptance.distribution.final_acceptance import (
    CHALLENGES,
    MODALITIES,
    REGIONS,
)

# ── display window & overlay tuning ────────────────────────────────────────
WINDOW_LOW_PERCENTILE = 1.0
WINDOW_HIGH_PERCENTILE = 99.0
OVERLAY_ALPHA = 0.55


class BraTSRegionPalette:
    """BraTS region label -> RGBA colour mapping (WT/TC/ET).

    The instrument encodes labels as 1 = non-enhancing/necrotic (NCR/NET),
    2 = peritumoral oedema (ED), 3 = enhancing tumour (ET).  The three regions
    nest (ET subset TC subset WT), so a label maps to its innermost region:

    label 2 (ED)        -> WT (blue)   -- the outermost, whole-tumour ring
    label 1 (NCR/NET)   -> TC (yellow) -- the core
    label 3 (ET)        -> ET (red)    -- the innermost enhancing part

    Hues reuse the dataviz-validated categorical slots (blue #2a78d6, yellow
    #eda100, red #e34948), which stand apart from the status palette used for
    pass/fail, and are defensible by colour alone only as an overlay *on top of*
    the anatomical image; the web legend pairs every region with its name.
    """

    _LABEL_COLOR = {
        2: (0x2A, 0x78, 0xD6),  # WT  blue
        1: (0xED, 0xA1, 0x00),  # TC  yellow
        3: (0xE3, 0x49, 0x48),  # ET  red
    }

    def color_for_label(self, label):
        """Return the (r, g, b) for a label value, or None for background."""
        return self._LABEL_COLOR.get(label)

    def label_hex(self, label):
        """Return a hex string for a label value (used for the legend swatch)."""
        rgb = self._LABEL_COLOR[label]
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def legend_entries(self):
        """Return ordered (region, rgb) tuples for the web legend."""
        # Region order innermost-out: ET, TC, WT; each drawn as the colour of the
        # label that produces it.
        return [
            ("ET (enhancing)", self._LABEL_COLOR[3]),
            ("TC (core)", self._LABEL_COLOR[1]),
            ("WT (whole tumour)", self._LABEL_COLOR[2]),
        ]


class GrayscaleWindowing:
    """Percentile windowing of a MR slice into a 0-255 uint8 gray grid.

    Pure-Python so the renderer stays numpy-free (fine for the sampled case
    count; the sugon side passes the array through unchanged).  The window is
    per-slice, fixed at (WINDOW_LOW_PERCENTILE, WINDOW_HIGH_PERCENTILE) so the
    generated and real images are normalised the same way for comparison.
    """

    def apply(self, grid):
        """Return a list of lists of ints in 0..255 spanning the slice range.

        ``grid`` is any 2D iterable of real numbers (list of lists, or a numpy
        2D array).  Pixels at/below the low percentile map to 0, at/above the
        high percentile to 255, linear in between.
        """
        rows = [list(row) for row in grid]
        flat = []
        for row in rows:
            flat.extend(v for v in row if v is not None)
        if not flat:
            return [[0] * len(rows[0]) for _ in rows]
        flat.sort()
        n = len(flat)
        lo = flat[min(n - 1, int(WINDOW_LOW_PERCENTILE / 100.0 * (n - 1)))]
        hi = flat[min(n - 1, int(WINDOW_HIGH_PERCENTILE / 100.0 * (n - 1)))]
        span = hi - lo
        if span <= 0:
            span = 1
        out = []
        for row in rows:
            g = []
            for v in row:
                if v is None:
                    g.append(0)
                else:
                    g.append(int(255 * min(1.0, max(0.0, (v - lo) / span))))
            out.append(g)
        return out


class SliceRenderer:
    """Turn a 2D gray slice (and optional label slice) into a base64 PNG.

    Owns windowing, label colouring, alpha compositing and PNG/base64 encoding.
    ``render_gray`` yields the anatomical slice; ``render_overlay`` composites
    the instrument labels on top of the same windowed slice.  All PNGs are
    produced from a single row-by-row byte build so the renderer needs no numpy.
    """

    def __init__(self):
        self._window = GrayscaleWindowing()
        self._palette = BraTSRegionPalette()

    def _slice_geometry(self, grid):
        return len(grid), len(grid[0])

    def _gray_bytes(self, grid):
        """Encode a windowed (already 0..255) grid row-major as bytes."""
        out = bytearray()
        for row in grid:
            out.extend(int(v) & 0xFF for v in row)
        return bytes(out)

    def _encode_png(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_gray(self, grid):
        """Return the base64 PNG of a grayscale slice."""
        h, w = self._slice_geometry(grid)
        windowed = self._window.apply(grid)
        img = Image.frombytes("L", (w, h), self._gray_bytes(windowed))
        return self._encode_png(img)

    def _composite(self, windowed, label_grid):
        """Return an RGBA image: gray slice with the label overlay blended in."""
        h, w = self._slice_geometry(windowed)
        gray = Image.frombytes("L", (w, h), self._gray_bytes(windowed)).convert("RGBA")
        # Pixel-interleaved RGBA: PIL's Image.frombytes("RGBA") reads R,G,B,A per
        # pixel, so the mask must be built interleaved (not as four planar bands).
        rgba = bytearray(w * h * 4)
        for y in range(h):
            row = label_grid[y]
            base = y * w
            for x in range(w):
                rgb = self._palette.color_for_label(row[x]) if row[x] is not None else None
                i = (base + x) * 4
                if rgb is None:
                    rgba[i + 3] = 0
                else:
                    rgba[i] = rgb[0]
                    rgba[i + 1] = rgb[1]
                    rgba[i + 2] = rgb[2]
                    rgba[i + 3] = int(255 * OVERLAY_ALPHA)
        mask = Image.frombytes("RGBA", (w, h), bytes(rgba))
        return Image.alpha_composite(gray, mask)

    def render_overlay(self, grid, label_grid):
        """Return the base64 PNG of a slice with the label overlay blended in.

        ``label_grid`` is a 2D iterable of integer label values (0 = background).
        """
        windowed = self._window.apply(grid)
        img = self._composite(windowed, label_grid)
        return self._encode_png(img)


class MeasurementPresenter:
    """Present one measurement row (real or gen) as human-readable fields.

    The input is a dict keyed by MEASUREMENT_FIELDS.  Missing or undefined
    cells render as an em-dash, never as a silent drop or a zero.
    """

    _VOLUME_FIELDS = ("vol_wt_ml", "vol_tc_ml", "vol_et_ml")
    _RATIO_FIELDS = ("wt_brain", "et_wt")
    _DICE_FIELDS = ("cond_dice_wt", "cond_dice_tc", "cond_dice_et")

    def volume_fields(self, row):
        """Return a dict of region -> formatted volume (ml) for a row."""
        return {region: self._fmt(row.get(f"vol_{region.lower()}_ml"), 1) for region in REGIONS}

    def ratio_fields(self, row):
        """Return ('wt_brain', 'et_wt') formatted as percentages."""
        return {name: self._fmt((row.get(name) if row.get(name) not in (None, "") else None), 1, percent=True) for name in self._RATIO_FIELDS}

    def dice_fields(self, row):
        """Return a dict of region -> formatted conditioned Dice for a row."""
        return {region: self._fmt(row.get(f"cond_dice_{region.lower()}"), 2) for region in REGIONS}

    def vol_present(self, row):
        """Whether every region volume is a defined number (for the fail/empty cue)."""
        return all(row.get(f"vol_{region.lower()}_ml") not in (None, "") for region in REGIONS)

    def _fmt(self, value, digits, percent=False):
        if value in (None, ""):
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if percent:
            return f"{number * 100:.{digits}f}%"
        return f"{number:.{digits}f}"


class CaseSampler:
    """Choose an evenly spaced sample of cases per challenge from measurement rows.

    ``rows`` is the list of measurement row dicts (one per observation; real and
    gen both appear for each case).  Cases are deduplicated per challenge and
    taken evenly across the (sorted) case list so a report stays readable for a
    large cohort.  ``per_challenge`` of 0 or None means every case.
    """

    def sample(self, rows, per_challenge):
        """Return a list of (challenge, case_id) tuples, one per selected case."""
        by_chal = {ch: sorted({row["case"] for row in rows if row.get("challenge") == ch and row.get("case")}) for ch in CHALLENGES}
        selected = []
        for ch in CHALLENGES:
            cases = by_chal[ch]
            if not cases or not per_challenge:
                for case in cases:
                    selected.append((ch, case))
                continue
            step = max(1, len(cases) / per_challenge)
            for i in range(per_challenge):
                idx = min(len(cases) - 1, int(i * step))
                case = cases[idx]
                if (ch, case) not in selected:
                    selected.append((ch, case))
        return selected


class IndexSummarizer:
    """Roll measurement rows + the L2 report JSON into the hero summary dict.

    Without a report JSON the verdict stays ``unknown`` and the per-challenge
    counts come from the rows; with it, matching per-challenge verdicts and TOST
    counts are surfaced (best-effort against the published L2 report schema, so a
    schema change degrades to ``unknown`` rather than erroring).
    """

    def summarize(self, rows, report_json):
        present = {(r.get("challenge"), r.get("case")) for r in rows if r.get("case")}
        index = {"overall": "unknown", "challenges": {}}
        for ch in CHALLENGES:
            n = sum(1 for (c, _) in present if c == ch)
            index["challenges"][ch] = {"verdict": "unknown", "tost_passed": "", "tost_total": "", "n_cases": n}
        if not isinstance(report_json, dict):
            return index
        # The L2 report schema (AcceptanceReport.build) exposes overall_verdict
        # and per_challenge[ch] = {verdict, tost: [{quantity, passed, ...}]};
        # fall back to the older naming so a change degrades, not errors.
        overall = report_json.get("overall_verdict", report_json.get("overall"))
        if isinstance(overall, str):
            index["overall"] = overall
        per_challenge = report_json.get("per_challenge", report_json.get("challenges", {}))
        for ch in CHALLENGES:
            row = index["challenges"][ch]
            entry = per_challenge.get(ch, {}) if isinstance(per_challenge, dict) else {}
            if isinstance(entry, dict):
                row["verdict"] = entry.get("verdict", row["verdict"])
                tost = entry.get("tost")
                if isinstance(tost, list):
                    row["tost_passed"] = sum(1 for item in tost if isinstance(item, dict) and item.get("passed"))
                    row["tost_total"] = len(tost)
                elif entry.get("tost_passed") not in (None, ""):
                    row["tost_passed"] = entry.get("tost_passed")
                    row["tost_total"] = entry.get("tost_total", "")
        return index


class CaseCard:
    """Assemble one case card: modalities grid + L2 measurement + overlay panels.

    Owns the `used` tabular-num figure columns and the four-modality grid (real
    top row, generated bottom row), plus the generated/real overlay panels and
    the side-by-side volume comparison.  ``slice_data`` supplies windowed slices
    and labels via the callback hooks so this class stays unaware of arrays.
    """

    def __init__(self, renderer, presenter, palette):
        self._renderer = renderer
        self._presenter = presenter
        self._palette = palette

    def _img(self, label, src):
        return f'<figure class="cell"><img src="data:image/png;base64,{src}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'

    def _overlay_cell(self, title, gray, labels):
        if gray is None:
            return f'<figure class="cell empty"><figcaption>{html.escape(title)} — no image</figcaption></figure>'
        if labels is None:
            src = self._renderer.render_gray(gray)
        else:
            src = self._renderer.render_overlay(gray, labels)
        return self._img(title, src)

    def _row(self, label, values):
        cells = "".join(f"<td>{v}</td>" for v in values)
        return f"<tr><th>{html.escape(label)}</th>{cells}</tr>"

    def render(self, case):
        """Return the HTML fragment for one case card.

        ``case`` is a dict with keys: case_id, challenge, modality_slices
        (mod -> {'real': _, 'gen': _}), overlays ({'gen': _, 'real': _, optional}),
        measurements ({'real': row, 'gen': row}), slice_label.
        """
        real_row = case.get("measurements", {}).get("real", {})
        gen_row = case.get("measurements", {}).get("gen", {})
        pres = self._presenter

        modality_cols = []
        for mod in MODALITIES:
            slices = case["modality_slices"][mod]
            real_img = self._overlay_cell(f"real {mod}", slices.get("real"), None)
            gen_img = self._overlay_cell(f"gen {mod}", slices.get("gen"), None)
            modality_cols.append(f'<div class="col"><div class="row-title mod">{mod}</div>{real_img}{gen_img}</div>')

        gen_overlay = self._overlay_cell(
            "gen + L2 pred",
            case["modality_slices"]["t1c"]["gen"],
            case.get("overlays", {}).get("gen"),
        )
        real_overlay = self._overlay_cell(
            "real + L2 pred",
            case["modality_slices"]["t1c"]["real"],
            case.get("overlays", {}).get("real"),
        )

        vol_real = pres.volume_fields(real_row)
        vol_gen = pres.volume_fields(gen_row)
        vol_rows = [self._row(region, [vol_real[region], vol_gen[region]]) for region in REGIONS]

        dice = pres.dice_fields(gen_row)
        dice_cells = "".join(
            f'<span class="chip dice"><span class="swatch" style="background:{self._palette.label_hex(lbl)}"></span>{region} {dice[region]}</span>'
            for region, lbl in (("WT", 2), ("TC", 1), ("ET", 3))
        )

        empty_cue = ""
        if not pres.vol_present(gen_row):
            empty_cue = '<div class="cue warn">generated prediction empty — tumour volumes undefined (measurement, not failure)</div>'

        ch = html.escape(case["challenge"])
        case_id = html.escape(case["case_id"])
        slice_label = html.escape(case["slice_label"])
        return (
            f'<article class="card" data-challenge="{ch}" data-case="{case_id}">'
            '<header class="card-head">'
            f'<span class="chip challenge">{ch}</span>'
            f'<h3 class="case-id">{case_id}</h3>'
            f'<span class="slice-label">{slice_label}</span>'
            "</header>"
            f"{empty_cue}"
            '<div class="grid">'
            f'<div class="grid-row mods">{"".join(modality_cols)}<div class="col"><div class="row-title mod">overlay</div>{real_overlay}{gen_overlay}</div></div>'
            "</div>"
            '<footer class="card-foot">'
            '<div class="measure"><table class="measure">'
            "<thead><tr><th>region</th><th>real ml</th><th>gen ml</th></tr></thead>"
            f"<tbody>{''.join(vol_rows)}</tbody>"
            "</table></div>"
            f'<div class="legend">{dice_cells}</div>'
            "</footer>"
            "</article>"
        )


class L2HtmlReport:
    """Assemble the whole self-contained HTML report from index + case cards.

    ``index`` is an optional summary dict (overall verdict, per-challenge verdict
    and TOST counts); if absent the report falls back to a cohort-count summary
    derived from the case list.  Output is a single, offline-openable file with
    embedded PNGs, a challenge filter bar, a case search box and the region
    legend.
    """

    _CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--hair:#e1e0d9;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--hair:#2c2c2a;--good:#0ca30c;--bad:#e66767;--warn:#fab219}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--hair:#2c2c2a;--good:#0ca30c;--bad:#e66767;--warn:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:24px 20px}h1{font-size:22px;margin:0 0 4px;font-weight:650}h2{font-size:15px;font-weight:600}.muted{color:var(--muted)}
.hero{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:18px 20px;margin-bottom:16px}
.verdict{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:650}.badge{padding:2px 10px;border-radius:999px;color:#fff;font-size:13px;font-weight:650}
.badge.pass{background:var(--good)}.badge.fail{background:var(--bad)}.badge.warn{background:var(--warn);color:#3a2b00}
.challenges{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.challenge_row{background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:6px 10px;font-size:13px;display:flex;gap:8px;align-items:center}
.challenge_row .name{font-weight:600}.challenge_row .stat{color:var(--muted)}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}.pills{display:flex;gap:6px;flex-wrap:wrap}.pill{border:1px solid var(--hair);background:var(--surface);color:var(--ink2);border-radius:999px;padding:4px 12px;cursor:pointer;font-size:13px}
.pill.active{background:var(--ink);color:var(--surface);border-color:var(--ink)}.search{flex:1;min-width:160px;border:1px solid var(--hair);background:var(--surface);color:var(--ink);border-radius:8px;padding:6px 10px;font-size:14px}
.cards{display:flex;flex-wrap:wrap;gap:14px}.card{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:14px;width:100%}
.card-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}.case-id{margin:0;font-size:16px;font-weight:650}.slice-label{color:var(--muted);font-size:12px}
.chip{border:1px solid var(--hair);background:var(--page);border-radius:8px;padding:2px 8px;font-size:12px;color:var(--ink2)}
.cue{font-size:12px;color:#7a5b00;border:1px solid var(--warn);background:rgba(250,178,25,.08);border-radius:8px;padding:6px 10px;margin-bottom:8px}
.grid{overflow-x:auto}.grid-row{display:flex;gap:8px}.grid-row .col{display:flex;flex-direction:column;gap:6px}
.row-title{font-size:12px;color:var(--muted);text-align:center}.row-title.mod{font-weight:600}
.cell{margin:0;border:1px solid var(--hair);border-radius:6px;padding:0;background:#000}.cell img{display:block;width:128px;height:128px;object-fit:contain;image-rendering:pixelated}.cell figcaption{font-size:11px;color:var(--ink2);text-align:center;padding:2px 0}
.card-foot{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;align-items:flex-start}.measure{border-collapse:collapse;font-size:13px}.measure th,.measure td{border-bottom:1px solid var(--hair);padding:4px 10px;text-align:right}.measure th:first-child,.measure td:first-child{text-align:left}.measure thead th{color:var(--ink2);font-weight:600;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-direction:column;gap:6px;font-size:13px}.chip.dice{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:8px;border:1px solid var(--hair);background:var(--page)}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}.legend .swatch{width:14px;height:14px}
footer{margin-top:24px;color:var(--muted);font-size:12px;text-align:center}
"""

    def __init__(self):
        self._renderer = SliceRenderer()
        self._presenter = MeasurementPresenter()
        self._palette = BraTSRegionPalette()

    def build(self, cases, index=None, note=""):
        """Return the full HTML string.

        ``cases`` is a list of case dicts (see CaseCard.render); ``index`` is an
        optional summary dict; ``note`` is an optional explanatory paragraph for
        the hero.
        """
        head = self._build_hero(index, len(cases), note)
        toolbar = self._build_toolbar()
        cards = "".join(CaseCard(self._renderer, self._presenter, self._palette).render(c) for c in cases)
        legend = self._build_legend()
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>L2 generated-case report</title>\n<style>{self._CSS}</style>\n</head>\n<body>\n"
            '<div class="wrap">'
            f"{head}{toolbar}"
            '<div class="cards" id="cards">'
            f"{cards}"
            "</div>"
            f"{legend}"
            "</div>\n"
            f"{self._SCRIPT}\n"
            "</body>\n</html>\n"
        )

    def _build_hero(self, index, n_cases, note):
        if not index:
            return (
                '<section class="hero"><h1>L2 generated-case report</h1>'
                f'<p class="muted">{n_cases} sampled case(s); L2 verdict summary unavailable.</p></section>'
            )
        overall = index.get("overall", "unknown")
        badge = {"pass": "pass", "fail": "fail"}.get(overall, "warn")
        verdict_line = f'<div class="verdict">L2 final acceptance: <span class="badge {badge}">{html.escape(overall)}</span></div>'
        challenge_rows = []
        for ch in CHALLENGES:
            row = index.get("challenges", {}).get(ch, {})
            verdict = row.get("verdict", "unknown")
            b = {"pass": "pass", "fail": "fail", "undecided": "warn"}.get(verdict, "warn")
            tost = row.get("tost_passed", row.get("tost", ""))
            total = row.get("tost_total", "")
            stat = f"{tost}/{total} TOST passed" if tost != "" and total != "" else ""
            challenge_rows.append(
                f'<div class="challenge_row"><span class="name">{html.escape(ch)}</span>'
                f'<span class="badge {b}">{html.escape(verdict)}</span>'
                f'<span class="stat">{html.escape(str(stat))}</span></div>'
            )
        note_html = f'<p class="muted">{html.escape(note)}</p>' if note else ""
        return (
            '<section class="hero"><h1>L2 generated-case report</h1>'
            f"{verdict_line}"
            f'<p class="muted">{n_cases} sampled case(s)</p>'
            f'<div class="challenges">{"".join(challenge_rows)}</div>'
            f"{note_html}</section>"
        )

    def _build_toolbar(self):
        pills = "".join(f'<button class="pill" data-ch="{ch}">{ch}</button>' for ch in CHALLENGES)
        return (
            '<div class="toolbar">'
            f'<div class="pills"><button class="pill active" data-ch="">all</button>{pills}</div>'
            '<input class="search" id="search" type="text" placeholder="search case id…" aria-label="search cases">'
            "</div>"
        )

    def _build_legend(self):
        entries = "".join(
            f'<span class="chip dice"><span class="swatch" style="background:{self._palette.label_hex(lbl)}"></span>{html.escape(name)}</span>'
            for name, lbl in zip(
                (e[0] for e in self._palette.legend_entries()),
                (3, 1, 2),
            )
        )
        return (
            '<footer><div class="legend">'
            f"{entries}"
            "</div><p>Generated vs real four modalities, L2 instrument predictions (WT/TC/ET) overlaid. "
            "Subject ids and per-case measurements stay in controlled storage.</p></footer>"
        )

    _SCRIPT = """
<script>
(function(){
  var pills=document.querySelectorAll('.pill');
  var cards=document.querySelectorAll('.card');
  var search=document.getElementById('search');
  function apply(){
    var ch=(pillsActive());
    var q=(search.value||'').toLowerCase();
    cards.forEach(function(c){
      var okCh=!ch||c.getAttribute('data-challenge')===ch;
      var okQ=!q||c.getAttribute('data-case').toLowerCase().indexOf(q)>=0;
      c.style.display=(okCh&&okQ)?'':'none';
    });
  }
  function pillsActive(){var a=document.querySelector('.pill.active');return a?a.getAttribute('data-ch'):'';}
  pills.forEach(function(p){
    p.addEventListener('click',function(){
      pills.forEach(function(x){x.classList.remove('active');});
      p.classList.add('active');apply();
    });
  });
  if(search){search.addEventListener('input',apply);}
  apply();
})();
</script>
"""
