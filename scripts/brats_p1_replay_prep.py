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

"""P1 MR-RATE replay cohort pipeline (issue #57, spec #51 decision 6).

The P1 full-parameter DM continuation mixes BraTS train entries with a 1:1
MR-RATE replay cohort at the data-list level to anchor the frozen v1 trunk
against catastrophic forgetting (issue #10 resolution §3). This script builds
that cohort from the gated HF dataset ``Forithmus/MR-RATE`` (native-space
defaced whole-brain volumes — the same variant the v1 base model trained on,
with the original modality tokens 9/10/11).

Cohort rules (pinned, recorded in ``replay_selection.json``):
- brain studies only — a study qualifies if it has an MR-RATE-atlas
  registration (the atlas repo covers the brain cohort; v1 trained brains);
- MR-RATE patient-level ``train`` split only (their val/test stay untouched);
- series: T1w / T2w / FLAIR (the three BraTS-overlapping modalities),
  not derived, not localizer, not subtraction;
- native shape 32..320 per axis, so the nearest-multiple-of-128 training grid
  stays <= 256 per axis (latent <= 64) alongside the BraTS 64x64x32 latents;
- deterministic order: SHA-256 of ``<study>/<series>`` ascending, first N.

Stages (each standalone, idempotent):

    python scripts/brats_p1_replay_prep.py select \
        --metadata-dir CSV_DIR --splits splits.csv --brain-studies atlas_studies.txt \
        --target-count 7404 --out replay_selection.json
    python scripts/brats_p1_replay_prep.py download \
        --selection replay_selection.json --raw-root RAW_DIR [--hf-token-file PATH]
    python scripts/brats_p1_replay_prep.py encode-list \
        --selection replay_selection.json --raw-root RAW_DIR --emb-root EMB_DIR \
        --autoencoder AE.pt --out-list p1_mrrate_encode_source.json --out-env environment_mrrate_encode.json
    python scripts/brats_p1_replay_prep.py companions \
        --selection replay_selection.json --emb-root EMB_DIR
    python scripts/brats_p1_replay_prep.py lists \
        --selection replay_selection.json --out lists/p1_mrrate_replay.json
    python scripts/brats_p1_replay_prep.py verify \
        --selection replay_selection.json --raw-root RAW_DIR --emb-root EMB_DIR \
        --list lists/p1_mrrate_replay.json --manifest phase_manifest.json --out verify/report.json
    python scripts/brats_p1_replay_prep.py selftest --workdir TMP

DUA: MR-RATE volumes, embeddings and study lists stay in controlled storage
(gauss / sugon private_data); nothing here writes into a git work tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# modality_mapping.json keys for the original (non-skull-stripped) tokens —
# the labels the v1 base model saw for whole-brain native volumes.
REPLAY_MODALITIES = {"T1w": "mri_t1", "T2w": "mri_t2", "FLAIR": "mri_flair"}

# Native-shape window per axis: lower bound drops tiny scout/localizer-like
# series; upper bound keeps the round-to-128 grid at <=256 per axis (latent <=64).
SHAPE_MIN = 32
SHAPE_MAX = 320

REPLAY_SUB = "MRRATE"

# Read-ahead window for ranged zip reads: zipfile reads members in small
# blocks; one HTTP Range request per read would pay the proxy RTT dozens of
# times per study. A 16 MiB window keeps it to a handful of requests.
READ_AHEAD_BYTES = 16 * 1024 * 1024


class ReplayMetadataReader:
    """Reads the pinned MR-RATE metadata CSVs (one per batch) as series records."""

    SERIES_FIELDS = (
        "patient_uid",
        "study_uid",
        "series_id",
        "classified_modality",
        "is_derived",
        "is_localizer",
        "is_subtraction",
        "array_shape",
    )

    def __init__(self, metadata_dir):
        self._dir = Path(metadata_dir)

    def csv_paths(self):
        paths = sorted(self._dir.glob("batch*_metadata.csv"))
        if not paths:
            raise FileNotFoundError(f"no batch*_metadata.csv under {self._dir}")
        return paths

    def iter_series(self):
        """Yields one dict per metadata row (batch inferred from the file name)."""
        for path in self.csv_paths():
            batch = path.name.split("_")[0]
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    record = {field: row.get(field) for field in self.SERIES_FIELDS}
                    record["batch"] = batch
                    yield record


class ReplaySelector:
    """Deterministic replay-cohort selection from MR-RATE metadata (spec #57)."""

    def __init__(self, metadata_dir, splits_csv, brain_studies, target_count):
        self._reader = ReplayMetadataReader(metadata_dir)
        self._splits_csv = Path(splits_csv)
        self._brain_studies = brain_studies
        self._target = target_count

    def train_patients(self):
        with open(self._splits_csv, newline="") as handle:
            return {row["patient_uid"] for row in csv.DictReader(handle) if row["split"] == "train"}

    @staticmethod
    def parse_shape(array_shape):
        try:
            values = json.loads(array_shape)
            return [int(v) for v in values] if len(values) == 3 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def order_key(study, series):
        return hashlib.sha256(f"{study}/{series}".encode()).hexdigest()

    def candidates(self, train_patients):
        for record in self._reader.iter_series():
            if record["classified_modality"] not in REPLAY_MODALITIES:
                continue
            if str(record["is_derived"]).lower() == "true":
                continue
            if str(record["is_localizer"]).lower() == "true":
                continue
            if str(record["is_subtraction"]).lower() == "true":
                continue
            if record["study_uid"] not in self._brain_studies:
                continue
            if record["patient_uid"] not in train_patients:
                continue
            shape = self.parse_shape(record["array_shape"])
            if shape is None or not all(SHAPE_MIN <= dim <= SHAPE_MAX for dim in shape):
                continue
            yield {
                "batch": record["batch"],
                "study": record["study_uid"],
                "series": record["series_id"],
                "modality": REPLAY_MODALITIES[record["classified_modality"]],
                "patient_uid": record["patient_uid"],
                "shape": shape,
            }

    def select(self):
        train_patients = self.train_patients()
        ordered = sorted(self.candidates(train_patients), key=lambda e: self.order_key(e["study"], e["series"]))
        chosen = ordered[: self._target]
        by_modality = {}
        for entry in chosen:
            by_modality[entry["modality"]] = by_modality.get(entry["modality"], 0) + 1
        studies = len({entry["study"] for entry in chosen})
        return {
            "target_count": self._target,
            "candidate_count": len(ordered),
            "selected_count": len(chosen),
            "selected_studies": studies,
            "selected_by_modality": by_modality,
            "rules": {
                "source": "Forithmus/MR-RATE mri/batch00..27 native-space defaced volumes",
                "brain_filter": "study present in the MR-RATE-atlas registration set",
                "split": "MR-RATE patient-level train split only",
                "modalities": sorted(REPLAY_MODALITIES.values()),
                "series_flags_excluded": ["is_derived", "is_localizer", "is_subtraction"],
                "shape_window_per_axis": [SHAPE_MIN, SHAPE_MAX],
                "order": "sha256('<study>/<series>') ascending, first target_count",
            },
            "entries": chosen,
        }


class HttpRangeFile:
    """Seekable read-only file over HTTP Range requests against a resolved LFS URL.

    Uses the 0-0 probe GET to follow the resolve redirect once (Content-Range
    carries the total size); subsequent reads hit the signed CDN URL directly.
    """

    def __init__(self, url, token=None, timeout=120):
        import requests  # deferred: execution side only (gauss system python)

        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        probe = self._session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=timeout)
        probe.raise_for_status()
        content_range = probe.headers.get("Content-Range")
        if content_range is None:
            raise ValueError(f"server did not honour Range (no Content-Range): {url}")
        self._url = probe.url
        self._size = int(content_range.rsplit("/", 1)[1])
        self._pos = 0
        self._cache_start = -1
        self._cache = b""

    def _fetch(self, start, end):
        response = self._session.get(
            self._url, headers={"Range": f"bytes={start}-{end}"}, timeout=self._timeout
        )
        response.raise_for_status()
        return response.content

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._pos
        if size == 0 or self._pos >= self._size:
            return b""
        cached_end = self._cache_start + len(self._cache)
        if 0 <= self._cache_start <= self._pos and self._pos + size <= cached_end:
            data = self._cache[self._pos - self._cache_start : self._pos - self._cache_start + size]
            self._pos += size
            return data
        fetch_end = min(self._pos + max(size, READ_AHEAD_BYTES), self._size) - 1
        self._cache_start = self._pos
        self._cache = self._fetch(self._pos, fetch_end)
        data = self._cache[:size]
        self._pos += len(data)
        return data


class ReplayDownloader:
    """Fetches only the chosen members via ranged zip reads (whole-zip download avoided)."""

    def __init__(self, selection, raw_root, token=None):
        self._selection = selection
        self._raw_root = Path(raw_root)
        self._token = token

    def target_path(self, entry):
        return self._raw_root / "MR-RATE" / entry["batch"] / entry["study"] / f"{entry['study']}_{entry['series']}.nii.gz"

    @staticmethod
    def resolve_url(repo_id, entry):
        # HF_ENDPOINT covers mirror deployments (sugon: https://hf-mirror.com);
        # requests picks up http_proxy/https_proxy from the environment.
        base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{base}/datasets/{repo_id}/resolve/main/mri/{entry['batch']}/{entry['study']}.zip"

    def run(self, repo_id="Forithmus/MR-RATE", shard=None):
        """shard: optional (index, total) — only entries with pos % total == index."""
        shard_index, shard_total = shard if shard else (0, 1)
        failures = []
        todo = [
            (index, entry)
            for index, entry in enumerate(self._selection["entries"])
            if index % shard_total == shard_index
        ]
        for done, (index, entry) in enumerate(todo):
            target = self.target_path(entry)
            if target.is_file() and target.stat().st_size > 0:
                continue
            member = f"{entry['study']}/img/{entry['study']}_{entry['series']}.nii.gz"
            try:
                remote = HttpRangeFile(self.resolve_url(repo_id, entry), token=self._token)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(remote) as archive:
                    with archive.open(member) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
            except Exception as error:  # noqa: BLE001 - one bad study must not kill the cohort
                failures.append({"study": entry["study"], "series": entry["series"], "error": str(error)})
                continue
            if (done + 1) % 100 == 0:
                print(f"[download shard {shard_index}/{shard_total}] {done + 1}/{len(todo)} series extracted", flush=True)
        return failures


class EncodeListWriter:
    """Emits the upstream create-training-data inputs for the replay cohort."""

    def __init__(self, selection):
        self._selection = selection

    def entries(self):
        return [
            {
                "image": f"MR-RATE/{entry['batch']}/{entry['study']}/{entry['study']}_{entry['series']}.nii.gz",
                "modality": entry["modality"],
                "sub": REPLAY_SUB,
                "case": entry["study"],
            }
            for entry in self._selection["entries"]
        ]

    def write(self, out_list, out_env, raw_root, emb_root, autoencoder_path):
        list_path = Path(out_list)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text(json.dumps({"training": self.entries()}, indent=1) + "\n")
        env_path = Path(out_env)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            json.dumps(
                {
                    "data_base_dir": str(Path(raw_root).resolve()),
                    "embedding_base_dir": str(Path(emb_root).resolve()),
                    "json_data_list": str(list_path.resolve()),
                    "trained_autoencoder_path": str(Path(autoencoder_path).resolve()),
                },
                indent=4,
            )
            + "\n"
        )
        print(f"encode list: {len(self.entries())} entries -> {list_path}\nencode env -> {env_path}")
        return list_path


class CompanionWriter:
    """Writes the per-embedding {spacing, modality} companion the DM loader reads."""

    def __init__(self, selection, emb_root):
        self._selection = selection
        self._emb_root = Path(emb_root)

    def emb_path(self, entry):
        rel = f"MR-RATE/{entry['batch']}/{entry['study']}/{entry['study']}_{entry['series']}.nii.gz"
        return self._emb_root / rel.replace(".nii.gz", "_emb.nii.gz")

    def write_all(self):
        import nibabel as nib  # deferred: pure-stdlib stages must run anywhere

        written = 0
        for entry in self._selection["entries"]:
            emb = self.emb_path(entry)
            if not emb.is_file():
                raise FileNotFoundError(f"embedding missing (encode first): {emb}")
            spacing = [float(v) for v in nib.load(str(emb)).header["pixdim"][1:4]]
            emb.with_name(emb.name + ".json").write_text(json.dumps({"spacing": spacing, "modality": entry["modality"]}))
            written += 1
        print(f"companions: {written} written under {self._emb_root}")
        return written


class ReplayListBuilder:
    """Writes the contract-pinned replay data list (list-level 1:1 replay input)."""

    def __init__(self, selection):
        self._selection = selection

    def write(self, out_path):
        entries = EncodeListWriter(self._selection).entries()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"training": entries}, indent=1) + "\n")
        print(f"replay data list: {len(entries)} entries -> {path}")
        return path


class ReplayVerifier:
    """Cross-checks the finished replay artifacts against the cohort rules."""

    def __init__(self, selection, raw_root, emb_root, list_path, manifest_path):
        self._selection = selection
        self._raw_root = Path(raw_root)
        self._emb_root = Path(emb_root)
        self._list_path = Path(list_path)
        self._manifest_path = Path(manifest_path)

    def failures(self):
        problems = []
        entries = self._selection["entries"]
        manifest_cases = set()
        manifest = json.loads(self._manifest_path.read_text())
        for info in manifest["challenges"].values():
            for side in ("train", "dev", "holdout"):
                manifest_cases.update(info["cases"][side])
        seen_pairs = set()
        data_list = json.loads(self._list_path.read_text())["training"]
        if len(data_list) != len(entries):
            problems.append(f"replay list carries {len(data_list)} entries, selection has {len(entries)}")
        for entry in entries:
            pair = (entry["study"], entry["series"])
            if pair in seen_pairs:
                problems.append(f"duplicate (study, series) pair in selection: {entry['study']}/{entry['series']}")
            seen_pairs.add(pair)
            if entry["study"] in manifest_cases:
                problems.append(f"replay study collides with a BraTS manifest case id: {entry['study']}")
            if entry["modality"] not in REPLAY_MODALITIES.values():
                problems.append(f"unexpected replay modality token: {entry['modality']}")
            raw = ReplayDownloader(self._selection, self._raw_root).target_path(entry)
            if not raw.is_file():
                problems.append(f"raw volume missing: {raw}")
            emb = CompanionWriter(self._selection, self._emb_root).emb_path(entry)
            if not emb.is_file():
                problems.append(f"embedding missing: {emb}")
                continue
            companion = emb.with_name(emb.name + ".json")
            if not companion.is_file():
                problems.append(f"companion missing: {companion}")
                continue
            payload = json.loads(companion.read_text())
            if payload.get("modality") != entry["modality"]:
                problems.append(f"companion modality mismatch: {emb}")
            spacing = payload.get("spacing", [])
            if len(spacing) != 3 or not all(0.05 < value < 5.0 for value in spacing):
                problems.append(f"implausible replay spacing {spacing}: {emb}")
        return problems


class ReplayPrepSelfTest:
    """Synthetic-fixture check of select/list/verify (no network, no volumes)."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def write_fixture(self):
        root = self._workdir / "fixture"
        meta_dir = root / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            # (batch, patient, study, series, modality, derived, localizer, shape, brain, split)
            ("batch00", "P1", "S1", "t1w-raw-axi", "T1w", "False", "False", "[256, 256, 176]", True, "train"),
            ("batch00", "P1", "S1", "t2w-raw-axi", "T2w", "False", "False", "[256, 256, 176]", True, "train"),
            ("batch00", "P2", "S2", "flair-raw-sag", "FLAIR", "False", "False", "[240, 240, 155]", True, "train"),
            ("batch00", "P3", "S3", "t1w-raw-axi", "T1w", "False", "False", "[256, 256, 176]", False, "train"),  # not brain
            ("batch00", "P4", "S4", "t1w-raw-axi", "T1w", "False", "False", "[256, 256, 176]", True, "test"),  # wrong split
            ("batch01", "P5", "S5", "swi-raw-axi", "SWI", "False", "False", "[256, 256, 176]", True, "train"),  # wrong modality
            ("batch01", "P6", "S6", "t1-derived", "T1w", "True", "False", "[256, 256, 176]", True, "train"),  # derived
            ("batch01", "P7", "S7", "t1w-loc", "T1w", "False", "True", "[512, 512, 12]", True, "train"),  # localizer + shape
            ("batch01", "P8", "S8", "t1w-huge", "T1w", "False", "False", "[512, 512, 320]", True, "train"),  # shape too large
        ]
        batches = {}
        for row in rows:
            batches.setdefault(row[0], []).append(row)
        for batch, batch_rows in batches.items():
            with open(meta_dir / f"{batch}_metadata.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ReplayMetadataReader.SERIES_FIELDS)
                writer.writeheader()
                for (b, patient, study, series, modality, derived, localizer, shape, _brain, _split) in batch_rows:
                    writer.writerow(
                        {
                            "patient_uid": patient,
                            "study_uid": study,
                            "series_id": series,
                            "classified_modality": modality,
                            "is_derived": derived,
                            "is_localizer": localizer,
                            "is_subtraction": "False",
                            "array_shape": shape,
                        }
                    )
        splits = root / "splits.csv"
        with open(splits, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient_uid", "split"])
            writer.writeheader()
            for patient in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
                writer.writerow({"patient_uid": patient, "split": "train" if patient != "P4" else "test"})
        brain = root / "atlas_studies.txt"
        brain.write_text("S1\nS2\nS4\nS5\nS6\nS7\nS8\n")
        manifest = root / "phase_manifest.json"
        manifest.write_text(
            json.dumps({"split_id": "selftest", "challenges": {"GLI": {"cases": {"train": ["BraTS-GLI-0000-000"], "dev": [], "holdout": []}}}})
        )
        return root, brain, splits

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        root, brain, splits = self.write_fixture()
        selector = ReplaySelector(root / "metadata", splits, {line.strip() for line in brain.read_text().splitlines() if line.strip()}, 7404)
        selection = selector.select()
        if selection["selected_count"] != 3:
            self.failures.append(f"select: expected 3 qualifying series (S1 t1w, S1 t2w, S2 flair), got {selection['selected_count']}")
        if {entry["series"] for entry in selection["entries"]} != {"t1w-raw-axi", "t2w-raw-axi", "flair-raw-sag"}:
            self.failures.append("select: filter rules admitted or dropped the wrong series")
        # Determinism: same inputs, same order.
        again = ReplaySelector(root / "metadata", splits, {line.strip() for line in brain.read_text().splitlines() if line.strip()}, 7404).select()
        if again["entries"] != selection["entries"]:
            self.failures.append("select: not deterministic across runs")

        raw_root = root / "raw"
        emb_root = root / "emb"
        for entry in selection["entries"]:
            raw = ReplayDownloader(selection, raw_root).target_path(entry)
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(b"fixture-volume")
            emb = CompanionWriter(selection, emb_root).emb_path(entry)
            emb.parent.mkdir(parents=True, exist_ok=True)
            emb.write_bytes(b"fixture-emb")
            spacing = [round(fov / dim, 6) for fov, dim in zip((240.0, 240.0, 180.0), entry["shape"])]
            emb.with_name(emb.name + ".json").write_text(json.dumps({"spacing": spacing, "modality": entry["modality"]}))
        list_path = ReplayListBuilder(selection).write(root / "lists" / "p1_mrrate_replay.json")
        problems = ReplayVerifier(selection, raw_root, emb_root, list_path, root / "phase_manifest.json").failures()
        self.failures += [f"verify positive: {problem}" for problem in problems]

        # A manifest-colliding study id must be flagged.
        tampered = json.loads(list_path.read_text())
        colliding_study = selection["entries"][0]["study"]
        manifest = json.loads((root / "phase_manifest.json").read_text())
        manifest["challenges"]["GLI"]["cases"]["train"].append(colliding_study)
        (root / "phase_manifest_collision.json").write_text(json.dumps(manifest))
        problems = ReplayVerifier(selection, raw_root, emb_root, list_path, root / "phase_manifest_collision.json").failures()
        if not any("collides with a BraTS manifest case" in problem for problem in problems):
            self.failures.append("verify: manifest case-id collision not detected")
        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select", help="deterministic replay-cohort selection from MR-RATE metadata")
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--splits", required=True, help="MR-RATE splits.csv (patient-level)")
    p.add_argument("--brain-studies", required=True, help="text file with one atlas-registered study uid per line")
    p.add_argument("--target-count", type=int, default=7404, help="1:1 against the 7404-entry BraTS p1 train list")
    p.add_argument("--out", required=True)

    p = sub.add_parser("download", help="fetch the chosen series via ranged zip member reads (gauss)")
    p.add_argument("--selection", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--failures-out", default=None, help="where to write the per-study failure list")
    p.add_argument("--hf-token-file", default=None, help="file holding the gated-dataset HF token")
    p.add_argument("--shard", default=None, help="i/N — only entries with pos %% N == i (parallel shards)")

    p = sub.add_parser("encode-list", help="emit the upstream encode inputs + env config")
    p.add_argument("--selection", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("--autoencoder", required=True)
    p.add_argument("--out-list", required=True)
    p.add_argument("--out-env", required=True)

    p = sub.add_parser("companions", help="write per-embedding {spacing, modality} companions")
    p.add_argument("--selection", required=True)
    p.add_argument("--emb-root", required=True)

    p = sub.add_parser("lists", help="write the contract-pinned replay data list")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("verify", help="verify finished replay artifacts against the cohort rules")
    p.add_argument("--selection", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="verify report json (path must not be inside a git work tree)")

    p = sub.add_parser("selftest", help="fixture-driven check of select/lists/verify (stdlib only)")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    if args.command == "select":
        brain = {line.strip() for line in Path(args.brain_studies).read_text().splitlines() if line.strip()}
        selection = ReplaySelector(args.metadata_dir, args.splits, brain, args.target_count).select()
        if selection["selected_count"] < selection["target_count"]:
            print(
                f"WARNING: only {selection['selected_count']} of {selection['target_count']} candidates qualified "
                f"(pool {selection['candidate_count']}); the 1:1 replay ratio requires the full target",
                file=sys.stderr,
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(selection, indent=1) + "\n")
        print(f"replay selection: {selection['selected_count']} series / {selection['selected_studies']} studies -> {out}")
        return 0

    if args.command == "selftest":
        failures = ReplayPrepSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0

    selection = json.loads(Path(args.selection).read_text())

    if args.command == "download":
        token = Path(args.hf_token_file).read_text().strip() if args.hf_token_file else None
        shard = None
        if args.shard:
            index_text, _, total_text = args.shard.partition("/")
            shard = (int(index_text), int(total_text))
        failures = ReplayDownloader(selection, args.raw_root, token=token).run(shard=shard)
        if args.failures_out:
            Path(args.failures_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.failures_out).write_text(json.dumps(failures, indent=1) + "\n")
        print(f"download finished: {len(failures)} failed series")
        return 1 if failures else 0

    if args.command == "encode-list":
        EncodeListWriter(selection).write(args.out_list, args.out_env, args.raw_root, args.emb_root, args.autoencoder)
        return 0

    if args.command == "companions":
        CompanionWriter(selection, args.emb_root).write_all()
        return 0

    if args.command == "lists":
        ReplayListBuilder(selection).write(args.out)
        return 0

    if args.command == "verify":
        problems = ReplayVerifier(selection, args.raw_root, args.emb_root, args.list, args.manifest).failures()
        out = Path(args.out)
        if any(parent.name == ".git" for parent in out.parents):
            print("CONTRACT VIOLATION: verify report must not live inside a git work tree (DUA)", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"failures": problems, "checked": len(selection["entries"]), "passed": not problems}, indent=1) + "\n")
        for problem in problems:
            print("FAIL " + problem, file=sys.stderr)
        print(f"VERIFY {'PASS' if not problems else 'FAIL'} ({len(problems)} problems, {len(selection['entries'])} entries)")
        return 0 if not problems else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
