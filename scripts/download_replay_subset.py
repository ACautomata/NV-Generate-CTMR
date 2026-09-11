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

"""Manifest-driven replay download: study zips -> extracted series -> skull-stripped twin
(spec #13 sections 3.1 / 3.3, ticket T6 #22, stage C2).

Downloads MR-RATE at **study-zip granularity** driven by a replay manifest
(``hf download --include "mri/batchXX/<study_uid>.zip"``; the official batch-level mode is
~290 GB per batch and is not used), extracts only the manifest-listed series, applies the
``scripts.spine_filter`` heuristic to each extracted image/mask pair, derives the
skull-stripped twin with ``scripts.create_skull_stripped``, and records a per-series verdict.

A study zip holds many series, so candidates are grouped by study and each zip is fetched
exactly once.  The zip is deleted as soon as the study's last listed series has been
processed: a zip carries the whole study (~74 MB on average across the N=1000 manifest)
and keeping 3,737 of them would cost ~277 GB for two files each.

Resumability is the point of the verdict CSV: every processed series appends one line, and
a rerun skips whatever is already on record.  That is also the top-up closed loop -- the
spine filter rejects some series, the manifest is regenerated along each modality's seeded
ordering to refill the shortage, and only the new rows are downloaded.

The verdict CSV repeats the manifest columns so the accepted CSV is a manifest in its own
right; the top-up step therefore consumes the same format it always did.

Usage::

    python -m scripts.download_replay_subset \\
        --manifest    $RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000.csv \\
        --download-dir $RUN_ROOT/runs/replay-<yyyymmdd>/data \\
        --verdicts-csv $RUN_ROOT/datasets/MR-RATE/manifests/replay_verdicts.csv \\
        --accepted-csv $RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000_accepted.csv \\
        --rejected-csv $RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000_rejected.csv \\
        --workers 32
"""

import argparse
import csv
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import snapshot_download

from .create_replay_manifest import MANIFEST_COLUMNS
from .create_skull_stripped import SkullStrippedCreator
from .mrrate_series import MrRateVariant
from .spine_filter import SpineFilter, SpineFilterResult

REPO_ID = "Forithmus/MR-RATE"
DEFAULT_WORKERS = 16
MEASUREMENT_COLUMNS = ("verdict", "mask_voxel_ratio", "fov_mm", "reasons")
VERDICT_COLUMNS = (*MANIFEST_COLUMNS, *MEASUREMENT_COLUMNS)
VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"


@dataclass(frozen=True)
class ManifestCandidate:
    """One manifest row: a series to download, filter, derive and hand to the encoder."""

    patient_uid: str
    study_uid: str
    series_id: str
    modality: str
    label: str
    split: str
    image_path: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ManifestCandidate":
        """Build from a manifest CSV row, refusing a row whose label column lies about its series id.

        The manifest's ``label`` drives the training condition, while the dual twin's label is
        derived from the series-id prefix -- a disagreement would land as a wrong label in the
        training set, or as a ``KeyError`` hours into the run for an unknown modality.  Checking
        in the factory rather than at each call site is what makes the ticket's "与 manifest 一致"
        hold for *every* consumer: the download, the encoder input list and the sidecar finalize
        all go through here, so none of them can silently skip the check.
        """
        candidate = cls(**{name: row[name] for name in MANIFEST_COLUMNS})
        if candidate.label != candidate.variant.label:
            raise ValueError(
                f"{candidate.study_uid}/{candidate.series_id}: manifest label {candidate.label!r} "
                f"contradicts the series id (implies {candidate.variant.label!r})"
            )
        return candidate

    @property
    def batch(self) -> str:
        """The MR-RATE batch the study zip lives in (``mri/<batch>/<study_uid>.zip``)."""
        return self.variant.batch

    @property
    def variant(self) -> MrRateVariant:
        """The path/label view the download, filter and derive stages operate on."""
        return MrRateVariant(self.image_path)

    @property
    def repo_zip_path(self) -> str:
        """The zip's path inside the HuggingFace dataset repository."""
        return f"mri/{self.batch}/{self.study_uid}.zip"

    def zip_member(self, path: str) -> str:
        """A materialized path's name inside the study zip (the zip root is the study directory)."""
        return path.removeprefix(f"mri/{self.batch}/")

    @property
    def key(self) -> tuple[str, str]:
        """``(study_uid, series_id)``: the identity the verdict log resumes on."""
        return (self.study_uid, self.series_id)

    def as_row(self) -> dict[str, str]:
        """This candidate as a manifest CSV row, in the frozen column order."""
        return {name: getattr(self, name) for name in MANIFEST_COLUMNS}


@dataclass(frozen=True)
class SeriesVerdict:
    """One filtered series: the spine heuristic's verdict plus the measured quantities."""

    status: str
    mask_voxel_ratio: float
    fov_mm: tuple[float, ...]
    reasons: tuple[str, ...]

    @property
    def is_accepted(self) -> bool:
        return self.status == VERDICT_ACCEPT

    @classmethod
    def from_filter_result(cls, result: SpineFilterResult) -> "SeriesVerdict":
        return cls(
            status=VERDICT_ACCEPT if result.is_brain else VERDICT_REJECT,
            mask_voxel_ratio=result.mask_voxel_ratio,
            fov_mm=result.fov_mm,
            reasons=result.reasons,
        )

    @classmethod
    def unusable(cls, reason: str) -> "SeriesVerdict":
        """A series whose image and mask cannot be combined at all, so nothing could be measured.

        The measurements are left empty rather than zeroed: a row reading ``0.0 %`` and a
        0 mm FOV looks like a measurement, and this series was never measured.  Recording it as
        a *reject* is what keeps the top-up loop closed -- a series with no verdict stays in the
        todo list, so every rerun stops on the same row and the roster never advances.
        """
        return cls(status=VERDICT_REJECT, mask_voxel_ratio=float("nan"), fov_mm=(), reasons=(reason,))

    def to_row(self, candidate: ManifestCandidate) -> dict[str, str]:
        """The verdict CSV line for this candidate: manifest columns + measurements + verdict.

        The keys are emitted in ``VERDICT_COLUMNS`` order, so the header the log writes on its
        first append lines up with every row written afterwards.
        """
        values = {
            **candidate.as_row(),
            "verdict": self.status,
            "mask_voxel_ratio": f"{self.mask_voxel_ratio:.6f}",
            "fov_mm": " ".join(f"{extent:.2f}" for extent in self.fov_mm),
            "reasons": "; ".join(self.reasons),
        }
        return {name: values[name] for name in VERDICT_COLUMNS}


@dataclass(frozen=True)
class StudyOutcome:
    """What happened to one series: its verdict, or nothing when its zip could not be read."""

    verdict: SeriesVerdict | None


class ZipSource(Protocol):
    """Fetches study zips from the dataset repository into a local directory."""

    def fetch(self, repo_paths: list[str]) -> Path:
        """Download every ``repo_paths`` entry and return the local root the files landed under."""
        ...


class HuggingFaceZipSource:
    """``snapshot_download``-backed fetch, authenticated from the ambient HF credentials.

    The token is resolved explicitly rather than left to the client's own lookup: the run
    sets ``HF_HOME`` to a cache on the scratch disk (a plain ``hf auth login`` would land in
    a nearly-full ``$HOME``), and the client only looks for the token file under
    ``HF_HOME``.  Both layout generations are supported -- ``~/.cache/huggingface/token``
    holding the bare token, and ``stored_tokens`` holding an INI-style ``[section]``/``key
    = value`` block.

    The run is hours long over a few thousand zips, so a transient network failure must not
    kill it: each fetch retries a few times with a widening backoff before giving up.  A
    chunk that still fails propagates -- its studies stay unrecorded, and rerunning the
    script picks up exactly those.

    ``downloader`` is a collaborator, not a hard dependency: it defaults to the real
    ``snapshot_download`` call and tests inject a scripted one instead of reaching for the
    network.
    """

    ATTEMPTS = 4
    BACKOFF_SECONDS = 30.0

    def __init__(self, repo_id: str = REPO_ID, max_workers: int = 8, downloader: Callable[[list[str]], str] | None = None) -> None:
        self._repo_id = repo_id
        self._max_workers = max_workers
        self._token = self._resolve_token()
        self._downloader = downloader or self.download

    def fetch(self, repo_paths: list[str]) -> Path:
        for attempt in range(1, self.ATTEMPTS + 1):
            try:
                return Path(self._downloader(repo_paths))
            except Exception as error:  # noqa: BLE001 - one bad chunk must not end a multi-hour run
                if attempt == self.ATTEMPTS:
                    raise
                print(f"  fetch attempt {attempt}/{self.ATTEMPTS} failed ({error}); retrying in {self.BACKOFF_SECONDS * attempt:.0f}s")
                time.sleep(self.BACKOFF_SECONDS * attempt)
        raise RuntimeError("unreachable")

    def download(self, repo_paths: list[str]) -> str:
        """One ``snapshot_download`` call: fetch every listed repo path into the ambient HF cache.

        ``huggingface_hub`` is a declared dependency (``requirements.txt``) rather than an
        optional one, so it is imported at module level like every other dependency.
        """
        return str(
            snapshot_download(
                self._repo_id,
                repo_type="dataset",
                allow_patterns=repo_paths,
                max_workers=self._max_workers,
                token=self._token,
            )
        )

    @staticmethod
    def _resolve_token() -> str | None:
        """The HuggingFace token: ``HF_TOKEN`` if set, else the first readable credential file, else ``None``."""
        if token := os.environ.get("HF_TOKEN"):
            return token.strip()
        for candidate in HuggingFaceZipSource._credential_paths():
            if candidate.is_file():
                parsed = HuggingFaceZipSource._parse_credential_file(candidate)
                if parsed:
                    return parsed
        return None

    @staticmethod
    def _credential_paths() -> list[Path]:
        roots = [Path(home) for home in (os.environ.get("HF_HOME"),) if home] + [Path.home() / ".cache" / "huggingface"]
        return [root / name for root in roots for name in ("token", "stored_tokens")]

    @staticmethod
    def _parse_credential_file(path: Path) -> str | None:
        """Read a token out of either credential layout; the token file wins over INI parsing."""
        content = path.read_text().strip()
        if content.startswith("hf_"):
            return content.splitlines()[0].strip()
        for line in content.splitlines():
            _, separator, value = line.partition("=")
            if separator and value.strip().startswith("hf_"):
                return value.strip()
        return None


class StudyZipExtractor:
    """Extracts exactly the manifest-listed members of a study zip into the data tree."""

    def materialize(self, archive: Path, candidates: list[ManifestCandidate], staging_dir: Path, download_dir: Path) -> dict[Path, Path]:
        """Extract every candidate's image and mask; return the staged-path -> destination mapping.

        Destinations follow the official unzip layout (``mri/<batch>/<study>/img/...``), which is
        the layout the manifest's ``image_path`` assumes -- the paths in dataset.json entries must
        resolve under the training env's ``data_base_dir``, and the encoding stage reads those same
        relative paths back.
        """
        members = set()
        for candidate in candidates:
            members.add(candidate.zip_member(candidate.variant.whole_brain_path))
            members.add(candidate.zip_member(candidate.variant.mask_path))
        staging_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            available = set(zipped.namelist())
            missing = sorted(members - available)
            if missing:
                raise FileNotFoundError(f"{archive.name}: missing {missing}; the zip carries {len(available)} members")
            zipped.extractall(staging_dir, members=members)
        mapping = {}
        for candidate in candidates:
            for path in (candidate.variant.whole_brain_path, candidate.variant.mask_path):
                mapping[staging_dir / candidate.zip_member(path)] = download_dir / path
        return mapping

    @staticmethod
    def move_into_place(mapping: dict[Path, Path]) -> set[Path]:
        """Move each staged file to its final path; return the destinations that now exist."""
        placed = set()
        for staged, destination in mapping.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(staged, destination)
            placed.add(destination)
        return placed


class VerdictLog:
    """Append-only record of processed series; the resume point and the top-up audit trail."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def completed(self) -> set[tuple[str, str]]:
        """``(study_uid, series_id)`` of every series with a verdict on record."""
        return {(row["study_uid"], row["series_id"]) for row in self.read_rows()}

    def read_rows(self) -> list[dict[str, str]]:
        if not self._path.is_file():
            return []
        with self._path.open(newline="") as file:
            return list(csv.DictReader(file))

    def append(self, row: dict[str, str]) -> None:
        """Append one verdict line, writing the header on first use (never truncates: resume-safe)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.is_file()
        with self._path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(VERDICT_COLUMNS))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def rewrite_in_order(self, order: dict[tuple[str, str], int]) -> None:
        """Rewrite the log in manifest order so reruns diff cleanly; rows with no position keep their relative order."""
        rows = self.read_rows()
        rows.sort(key=lambda row: order.get((row["study_uid"], row["series_id"]), len(order)))
        self._path.unlink(missing_ok=True)
        for row in rows:
            self.append(row)

    def save_subset(self, destination: Path, verdict: str) -> int:
        """Write the rows carrying ``verdict`` to ``destination`` in manifest columns; return the row count."""
        rows = [row for row in self.read_rows() if row["verdict"] == verdict]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)


class ReplayDownloader:
    """Manifest-driven download of the replay subset: fetch, extract, filter, derive, record."""

    def __init__(
        self,
        manifest_path: Path,
        download_dir: Path,
        source: ZipSource,
        verdict_log: VerdictLog,
        filter_service: SpineFilter | None = None,
        stripper: SkullStrippedCreator | None = None,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self._manifest_path = manifest_path
        self._download_dir = download_dir
        self._source = source
        self._log = verdict_log
        self._filter = filter_service or SpineFilter()
        self._stripper = stripper or SkullStrippedCreator()
        self._workers = workers
        self._extractor = StudyZipExtractor()

    def candidates(self) -> list[ManifestCandidate]:
        """The manifest rows in manifest order, each checked against the path convention it implies."""
        with self._manifest_path.open(newline="") as file:
            return [ManifestCandidate.from_row(row) for row in csv.DictReader(file)]

    def run(self, accepted_csv: Path, rejected_csv: Path) -> dict:
        """Process every not-yet-recorded candidate, then write the accepted/rejected manifests."""
        candidates = self.candidates()
        todo = [candidate for candidate in candidates if candidate.key not in self._log.completed()]
        print(f"manifest={len(candidates)} already_recorded={len(candidates) - len(todo)} to_process={len(todo)}")
        failed = 0
        for chunk, download_root in self._chunks(todo):
            for group in self._group_by_study(chunk):
                failed += sum(outcome.verdict is None for outcome in self._process_study(group, download_root))
            print(f"  recorded={len(self._log.completed())} failed={failed}")
        self._clear_staging()
        self._log.rewrite_in_order({candidate.key: index for index, candidate in enumerate(candidates)})
        accepted = self._log.save_subset(accepted_csv, VERDICT_ACCEPT)
        rejected = self._log.save_subset(rejected_csv, VERDICT_REJECT)
        print(f"accepted={accepted} rejected={rejected} failed={failed}")
        print(f"wrote {accepted_csv} and {rejected_csv}")
        return {"candidates": len(candidates), "accepted": accepted, "rejected": rejected, "failed": failed}

    def _process_study(self, group: list[ManifestCandidate], download_root: Path) -> list[StudyOutcome]:
        """Extract, filter and derive every listed series of one study, then drop its zip."""
        candidates = self._unique(group)
        archive = download_root / candidates[0].repo_zip_path
        staging = self._download_dir / "staging" / candidates[0].study_uid
        try:
            staged = self._extractor.materialize(archive, candidates, staging, self._download_dir)
            placed = self._extractor.move_into_place(staged)
            return [self._derive(candidate, placed) for candidate in candidates]
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as error:
            print(f"  FAILED {candidates[0].study_uid}: {error}")
            return [StudyOutcome(verdict=None) for _ in candidates]
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)

    def _derive(self, candidate: ManifestCandidate, placed: set[Path]) -> StudyOutcome:
        """Filter one extracted series, derive its twin when it survives, and record the verdict."""
        variant = candidate.variant
        image = self._download_dir / variant.whole_brain_path
        mask = self._download_dir / variant.mask_path
        if image not in placed or mask not in placed:
            # Partial extraction: raise so the study is reported failed and retried on the next run.
            raise FileNotFoundError(f"{candidate.study_uid}/{candidate.series_id}: image or mask missing after extraction")
        try:
            verdict = SeriesVerdict.from_filter_result(self._filter.check(image, mask))
            if verdict.is_accepted:
                self._stripper.create(image, mask, self._download_dir / variant.skull_stripped_path)
        except ValueError as error:
            # An off-grid pair can be neither judged nor multiplied. Letting this escape would
            # kill an hours-long run on one bad series; leaving it unrecorded would wedge the
            # refill loop on it forever. Rejecting it is the closed-loop answer.
            verdict = SeriesVerdict.unusable(str(error))
        if not verdict.is_accepted:
            image.unlink()
            mask.unlink()
        self._log.append(verdict.to_row(candidate))
        return StudyOutcome(verdict=verdict)

    def _clear_staging(self) -> None:
        """Drop the now-empty per-study staging scaffolding so the data tree holds only series files."""
        shutil.rmtree(self._download_dir / "staging", ignore_errors=True)

    def _chunks(self, candidates: list[ManifestCandidate]) -> Iterator[tuple[list[ManifestCandidate], Path]]:
        """Yield ``(chunk, download_root)`` one chunk at a time, fetching just before the chunk is processed.

        Lazy on purpose: the run is hours long and interruptible, so verdicts must land
        incrementally rather than after the whole ~277 GB has been fetched.  A chunk holds
        at most ``workers`` distinct studies and is never split across fetches.
        """
        for chunk in self._fetch_batches(candidates):
            root = self._source.fetch(sorted({candidate.repo_zip_path for candidate in chunk}))
            yield chunk, root

    def _fetch_batches(self, candidates: list[ManifestCandidate]) -> list[list[ManifestCandidate]]:
        """Split the todo list into download batches of at most ``workers`` distinct studies.

        The batch is the unit of fetching, and every series of one study is extracted from the
        one zip the batch pulls, so a batch may not straddle a study; manifest order is kept so
        the verdict log still reads in roster order.  ``_group_by_study`` then walks each batch
        study by study -- the two split different things for different reasons.
        """
        groups: list[list[ManifestCandidate]] = []
        current: list[ManifestCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.study_uid not in seen and len(seen) >= self._workers:
                groups.append(current)
                current, seen = [], set()
            seen.add(candidate.study_uid)
            current.append(candidate)
        if current:
            groups.append(current)
        return groups

    def _group_by_study(self, chunk: list[ManifestCandidate]) -> list[list[ManifestCandidate]]:
        """Split one fetched batch into per-study groups, preserving order.

        Repeats of a series survive this grouping on purpose: collapsing them belongs next to
        the extraction they would otherwise repeat, in ``_process_study``.
        """
        groups: dict[str, list[ManifestCandidate]] = {}
        for candidate in chunk:
            groups.setdefault(candidate.study_uid, []).append(candidate)
        return list(groups.values())

    @staticmethod
    def _unique(candidates: list[ManifestCandidate]) -> list[ManifestCandidate]:
        """Collapse repeats of the same series, preserving order.

        A study zip names each of its series once, but a manifest can list one twice; the
        pipeline must extract, filter and record it once either way.
        """
        return list({candidate.key: candidate for candidate in candidates}.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="replay manifest CSV to download")
    parser.add_argument("--download-dir", type=Path, required=True, help="data root the mri/ tree lands in")
    parser.add_argument("--verdicts-csv", type=Path, required=True, help="append-only per-series verdict record (resume point)")
    parser.add_argument("--accepted-csv", type=Path, required=True, help="where to write the accepted-series manifest")
    parser.add_argument("--rejected-csv", type=Path, required=True, help="where to write the rejected-series manifest")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel study zips (also the group size)")
    parser.add_argument("--repo-id", default=REPO_ID, help="HuggingFace dataset repo id")
    args = parser.parse_args()

    downloader = ReplayDownloader(
        manifest_path=args.manifest,
        download_dir=args.download_dir,
        source=HuggingFaceZipSource(repo_id=args.repo_id, max_workers=args.workers),
        verdict_log=VerdictLog(args.verdicts_csv),
        workers=args.workers,
    )
    downloader.run(accepted_csv=args.accepted_csv, rejected_csv=args.rejected_csv)


if __name__ == "__main__":
    main()
