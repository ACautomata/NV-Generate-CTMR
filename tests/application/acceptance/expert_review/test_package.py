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

"""The blinding-package renderer, observed as pytest (#141).

The resident ``L3SelfTest`` package-build assertions of
``brats_l3_blind_eval.py`` (retired scripts layer, git history) became this file when the chain moved into
the expert_review package (ADR-0015 §6). Every assertion is the blinding
protocol on a synthetic catalog: reproducible seeded draw, fully blinded
entry ids, no source/case leakage into the reviewer package, exact per-cell
quotas, refusal to build a sparse cell, and the CLI writing both artifacts.

Light stack (stdlib only): the renderer imports no numpy/torch.
"""

import json

import pytest

from ctmr.application.acceptance.expert_review.catalog import (
    BLIND_MAP_SCHEMA,
    CATALOG_SCHEMA,
    PACKAGE_SCHEMA,
    Catalog,
    L3Error,
)
from ctmr.application.acceptance.expert_review.package import BlindPackageBuilder
from ctmr.application.acceptance.expert_review.package import main as build_package_main


def test_package_build_is_reproducible_and_blinded(run_record, catalog):
    package_doc, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run_record, catalog)
    package_again, blind_map_again = BlindPackageBuilder(seed=20260821, per_cell=5).build(run_record, catalog)

    assert package_doc["schema"] == PACKAGE_SCHEMA
    assert blind_map_doc["schema"] == BLIND_MAP_SCHEMA
    assert len(package_doc["entries"]) == 200
    assert package_again["entries"] == package_doc["entries"]  # deterministic for a fixed seed and catalog
    assert blind_map_again["entries"] == blind_map_doc["entries"]
    entry_ids = [entry["entry_id"] for entry in blind_map_doc["entries"]]
    assert entry_ids == sorted(entry_ids)  # fully blinded L3-XXXX ids in presentation order
    assert len(set(entry_ids)) == 200
    assert all("source" not in entry and "case" not in entry for entry in package_doc["entries"])  # no leak into the reviewer face


def test_package_cells_hold_exact_quotas(blind_map_doc):
    per_cell_counts = {}
    for entry in blind_map_doc["entries"]:
        key = (entry["challenge"], entry["target_modality"])
        per_cell_counts[key] = per_cell_counts.get(key, 0) + 1

    assert len(per_cell_counts) == 20  # 5 challenges x 4 modalities
    assert all(count == 10 for count in per_cell_counts.values())  # 5 real + 5 synth per cell


def test_sparse_cell_refuses_to_build(run_record, catalog_payload):
    sparse = {"schema": CATALOG_SCHEMA, "records": catalog_payload["records"][:4]}

    with pytest.raises(L3Error, match="available images"):
        BlindPackageBuilder(seed=20260821, per_cell=5).build(run_record, Catalog(sparse))


def test_build_package_command_writes_both_artifacts(tmp_path, run_record, catalog_payload):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload))
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run_record))
    out_dir = tmp_path / "pkg"

    status = build_package_main(
        ["--run", str(run_path), "--catalog", str(catalog_path), "--output", str(out_dir), "--seed", "20260821", "--per-cell", "5"]
    )

    assert status == 0
    package_doc = json.loads((out_dir / "package.json").read_text())
    blind_map_doc = json.loads((out_dir / "blind_map.json").read_text())
    assert package_doc["schema"] == PACKAGE_SCHEMA
    assert blind_map_doc["schema"] == BLIND_MAP_SCHEMA
    assert len(package_doc["entries"]) == 200
