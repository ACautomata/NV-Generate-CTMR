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

"""Behaviour gates for ctmr.infrastructure.dataio.find_masks (#132).

Verbatim lift of scripts/find_masks.py: body-region name normalization plus
the candidate-mask database filter (anatomy subset inclusion, tumor-free
requirement, one-hot region window, strict spacing/dimension mode). The zip
extraction side stays untouched behind its existence guard -- tests seed the
mask folder so only selection logic is exercised. Needs monai for import;
selection checks run on CPU from tmp fixture files.
"""

import json

import pytest

pytest.importorskip("monai")

from ctmr.infrastructure.dataio.find_masks import convert_body_region, find_masks  # noqa: E402


def _item(label_list, name="a", top=None, bottom=None, dim=(64, 64, 64), spacing=(1.0, 1.0, 1.0), label=False):
    entry = {
        "label_list": list(label_list),
        "pseudo_label_filename": f"{name}_pseudo.nii.gz",
        "dim": list(dim),
        "spacing": list(spacing),
    }
    if top is not None:
        onehot = [0, 0, 0, 0]
        onehot[top] = 1
        entry["top_region_index"] = onehot
        onehot_b = [0, 0, 0, 0]
        onehot_b[bottom] = 1
        entry["bottom_region_index"] = onehot_b
    if label:
        entry["label_filename"] = f"{name}_label.nii.gz"
    return entry


def _write_db(tmp_path, entries):
    db = tmp_path / "database.json"
    db.write_text(json.dumps(entries))
    masks = tmp_path / "masks"
    masks.mkdir(exist_ok=True)
    return str(db), str(masks) + "/"


def test_convert_body_region_normalizes_and_expands():
    assert convert_body_region("head") == [0]
    assert convert_body_region(["HEAD"]) == [0]
    assert convert_body_region("chest/thorax") == [1]
    assert convert_body_region(["abdomen", "pelvis/lower"]) == [2, 3]


def test_convert_body_region_rejects_unknown_name():
    with pytest.raises(ValueError, match="Invalid region"):
        convert_body_region("brain")


def test_find_masks_keeps_tumor_free_candidates_matching_anatomy(tmp_path):
    db, masks = _write_db(tmp_path, [_item([1], name="keep", top=0, bottom=2)])
    got = find_masks(body_region="abdomen", anatomy_list=[1], database_filepath=db, mask_foldername=masks)
    assert len(got) == 1
    cand = got[0]
    assert cand["pseudo_label"].endswith("keep_pseudo.nii.gz")
    assert cand["top_region_index"][0] == 1 and cand["bottom_region_index"][2] == 1


def test_find_masks_carries_optional_real_label_when_present(tmp_path):
    db, masks = _write_db(tmp_path, [_item([1], name="withlabel", top=0, bottom=2, label=True)])
    got = find_masks("abdomen", [1], database_filepath=db, mask_foldername=masks)
    assert got[0]["label"].endswith("withlabel_label.nii.gz")


def test_find_masks_skips_items_holding_unrequested_tumors(tmp_path):
    # a filter that empties the candidate set cannot quietly return [] on the
    # loose path -- it raises instead
    db, masks = _write_db(tmp_path, [_item([1, 23], name="tumorous", top=0, bottom=2)])
    with pytest.raises(ValueError, match="Cannot find body region"):
        find_masks("abdomen", [1], database_filepath=db, mask_foldername=masks)


def test_find_masks_enforces_one_hot_region_window(tmp_path):
    db, masks = _write_db(tmp_path, [_item([1], name="pelvis_only", top=1, bottom=3)])
    with pytest.raises(ValueError, match="Cannot find body region"):
        find_masks("head", [1], database_filepath=db, mask_foldername=masks)
    assert len(find_masks("chest", [1], database_filepath=db, mask_foldername=masks)) == 1


def test_strict_mode_filters_on_dimensions_and_spacing(tmp_path):
    entry = _item([1], name="strict", top=0, bottom=2, dim=(96, 96, 96), spacing=(1.5, 1.5, 1.5))
    db, masks = _write_db(tmp_path, [entry])
    ok_kwargs = dict(
        body_region="abdomen",
        anatomy_list=[1],
        check_spacing_and_output_size=True,
        output_size=[96, 96, 96],
        spacing=[1.5, 1.5, 1.5],
    )
    assert len(find_masks(database_filepath=db, mask_foldername=masks, **ok_kwargs)) == 1
    mismatch = find_masks(
        database_filepath=db,
        mask_foldername=masks,
        body_region="abdomen",
        anatomy_list=[1],
        check_spacing_and_output_size=True,
        output_size=[64, 64, 64],
        spacing=[1.5, 1.5, 1.5],
    )
    assert mismatch == []  # strict misses stay quiet instead of raising


def test_loose_mode_raises_when_nothing_matches(tmp_path):
    db, masks = _write_db(tmp_path, [])
    with pytest.raises(ValueError, match="Cannot find body region"):
        find_masks("abdomen", [7], database_filepath=db, mask_foldername=masks)


def test_missing_mask_archive_is_a_guided_error(tmp_path):
    db = tmp_path / "database.json"
    db.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="Please download.*zip"):
        find_masks(
            "abdomen",
            [1],
            database_filepath=str(db),
            mask_foldername=str(tmp_path / "absent") + "/",
        )
