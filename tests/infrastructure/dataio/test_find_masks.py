"""Tests for ctmr.infrastructure.dataio.find_masks (candidate mask database lookup, migrated from scripts/find_masks)."""

import json

import pytest

from ctmr.infrastructure.dataio.find_masks import convert_body_region, find_masks


def test_convert_body_region_single_and_plural():
    assert convert_body_region("head") == [0]
    assert convert_body_region(["head", "abdomen"]) == [0, 2]
    assert convert_body_region("chest/thorax") == [1]
    assert convert_body_region(["pelvis/lower"]) == [3]


def test_convert_body_region_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid region"):
        convert_body_region("leg")


def _write_database(tmp_path, entries):
    db_path = tmp_path / "database.json"
    db_path.write_text(json.dumps(entries))
    return str(db_path)


def _entry(label_list, top=None, bottom=None, spacing=(1.0, 1.0, 1.0), dim=(512, 512, 512), name="cand", with_label=True):
    entry = {
        "pseudo_label_filename": f"{name}_pseudo.nii.gz",
        "spacing": list(spacing),
        "dim": list(dim),
        "label_list": label_list,
    }
    if top is not None:
        entry["top_region_index"] = top
        entry["bottom_region_index"] = bottom
    if with_label:
        entry["label_filename"] = f"{name}_label.nii.gz"
    return entry


@pytest.fixture
def masks_env(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    entries = [
        _entry([1, 5, 14, 23, 200], top=[1, 0, 0, 0], bottom=[0, 0, 0, 1], spacing=[1, 1, 1], dim=[512, 512, 512], name="tumorful"),
        _entry([1, 5, 14, 200], top=[1, 0, 0, 0], bottom=[0, 0, 0, 1], spacing=[2, 2, 2], dim=[256, 256, 256], name="clean", with_label=False),
        _entry([1, 5, 14, 200], top=[1, 0, 0, 0], bottom=[1, 0, 0, 0], spacing=[1, 1, 1], dim=[512, 512, 512], name="head-only"),
    ]
    return {"dir": masks_dir, "db": _write_database(tmp_path, entries)}


def test_find_masks_skips_tumorful_mask_when_tumor_not_requested(masks_env):
    found = find_masks(
        body_region="abdomen",
        anatomy_list=[1, 5, 14],
        database_filepath=masks_env["db"],
        mask_foldername=str(masks_env["dir"]),
    )
    assert len(found) == 1
    cand = found[0]
    assert cand["pseudo_label"].endswith("clean_pseudo.nii.gz")
    assert cand["spacing"] == [2, 2, 2]
    assert cand["dim"] == [256, 256, 256]
    assert "label" not in cand


def test_find_masks_requests_tumor_and_gets_tumorful_candidate(masks_env):
    found = find_masks(
        body_region="head",
        anatomy_list=[1, 5, 14, 23],
        database_filepath=masks_env["db"],
        mask_foldername=str(masks_env["dir"]),
    )
    assert len(found) == 1
    assert found[0]["label"].endswith("tumorful_label.nii.gz")


def test_find_masks_out_of_region_excludes_candidate(masks_env):
    # tumor-free query: clean (abdomen..pelvis) and head-only both qualify anatomically,
    # so the region gate is what separates them.
    common = dict(
        anatomy_list=[1, 5, 14],
        database_filepath=masks_env["db"],
        mask_foldername=str(masks_env["dir"]),
    )
    head = find_masks(body_region="head", **common)
    assert len(head) == 2  # clean + head-only (tumorful is excluded by the tumor-free constraint)
    abdomen = find_masks(body_region="abdomen", **common)
    assert len(abdomen) == 1  # head-only is out of the abdomen region
    assert not any("head-only" in cand["pseudo_label"] for cand in abdomen)


def test_find_masks_spacing_size_check_filters(masks_env):
    found = find_masks(
        body_region="abdomen",
        anatomy_list=[1, 5, 14],
        spacing=[2, 2, 2],
        output_size=[256, 256, 256],
        check_spacing_and_output_size=True,
        database_filepath=masks_env["db"],
        mask_foldername=str(masks_env["dir"]),
    )
    assert len(found) == 1
    assert found[0]["dim"] == [256, 256, 256]


def test_find_masks_no_candidates_raises_when_not_checked(masks_env):
    with pytest.raises(ValueError, match="Cannot find body region"):
        find_masks(
            body_region="head",
            anatomy_list=[127],  # no entry contains 127
            database_filepath=masks_env["db"],
            mask_foldername=str(masks_env["dir"]),
        )
    # ...but the same query passes with check flag on (empty list is legal there)
    assert (
        find_masks(
            body_region="head",
            anatomy_list=[127],
            database_filepath=masks_env["db"],
            mask_foldername=str(masks_env["dir"]),
            check_spacing_and_output_size=True,
        )
        == []
    )


def test_find_masks_missing_database_raises(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    with pytest.raises(ValueError, match="Please download"):
        find_masks(
            body_region="head",
            anatomy_list=[1],
            database_filepath=str(tmp_path / "nope.json"),
            mask_foldername=str(masks_dir),
        )


def test_find_masks_missing_masks_dir_and_zip_raises(tmp_path):
    with pytest.raises(ValueError, match="Please download"):
        find_masks(
            body_region="head",
            anatomy_list=[1],
            database_filepath=_write_database(tmp_path, []),
            mask_foldername=str(tmp_path / "absent-masks"),
        )
