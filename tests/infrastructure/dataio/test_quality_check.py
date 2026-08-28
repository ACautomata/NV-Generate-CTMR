"""Numpy-only tests for ctmr.infrastructure.dataio.quality_check (migrated from the retired scripts layer (git history; ``quality_check``))."""

import numpy as np
import pytest

from ctmr.infrastructure.dataio.quality_check import get_masked_data, is_outlier


def test_get_masked_data_returns_only_matching_voxels():
    image = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    labels = np.zeros((3, 3, 3), dtype=np.int64)
    labels[1, 1, 1] = 5
    labels[0, 0, 0] = 5
    out = get_masked_data(labels, image, [5])
    # boolean indexing walks C-order, so (0,0,0) comes before (1,1,1)
    assert out.tolist() == [0.0, 13.0]


def test_get_masked_data_handles_duplicate_and_empty_labels():
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    labels = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=np.int64)
    assert get_masked_data(labels, image, [1, 1]).size == 4
    assert get_masked_data(labels, image, []).size == 0


def test_get_masked_data_many_labels_uses_isin_path():
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    labels = np.arange(64).reshape(8, 8) % 8
    out = get_masked_data(labels, image, [0, 1, 2, 3])
    assert out.size > 0
    assert out.size == np.isin(labels, [0, 1, 2, 3]).sum()


def test_get_masked_data_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        get_masked_data(np.zeros((2, 2)), np.zeros((3, 3)), [1])


def test_is_outlier_above_high_threshold():
    statistics = {"liver": {"sigma_6_low": 0.0, "sigma_6_high": 10.0, "percentile_0_5": -1.0, "percentile_99_5": 9.0}}
    image = np.full((4, 4, 4), 50.0)
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[1:3, 1:3, 1:3] = 1
    result = is_outlier(statistics, image, labels, {"liver": [1]})
    assert bool(result["liver"]["is_outlier"]) is True
    assert result["liver"]["median_value"] == pytest.approx(50.0)


def test_is_outlier_inside_range():
    statistics = {"liver": {"sigma_6_low": 0.0, "sigma_6_high": 10.0, "percentile_0_5": -1.0, "percentile_99_5": 9.0}}
    image = np.full((4, 4, 4), 5.0)
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[1:3, 1:3, 1:3] = 1
    result = is_outlier(statistics, image, labels, {"liver": [1]})
    assert bool(result["liver"]["is_outlier"]) is False


def test_is_outlier_absent_label_reports_no_outlier_none_median():
    statistics = {"liver": {"sigma_6_low": 0.0, "sigma_6_high": 10.0, "percentile_0_5": -1.0, "percentile_99_5": 9.0}}
    image = np.full((4, 4, 4), 5.0)
    labels = np.zeros((4, 4, 4), dtype=np.int64)  # no liver voxels at all
    result = is_outlier(statistics, image, labels, {"liver": [1]})
    assert bool(result["liver"]["is_outlier"]) is False
    assert result["liver"]["median_value"] is None


def test_is_outlier_bone_overrides_high_threshold():
    # bone: any median below 1000 HU is fine regardless of statistical envelope
    statistics = {"bone": {"sigma_6_low": 10.0, "sigma_6_high": 20.0, "percentile_0_5": 15.0, "percentile_99_5": 18.0}}
    image = np.full((4, 4, 4), 500.0)
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[1:3, 1:3, 1:3] = 1
    result = is_outlier(statistics, image, labels, {"bone": [1]})
    assert bool(result["bone"]["is_outlier"]) is False
    assert result["bone"]["high_thresh"] == 1000.0
