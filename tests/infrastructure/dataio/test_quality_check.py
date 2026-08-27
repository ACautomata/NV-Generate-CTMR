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

"""Behaviour gates for ctmr.infrastructure.dataio.quality_check (#132).

Verbatim lift of scripts/quality_check.py: the label-masked readout
(``get_masked_data``) with its two performance branches, and the
median-vs-envelope outlier verdict (``is_outlier``) including the bone
ceiling override and NaN/empty-region handling. Pure numpy -- runs on any
machine (no importorskip needed).
"""

import numpy as np
import pytest

from ctmr.infrastructure.dataio.quality_check import get_masked_data, is_outlier


def _volume():
    image = np.zeros((6, 6, 6), dtype=np.float32)
    label = np.zeros((6, 6, 6), dtype=np.int64)
    label[0:2, :, :] = 1  # "liver"
    label[4:5, :, :] = 5  # "kidney"
    label[:, :, 2:3] = 10  # extra organ
    return image + 50.0, label


def test_get_masked_data_selects_only_requested_labels():
    image, label = _volume()
    got = get_masked_data(label, image, [1])
    assert np.allclose(got, 50.0)
    assert got.shape == (image[label == 1].shape)


def test_get_masked_data_deduplicates_and_spans_multiple_labels():
    image, label = _volume()
    single = get_masked_data(label, image, [1])
    dup = get_masked_data(label, image, [1, 1, 1])
    assert single.shape == dup.shape


def test_get_masked_data_matches_across_both_branch_thresholds():
    # >=3 labels goes through np.isin; <3 labels through the logical-OR loop.
    image, label = _volume()
    few = get_masked_data(label, image, [1, 5])
    many = get_masked_data(label, image, [1, 5, 10, 99])  # 99 absent
    assert set(np.unique(few).tolist()) == {50.0}
    assert set(np.unique(many).tolist()) == {50.0}


def test_get_masked_data_empty_label_list_returns_empty_array():
    _, label = _volume()
    assert get_masked_data(label, np.zeros_like(label), []).size == 0


def test_get_masked_data_rejects_shape_mismatch():
    image, label = _volume()
    with pytest.raises(ValueError, match="Shape mismatch"):
        get_masked_data(label, image[:3], [1])


def test_is_outlier_flags_off_envelope_median():
    image, label = _volume()
    stats = {"liver": {"sigma_6_low": -100.0, "sigma_6_high": 40.0, "percentile_0_5": -90.0, "percentile_99_5": 45.0}}
    result = is_outlier(stats, image, label, {"liver": [1]})
    assert result["liver"]["is_outlier"]
    assert result["liver"]["median_value"] == pytest.approx(50.0)


def test_is_outlier_passes_inlier_region():
    image, label = _volume()
    stats = {"liver": {"sigma_6_low": 0.0, "sigma_6_high": 100.0, "percentile_0_5": 0.0, "percentile_99_5": 100.0}}
    assert not is_outlier(stats, image, label, {"liver": [1]})["liver"]["is_outlier"]


def test_is_outlier_bone_ceiling_overrides_statistics_high_side():
    image, label = _volume()
    stats = {"bone": {"sigma_6_low": -10.0, "sigma_6_high": 10.0, "percentile_0_5": -5.0, "percentile_99_5": 15.0}}
    result = is_outlier(stats, image, label, {"bone": [1]})
    assert result["bone"]["high_thresh"] == 1000.0
    assert not result["bone"]["is_outlier"]  # median 50 stays below the bone ceiling


def test_is_outlier_missing_labels_report_no_data_not_failure():
    image, label = _volume()
    stats = {"ghost": {"sigma_6_low": 0.0, "sigma_6_high": 1.0, "percentile_0_5": 0.0, "percentile_99_5": 1.0}}
    result = is_outlier(stats, image, label, {})  # no entry for "ghost"
    assert result["ghost"] == {
        "is_outlier": False,
        "median_value": None,
        "low_thresh": 0.0,
        "high_thresh": 1.0,
    }


def test_is_outlier_all_nan_region_reports_none_median():
    image, label = _volume()
    nan_image = image.copy()
    nan_image[label == 1] = np.nan
    stats = {"liver": {"sigma_6_low": 0.0, "sigma_6_high": 100.0, "percentile_0_5": 0.0, "percentile_99_5": 100.0}}
    result = is_outlier(stats, nan_image, label, {"liver": [1]})
    assert result["liver"]["median_value"] is None
    assert result["liver"]["is_outlier"] is False
