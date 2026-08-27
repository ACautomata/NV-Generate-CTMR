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


"""Behaviour gates for ctmr.infrastructure.dataio.sample_mask (#132).

Covers the portion landed in this ticket from scripts/sample_mask.py:
``filter_mask_with_organs`` compact-id relabeling and the CT/MR pipeline input
validators (geometry windows, FOV floor, controllable-anatomy vocabulary and
the label-dictionary cross-check). The DDPM sampler core stays on the old leaf
until its MAISI engine dependencies are collected -- asserted here so the
partial move is explicit. Torch-level, CPU-only.
"""

import json

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip must precede the torch-dependent import)

from ctmr.infrastructure.dataio import sample_mask  # noqa: E402


def test_filter_mask_with_organs_compacts_to_requested_ids():
    mask = torch.zeros(1, 4, 4)
    mask[0, 0] = 1  # liver
    mask[0, 1] = 5  # kidney
    mask[0, 2] = 62  # colon -> dropped
    out = sample_mask.filter_mask_with_organs(mask, [1, 5])
    assert bool((out[0, 0] == 1).all()) and bool((out[0, 1] == 2).all())
    assert int(out[0, 2].abs().sum()) == 0
    assert set(out.unique().tolist()) <= {0, 1, 2}


def _label_dict(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"liver": 1, "gallbladder": 10}))
    return str(path)


# label_dict_json sits early enough in every signature that only paths actually
# opening it need a real file; "" suffices for the fast-fail branches.
VALID_CT = dict(
    body_region=["abdomen"],
    anatomy_list=["liver"],
    label_dict_json="",
    output_size=[512, 512, 256],
    spacing=[1.0, 1.0, 2.0],
)


def test_check_input_ct_accepts_valid_geometry(tmp_path):
    ok = {**VALID_CT, "label_dict_json": _label_dict(tmp_path)}
    assert sample_mask.check_input_ct(**ok) is None


def test_check_input_ct_rejects_asymmetric_xy_sizes():
    with pytest.raises(ValueError, match="first two components of output_size"):
        sample_mask.check_input_ct(output_size=[256, 384, 256], spacing=[1.0, 1.0, 1.0], body_region=[], anatomy_list=[], label_dict_json="")


def test_check_input_ct_rejects_out_of_vocabulary_dimension():
    with pytest.raises(ValueError, match="output_size"):
        sample_mask.check_input_ct(output_size=[300, 300, 256], spacing=[1.0, 1.0, 1.0], body_region=[], anatomy_list=[], label_dict_json="")


def test_check_input_ct_enforces_fov_floor(tmp_path):
    with pytest.raises(ValueError, match="field of view"):
        sample_mask.check_input_ct(
            body_region=[],
            anatomy_list=[],
            output_size=[256, 256, 256],
            spacing=[0.9, 0.9, 1.0],
            label_dict_json=_label_dict(tmp_path),
        )


def test_check_input_ct_rejects_unknown_controllable_anatomy():
    with pytest.raises(ValueError, match="controllable_anatomy have to be chosen"):
        sample_mask.check_input_ct(**VALID_CT, controllable_anatomy_size=[("spleen", 0.5)])


def test_check_input_ct_rejects_repeated_or_multiple_tumors():
    with pytest.raises(ValueError, match="do not repeat controllable_anatomy"):
        sample_mask.check_input_ct(**VALID_CT, controllable_anatomy_size=[("lung tumor", 0.5), ("lung tumor", 0.6)])
    with pytest.raises(ValueError, match="Only one controllable tumor"):
        sample_mask.check_input_ct(**VALID_CT, controllable_anatomy_size=[("lung tumor", 0.5), ("bone lesion", 0.5)])


def test_check_input_ct_validates_anatomy_against_label_dict(tmp_path):
    with pytest.raises(ValueError, match="anatomy_list have to be chosen"):
        sample_mask.check_input_ct(
            body_region=["abdomen"],
            anatomy_list=["spleen"],
            label_dict_json=_label_dict(tmp_path),
            output_size=[512, 512, 256],
            spacing=[1.0, 1.0, 2.0],
            controllable_anatomy_size=[],
        )


VALID_MR = dict(body_region=["head"], anatomy_list=["liver"], label_dict_json="", output_size=[128, 256, 256], spacing=[0.8, 1.2, 1.0])


def test_check_input_mr_accepts_swapped_pair_layout(tmp_path):
    ok = {**VALID_MR, "label_dict_json": _label_dict(tmp_path)}
    assert sample_mask.check_input_mr(**ok) is None


def test_check_input_mr_rejects_bad_combo_at_256():
    with pytest.raises(ValueError, match="only be"):
        sample_mask.check_input_mr(body_region=[], anatomy_list=[], output_size=[384, 384, 256], spacing=[1.0, 1.0, 1.0], label_dict_json="")


def test_check_input_mr_requires_equal_third_axis_choice():
    with pytest.raises(ValueError, match="output_size"):
        sample_mask.check_input_mr(body_region=[], anatomy_list=[], output_size=[256, 256, 384], spacing=[1.0, 1.0, 1.0], label_dict_json="")


def test_sampler_core_stays_on_the_old_leaf_until_engine_arrives():
    """Documents the partial move declared in the module docstring."""
    assert not hasattr(sample_mask, "ldm_conditional_sample_one_mask")
