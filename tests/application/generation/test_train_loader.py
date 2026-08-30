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

"""Train-side loader gate of the shared bypass-family loader (issue #226).

``BypassTrainLoader`` is the one parameterized definition of the MAISI
JSON-list loader the mask and cross-modal families used to hand-copy. The
contract under test: only the train side is constructed (the fold split's val
side is never built -- the families select their candidate by the dev-eval
sidecar, never by a validation loss, spec #51 decision 7), the family
transform set rides the constructor parameters (the mask companions vs the
``src_image`` load), and the ``src_image`` condition joins its per-list data
root. Real tiny NIfTIs keep the transform chain honest; the dataset (not the
worker pool) is iterated so the gate stays single-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from ctmr.application.generation.train_loader import BypassTrainLoader

pytestmark = pytest.mark.torch

MODALITY_MAPPING = {"mri_t1_skull_stripped": 29, "mri_t1c_skull_stripped": 34}

MASK_SPEC = dict(load_keys=("image", "label"), companion_keys=("top_region_index", "bottom_region_index"))
CROSS_MODAL_SPEC = dict(load_keys=("image", "label", "src_image"), join_keys=("src_image",))


def _write_volume(path, array):
    nib.save(nib.Nifti1Image(array, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))


def _fixture(tmp_path):
    """Three cases on the image grid: two on the train side (fold=1), one held out (fold=0)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    rng = np.random.default_rng(7)
    entries = []
    for index, fold in enumerate((1, 1, 0)):
        stem = f"case{index}"
        _write_volume(data_root / f"{stem}_img.nii.gz", rng.standard_normal((4, 2, 2, 2)).astype(np.float32))
        _write_volume(data_root / f"{stem}_label.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8))
        _write_volume(data_root / f"{stem}_src.nii.gz", rng.standard_normal((4, 2, 2, 2)).astype(np.float32))
        entries.append(
            {
                "image": f"{stem}_img.nii.gz",
                "label": f"{stem}_label.nii.gz",
                "src_image": f"{stem}_src.nii.gz",
                "spacing": [1.0, 1.2, 0.8],
                "modality": "mri_t1_skull_stripped" if index == 0 else "mri_t1c_skull_stripped",
                "fold": fold,
                "sub": "GLI",
                "case": f"SYNTH-{index:04d}",
                "top_region_index": [0, 1, 0, 0],
                "bottom_region_index": [0, 0, 0, 1],
            }
        )
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps({"training": entries}))
    return str(data_root), str(list_path)


def _build(spec, tmp_path, fold=0):
    data_root, list_path = _fixture(tmp_path)
    return BypassTrainLoader(**spec).build(
        json_data_list=list_path,
        data_base_dir=data_root,
        batch_size=1,
        cache_rate=0.0,
        fold=fold,
        rank=0,
        world_size=1,
        modality_mapping=dict(MODALITY_MAPPING),
    )


def test_mask_contract_returns_only_the_train_side_of_the_fold_split(tmp_path):
    loader = _build(MASK_SPEC, tmp_path, fold=0)

    assert isinstance(loader, torch.utils.data.DataLoader)  # the val loader is gone from the contract
    assert len(loader.dataset) == 2  # only the fold!=0 entries reach the train side


def test_mask_transform_set_carries_the_region_companions(tmp_path):
    batch = _build(MASK_SPEC, tmp_path).dataset[0]

    # the loaded family keys (the list's metadata keys ride along, as in production)
    assert {"image", "label", "top_region_index", "bottom_region_index", "spacing", "modality"} <= set(batch)
    assert batch["label"].dtype == torch.long
    assert batch["label"].shape[0] == 1  # ensure_channel_first moves the mask channel to the front
    assert list(batch["top_region_index"]) == [0.0, 100.0, 0.0, 0.0]  # float tensor, scaled x1e2
    assert float(batch["spacing"][0]) == 100.0  # x1e2
    assert int(batch["modality"]) == 29  # mapped through modality_mapping


def test_cross_modal_transform_set_loads_and_joins_the_src_latent(tmp_path):
    loader = _build(CROSS_MODAL_SPEC, tmp_path)

    data_root = str(Path(loader.dataset.data[0]["image"]).parent)
    assert loader.dataset.data[0]["src_image"].startswith(data_root)  # joined to the per-list data root
    batch = loader.dataset[0]
    assert {"image", "label", "src_image", "spacing", "modality"} <= set(batch)  # no region companions here
    assert batch["src_image"].dtype == torch.float  # the latent stays float, never binarized
    assert int(batch["modality"]) == 29
