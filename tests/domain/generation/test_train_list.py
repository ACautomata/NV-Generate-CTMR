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

"""Train-list fold split gate (issue #276 / ADR-0019 §3).

``add_data_dir2path`` floated up verbatim from
``infrastructure.dataio.list_assembly`` (retired with it; git history) -- the
pure stdlib assembly rule the bypass families' training lists obey: relative
``image``/``label`` paths join onto the per-list data root, the ``fold`` key
splits the train side (fold != N) from the held-out side (fold == N), and the
caller's list is left unmutated. The loader-level behavior (only the train
side reaches the dataset) is pinned in
tests/application/generation/test_train_loader.py; this gate pins the domain
rule itself, including the fold=None spelling the loader never exercises.
"""

import copy

from ctmr.domain.generation.train_list import add_data_dir2path

LIST = [
    {"image": "case0_img.nii.gz", "label": "case0_label.nii.gz", "fold": 1},
    {"image": "case1_img.nii.gz", "label": "case1_label.nii.gz", "fold": 0},
    {"image": "case2_img.nii.gz", "fold": 1},  # a label-less entry joins fine
]
ROOT = "/data/root"


def test_fold_key_splits_the_train_side_from_the_held_out_side():
    train, val = add_data_dir2path(LIST, ROOT, fold=0)

    assert [entry["image"] for entry in train] == ["/data/root/case0_img.nii.gz", "/data/root/case2_img.nii.gz"]
    assert [entry["image"] for entry in val] == ["/data/root/case1_img.nii.gz"]
    # the join reaches the label keys too; a missing label stays missing
    assert train[0]["label"] == "/data/root/case0_label.nii.gz"
    assert "label" not in train[1]


def test_fold_none_returns_every_entry_on_the_train_side():
    train, val = add_data_dir2path(LIST, ROOT, fold=None)

    assert len(train) == 3 and val == []
    assert train[0]["image"] == "/data/root/case0_img.nii.gz"


def test_the_callers_list_is_left_unmutated():
    snapshot = copy.deepcopy(LIST)
    add_data_dir2path(LIST, ROOT, fold=0)

    assert LIST == snapshot
