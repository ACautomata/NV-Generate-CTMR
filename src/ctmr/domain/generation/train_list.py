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

"""Train-list fold split: the pure assembly rule the bypass families' training lists obey (ADR-0019 §3, #276).

Floated up verbatim from ``infrastructure.dataio.list_assembly`` (retired with
the ratchet-zeroing sweep; git history) -- pure stdlib path assembly, no IO:
``add_data_dir2path`` joins the relative ``image``/``label`` paths of a
MAISI-style JSON data list onto its per-list data root and splits the train /
val sides by the ``fold`` key. The bypass train loader imports this domain
module directly.
"""

from __future__ import annotations

import copy
import os


def add_data_dir2path(list_files: list, data_dir: str, fold: int = None) -> tuple[list, list]:
    """
    Read a list of data dictionary.

    Args:
        list_files (list): input data to load and transform to generate dataset for model.
        data_dir (str): directory of files.
        fold (int, optional): fold index for cross validation. Defaults to None.

    Returns:
        tuple[list, list]: A tuple of two arrays (training, validation).
    """
    new_list_files = copy.deepcopy(list_files)
    if fold is not None:
        new_list_files_train = []
        new_list_files_val = []
    for d in new_list_files:
        d["image"] = os.path.join(data_dir, d["image"])

        if "label" in d:
            d["label"] = os.path.join(data_dir, d["label"])

        if fold is not None:
            if d["fold"] == fold:
                new_list_files_val.append(copy.deepcopy(d))
            else:
                new_list_files_train.append(copy.deepcopy(d))

    if fold is not None:
        return new_list_files_train, new_list_files_val
    else:
        return new_list_files, []
