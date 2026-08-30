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

"""BypassTrainLoader -- the one training-list loader of the bypass families (issue #226).

The MAISI JSON-list loader the mask (P2) and cross-modal (P3) families used to
hand-copy, collapsed into one parameterized definition: the family transform
set rides the constructor (load keys, the mask family's region-index
companions, the condition keys joined onto the per-list data root), the
runtime/recipe values come in per ``build`` call. The contract returns ONLY
the train loader: the fold split's val side is never constructed -- both
families select their candidate by the dev-eval sidecar, never by a
validation loss (spec #51 decision 7), so the old construct-and-discard val
loader (a startup and cache_rate>0 memory tax) is gone. Numerics of the train
side are byte-identical to the collapsed copies; the gate is
tests/application/generation/test_train_loader.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from monai.data import CacheDataset, partition_dataset
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from torch.utils.data import DataLoader

from ctmr.infrastructure.dataio.list_assembly import add_data_dir2path


class BypassTrainLoader:
    """The shared ControlNet-family train-list loader: fold-split list, family transforms, train loader only.

    ``load_keys`` are the LoadImaged/Orientationd keys (the mask pair vs the
    cross-modal triple), ``companion_keys`` the mask family's region-index
    entries (float tensor + x1e2, allow-missing), ``join_keys`` the condition
    paths joined onto the per-list data root beyond ``image``/``label``
    (cross-modal's ``src_image``).
    """

    def __init__(self, load_keys, companion_keys=(), join_keys=()):
        self._load_keys = tuple(load_keys)
        self._companion_keys = tuple(companion_keys)
        self._join_keys = tuple(join_keys)

    def build(self, *, json_data_list, data_base_dir, batch_size, cache_rate, fold, rank, world_size, modality_mapping):
        """One train DataLoader from the JSON list(s); ``rank``/``world_size`` partition it per DDP rank."""
        list_train = self._train_entries(json_data_list, data_base_dir, fold)
        if world_size > 1:
            list_train = partition_dataset(data=list_train, shuffle=True, num_partitions=world_size, even_divisible=True)[rank]
        dataset = CacheDataset(data=list_train, transform=Compose(self._transforms(modality_mapping)), cache_rate=cache_rate, num_workers=8)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def _train_entries(self, json_data_list, data_base_dir, fold):
        """The fold!=fold-indexed train side of the list(s), with the join keys rooted; the val side is dropped."""
        if isinstance(json_data_list, list):
            list_train = []
            for data_list, data_root in zip(json_data_list, data_base_dir):
                json_data = json.loads(Path(data_list).read_text())["training"]
                train, _val = add_data_dir2path(json_data, data_root, fold)
                list_train += self._join_conditions(train, data_root)
        else:
            json_data = json.loads(Path(json_data_list).read_text())["training"]
            list_train, _val = add_data_dir2path(json_data, data_base_dir, fold)
            list_train = self._join_conditions(list_train, data_base_dir)
        return list_train

    def _join_conditions(self, entries, data_root):
        for entry in entries:
            for key in self._join_keys:
                if key in entry:
                    entry[key] = os.path.join(data_root, entry[key])
        return entries

    def _transforms(self, modality_mapping):
        common_transform = [
            LoadImaged(keys=list(self._load_keys), image_only=True, ensure_channel_first=True),
            Orientationd(keys=list(self._load_keys), axcodes="RAS"),
            EnsureTyped(keys=["label"], dtype=torch.long, track_meta=True),
        ]
        common_transform += [Lambdad(keys=key, func=lambda x: torch.FloatTensor(x), allow_missing_keys=True) for key in self._companion_keys]
        common_transform += [
            Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            Lambdad(keys=[*self._companion_keys, "spacing"], func=lambda x: x * 1e2, allow_missing_keys=True),
            Lambdad(keys=["modality"], func=lambda x: modality_mapping[x], allow_missing_keys=True),
            EnsureTyped(keys=["modality"], dtype=torch.long, allow_missing_keys=True),
        ]
        return common_transform
