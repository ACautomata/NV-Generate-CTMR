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

"""The installable nnU-Net trainer variant required by the issue #35 training contract (migrated from the retired scripts layer (git history; ``nnunet_trainer_250_epochs``), ticket #140).

Before formal training starts, ``install_trainer`` copies this module into the
installed ``nnunetv2`` trainer variants package. It only changes the upstream
trainer's epoch count; the class name is pinned by nnunetv2's registry contract
(ADR-0015 §7 external-contract exemption, so no renaming).

``__init__`` must explicitly mirror the upstream signature rather than forwarding
``*args/**kwargs``: upstream indexes ``locals()`` by the parameter names of
``inspect.signature(self.__init__)`` to rebuild ``my_init_kwargs``; a forwarding
subclass's signature keys (``args``/``kwargs``) do not exist in the base frame's
``locals()``, which raises ``KeyError`` at construction. This matches how
nnunetv2's own ``nnUNetTrainer_Xepochs`` variants are written.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer250Epochs(nnUNetTrainer):
    """恰好运行 250 个 epoch，同时保留 upstream nnU-Net 其余训练配方。"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
