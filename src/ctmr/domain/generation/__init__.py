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

"""Generation behaviour domain entities (ADR-0016, issue #170).

``model`` is the rich ``DiffusionModel`` (train_step / sample behaviour, no
checkpoint identity); ``scheduler`` is the per-sample trajectory entity
``DiffusionScheduler``; ``update`` is the ``GradientExecutor`` protocol the
application injects the runtime precision strategy through; ``objective``
holds the modality-label perturbation (the VAE objective and the tumour-region
weighting arrive with their own migrations).
"""

from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import ModalityLabelPerturber
from ctmr.domain.generation.scheduler import DiffusionScheduler
from ctmr.domain.generation.update import GradientExecutor

__all__ = ["DiffusionModel", "DiffusionScheduler", "GradientExecutor", "ModalityLabelPerturber"]
