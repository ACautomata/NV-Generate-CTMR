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

"""ctmr.instrument -- superseded reverse shim (issue #140; ADR-0015 §2).

The execution side of the frozen instrument call moved to
``ctmr.infrastructure.nnunet_runner`` (predictor execution + the weights_only
allowlist), exposed canonically as ``ctmr measure predict``; the frozen command
construction had already moved to ``ctmr.domain.instrument_spec`` (#133). The
submodules here re-export the new home until their last not-yet-migrated
consumer (the modality-label chain, ticket #140's sibling tickets) switches;
they then go away with the ADR-0015 batches.
"""
