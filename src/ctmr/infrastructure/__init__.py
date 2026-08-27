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

"""``ctmr.infrastructure`` -- the IO/framework-adaptation layer (ADR-0015
section 2): filesystem, subprocess and third-party SDKs.

Batch provenance (ADR-0015 section 10):

- M3 (#135): ``checkpoints`` -- state_dict-payload storage: tmp+rename atomic
  publication plus the ``latest.json`` pointer protocol, sunk from the harness
  shell (ADR-0011, #111); ``dmsource`` -- dm_source.json lineage-ledger
  read/write, sunk from the phase run contract (#58)
- M1 (#132): ``dataio`` -- transforms/augmentation/morphology/quality_check/
  find_masks/sample_mask/plotting/download data-utility belt

This package re-exports nothing so ``import ctmr.infrastructure`` stays
runnable on a light stack: submodules carry their own dependency floors
(``checkpoints`` needs torch; ``dmsource`` is stdlib-only -- ADR-0013
section 4). Import submodules directly.
"""
