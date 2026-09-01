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

"""Image-conditioned cross-modality generation family (ADR-0015 §2, issue #129).

The cross-modal use case implemented as four word-named modules (ADR-0015 §7⑥):
``train`` (ControlNet candidate finetune), ``monitor`` (offline dev
acceptance: reference/watch/select assembly over the shell ``WatchEngine``,
early stop included), ``baseline`` (zero-training
img2img comparison floor) and ``candidate`` (the trained ControlNet
counterpart). ``plan`` holds the shared frozen vocabulary / seed rules /
contract markers. The frozen marker strings keep their byte identity by rule
(``variant=stage0-baseline`` / ``controlnet-candidate``, the ``brats-p3-*``
schemas) — code identifiers stay functional words.
"""
