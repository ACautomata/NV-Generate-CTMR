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

"""Modality-label-conditioned generation family (ADR-0015 §2, issue #129).

The image-only tumour-candidate use case as word-named modules (ADR-0015 §7⑥):
``train`` (full-parameter DM continuation with the 1:1 MR-RATE replay mix) and
``monitor`` (the offline dev light acceptance). The shared dev-trend machinery
lives in ``ctmr.application.generation.trend``; the watch/select engine
(``WatchEngine`` / ``SelectionEmitter``) in ``ctmr.application.shell``
-- ``monitor`` assembles the stage sampler/scorer/post-score collaborators
onto it. The historical phase codename stays in CONTEXT.md and ADR history
only.
"""
