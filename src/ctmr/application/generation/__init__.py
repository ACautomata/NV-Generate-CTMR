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

"""Generation use-case families (ADR-0015 §2, issue #129).

One subpackage per conditional generation family, named by the conditioning
signal (``modality_label`` / ``mask`` / ``cross_modal``); the historical phase
codenames stay in CONTEXT.md and ADR history only (ADR-0015 §7⑤). The family
kernels inject into ``ctmr.application.shell``'s ``PhaseHarness``; the ``ctmr
generate`` CLI verbs dispatch here.
"""
