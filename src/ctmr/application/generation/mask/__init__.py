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

"""Mask-conditioned generation family (ADR-0015 §2, issue #129).

The mask (tumour-mask-conditioned) use case as three word-named modules
(ADR-0015 §7⑥): ``train`` (ControlNet-only candidate finetune), ``monitor``
(dev sidecar: FID + instrument + round-trip-Dice trend, reference/watch/select
polling + early stop) and ``sample`` (final-holdout sample generation with the
selected candidate). ``inference`` holds the mask-conditioned sampling wrapper
shared with the user-facing mask→image tools. The family kernels inject into
``ctmr.application.shell``'s ``PhaseHarness``; the ``ctmr generate mask`` CLI
verbs dispatch here. The historical phase codename stays in CONTEXT.md and ADR
history only.
"""
