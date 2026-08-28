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

"""Quantitative acceptance layer: the FID / paired-error evidence chain.

The whole chain migrated from ``scripts/brats_l1_quantitative.py`` (#141 /
ADR-0015 §2): ``fid`` (Fréchet metric, case-level bootstrap, three-plane FID
assessment), ``paired`` (P3 candidate-vs-baseline MAE/SSIM with the t1n->t1c
exception), ``report`` (the versioned candidate-bound report assembly),
``evidence`` (controlled JSON/NIfTI/feature-manifest readers and the report
writer) and ``evaluate`` (the ``ctmr accept quantitative evaluate`` entry).
``fid_2d5`` (#140) is the 2.5D FID calculator shared with the dev-trend
machinery. Every domain quantity is frozen pre-registration -- this package
changed addresses only.
"""
