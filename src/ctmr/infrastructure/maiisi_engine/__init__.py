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

"""Vendored MAISI diffusion engine (issue #134 / ADR-0015 §2, batch M3).

The reusable upstream engine files are collected verbatim from ``scripts/``
(diff_model_setting, diff_model_infer, diff_model_train,
diff_model_create_training_data, sample, utils_infer). Vendoring is an
adoption, not a rewrite: the engine bodies are byte-identical to their
``scripts/`` originals (proven by tests/infrastructure/test_maiisi_engine_vendored.py),
so upstream diffs remain reviewable.

The bodies rely on sibling helper modules that live elsewhere in the
end-state layout (dataio base, ticket 03). Until those slices land, each
referenced sibling is represented here by a thin *bridge module* that
re-exports an explicit name list from the original ``scripts.*`` location:

=========================  ==========================================
Bridge module (relative)   Forwarded from
=========================  ==========================================
``utils``                  ``scripts.utils``
``transforms``             ``scripts.transforms``
``augmentation``           ``scripts.augmentation``
``find_masks``             ``scripts.find_masks``
``infer_image_from_mask``  ``scripts.infer_image_from_mask``
``quality_check``          ``scripts.quality_check``
``sample_mask``            ``scripts.sample_mask``
=========================  ==========================================

When the helpers move to their own homes (M1 dataio slice and friends),
update ONLY the bridge lines -- never edit the frozen engine bodies.

Consumer call sites stay untouched for now (existing ``scripts/`` entries
keep their old imports); switching them to this package belongs to the
expand-stage use-case-family slices (issues #137/#143).

Spawn-context constraint carried over from issue #123: DCU/HSA contexts
must not be fork()-inherited. Any driver in or around this engine package
that fans out worker processes MUST launch them with the ``spawn`` start
method (multiprocessing or DataLoader workers alike); forking on a DCU
host leaves workers permanently stuck in futex waits on inherited GPU/HSA
state. Do not regress to fork-style parallelism when refactoring here.
"""
