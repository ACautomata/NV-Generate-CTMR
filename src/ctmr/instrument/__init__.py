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

"""ctmr.instrument -- frozen-instrument execution side (ADR-0009, #107).

The command *spec* side moved to ``ctmr.domain.instrument_spec`` (#133);
the argv execution side still lives here until it lands as the M3
infrastructure nnunet-runner (ADR-0015 §2): ``safeglobals`` holds the single
allowlist definition and its scoped activation ``nnunet_safe_globals``, and
``predict`` is the canonical execution entry (``python -m ctmr.instrument.predict``,
the entry the spec-side builder emits). Mirror TTA stays ON by omission -- a
frozen invariant (see ``ctmr.domain.instrument_spec``).

This package re-exports nothing: ``predict`` needs nnunetv2 and ``safeglobals``
needs torch, while the domain spec side is stdlib-only -- import submodules
directly (ADR-0013 §4).
"""
