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

"""ctmr.instrument -- frozen instrument command, the unique module (ADR-0009, #107).

``command`` is the single construction point (``InstrumentSpec`` value object,
per-challenge frozen specs, ``FrozenInstrumentCommand.build`` -- pure argv onto
the canonical entry); ``safeglobals`` holds the single allowlist definition
and its scoped activation ``nnunet_safe_globals``; ``predict`` is the
canonical execution entry (``python -m ctmr.instrument.predict``). Mirror TTA
stays ON by omission -- a frozen invariant (see ``command``).

This package re-exports nothing: ``command`` is stdlib-only so its convergence
gate runs on any machine, while ``safeglobals`` needs torch and ``predict``
needs nnunetv2 -- import the submodules directly (ADR-0013 §4).
"""
