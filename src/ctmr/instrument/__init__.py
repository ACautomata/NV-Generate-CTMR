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

"""ctmr.instrument -- the execution side of the frozen instrument call (ADR-0009, #107).

Since the ADR-0015 §2 domain split (#133) the frozen command construction (the
``InstrumentSpec`` value object, per-challenge specs and
``FrozenInstrumentCommand.build`` -- pure argv) lives in
``ctmr.domain.instrument_spec``; this package keeps what executes it:
``safeglobals`` holds the single allowlist definition and its scoped activation
``nnunet_safe_globals``, and ``predict`` is the canonical execution entry
(``python -m ctmr.instrument.predict``). Mirror TTA stays ON by omission -- a
frozen invariant of the spec (see ``ctmr.domain.instrument_spec``).

This package re-exports nothing: ``safeglobals`` needs torch and ``predict``
needs nnunetv2 -- import the submodules directly (ADR-0013 §4).
"""
