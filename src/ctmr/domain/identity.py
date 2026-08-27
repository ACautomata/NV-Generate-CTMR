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

"""Weight-lineage identity, sha256 content-addressed (ADR-0015 §2, issue #133).

``WeightsRef`` is the identity entity of one weight artifact: its sha256 hex
digest IS the identity -- the same bytes are the same weights no matter which
path they were read from ("same weights, same identity"), and different bytes
are a different weight, born different. Paths, timestamps and any other
non-content facts never enter the identity.

Pure computation: ``of_bytes`` hashes an in-memory payload; file reading
stays with the callers -- the domain layer owns no IO (ADR-0015 §2).
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightsRef:
    """Content identity of one weight artifact: ``sha256`` hex digest."""

    sha256: str

    @classmethod
    def of_bytes(cls, payload: bytes) -> "WeightsRef":
        """Identity of one weight payload: the single addressing rule."""
        return cls(sha256=hashlib.sha256(payload).hexdigest())
