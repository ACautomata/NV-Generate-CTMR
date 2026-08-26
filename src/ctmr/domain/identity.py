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

"""Weight-lineage identity by content addressing (ADR-0015 section 2, issue #133).

A checkpoint's business identity is its weight-set payload content-addressed by
sha256 -- not a Python object, not a network class (CONTEXT.md,
"Checkpoint Identity"): the same architecture instantiated at different train
states yields different entities, while byte-equal payloads are the *same*
weights wherever they are stored. ``WeightsRef.of(payload)`` is a pure hash
transform -- stdlib hashlib only, no file IO (readers stay with the callers);
persistence lives with the M3 CheckpointRepository, lineage ledgers with
dm_source.json.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightsRef:
    """Content address of one weight-set payload (sha256 hex digest).

    Value semantics carry the identity rule: two refs are equal iff their
    payloads were byte-equal when hashed (same weights, same identity).
    """

    sha256: str

    @classmethod
    def of(cls, payload: bytes) -> "WeightsRef":
        """Hashes a weight-set payload into its identity ref (pure transform)."""
        return cls(sha256=hashlib.sha256(payload).hexdigest())
