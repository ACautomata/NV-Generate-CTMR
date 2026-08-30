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

"""Checkpoint-file addressing: one on-disk file folded into a ``WeightsRef`` (spec #221 candidate 6, issue #227).

The file-addressing half of checkpoint identity (CONTEXT.md). The domain
``WeightsRef`` pins the addressing rule in memory (``of_bytes``); the domain
layer owns no IO (ADR-0015 §2), so the single file read that folds an on-disk
checkpoint into the same identity lives here -- 1MB streaming, byte-identical
to the retired handwritten copies it consolidates (the two cross-modal run
guards and the dm_source ledger). Same bytes, same identity, whatever the read
path; this module is the single definition point of checkpoint-file addressing.
"""

import hashlib

from ctmr.domain.identity import WeightsRef


def weights_ref_of_file(path) -> WeightsRef:
    """Identity of one on-disk weight artifact: the ``of_bytes`` rule applied to the file's bytes, streamed in 1MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return WeightsRef(sha256=digest.hexdigest())
