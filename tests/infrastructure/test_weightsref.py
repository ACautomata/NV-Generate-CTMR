"""Convergence gate for the checkpoint-file addressing primitive (issue #227, spec #221 candidate 6).

``weights_ref_of_file`` is the single definition point of on-disk weight-file
hashing -- the file-addressing half of checkpoint identity (CONTEXT.md). The
suite pins the streaming block boundary once (files straddling the 1MB chunk
hash like the whole payload) and the identity contract: the same bytes fold
into the same ``WeightsRef`` whether read from a file or held in memory, so
file identity IS content identity and the path never enters it.
"""

import hashlib

from ctmr.domain.identity import WeightsRef
from ctmr.infrastructure.weightsref import weights_ref_of_file


def test_block_boundaries_hash_like_the_whole_payload(tmp_path):
    """The 1MB chunk boundary must be invisible: every size straddling it folds
    into the same identity as the single in-memory hash of the same bytes."""
    pattern = b"checkpoint-bytes-"
    path = tmp_path / "weights.pt"
    for size in (0, 1, (1 << 20) - 1, 1 << 20, (1 << 20) + 1, (2 << 20) + 7):
        payload = (pattern * (size // len(pattern) + 1))[:size]
        path.write_bytes(payload)
        assert weights_ref_of_file(path) == WeightsRef.of_bytes(payload), f"block boundary drifted at {size} bytes"


def test_the_file_identity_is_a_weights_ref_and_survives_rereads(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"frozen-p1-dm-fixture")
    ref = weights_ref_of_file(path)
    assert isinstance(ref, WeightsRef)
    assert ref == weights_ref_of_file(path)  # two independent reads collapse onto one identity
    assert ref.sha256 == hashlib.sha256(b"frozen-p1-dm-fixture").hexdigest()
