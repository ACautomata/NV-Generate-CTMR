"""Convergence-gate tests for the weight-lineage identity entity (ADR-0015 §2, #133).

``WeightsRef`` pins the content-addressing semantics at birth: a weight
artifact's identity IS its sha256 hex digest -- the same bytes are the same
weights no matter where they were read from, different bytes are a different
identity. Pure computation (hashlib in memory): file reading stays with the
callers, the domain layer owns no IO (ADR-0015 §2).
"""

import dataclasses
import hashlib

import pytest

from ctmr.domain.identity import WeightsRef

_WEIGHTS_A = b"checkpoint-bytes-a"
_WEIGHTS_B = b"checkpoint-bytes-b"


def test_same_bytes_are_the_same_identity():
    """同权重同身份: equal payloads meet in one ref, eq and hash alike."""
    assert WeightsRef.of_bytes(_WEIGHTS_A) == WeightsRef.of_bytes(_WEIGHTS_A)
    assert hash(WeightsRef.of_bytes(_WEIGHTS_A)) == hash(WeightsRef.of_bytes(_WEIGHTS_A))


def test_different_bytes_are_a_different_identity():
    ref_a = WeightsRef.of_bytes(_WEIGHTS_A)
    ref_b = WeightsRef.of_bytes(_WEIGHTS_B)
    assert ref_a != ref_b


def test_identity_is_independent_of_where_the_bytes_came_from():
    """Lineage follows content, not provenance: two independent reads of one
    checkpoint (e.g. raw dir vs published store) collapse onto one identity."""
    from_primary_dir = WeightsRef.of_bytes(_WEIGHTS_A)
    from_published_store = WeightsRef.of_bytes(b"checkpoint-bytes-" + b"a")
    assert from_primary_dir == from_published_store


def test_digest_is_the_canonical_sha256_hex_form():
    ref = WeightsRef.of_bytes(_WEIGHTS_A)
    assert ref.sha256 == hashlib.sha256(_WEIGHTS_A).hexdigest()
    assert len(ref.sha256) == 64
    assert ref.sha256 == ref.sha256.lower()


def test_ref_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        WeightsRef.of_bytes(_WEIGHTS_A).sha256 = "0" * 64
