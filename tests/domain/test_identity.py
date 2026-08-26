"""Birth-with-tests for the weight-lineage identity entity (ADR-0015 section 2, #133).

Pins the content-addressing semantics of ``WeightsRef``: the identity is the
sha256 of the weight-set payload -- byte-equal payloads are the same weights
(same identity), any difference in content is a different entity, and the ref
itself is a frozen value object. Stdlib-only, any machine (ADR-0013 §4).
"""

import dataclasses
import hashlib

import pytest

from ctmr.domain.identity import WeightsRef


def test_ref_is_the_sha256_of_the_payload():
    # the empty-payload vector pins the algorithm itself (sha256, hex digest)
    assert WeightsRef.of(b"").sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    payload = b"checkpoint bytes"
    assert WeightsRef.of(payload).sha256 == hashlib.sha256(payload).hexdigest()


def test_same_weights_same_identity():
    payload = b"P1-DM epoch-42 state_dict"
    first = WeightsRef.of(payload)
    second = WeightsRef.of(bytes(payload))  # an equal copy re-hashed
    assert first == second  # byte-equal payloads are the same weights...
    assert hash(first) == hash(second)
    assert {first: "P1 candidate"}[second] == "P1 candidate"  # ...one ledger key, not two


def test_different_weights_different_identity():
    base = b"base-model state_dict"
    finetuned = b"base-model state_dict" + b"\x00finetuned"
    assert WeightsRef.of(base) != WeightsRef.of(finetuned)
    assert len({WeightsRef.of(base), WeightsRef.of(finetuned), WeightsRef.of(base)}) == 2


def test_ref_is_a_frozen_value_object():
    ref = WeightsRef.of(b"weights")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.sha256 = "tampered"


def test_digest_is_canonical_hex_form():
    digest = WeightsRef.of(b"\x00\x01\x02").sha256
    assert len(digest) == 64
    assert digest == digest.lower()
    int(digest, 16)  # parses as hexadecimal
