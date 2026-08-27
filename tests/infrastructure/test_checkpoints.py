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

"""Behaviour gates for CheckpointRepository (ADR-0015 section 4, issue #135).

Pins the publication protocol sunk from the shell (ADR-0011 #111): a payload
passes through save/load with its key set and order untouched (the schema the
kernel ``checkpoint_payload`` hooks own -- ``unet_state_dict`` /
``controlnet_state_dict``); the ``latest.json`` pointer moves only after a
completed tmp+rename, so a failed torch.save leaves the pointer -- and hence
every polling reader of ``epoch_*.pt`` -- untouched; and no ``.tmp`` residue
survives a successful publication. Torch-level: skips itself on light stacks
via ``pytest.importorskip``.
"""

import json

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip must precede the torch-dependent import)

from ctmr.infrastructure.checkpoints import CheckpointRepository  # noqa: E402

PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict"]


def _payload(epoch):
    """A payload with the P1 schema shape; the repository must carry it opaquely."""
    return {
        "epoch": epoch,
        "loss": 0.25,
        "num_train_timesteps": 1000,
        "scale_factor": 1.0,
        "unet_state_dict": {"weight": torch.arange(6.0).reshape(2, 3)},
    }


def test_roundtrip_preserves_the_payload_schema_verbatim(tmp_path):
    payload = _payload(3)
    saved = CheckpointRepository(tmp_path).save(payload, 3)
    loaded = CheckpointRepository(tmp_path).load(saved)
    assert list(loaded) == PAYLOAD_KEYS  # key set AND order, verbatim against what went in
    assert loaded["epoch"] == 3
    assert loaded["loss"] == 0.25
    assert loaded["num_train_timesteps"] == 1000
    assert loaded["scale_factor"] == 1.0
    assert torch.equal(loaded["unet_state_dict"]["weight"], payload["unet_state_dict"]["weight"])


def test_publish_is_atomic_with_no_tmp_residue_and_a_valid_pointer(tmp_path):
    repo = CheckpointRepository(tmp_path)
    path = repo.publish(_payload(3), 3)
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    latest_text = (tmp_path / "latest.json").read_text()
    assert json.loads(latest_text) == {"epoch": 3, "checkpoint": str(path)}
    assert latest_text.endswith("\n")


def test_latest_pointer_tracks_the_last_completed_publication(tmp_path):
    repo = CheckpointRepository(tmp_path)
    repo.publish(_payload(1), 1)
    assert json.loads((tmp_path / "latest.json").read_text()) == {"epoch": 1, "checkpoint": str(tmp_path / "epoch_1.pt")}
    repo.publish(_payload(2), 2)
    assert json.loads((tmp_path / "latest.json").read_text()) == {"epoch": 2, "checkpoint": str(tmp_path / "epoch_2.pt")}


def test_failed_save_never_moves_the_pointer_onto_a_half_written_file(tmp_path, monkeypatch):
    repo = CheckpointRepository(tmp_path)
    repo.publish(_payload(1), 1)
    pointer_before = (tmp_path / "latest.json").read_text()

    def fail_mid_write(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr("ctmr.infrastructure.checkpoints.torch.save", fail_mid_write)
    with pytest.raises(RuntimeError):
        repo.publish(_payload(2), 2)

    # The rename never completed: the pointer still names the last completed
    # publication, so no reader can be pointed at a half-written checkpoint.
    assert (tmp_path / "latest.json").read_text() == pointer_before
    assert not (tmp_path / "epoch_2.pt").exists()
