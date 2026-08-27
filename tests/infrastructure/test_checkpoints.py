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

"""Behaviour gates for the CheckpointRepository port (ADR-0015 §4, #135).

The repository is the single persistence protocol for weight payloads: state_dict
payload store/fetch and the tmp-publish ``epoch_<N>.pt`` + rename protocol with
the ``latest.json`` pointer -- sunk verbatim out of the training shell. Real
torch execution (in this suite, CPU) is CI policy for torch-level tests: the
round-trip below stores a genuine state_dict, fetches it and loads it into a
fresh model.
"""

import json
from pathlib import Path

import pytest
import torch

from ctmr.infrastructure.checkpoints import CheckpointRepository

pytestmark = pytest.mark.torch

# The pre-#111 checkpoint payload key sets, verbatim (do not edit, mirrors
# tests/harness/test_payload_schema.py).
P1_PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict"]
P2_PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "controlnet_state_dict"]
P3_PAYLOAD_KEYS = P2_PAYLOAD_KEYS


def _payload(epoch, loss, state_dict=None):
    return {
        "epoch": epoch,
        "loss": loss,
        "num_train_timesteps": 1000,
        "scale_factor": 0.1,
        "unet_state_dict": {} if state_dict is None else state_dict,
    }


def test_save_publishes_the_payload_and_the_pointer(tmp_path):
    repo = CheckpointRepository(tmp_path)
    path = repo.save(_payload(1, 0.25), 1)

    assert path == tmp_path / "epoch_1.pt"
    assert (tmp_path / "epoch_1.pt").is_file()
    assert not (tmp_path / "epoch_1.pt.tmp").exists()  # 无 .tmp 残留
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest == {"epoch": 1, "checkpoint": str(tmp_path / "epoch_1.pt")}
    assert (tmp_path / "latest.json").read_text().endswith("\n")
    # the published file is a loadable torch artifact, not a partial write
    assert list(torch.load(tmp_path / "epoch_1.pt", weights_only=True)) == P1_PAYLOAD_KEYS


@pytest.mark.parametrize(
    "payload_key",
    [P1_PAYLOAD_KEYS, P2_PAYLOAD_KEYS, P3_PAYLOAD_KEYS],
)
def test_store_fetch_keeps_the_payload_key_set_verbatim(tmp_path, payload_key):
    state_key = "unet_state_dict" if "unet_state_dict" in payload_key else "controlnet_state_dict"
    payload = {
        "epoch": 3,
        "loss": 0.5,
        "num_train_timesteps": 1000,
        "scale_factor": 1.0,
        state_key: {"a": torch.tensor(1.0)},
    }
    repo = CheckpointRepository(tmp_path)
    path = repo.save(payload, 3)
    fetched = repo.load(path)

    assert list(fetched) == payload_key  # 切换前后逐一相等


def test_real_state_dict_roundtrip_loads_into_a_fresh_model(tmp_path):
    source = torch.nn.Linear(3, 2, bias=True)
    payload = _payload(1, 0.5, source.state_dict())
    repo = CheckpointRepository(tmp_path)

    path = repo.save(payload, 1)
    fetched = repo.load(path)

    target = torch.nn.Linear(3, 2, bias=True)
    state = target.load_state_dict(fetched["unet_state_dict"], strict=True)
    assert state.missing_keys == [] and state.unexpected_keys == []
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_failed_rename_never_moves_the_latest_pointer(tmp_path, monkeypatch):
    repo = CheckpointRepository(tmp_path)
    repo.save(_payload(1, 0.25), 1)

    # The pointer is written only after a successful rename: if the rename
    # fails, latest.json must keep pointing at the previous complete checkpoint.
    def _fail_rename(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", _fail_rename)
    with pytest.raises(OSError, match="rename failed"):
        repo.save(_payload(2, 0.5), 2)

    assert not (tmp_path / "epoch_2.pt").exists()
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest == {"epoch": 1, "checkpoint": str(tmp_path / "epoch_1.pt")}
    assert (tmp_path / "epoch_1.pt").is_file()


def test_a_serialization_failure_never_republishes_the_pointer(tmp_path):
    repo = CheckpointRepository(tmp_path)
    repo.save(_payload(1, 0.25), 1)

    with pytest.raises(Exception):
        repo.save({"epoch": 2, "unserializable": lambda: None}, 2)

    assert not (tmp_path / "epoch_2.pt").exists()
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["epoch"] == 1
    # the previous checkpoint is still intact and loadable
    assert list(repo.load(tmp_path / "epoch_1.pt")) == P1_PAYLOAD_KEYS
