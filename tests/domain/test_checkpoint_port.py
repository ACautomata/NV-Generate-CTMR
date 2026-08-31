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

"""Contract tests for the checkpoint-repository port (ADR-0019 §3, #269).

The port is what training shells rely on: ``save(payload, epoch)`` publishes
one epoch's payload and returns the published path, and ``load(path)`` fetches
it back verbatim with a repo-transparent payload key set. The same contract
runs against an in-memory fake adapter and the real json/torch repository --
the fake is how future application tests will drive checkpoint flows without
a filesystem. The tmp atomic publish and the ``latest.json`` pointer protocol
are the real adapter's on-disk semantics, pinned by
tests/infrastructure/test_checkpoints.py (ADR-0015 §4 辖区不变).
"""

from ctmr.domain.checkpoints import CheckpointRepository


class FakeCheckpointRepository:
    """In-memory stand-in: payloads keyed by epoch, one save remembered."""

    def __init__(self):
        self.published = {}
        self.pointer = None

    def save(self, payload, epoch):
        self.published[epoch] = payload
        self.pointer = epoch
        return f"memory://model/epoch_{epoch}.pt"

    def load(self, path):
        return self.published[int(path.rsplit("_", 1)[1].removesuffix(".pt"))]


PAYLOAD = {"epoch": 3, "loss": 0.5, "num_train_timesteps": 1000, "scale_factor": 1.0, "unet_state_dict": {}}

REAL_PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict"]


def real_repository(tmp_path):
    from ctmr.infrastructure.checkpoints import CheckpointRepository

    return CheckpointRepository(tmp_path)


def test_the_fake_adapter_satisfies_the_port_contract():
    repo = FakeCheckpointRepository()
    path = repo.save(PAYLOAD, 3)

    assert repo.load(path) == PAYLOAD


def test_the_real_adapter_satisfies_the_port_contract(tmp_path):
    repo = real_repository(tmp_path)
    assert isinstance(repo, CheckpointRepository)  # the infrastructure class IS the domain port, structurally

    path = repo.save(PAYLOAD, 3)
    fetched = repo.load(path)
    assert list(fetched) == REAL_PAYLOAD_KEYS  # 切换前后逐一相等;torch 值经既有 torch 级测试钉板
    for key, value in PAYLOAD.items():
        assert fetched[key] == value
