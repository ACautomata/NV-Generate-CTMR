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


"""Behaviour gates for ctmr.infrastructure.dataio.download (#132).

Verbatim lift of scripts/download_model_data.py (argparse front-end dropped --
CLI belongs to the future unified entry). ``hf_hub_download`` is monkeypatched
so every gate runs offline: tracking probe semantics, cache-copy/skip/
overwrite behaviour, per-repo tracking dedup, probe-failure tolerance, and
each generate-version manifest's exact file set (model paths stay repo-
relative while dataset paths get prefixed with root_dir). huggingface_hub
level; no network, CPU-only.
"""

import pytest

pytest.importorskip("huggingface_hub")

from ctmr.infrastructure.dataio import download as dl  # noqa: E402


class _FakeHub:
    """Scriptable stand-in for huggingface_hub.hf_hub_download."""

    def __init__(self, cache_dir, fail_probe_for=None):
        self.cache_dir = cache_dir
        self.fail_probe_for = list(fail_probe_for or ())
        self.calls = []

    def __call__(self, repo_id, filename, revision=None, token=None):
        self.calls.append((repo_id, filename))
        if filename == "config.json" and repo_id in self.fail_probe_for:
            raise ConnectionError("probe down")
        cached = self.cache_dir / (repo_id.replace("/", "_") + "__" + filename.replace("/", "_"))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"payload")
        return str(cached)


def _specs(root):
    return [
        {"repo_id": "org/a", "filename": "models/w1.pt", "path": str(root / "out/models/w1.pt")},
        {"repo_id": "org/a", "filename": "datasets/d1.json", "path": str(root / "root/datasets/d1.json")},
    ]


def test_fetch_copies_from_cache_and_creates_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "hf_hub_download", _FakeHub(tmp_path / "cache"))
    saved = dl.fetch_to_hf_path_cmd(_specs(tmp_path), track_download=False)
    assert len(saved) == 2
    assert (tmp_path / "out/models/w1.pt").exists()
    assert (tmp_path / "root/datasets/d1.json").exists()


def test_existing_destination_is_skipped_unless_overwrite(tmp_path, monkeypatch):
    dst = tmp_path / "out/models/w1.pt"
    dst.parent.mkdir(parents=True)
    dst.write_text("stale")
    hub = _FakeHub(tmp_path / "cache")
    monkeypatch.setattr(dl, "hf_hub_download", hub)

    first = dl.fetch_to_hf_path_cmd([_specs(tmp_path)[0]], track_download=False)
    assert first == [str(dst)]
    assert dst.read_text() == "stale"
    assert not hub.calls  # skip short-circuits before any hub traffic

    dl.fetch_to_hf_path_cmd([_specs(tmp_path)[0]], track_download=False, overwrite=True)
    assert dst.read_text() != "stale"


def test_tracking_pings_config_once_per_repo_and_tolerates_failure(tmp_path, monkeypatch):
    hub = _FakeHub(tmp_path / "cache")
    monkeypatch.setattr(dl, "hf_hub_download", hub)
    dl.fetch_to_hf_path_cmd(_specs(tmp_path), track_download=True)  # two items, same repo
    assert sum(1 for _, name in hub.calls if name == "config.json") == 1

    flaky = _FakeHub(tmp_path / "cache2", fail_probe_for={"org/a"})
    monkeypatch.setattr(dl, "hf_hub_download", flaky)
    flake_specs = [{"repo_id": "org/a", "filename": "m/x", "path": str(tmp_path / "x")}]
    saved = dl.fetch_to_hf_path_cmd(flake_specs, track_download=True)
    assert saved == [str(tmp_path / "x")]  # probe failure warns but does not abort


def test_tracking_pinger_passthrough_uses_config_json(tmp_path, monkeypatch):
    seen = {}

    def fake(repo_id, filename, revision=None, token=None):
        seen.update(repo_id=repo_id, filename=filename)
        return str(tmp_path / "cfg.json")

    monkeypatch.setattr(dl, "hf_hub_download", fake)
    dl.ensure_hf_download_tracked("org/z", revision="main")
    assert seen == {"repo_id": "org/z", "filename": "config.json"}


@pytest.fixture()
def capture(monkeypatch):
    calls = []

    def fake_fetch(items, **kwargs):
        calls.append(items)
        return [it["path"] for it in items]

    monkeypatch.setattr(dl, "fetch_to_hf_path_cmd", fake_fetch)
    return calls


def test_manifest_rflow_mr_brain_weight_only(capture):
    dl.download_model_data("rflow-mr-brain", "./")  # old leaf prints its saves; returns None
    items = capture[0]
    assert {(i["repo_id"], i["filename"]) for i in items} == {
        ("nvidia/NV-Generate-CT", "models/autoencoder_v1.pt"),
        ("nvidia/NV-Generate-MR-Brain", "models/diff_unet_3d_rflow-mr-brain_v1.pt"),
    }
    assert all(i["path"].startswith("models/") for i in items)


def test_manifest_rflow_ct_datasets_prefixed_with_root_dir(capture):
    dl.download_model_data("rflow-ct", "/srv/root")
    items = capture[0]
    names = [i["filename"] for i in items]
    assert "datasets/candidate_masks_flexible_size_and_spacing_4000.json" in names
    for it in items:
        if it["filename"].startswith("datasets/"):
            assert it["path"].startswith("/srv/root/")
        else:
            assert it["path"].startswith("models/")


def test_manifest_model_only_suppresses_datasets(capture):
    dl.download_model_data("ddpm-ct", "/", model_only=True)
    assert all(not i["filename"].startswith("datasets/") for i in capture[0])

    dl.download_model_data("ddpm-ct", "/")
    assert any(i["filename"].startswith("datasets/") for i in capture[1])


def test_unknown_version_is_rejected(capture):
    with pytest.raises(ValueError, match="generate_version has to be chosen"):
        dl.download_model_data("not-a-version", "./")
