"""Download-assembly tests with a mocked HF hub (no network): asset lists, caching and overwrite semantics."""

import inspect

import pytest

from ctmr.infrastructure.dataio import downloads


def _fake_hub_factory(monkeypatch, tmp_path):
    """Returns a fake hf_hub_download that materializes files in a fake HF cache and records calls."""
    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir()
    calls = []

    def fake_hf_download(repo_id, filename="", revision="main", token=None):
        calls.append((repo_id, filename, revision))
        cached = cache_dir / repo_id.replace("/", "__") / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(f"payload:{filename}")
        return str(cached)

    monkeypatch.setattr(downloads, "hf_hub_download", fake_hf_download)
    return calls


def test_fetch_downloads_and_copies_files(monkeypatch, tmp_path):
    calls = _fake_hub_factory(monkeypatch, tmp_path)
    items = [{"repo_id": "org/repo", "filename": "models/a.pt", "path": str(tmp_path / "dest" / "a.pt")}]
    saved = downloads.fetch_to_hf_path_cmd(items, root_dir=str(tmp_path))
    assert saved == [str(tmp_path / "dest" / "a.pt")]
    assert (tmp_path / "dest" / "a.pt").read_text() == "payload:models/a.pt"
    # tracking hits config.json once before the real file
    assert ("org/repo", "config.json", "main") in calls
    assert ("org/repo", "models/a.pt", "main") in calls


def test_fetch_skips_existing_without_overwrite_and_refetches_with(monkeypatch, tmp_path):
    calls = _fake_hub_factory(monkeypatch, tmp_path)
    items = [{"repo_id": "org/repo", "filename": "models/a.pt", "path": str(tmp_path / "dest" / "a.pt")}]
    downloads.fetch_to_hf_path_cmd(items)
    n_files_after_first = sum(1 for c in calls if c[1] == "models/a.pt")
    assert n_files_after_first == 1
    # second call: dst exists -> skip, no new fetch
    downloads.fetch_to_hf_path_cmd(items)
    assert sum(1 for c in calls if c[1] == "models/a.pt") == 1
    # overwrite=True removes dst and refetches
    downloads.fetch_to_hf_path_cmd(items, overwrite=True)
    assert sum(1 for c in calls if c[1] == "models/a.pt") == 2


def test_fetch_tracks_config_once_per_repo(monkeypatch, tmp_path):
    calls = _fake_hub_factory(monkeypatch, tmp_path)
    items = [
        {"repo_id": "org/one", "filename": "a", "path": str(tmp_path / "dest1")},
        {"repo_id": "org/one", "filename": "b", "path": str(tmp_path / "dest2")},
        {"repo_id": "org/two", "filename": "c", "path": str(tmp_path / "dest3")},
    ]
    downloads.fetch_to_hf_path_cmd(items)
    config_calls = [c for c in calls if c[1] == "config.json"]
    assert config_calls == [("org/one", "config.json", "main"), ("org/two", "config.json", "main")]


def test_fetch_tracking_failure_is_survived(monkeypatch, tmp_path):
    def flaky(repo_id, filename="", revision="main", token=None):
        if filename == "config.json":
            raise RuntimeError("network hiccup")
        cached = tmp_path / "hf-cache" / repo_id.replace("/", "__") / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text("x")
        return str(cached)

    monkeypatch.setattr(downloads, "hf_hub_download", flaky)
    saved = downloads.fetch_to_hf_path_cmd([{"repo_id": "org/r", "filename": "a", "path": str(tmp_path / "d")}])
    assert saved == [str(tmp_path / "d")]


def test_download_model_data_asset_lists_and_path_prefixing(monkeypatch, tmp_path):
    captured = []

    def fake_fetch(items, root_dir="./", revision="main", overwrite=False, token=None, track_download=True):
        captured.append(items)
        return [str(tmp_path / "x")]

    monkeypatch.setattr(downloads, "fetch_to_hf_path_cmd", fake_fetch)
    root = str(tmp_path / "root")

    downloads.download_model_data("rflow-mr-brain", root)
    assert len(captured[-1]) == 2
    assert all("datasets/" not in it["path"] for it in captured[-1])

    downloads.download_model_data("rflow-mr", root)
    assert len(captured[-1]) == 2

    downloads.download_model_data("ddpm-ct", root, model_only=True)
    assert len(captured[-1]) == 5

    downloads.download_model_data("ddpm-ct", root)
    assert len(captured[-1]) == 8

    downloads.download_model_data("rflow-ct", root, model_only=True)
    assert len(captured[-1]) == 5

    downloads.download_model_data("rflow-ct", root)
    assert len(captured[-1]) == 8
    # dataset assets are prefixed with root_dir, model assets are not
    assert all(it["path"].startswith("models/") for it in captured[-1] if "models" in it["path"])
    assert all(it["path"].startswith(root) for it in captured[-1] if "datasets" in it["path"])

    with pytest.raises(ValueError, match="chosen from"):
        downloads.download_model_data("bogus-v", root)


def test_bundle_asset_manifest_is_stable():
    """Guard against silent edits of the pinned resource manifest (rflow-ct, full bundle)."""
    src = inspect.getsource(downloads.download_model_data)
    assert '"repo_id": "nvidia/NV-Generate-MR-Brain"' in src
    assert "candidate_masks_flexible_size_and_spacing_4000.json" in src
    assert "all_masks_flexible_size_and_spacing_4000.zip" in src
