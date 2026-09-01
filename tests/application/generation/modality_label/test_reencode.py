"""The embedding re-encode job's gates (issue #251, series-② T4).

The clip=True encoding recipe invalidated the clip=False-encoded training
embeddings; this job re-encodes the full P1 image-only training list into a
new embedding root and registers the md5 manifest plus the in-domain
self-eval MAE spot-check (job C's clip=True reading convention, target
magnitude 0.006). The tests pin the orchestration contract on injected
stand-ins -- the encode runner (the vendored ``diff_model_create_training_data``
behind the composition-root injection) and the VAE decode callable -- never a
real autoencoder; the numeric readings themselves are a sugon-side verdict.
"""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError
from ctmr.application.generation.modality_label.reencode import EmbeddingManifest, ReencodeSelfCheck, main

# ----------------------------------------------------------------- manifest


def _write_fake_nifti(path, shape=(2, 2, 2, 4)):
    array = np.zeros(shape, dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))


def test_embedding_manifest_walks_tree_with_md5_and_shape(tmp_path):
    real = tmp_path / "embeddings" / "sub" / "case_a-t1c_emb.nii.gz"
    _write_fake_nifti(real)
    fake = tmp_path / "embeddings" / "case_b-t1c_emb.nii.gz"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not a nifti")
    (tmp_path / "embeddings" / "stray.txt").write_text("ignored")

    rows = EmbeddingManifest(tmp_path / "embeddings").walk()

    assert [row["path"] for row in rows] == ["case_b-t1c_emb.nii.gz", "sub/case_a-t1c_emb.nii.gz"]
    by_path = {row["path"]: row for row in rows}
    assert by_path["sub/case_a-t1c_emb.nii.gz"]["md5"] == hashlib.md5(real.read_bytes()).hexdigest()
    assert by_path["sub/case_a-t1c_emb.nii.gz"]["shape"] == [2, 2, 2, 4]
    assert by_path["case_b-t1c_emb.nii.gz"]["shape"] is None  # unreadable artifact recorded, not dropped
    assert by_path["sub/case_a-t1c_emb.nii.gz"]["bytes"] == real.stat().st_size


# --------------------------------------------------------------- self-check


def test_self_check_row_reads_like_job_c():
    """A decode stand-in returning the clip-encoded input itself reconstructs
    perfectly: clip_within ~ 0 (job C's 0.006-magnitude slot) and clip_over ~ 0,
    while the extrapolation scalars still record the noclip world's stress."""
    rng = np.random.default_rng(7)
    t1c = rng.uniform(0.0, 100.0, size=(8, 8, 8))
    t1c.flat[:2] = 1000.0  # a bright tail past the 99.5th percentile (under the 0.5% voxel share)
    latent = np.zeros((2, 2, 2, 4), dtype=np.float32)

    captured = {}

    def decode(z):
        captured["latent"] = z
        # stand-in reconstruction: the clip arm's own input (perfect fidelity)
        from ctmr.application.acceptance.distribution.intensity_domain import TrainingPreprocessing

        norm_clip, _l, _u = TrainingPreprocessing.normalize_percentile(t1c, clip=True)
        return TrainingPreprocessing.resize_image(norm_clip, (8, 8, 8), "trilinear")

    def encode(z):
        captured["encoded"] = z
        return latent  # the direct chain re-encodes into the same latent stand-in

    pool = ReencodeSelfCheck("", "", decode, encode=encode)
    row = pool.measure("case_a", t1c, latent, grid=(8, 8, 8))

    assert row["case"] == "case_a"
    assert row["excluded"] is None
    assert row["mae"]["clip_within"] == pytest.approx(0.0, abs=1e-6)
    assert row["mae"]["clip_over"] == pytest.approx(0.0, abs=1e-6)
    assert row["mae"]["n_over"] > 0  # the bright tail forms the extrapolated tier
    assert row["mae"]["extrapolation_max"] > 1.0  # the noclip world's tail, recorded as the tiering axis
    # the direct-chain control arm rides the same tier masks and target
    assert row["mae"]["direct_clip_within"] == pytest.approx(0.0, abs=1e-6)
    assert row["mae"]["direct_clip_over"] == pytest.approx(0.0, abs=1e-6)
    assert captured["latent"].shape == (2, 2, 2, 4)
    assert captured["encoded"].shape == (8, 8, 8)


def test_measure_without_encode_arm_leaves_direct_metrics_absent():
    rng = np.random.default_rng(3)
    t1c = rng.uniform(0.0, 100.0, size=(8, 8, 8))
    t1c.flat[:2] = 1000.0
    row = ReencodeSelfCheck("", "", lambda z: np.zeros((8, 8, 8), dtype=np.float32)).measure(
        "case_c", t1c, np.zeros((2, 2, 2, 4), dtype=np.float32), grid=(8, 8, 8)
    )
    assert "direct_clip_within" not in row["mae"]  # the control arm is opt-in


def test_measure_excludes_unusable_volume():
    flat = np.full((8, 8, 8), 3.0)  # no usable 0-99.5 percentile range
    row = ReencodeSelfCheck("", "", lambda z: z[:, :, :, 0]).measure("case_b", flat, np.zeros((2, 2, 2, 4), dtype=np.float32), grid=(8, 8, 8))
    assert row["excluded"] == "degenerate_percentile_range"


# ------------------------------------------------------------------- stride


def test_stride_keeps_position_spread_and_determinism():
    entries = [{"case": f"c{i}"} for i in range(10)]
    assert ReencodeSelfCheck.stride(entries, 3) == [entries[0], entries[3], entries[6]]
    assert ReencodeSelfCheck.stride(entries, None) == entries
    assert ReencodeSelfCheck.stride(entries, 0) == entries
    assert ReencodeSelfCheck.stride(entries, 10) == entries
    assert ReencodeSelfCheck.stride(entries, 99) == entries


# --------------------------------------------------------------------- main


def _write_t1c(path):
    rng = np.random.default_rng(11)
    volume = rng.uniform(0.0, 100.0, size=(8, 8, 8)).astype(np.float32)
    volume.flat[:2] = 1000.0  # under the 0.5% voxel share, so the 99.5p anchor stays in the bulk
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))


def _setup_job(tmp_path, n_cases=3):
    """A synthetic job tree: raw t1c files, the training list, the env/model/def
    json triple (env's embedding_base_dir points into the tmp tree), and an
    encode-runner stand-in that writes a latent-shaped embedding per entry
    (emb files only -- the real chain never writes the T7 sidecars). A sibling
    "old root" carries one valid sidecar per case, the copy source."""
    data_root = tmp_path / "raw"
    entries = []
    for i in range(n_cases):
        rel = f"case_{i}-t1c.nii.gz"
        _write_t1c(data_root / rel)
        entries.append({"image": rel, "modality": "mri"})
    train_list = tmp_path / "p1_image_only.json"
    train_list.write_text(json.dumps({"training": entries}))

    emb_root = tmp_path / "embeddings_cliptrue"
    old_root = tmp_path / "embeddings"
    for i in range(n_cases):
        stem = f"case_{i}-t1c"
        sidecar = {"spacing": [1.0, 1.0, 1.2], "modality": "t1c"}
        (old_root / f"{stem}_emb.nii.gz.json").parent.mkdir(parents=True, exist_ok=True)
        (old_root / f"{stem}_emb.nii.gz.json").write_text(json.dumps(sidecar))
    env = {
        "json_data_list": str(train_list),
        "data_base_dir": str(data_root),
        "embedding_base_dir": str(emb_root),
        "trained_autoencoder_path": "models/autoencoder_v1.pt",
    }
    env_config = tmp_path / "env.json"
    env_config.write_text(json.dumps(env))
    model_config = tmp_path / "model.json"
    model_config.write_text(json.dumps({}))
    model_def = tmp_path / "def.json"
    model_def.write_text(json.dumps({}))

    def encode_runner(env_config_path, model_config_path, model_def_path, num_gpus):
        # mirrors the real chain: encode exactly the entries of the env's list
        cfg = json.loads(Path(env_config_path).read_text())
        listing = json.loads(Path(cfg["json_data_list"]).read_text())["training"]
        for entry in listing:
            stem = entry["image"].replace(".nii.gz", "")
            out = Path(cfg["embedding_base_dir"]) / f"{stem}_emb.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4), dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0])), str(out))

    return {
        "train_list": str(train_list),
        "env_config": str(env_config),
        "model_config": str(model_config),
        "model_def": str(model_def),
        "emb_root": emb_root,
        "data_root": data_root,
        "old_root": old_root,
        "entries": entries,
        "encode_runner": encode_runner,
    }


def test_main_writes_manifest_and_report(tmp_path):
    job = _setup_job(tmp_path)

    def decode(z):
        # stand-in reconstruction: a flat mid-range volume (finite, imperfect)
        return np.full((8, 8, 8), 0.3, dtype=np.float32)

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--run-id",
        "p1-20260822T131947Z",
        "--bootstrap-b",
        "100",
        "--self-check-limit",
        "2",
    ]
    main(argv, encode_runner=job["encode_runner"], decode=decode)

    # the manifest rides the new embedding root: one row per re-encoded case
    manifest_lines = (job["emb_root"] / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 3
    assert json.loads(manifest_lines[0])["md5"]

    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    assert report["schema"] == "embedding-reencode/1"
    assert report["issue"] == 251
    assert report["variant"] == "diagnostic"
    assert report["run_id"] == "p1-20260822T131947Z"
    assert report["reencode"]["n_entries"] == 3
    assert report["reencode"]["n_train_list"] == 3
    assert report["reencode"]["missing"] == []
    check = report["self_check"]
    assert check["n_cases"] == 2  # the spot-check subsample
    for metric in ("clip_within", "clip_over", "extrapolation_max", "raw_percentile_upper"):
        assert metric in check["aggregate"]
        assert check["aggregate"][metric]["n_cases"] == 2
    # the direct-chain control arm aggregates exist but carry no readings when
    # the encode callable is not injected (the injected-decode test path)
    assert check["aggregate"]["direct_clip_within"]["n_cases"] == 0
    assert check["aggregate"]["direct_clip_over"]["n_cases"] == 0
    assert check["job_c_reference"]["direct_clip_within_median"] == 0.0062
    assert check["job_c_reference"]["artifact_within_median"] == 0.0823
    assert (tmp_path / "report" / "embedding_reencode_report.md").exists()


def test_main_records_missing_entries(tmp_path):
    """An entry the encode runner failed to write is reconciled as missing, not
    silently dropped -- the manifest count must not just echo the list length.
    The job must then FAIL nonzero (after the report lands): a tree the
    downstream training list cannot satisfy must never exit green."""
    job = _setup_job(tmp_path, n_cases=2)

    def encode_runner_missing_one(env_config_path, model_config_path, model_def_path, num_gpus):
        cfg = json.loads(Path(env_config_path).read_text())
        out = Path(cfg["embedding_base_dir"]) / "case_0-t1c_emb.nii.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4), dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0])), str(out))

    def decode(z):
        return np.full((8, 8, 8), 0.3, dtype=np.float32)

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--self-check-limit",
        "0",
    ]
    with pytest.raises(DiagnosticError, match="missing"):
        main(argv, encode_runner=encode_runner_missing_one, decode=decode)

    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    assert report["reencode"]["n_entries"] == 1
    assert report["reencode"]["n_train_list"] == 2
    assert report["reencode"]["missing"] == ["case_1-t1c.nii.gz"]
    assert report["self_check"] is None  # the spot-check was switched off


def test_main_rejects_env_root_mismatch(tmp_path):
    job = _setup_job(tmp_path)
    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--embedding-root",
        str(tmp_path / "somewhere_else"),
    ]
    with pytest.raises(DiagnosticError, match="embedding_base_dir"):
        main(argv, encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])


# ---------------------------------------------------- two-pass shard orchestration


def test_main_encode_only_stops_after_the_encode_chain(tmp_path):
    """The shard-worker pass: the encode chain runs, nothing else does -- the
    manifest and the report are the finishing pass's business."""
    job = _setup_job(tmp_path)

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--encode-only",
    ]
    main(argv, encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    assert (job["emb_root"] / "case_0-t1c_emb.nii.gz").exists()  # the chain wrote its artifacts
    assert not (job["emb_root"] / "manifest.jsonl").exists()  # but nothing else ran
    assert not (tmp_path / "report").exists()


def test_main_skip_encode_writes_report_without_the_encode_chain(tmp_path):
    """The finishing pass: the encode chain is NOT called again, the manifest
    and the report ride the artifacts the workers already wrote."""
    job = _setup_job(tmp_path)
    job["encode_runner"](job["env_config"], job["model_config"], job["model_def"], 1)  # the workers' pass, ahead of main

    def decode(z):
        return np.full((8, 8, 8), 0.3, dtype=np.float32)

    def encode_runner_must_not_run(*_args):
        raise AssertionError("the finishing pass must not invoke the encode chain")

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--self-check-limit",
        "0",
        "--skip-encode",
    ]
    main(argv, encode_runner=encode_runner_must_not_run, decode=decode)

    assert json.loads((job["emb_root"] / "manifest.jsonl").read_text().splitlines()[0])["md5"]
    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    assert report["reencode"]["n_entries"] == 3
    assert report["reencode"]["missing"] == []


def test_main_rejects_both_pass_flags(tmp_path):
    job = _setup_job(tmp_path)
    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--encode-only",
        "--skip-encode",
    ]
    with pytest.raises(DiagnosticError, match="mutually exclusive"):
        main(argv, encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])


# ------------------------------------------------- PR #299 review findings (F2/F4/F5)


def test_self_check_pool_is_t1c_only(tmp_path):
    """Job C's anchors are t1c-only readings; the spot-check pool must be the
    t1c subset of the primary list (the reconciliation stays full-list). A
    mixed-modality pool would compare t1n/t2w/t2f reconstructions against
    t1c anchors (PR #299 review F2)."""
    job = _setup_job(tmp_path, n_cases=1)
    # extend the primary list across modalities; raw files exist for all
    extra_modalities = [("case_1", "t1n"), ("case_2", "t2w"), ("case_3", "t2f")]
    entries = list(job["entries"])
    for case, mod in extra_modalities:
        rel = f"{case}-{mod}.nii.gz"
        _write_t1c(job["data_root"] / rel)
        entries.append({"image": rel, "modality": "mri"})
    Path(job["train_list"]).write_text(json.dumps({"training": entries}))

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--self-check-limit",
        "99",
    ]
    main(argv, encode_runner=job["encode_runner"], decode=lambda z: np.full((8, 8, 8), 0.3, dtype=np.float32))

    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    cases = [row["case"] for row in report["self_check"]["per_case"]]
    assert cases == ["case_0"]  # the only t1c case: t1n/t2w/t2f never enter the pool
    assert report["reencode"]["n_train_list"] == 4  # the reconciliation still covers the full list


def test_main_copies_and_validates_sidecars(tmp_path):
    """T7's loader reads <emb>.json sidecars (spacing/modality) unconditionally;
    the encode chain never writes them. With --sidecar-source the finishing
    pass copies each sidecar into the new root (PR #299 review F4), and a
    source sidecar that fails validation counts the entry missing (F3)."""
    job = _setup_job(tmp_path, n_cases=2)
    (job["old_root"] / "case_1-t1c_emb.nii.gz.json").write_text('{"spacing": "not-a-list"}')  # invalid sidecar

    argv = [
        "--train-list",
        job["train_list"],
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--sidecar-source",
        str(job["old_root"]),
        "--self-check-limit",
        "0",
    ]
    with pytest.raises(DiagnosticError, match="missing"):
        main(argv, encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    # the valid case's sidecar landed beside its embedding
    copied = json.loads((job["emb_root"] / "case_0-t1c_emb.nii.gz.json").read_text())
    assert copied["spacing"] == [1.0, 1.0, 1.2] and copied["modality"] == "t1c"
    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    assert report["reencode"]["n_entries"] == 2  # both embeddings exist on disk
    assert report["reencode"]["missing"] == ["case_1-t1c.nii.gz"]  # unusable without a valid sidecar


def test_extra_list_joins_the_reconciliation_but_not_the_pool(tmp_path):
    """T7's DataCatalog concatenates the primary list with the replay list and
    resolves both under one embedding root; the re-encode must cover both
    cohorts (PR #299 review F5). Extra entries join the manifest/reconciliation
    while the spot-check pool stays the primary list's t1c subset."""
    job = _setup_job(tmp_path, n_cases=1)
    replay_entries = []
    for i in range(2):
        rel = f"replay_{i}-t1c.nii.gz"
        _write_t1c(job["data_root"] / rel)
        replay_entries.append({"image": rel, "modality": "mri"})
    replay_list = tmp_path / "p1_mrrate_replay.json"
    replay_list.write_text(json.dumps({"training": replay_entries}))

    argv = [
        "--train-list",
        job["train_list"],
        "--extra-list",
        str(replay_list),
        "-e",
        job["env_config"],
        "-c",
        job["model_config"],
        "-t",
        job["model_def"],
        "--output-dir",
        str(tmp_path / "report"),
        "--self-check-limit",
        "0",
    ]
    main(argv, encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    assert (job["emb_root"] / "replay_0-t1c_emb.nii.gz").exists()  # the extra cohort got encoded too
    report = json.loads((tmp_path / "report" / "embedding_reencode_report.json").read_text())
    assert report["reencode"]["n_train_list"] == 1
    assert report["reencode"]["n_extra_list"] == 2
    assert report["reencode"]["n_entries"] == 3  # manifest walks the combined corpus
    assert report["reencode"]["missing"] == []
