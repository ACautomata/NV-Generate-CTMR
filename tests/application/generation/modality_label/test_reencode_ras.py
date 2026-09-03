"""The T2 re-encode job's gates (issue #312, series-③ T2).

The #312 dim-decision fix (resize target read off the RAS-reoriented shape)
invalidates every clip=True-era training embedding whose volume direction
permutes axes under reorientation; this job re-encodes the full corpus (P1
primary + MR-RATE replay + the dev cohort the P2/P3 arms consume) into a NEW
tree and guards every artifact against the expected latent shape derived from
its raw volume. The tests pin the orchestration contract on injected stand-ins
(the encode runner, the VAE decode callable) -- never a real autoencoder; the
numeric readings themselves are a sugon-side verdict.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError
from ctmr.application.generation.modality_label.reencode_ras import (
    LATENT_CHANNELS,
    EmbCohortList,
    EmbeddingShapeGuard,
    SidecarMultiSource,
    main,
)

# A volume whose direction permutes axes under RAS reorientation, with
# rounding-distinct axes: stored (96, 96, 200) under a SLA affine reorients to
# (200, 96, 96) -> rounded (256, 128, 128) -> latent (64, 32, 32, 4). The
# pre-T2 header-dim path would have produced (32, 32, 64, 4) instead.
SLA_AFFINE = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
ASYM_SHAPE = (96, 96, 200)
ASYM_EXPECTED_LATENT = (64, 32, 32, LATENT_CHANNELS)
# The round-trip-stable axis-scrambled shape the buggy chain wrote for it.
ASYM_BUGGY_LATENT = (32, 32, 64, LATENT_CHANNELS)


def _write_raw(path, shape=ASYM_SHAPE, affine=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine if affine is not None else SLA_AFFINE), str(path))


# ----------------------------------------------------------------- shape guard


def test_guard_grid_rounding_matches_the_vendored_chain_pointwise():
    """The guard restates the encode chain's grid rounding locally (the
    modality_label family imports zero infrastructure, ADR-0019 §1); the
    restatement must stay pointwise-identical to the vendored
    ``round_number`` or guard and chain would drift apart."""
    from ctmr.infrastructure.maisi_engine.create_training_data import round_number as chain_round

    guard = EmbeddingShapeGuard()
    for size in (0, 1, 32, 63, 64, 100, 127, 128, 129, 155, 192, 200, 256, 500):
        assert guard.round_to_encode_grid(size) == chain_round(size, 128), size


def test_shape_guard_derives_expected_latent_from_ras_reorientation(tmp_path):
    """The expected latent shape derives from the RAS-reoriented raw shape
    (round_number per axis, then the 4x latent downsample, channels last) --
    NOT from the header's storage order, which the axis-permuting direction
    scrambles relative to the reoriented shape."""
    raw = tmp_path / "raw" / "asym-t1n.nii.gz"
    _write_raw(raw)

    guard = EmbeddingShapeGuard()
    expected = guard.derive_expected_latent_shape(raw)

    assert expected == ASYM_EXPECTED_LATENT
    assert expected != ASYM_BUGGY_LATENT


def test_shape_guard_flags_scrambled_and_unreadable_shapes(tmp_path):
    """An embedding whose shape disagrees with the derived expectation is a
    violation carrying both sides; an unreadable artifact (manifest shape
    null) is a violation too -- never silently passed."""
    raw = tmp_path / "raw" / "asym-t1n.nii.gz"
    _write_raw(raw)

    guard = EmbeddingShapeGuard()
    shapes = {
        "scrambled-t1n_emb.nii.gz": list(ASYM_BUGGY_LATENT),
        "unreadable-t1n_emb.nii.gz": None,
    }

    rows = guard.check([("scrambled-t1n_emb.nii.gz", raw), ("unreadable-t1n_emb.nii.gz", raw)], shapes)

    assert [row["path"] for row in rows] == ["scrambled-t1n_emb.nii.gz", "unreadable-t1n_emb.nii.gz"]
    assert rows[0]["expected"] == list(ASYM_EXPECTED_LATENT)
    assert rows[0]["actual"] == list(ASYM_BUGGY_LATENT)
    assert rows[1]["reason"] == "unreadable_shape"


def test_shape_guard_flags_unreadable_raw_instead_of_crashing(tmp_path):
    """A P2/P3 reference whose raw volume is gone cannot have an expectation
    derived; the guard reports it as a violation class instead of crashing the
    finishing pass."""
    guard = EmbeddingShapeGuard()
    rows = guard.check([("ghost-t1n_emb.nii.gz", tmp_path / "raw" / "ghost-t1n.nii.gz")], {"ghost-t1n_emb.nii.gz": [2, 2, 2, 4]})

    assert rows[0]["reason"] == "unreadable_raw"
    assert rows[0]["expected"] is None


def test_shape_guard_passes_conforming_embedding(tmp_path):
    raw = tmp_path / "raw" / "asym-t1n.nii.gz"
    _write_raw(raw)

    guard = EmbeddingShapeGuard()
    rows = guard.check([("asym-t1n_emb.nii.gz", raw)], {"asym-t1n_emb.nii.gz": list(ASYM_EXPECTED_LATENT)})

    assert rows == []


# --------------------------------------------------------------- emb cohorts


def test_emb_cohort_list_resolves_image_and_src_image_under_tree_prefix(tmp_path):
    """P2/P3 list entries reference embeddings by their legacy-tree path
    (`embeddings/..._emb.nii.gz`, relative to the phase root); the cohort
    resolves each to (new-root-relative emb path, raw path) and dedupes the
    image/src_image overlap."""
    entries = [
        {"image": "embeddings/B/c1-t2w_emb.nii.gz", "src_image": "embeddings/B/c1-t1n_emb.nii.gz"},
        {"image": "embeddings/B/c1-t2w_emb.nii.gz"},  # duplicate across rows
    ]
    listing = tmp_path / "p3_pairs.json"
    listing.write_text(json.dumps({"training": entries}))

    cohort = EmbCohortList(listing)
    pairs = cohort.pairs(tmp_path / "raw_relinked")

    assert [emb_rel for emb_rel, _ in pairs] == ["B/c1-t2w_emb.nii.gz", "B/c1-t1n_emb.nii.gz"]
    assert pairs[0][1] == tmp_path / "raw_relinked" / "B" / "c1-t2w.nii.gz"


def test_emb_cohort_list_rejects_unknown_tree_prefix(tmp_path):
    listing = tmp_path / "weird.json"
    listing.write_text(json.dumps({"training": [{"image": "somewhere_else/B/c1_emb.nii.gz"}]}))

    with pytest.raises(DiagnosticError, match="embeddings"):
        EmbCohortList(listing).pairs(tmp_path / "raw_relinked")


# ------------------------------------------------------------- sidecar sources


def test_sidecar_multisource_falls_back_across_roots(tmp_path):
    """The new tree's sidecars live in different old roots per arm (primary +
    replay in the clip=True tree, the dev cohort in the legacy tree); each
    entry's sidecar is fetched from the first root that carries a valid one,
    creating any missing parent directories along the way (the encode chain
    only creates dirs under embeddings it actually wrote)."""
    emb_root = tmp_path / "embeddings_ras"
    emb_root.mkdir()
    cliptrue = tmp_path / "embeddings_cliptrue"
    legacy = tmp_path / "embeddings"
    for root, name in [(cliptrue, "replay-t1n_emb.nii.gz.json"), (legacy, "sub/dev-t1n_emb.nii.gz.json")]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps({"spacing": [1.0, 1.0, 1.2], "modality": "t1n"}))
    (legacy / "broken-t1n_emb.nii.gz.json").parent.mkdir(parents=True, exist_ok=True)
    (legacy / "broken-t1n_emb.nii.gz.json").write_text("{not json")

    copier = SidecarMultiSource(emb_root, [cliptrue, legacy])
    statuses = copier.ensure(["replay-t1n_emb.nii.gz", "sub/dev-t1n_emb.nii.gz", "broken-t1n_emb.nii.gz"])

    assert statuses["replay-t1n_emb.nii.gz"] == "copied"
    assert statuses["sub/dev-t1n_emb.nii.gz"] == "copied"  # absent from the first root, fetched from the second
    assert statuses["broken-t1n_emb.nii.gz"] == "invalid"  # no root carries a VALID sidecar
    assert json.loads((emb_root / "sub/dev-t1n_emb.nii.gz.json").read_text())["modality"] == "t1n"


# ------------------------------------------------------------------- main


def _setup_job(tmp_path, encode_shapes="expected", n_cohorts=("replay", "dev")):
    """A synthetic job tree: raw volumes (SLA, asymmetric), the P1 list + extra
    cohort lists, the P2/P3 embedding-reference lists, the env json triple, and
    an encode-runner stand-in that mirrors the fixed chain: per entry, write an
    embedding of the derived expected latent shape (encode_shapes="expected")
    or the pre-T2 scrambled shape ("buggy")."""
    data_root = tmp_path / "raw_relinked"
    entries = []
    for rel in ["case_a-t1n.nii.gz", "case_b-t2w.nii.gz"]:
        _write_raw(data_root / rel)
        entries.append({"image": rel, "modality": "mri"})
    train_list = tmp_path / "p1_image_only.json"
    train_list.write_text(json.dumps({"training": entries}))

    extra_lists = []
    for cohort in n_cohorts:
        rel = f"{cohort}_x-flair.nii.gz"
        _write_raw(data_root / rel)
        extra = tmp_path / f"{cohort}.json"
        extra.write_text(json.dumps({"training": [{"image": rel, "modality": "mri"}]}))
        extra_lists.append(str(extra))

    emb_names = [e["image"].replace(".nii.gz", "_emb.nii.gz") for e in entries]
    emb_names += [f"{c}_x-flair_emb.nii.gz" for c in n_cohorts]
    p2 = tmp_path / "p2_mask_cond.json"
    p2.write_text(json.dumps({"training": [{"image": f"embeddings/{name}"} for name in emb_names[:2]]}))
    p3 = tmp_path / "p3_pairs.json"
    p3.write_text(json.dumps({"training": [{"image": f"embeddings/{emb_names[1]}", "src_image": f"embeddings/{emb_names[0]}"}]}))

    old_roots = []
    for source_i in range(2):
        old_root = tmp_path / f"old{source_i}"
        for name in emb_names:
            sidecar = old_root / (name + ".json")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps({"spacing": [1.0, 1.0, 1.2], "modality": "t1n"}))
        old_roots.append(old_root)
    # the dev cohort's sidecar exists only in the second old root
    for cohort in n_cohorts:
        (old_roots[0] / f"{cohort}_x-flair_emb.nii.gz.json").unlink()

    emb_root = tmp_path / "embeddings_ras"
    env = {
        "json_data_list": str(train_list),
        "data_base_dir": str(data_root),
        "embedding_base_dir": str(emb_root),
        "trained_autoencoder_path": "models/autoencoder_v1.pt",
    }
    env_config = tmp_path / "env.json"
    env_config.write_text(json.dumps(env))

    def encode_runner(env_config_path, model_config_path, model_def_path, num_gpus):
        cfg = json.loads(Path(env_config_path).read_text())
        listing = json.loads(Path(cfg["json_data_list"]).read_text())["training"]
        for entry in listing:
            rel_emb = entry["image"].replace(".nii.gz", "_emb.nii.gz")
            out = Path(cfg["embedding_base_dir"]) / rel_emb
            out.parent.mkdir(parents=True, exist_ok=True)
            if encode_shapes == "expected":
                shape = EmbeddingShapeGuard().derive_expected_latent_shape(Path(cfg["data_base_dir"]) / entry["image"])
            else:
                shape = ASYM_BUGGY_LATENT
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0])), str(out))

    return {
        "train_list": str(train_list),
        "extra_lists": extra_lists,
        "emb_lists": [str(p2), str(p3)],
        "env_config": str(env_config),
        "model_config": str(tmp_path / "model.json"),
        "model_def": str(tmp_path / "def.json"),
        "emb_root": emb_root,
        "old_roots": old_roots,
        "emb_names": emb_names,
        "encode_runner": encode_runner,
    }


def _argv(job, tmp_path, sidecars=True, extra=()):
    argv = [
        "--train-list",
        job["train_list"],
        "--emb-list",
        job["emb_lists"][0],
        "--emb-list",
        job["emb_lists"][1],
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
        *extra,
    ]
    for extra_list in job["extra_lists"]:
        argv += ["--extra-list", extra_list]
    if sidecars:
        for root in job["old_roots"]:
            argv += ["--sidecar-source", str(root)]
    return argv


def test_main_writes_manifest_guard_and_report(tmp_path):
    job = _setup_job(tmp_path)

    main(_argv(job, tmp_path), encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    # all four arms encoded: 2 primary + 2 cohort entries
    for name in job["emb_names"]:
        assert (job["emb_root"] / name).exists()
    manifest_lines = (job["emb_root"] / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 4

    report = json.loads((tmp_path / "report" / "embedding_reencode_ras_report.json").read_text())
    assert report["schema"] == "embedding-reencode-ras/1"
    assert report["issue"] == 312
    assert report["variant"] == "diagnostic"
    assert report["reencode"]["n_train_list"] == 2
    assert report["reencode"]["n_extra_list"] == 2
    assert report["reencode"]["n_entries"] == 4
    assert report["reencode"]["missing"] == []
    assert report["reencode"]["shape_violations"] == []
    # the P2/P3 cohort pairs dedupe against each other AND the raw cohorts:
    # p2's two paths and p3's image+src_image all name the primary entries
    assert report["reencode"]["n_emb_cohort_paths"] == 2
    assert report["reencode"]["sidecars"]["copied"] == 4
    assert (tmp_path / "report" / "embedding_reencode_ras_report.md").exists()


def test_main_fails_on_shape_violation_after_the_report_lands(tmp_path):
    """An axis-scrambled embedding (the pre-T2 bug's shape signature) passes the
    missing-check but must fail the shape guard: report first, then a nonzero
    exit -- a tree the guard cannot bless must never exit green."""
    job = _setup_job(tmp_path, encode_shapes="buggy")

    with pytest.raises(DiagnosticError, match="shape"):
        main(_argv(job, tmp_path), encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    report = json.loads((tmp_path / "report" / "embedding_reencode_ras_report.json").read_text())
    assert report["reencode"]["missing"] == []
    assert len(report["reencode"]["shape_violations"]) == 4
    first = report["reencode"]["shape_violations"][0]
    assert first["expected"] == list(ASYM_EXPECTED_LATENT)
    assert first["actual"] == list(ASYM_BUGGY_LATENT)


def test_main_records_missing_entries(tmp_path):
    """An entry the encode runner failed to write reconciles missing (T4
    semantics) and fails the job after the report lands."""
    job = _setup_job(tmp_path)

    def runner_missing_one(env_config_path, model_config_path, model_def_path, num_gpus):
        cfg = json.loads(Path(env_config_path).read_text())
        listing = json.loads(Path(cfg["json_data_list"]).read_text())["training"]
        for entry in listing[1:]:
            rel_emb = entry["image"].replace(".nii.gz", "_emb.nii.gz")
            out = Path(cfg["embedding_base_dir"]) / rel_emb
            out.parent.mkdir(parents=True, exist_ok=True)
            shape = EmbeddingShapeGuard().derive_expected_latent_shape(Path(cfg["data_base_dir"]) / entry["image"])
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0])), str(out))

    with pytest.raises(DiagnosticError, match="missing"):
        main(_argv(job, tmp_path), encode_runner=runner_missing_one, decode=lambda z: z[:, :, :, 0])

    report = json.loads((tmp_path / "report" / "embedding_reencode_ras_report.json").read_text())
    assert report["reencode"]["missing"] == ["case_a-t1n_emb.nii.gz"]


def test_main_encode_only_stops_after_the_encode_chain(tmp_path):
    """The shard-worker pass: encode only, no manifest/guard/report."""
    job = _setup_job(tmp_path)

    main(
        _argv(job, tmp_path, sidecars=False, extra=["--encode-only"]),
        encode_runner=job["encode_runner"],
        decode=lambda z: z[:, :, :, 0],
    )

    assert (job["emb_root"] / job["emb_names"][0]).exists()
    assert not (job["emb_root"] / "manifest.jsonl").exists()
    assert not (tmp_path / "report").exists()


def test_main_skip_encode_never_invokes_the_chain(tmp_path):
    """The finishing pass rides the artifacts the shard workers wrote."""
    job = _setup_job(tmp_path)
    # the workers' pass, ahead of main: the derived combined env covers the
    # raw cohorts exactly as main's encode step would have built it
    env = json.loads(Path(job["env_config"]).read_text())
    entries = [entry for path in [job["train_list"], *job["extra_lists"]] for entry in json.loads(Path(path).read_text())["training"]]
    env["json_data_list"] = str(tmp_path / "combined.json")
    Path(env["json_data_list"]).write_text(json.dumps({"training": entries}))
    combined_env = tmp_path / "env_combined.json"
    combined_env.write_text(json.dumps(env))
    job["encode_runner"](str(combined_env), job["model_config"], job["model_def"], 1)

    def must_not_run(*_args):
        raise AssertionError("the finishing pass must not invoke the encode chain")

    main(
        _argv(job, tmp_path, extra=["--skip-encode"]),
        encode_runner=must_not_run,
        decode=lambda z: z[:, :, :, 0],
    )

    report = json.loads((tmp_path / "report" / "embedding_reencode_ras_report.json").read_text())
    assert report["reencode"]["n_entries"] == 4
    assert report["reencode"]["missing"] == []


def test_main_rejects_both_pass_flags(tmp_path):
    job = _setup_job(tmp_path)
    with pytest.raises(DiagnosticError, match="mutually exclusive"):
        main(
            _argv(job, tmp_path, sidecars=False, extra=["--encode-only", "--skip-encode"]),
            encode_runner=job["encode_runner"],
            decode=lambda z: z[:, :, :, 0],
        )


def test_main_without_sidecar_source_skips_sidecar_reconciliation(tmp_path):
    """Sidecar enforcement is opt-in (T4 semantics): with no --sidecar-source,
    the sidecar status map is absent from the report and never counts missing."""
    job = _setup_job(tmp_path)

    main(_argv(job, tmp_path, sidecars=False), encode_runner=job["encode_runner"], decode=lambda z: z[:, :, :, 0])

    report = json.loads((tmp_path / "report" / "embedding_reencode_ras_report.json").read_text())
    assert report["reencode"]["sidecars"] is None
    assert report["reencode"]["missing"] == []


def test_main_rejects_env_root_mismatch(tmp_path):
    job = _setup_job(tmp_path)
    with pytest.raises(DiagnosticError, match="embedding_base_dir"):
        main(
            _argv(job, tmp_path, sidecars=False, extra=["--embedding-root", str(tmp_path / "elsewhere")]),
            encode_runner=job["encode_runner"],
            decode=lambda z: z[:, :, :, 0],
        )
