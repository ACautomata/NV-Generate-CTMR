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

"""Tests for scripts.run_acceptance_suite (ticket T10, issue #28).

The orchestrator is driven by the T10 task table (18 snapshots x 44 FID
comparisons). Tests pin the three things that must never drift: the two
generation invocations carry the right labels/config/prefixes per side, the
44 comparisons route gen-real/BRATS through the GPU tool with ``--cache_only
real`` while gen-gen goes through the CPU feature-only tool, and a comparison
whose result json already exists is skipped (the suite resumes, it does not
recompute). All subprocess calls are mocked; a dry-run flag lets the real
suite be eyeballed on the cluster without GPUs.
"""

import json
from pathlib import Path

import pytest
import torch

from scripts.run_acceptance_suite import AcceptanceCommandPlanner, AcceptanceOrchestrator, SubprocessExecutor, ValFullFeatureAssembler

BRAINS_CONFIG = "configs/config_maisi_diff_model_rflow-mr-brain.json"
BRATS_CONFIG = "configs/config_maisi_diff_model_rflow-mr-brain-brats.json"
NETWORK_CONFIG = "configs/config_network_rflow.json"
def _write_lines(path: str, count: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"case{i:03d}/vol.nii.gz\n" for i in range(count)))


def make_task_table(tmp_path: Path, epoch: int = 50) -> dict:
    run_root = tmp_path / "runs" / "t10"
    features_root = run_root / "freeze" / "features"
    for rel, count in (
        ("freeze/fid_filelists/ref/mri_t1.txt", 50),
        ("freeze/fid_filelists/ref/mri_t2.txt", 50),
        ("freeze/fid_filelists/ref/mri_flair.txt", 50),
        ("freeze/fid_filelists/ref/mri_mra.txt", 2),
        ("freeze/fid_filelists/ref/mri_swi.txt", 50),
        ("freeze/fid_filelists/synth/label9_seed42.txt", 50),
        ("freeze/fid_filelists/synth/label9_seed1337.txt", 50),
        ("freeze/floor-split/filelist_val64_mri_t1n.txt", 64),
    ):
        _write_lines(str(run_root / rel), count)
    for rel in (
        "N300/e050/fid_filelists/synth/ftN300e050_label9_seed42.txt",
        "N300/e050/fid_filelists/synth/ftN300e050_label9_seed1337.txt",
        "N300/e050/fid_filelists/synth/bratsftN300e050_label40_seed42.txt",
    ):
        _write_lines(str(run_root / rel), 50)
    comparisons = [
        {
            "tag": "genreal_label9_seed42",
            "kind": "gen_real",
            "label": 9,
            "seed": 42,
            "real_dataset_root": "/gone/scratch/reference/data",
            "real_filelist": str(run_root / "freeze/fid_filelists/ref/mri_t1.txt"),
            "real_features_dir": "ref_label9",
            "synth_dataset_root": str(run_root / f"N300/e{epoch:03d}/outputs_forgetting"),
            "synth_filelist": str(run_root / f"N300/e{epoch:03d}/fid_filelists/synth/ftN300e050_label9_seed42.txt"),
            "synth_features_dir": "ftN300e050_label9_seed42",
            "num_images": 50,
            "result_json": str(run_root / f"N300/e{epoch:03d}/fid/genreal_label9_seed42.json"),
        },
        {
            "tag": "gengen_label9_seed42",
            "kind": "gen_gen",
            "label": 9,
            "seed": 42,
            "real_dataset_root": "/gone/scratch/baseline/outputs",
            "real_filelist": str(run_root / "freeze/fid_filelists/synth/label9_seed42.txt"),
            "real_features_dir": "synth_label9_seed42",
            "synth_dataset_root": str(run_root / f"N300/e{epoch:03d}/outputs_forgetting"),
            "synth_filelist": str(run_root / f"N300/e{epoch:03d}/fid_filelists/synth/ftN300e050_label9_seed42.txt"),
            "synth_features_dir": "ftN300e050_label9_seed42",
            "num_images": 50,
            "result_json": str(run_root / f"N300/e{epoch:03d}/fid/gengen_label9_seed42.json"),
        },
        {
            "tag": "brats_label40_seed42",
            "kind": "brats",
            "label": 40,
            "seed": 42,
            "real_dataset_root": "/gone/scratch/datasets",
            "real_filelist": str(run_root / "freeze/floor-split/filelist_val64_mri_t1n.txt"),
            "real_features_dir": "val_full_mri_t1n",
            "synth_dataset_root": str(run_root / f"N300/e{epoch:03d}/outputs_brats"),
            "synth_filelist": str(run_root / f"N300/e{epoch:03d}/fid_filelists/synth/bratsftN300e050_label40_seed42.txt"),
            "synth_features_dir": "bratsftN300e050_label40_seed42",
            "num_images": 50,
            "result_json": str(run_root / f"N300/e{epoch:03d}/fid/brats_label40_seed42.json"),
        },
    ]
    snapshot = {
        "run_id": f"N300e{epoch:03d}",
        "tier": 300,
        "epoch": epoch,
        "ckpt_path": str(run_root / "models" / "ckpt_epoch50.pt"),
        "env": str(run_root / f"N300/e{epoch:03d}/env_forgetting.json"),
        "outputs_forgetting": str(run_root / f"N300/e{epoch:03d}/outputs_forgetting"),
        "outputs_brats": str(run_root / f"N300/e{epoch:03d}/outputs_brats"),
        "generation_prefix": "ftN300e050",
        "brats_generation_prefix": "bratsftN300e050",
        "comparisons": comparisons,
    }
    return {
        "protocol": {
            "model_name": "radimagenet_resnet50",
            "modality": "mr",
            "center_slices_ratio": 0.4,
            "padding": True,
            "center_cropping": True,
            "resampling_spacing": "1.0x1.0x1.0",
            "target_shape": "256x256x128",
            "num_images": 50,
            "decoder_roi": "40x40x16",
        },
        "assets": {"features_root": str(features_root), "fid_freeze_json": str(run_root / "freeze" / "fid_freeze.json")},
        "snapshots": [snapshot],
    }


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs) -> None:
        self.calls.append(command)


OLD_LABELS = [9, 10, 11, 16, 20, 29, 30, 31, 32, 33]
BRATS_LABELS = [40, 41, 42, 43]
SEEDS = [42, 1337]


@pytest.fixture
def planner(tmp_path: Path) -> AcceptanceCommandPlanner:
    return AcceptanceCommandPlanner(
        task_table=make_task_table(tmp_path),
        code_root=Path("/opt/nv-ctmr"),
        num_gpus=8,
        forgetting_config=Path(BRAINS_CONFIG),
        brats_config=Path(BRATS_CONFIG),
        network_config=Path(NETWORK_CONFIG),
    )


class TestValFullFeatureAssembler:
    def test_merges_half_trees_and_writes_val64_filelist(self, tmp_path: Path) -> None:
        features = tmp_path / "features" / "mr"
        half_a = ["brats2023-gli/data/caseA/img/caseA-t1n.nii.gz"]
        half_b = ["brats2023-gli/data/caseB/img/caseB-t1n.nii.gz"]
        for side, rels in (("floor_a_mri_t1n", half_a), ("floor_b_mri_t1n", half_b)):
            for rel in rels:
                target = features / side / rel.replace(".nii.gz", ".pt")
                target.parent.mkdir(parents=True, exist_ok=True)
                torch_save_marker(target, rel)
        (tmp_path / "split").mkdir()
        (tmp_path / "split" / "filelist_half_a_mri_t1n.txt").write_text("\n".join(half_a) + "\n")
        (tmp_path / "split" / "filelist_half_b_mri_t1n.txt").write_text("\n".join(half_b) + "\n")

        assembler = ValFullFeatureAssembler(features_root=tmp_path / "features", floor_split=tmp_path / "split")
        assembler.assemble("mri_t1n")

        merged = features / "val_full_mri_t1n"
        assert (merged / "brats2023-gli/data/caseA/img/caseA-t1n.pt").exists()
        assert (merged / "brats2023-gli/data/caseB/img/caseB-t1n.pt").exists()
        filelist = (tmp_path / "split" / "filelist_val64_mri_t1n.txt").read_text().splitlines()
        assert filelist == sorted(half_a + half_b)

    def test_is_idempotent(self, tmp_path: Path) -> None:
        features = tmp_path / "features" / "mr"
        rel = "brats2023-gli/data/caseA/img/caseA-t1n.nii.gz"
        target = features / "floor_a_mri_t1n" / rel.replace(".nii.gz", ".pt")
        target.parent.mkdir(parents=True)
        torch_save_marker(target, "original")
        (features / "floor_b_mri_t1n").mkdir(parents=True)
        (tmp_path / "split").mkdir()
        (tmp_path / "split" / "filelist_half_a_mri_t1n.txt").write_text(rel + "\n")
        (tmp_path / "split" / "filelist_half_b_mri_t1n.txt").write_text("")

        assembler = ValFullFeatureAssembler(features_root=tmp_path / "features", floor_split=tmp_path / "split")
        assembler.assemble("mri_t1n")
        merged_file = features / "val_full_mri_t1n" / "brats2023-gli/data/caseA/img/caseA-t1n.pt"

        torch_save_marker(merged_file, "mutated")
        assembler.assemble("mri_t1n")  # second run must not overwrite

        assert torch.load(merged_file, weights_only=True) == "mutated"

    def test_missing_floor_half_is_refused(self, tmp_path: Path) -> None:
        assembler = ValFullFeatureAssembler(features_root=tmp_path / "features", floor_split=tmp_path / "split")
        with pytest.raises(FileNotFoundError, match="floor_a_mri_t1n"):
            assembler.assemble("mri_t1n")


class TestGenerationCommands:
    def test_forgetting_side_uses_native_config_and_old_labels(self, planner: AcceptanceCommandPlanner) -> None:
        snapshot = planner.snapshots[0]
        command = planner.generation_command(snapshot, forgetting_side=True)

        assert command[:4] == ["torchrun", "--nproc_per_node=8", "--nnodes=1", "--master_addr=localhost"]
        assert "-m" in command and "scripts.diff_model_infer_batch" in command[command.index("-m") + 1]
        joined = " ".join(command)
        assert BRAINS_CONFIG in joined
        assert BRATS_CONFIG not in joined
        assert "--seeds 42 1337" in joined
        assert "--count 50" in joined
        assert f"--output-prefix {snapshot['generation_prefix']}" in joined
        assert snapshot["outputs_forgetting"] in joined
        assert "--decoder-roi 40x40x16" in joined

    def test_brats_side_uses_brats_config_and_new_labels(self, planner: AcceptanceCommandPlanner) -> None:
        snapshot = planner.snapshots[0]
        command = planner.generation_command(snapshot, forgetting_side=False)

        joined = " ".join(command)
        assert BRATS_CONFIG in joined
        assert BRAINS_CONFIG not in joined
        assert "--labels 40 41 42 43" in joined
        assert f"--output-prefix {snapshot['brats_generation_prefix']}" in joined
        assert snapshot["outputs_brats"] in joined

    def test_label_list_is_complete(self, planner: AcceptanceCommandPlanner) -> None:
        snapshot = planner.snapshots[0]
        forgetting = planner.generation_command(snapshot, forgetting_side=True)
        brats = planner.generation_command(snapshot, forgetting_side=False)

        assert forgetting[forgetting.index("--labels") + 1 : forgetting.index("--labels") + 11] == [str(x) for x in OLD_LABELS]
        assert brats[brats.index("--labels") + 1 : brats.index("--labels") + 5] == [str(x) for x in BRATS_LABELS]


class TestFidCommands:
    def test_gen_real_routes_through_gpu_tool_with_cache_only_real(self, planner: AcceptanceCommandPlanner) -> None:
        comparison = planner.snapshots[0]["comparisons"][0]
        command = planner.fid_command(comparison)

        joined = " ".join(command)
        assert "scripts.compute_fid_2-5d_ct" in command
        assert "--cache_only real" in joined
        assert f"--real_features_dir {comparison['real_features_dir']}" in joined
        assert f"--synth_features_dir {comparison['synth_features_dir']}" in joined
        assert f"--comparison_tag {comparison['tag']}" in joined
        assert f"--result_json {comparison['result_json']}" in joined
        assert "--modality mr" in joined
        assert "--target_shape 256x256x128" in joined
        assert "--enable_resampling_spacing 1.0x1.0x1.0" in joined
        assert f"--output_root {planner.features_root}" in joined

    def test_gen_gen_routes_through_cpu_feature_tool(self, planner: AcceptanceCommandPlanner) -> None:
        comparison = planner.snapshots[0]["comparisons"][1]
        command = planner.fid_command(comparison)

        assert "scripts.compute_fid_from_features" in command
        assert "torchrun" not in command
        assert f"--real_features_dir {planner.features_root}/mr/{comparison['real_features_dir']}" in " ".join(command)
        assert f"--synth_features_dir {planner.features_root}/mr/{comparison['synth_features_dir']}" in " ".join(command)

    def test_brats_routes_through_gpu_tool(self, planner: AcceptanceCommandPlanner) -> None:
        comparison = planner.snapshots[0]["comparisons"][2]
        command = planner.fid_command(comparison)

        assert "scripts.compute_fid_2-5d_ct" in command
        assert "--cache_only real" in " ".join(command)


class TestOrchestrator:
    def test_skips_comparison_with_existing_result_json(self, tmp_path: Path) -> None:
        table = make_task_table(tmp_path)
        done = Path(table["snapshots"][0]["comparisons"][0]["result_json"])
        done.parent.mkdir(parents=True)
        done.write_text("{}")

        seen: list[str] = []

        def fake_run(command: list[str], **kwargs) -> None:
            seen.append(" ".join(command))

        orchestrator = AcceptanceOrchestrator(
            task_table=table,
            code_root=tmp_path,
            num_gpus=1,
            executor=fake_run,
        )
        orchestrator.run_snapshot(table["snapshots"][0], generate=False)

        assert not any("genreal_label9_seed42" in call for call in seen)
        assert any("gengen_label9_seed42" in call for call in seen)
        assert any("brats_label40_seed42" in call for call in seen)

    def test_generation_skipped_when_all_volumes_present(self, tmp_path: Path) -> None:
        table = make_task_table(tmp_path)
        snapshot = table["snapshots"][0]
        for directory, labels, prefix in (
            (snapshot["outputs_forgetting"], OLD_LABELS, snapshot["generation_prefix"]),
            (snapshot["outputs_brats"], BRATS_LABELS, snapshot["brats_generation_prefix"]),
        ):
            for label in labels:
                for seed in SEEDS:
                    for index in range(50):
                        path = Path(directory) / f"{prefix}_label{label}_seed{seed}_{index:03d}.nii.gz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"stub")

        calls: list[str] = []

        def fake_run(command: list[str], **kwargs) -> None:
            calls.append(" ".join(command))

        planner = AcceptanceCommandPlanner(
            task_table=table,
            code_root=tmp_path,
            num_gpus=1,
            forgetting_config=Path(BRAINS_CONFIG),
            brats_config=Path(BRATS_CONFIG),
            network_config=Path(NETWORK_CONFIG),
        )
        assert planner.generation_complete(snapshot, forgetting_side=True)
        assert planner.generation_complete(snapshot, forgetting_side=False)

    def test_dry_run_prints_without_executing(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        table = make_task_table(tmp_path)
        executed: list[str] = []

        orchestrator = AcceptanceOrchestrator(
            task_table=table,
            code_root=tmp_path,
            num_gpus=1,
            executor=lambda command, **kw: executed.append(command),
            dry_run=True,
        )
        orchestrator.run_snapshot(table["snapshots"][0], generate=False)

        assert executed == []
        assert "gengen_label9_seed42" in capsys.readouterr().out


def torch_save_marker(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(marker, path)


def test_real_subprocess_executor_runs_and_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    executor = SubprocessExecutor(code_root=tmp_path, log_dir=log_dir)

    executor(["echo", "hello"], check=True)

    logs = list(log_dir.glob("*.log"))
    assert len(logs) == 1
    assert "hello" in logs[0].read_text()
