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

"""Drive the per-snapshot acceptance suite from the T10 task table (ticket T10, issue #28).

For each of the 18 snapshots (3 tiers x 6 epochs) the protocol of spec #13 §⑤
is: generate the forgetting side (10 old labels x 2 seeds x 50 volumes, native
v1 conditions), generate the BRATS side (4 new labels, BRATS geometry), then
compute 44 FID comparisons. The frozen-reference side of every comparison
lives in the T8/T17 feature cache, so gen-real and BRATS comparisons run the
GPU extractor with ``--cache_only real`` (only the finetuned synth side is
extracted) while gen-gen comparisons are frozen features on both sides and run
on the CPU feature-only tool.

Resumability is by artifact existence: a generation stage whose volume count
is complete is skipped, a comparison whose result json exists is skipped. The
suite may be interrupted and relaunched at any point; volume-level determinism
comes from ``diff_model_infer_batch``'s seeding contract, not from this driver.

The task table's ``real_dataset_root`` entries name volumes on an expired
scratch disk and are carried as opaque strings only -- cache-only real sides
never touch them.

Usage (on the GPU node)::

    python -m scripts.run_acceptance_suite \
        --task_table runs/t10-acceptance-20260920/task_table.json \
        --code_root /root/nv-ctmr --num_gpus 8 \
        [--snapshots N300e050 N300e100 ...] [--dry_run]
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FORGETTING_LABELS = [9, 10, 11, 16, 20, 29, 30, 31, 32, 33]
BRATS_GENERATION_LABELS = [40, 41, 42, 43]
GENERATION_SEEDS = [42, 1337]
VAL_FULL_LABELS = ("mri_t1n", "mri_t1ce", "mri_t2w", "mri_t2f")
GPU_TOOL = "scripts.compute_fid_2-5d_ct"
CPU_TOOL = "scripts.compute_fid_from_features"


class ValFullFeatureAssembler:
    """Builds the BRATS val64 full-reference feature tree from the frozen floor halves.

    The floor split (T8) divides BRATS val 64 into two case-disjoint halves and
    froze each half's features separately; the BRATS comparisons of T10 need the
    64-case whole as one reference. Merging the two trees reconstructs it
    losslessly, and the merged filelist is the union of the two half filelists.
    Existing outputs are never rewritten: the tree is a frozen reference, and
    reassembly must stay idempotent across suite resumes.
    """

    def __init__(self, features_root: Path, floor_split: Path) -> None:
        self._features_root = Path(features_root)
        self._floor_split = Path(floor_split)

    def assemble_labels(self, label_names: tuple[str, ...] = VAL_FULL_LABELS) -> None:
        for label_name in label_names:
            self.assemble(label_name)

    def assemble(self, label_name: str) -> None:
        merged_dir = self._features_root / "mr" / f"val_full_{label_name}"
        if merged_dir.is_dir() and any(merged_dir.rglob("*.pt")):
            return

        sources = []
        filelist_lines = []
        for half in ("a", "b"):
            half_dir = self._features_root / "mr" / f"floor_{half}_{label_name}"
            if not half_dir.is_dir():
                raise FileNotFoundError(f"floor half tree missing: {half_dir}")
            half_list = self._floor_split / f"filelist_half_{half}_{label_name}.txt"
            if not half_list.is_file():
                raise FileNotFoundError(f"floor half filelist missing: {half_list}")
            sources.append((half_dir, [line.strip() for line in half_list.read_text().splitlines() if line.strip()]))
            filelist_lines.extend(line.strip() for line in half_list.read_text().splitlines() if line.strip())

        merged_dir.mkdir(parents=True, exist_ok=True)
        for half_dir, rels in sources:
            for rel in rels:
                source = half_dir / rel.replace(".nii.gz", ".pt")
                if not source.is_file():
                    raise FileNotFoundError(f"floor feature missing for {rel}: {source}")
                target = merged_dir / rel.replace(".nii.gz", ".pt")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())

        val64_list = self._floor_split / f"filelist_val64_{label_name}.txt"
        val64_list.write_text("\n".join(sorted(filelist_lines)) + "\n")


class SubprocessExecutor:
    """Runs one staged command with its output captured to a timestamped log."""

    def __init__(self, code_root: Path, log_dir: Path) -> None:
        self._code_root = Path(code_root)
        self._log_dir = Path(log_dir)

    def __call__(self, command: list[str], check: bool = True) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = self._log_dir / f"{stamp}-{Path(command[0]).name}.log"
        with log_path.open("w") as log_file:
            result = subprocess.run(command, cwd=self._code_root, check=check, stdout=log_file, stderr=subprocess.STDOUT)
        log_path.write_text(log_path.read_text() + f"\n[exit {result.returncode}] {' '.join(command)}\n")


class AcceptanceCommandPlanner:
    """Translates task-table entries into the concrete commands of the protocol."""

    def __init__(
        self,
        task_table: dict,
        code_root: Path,
        num_gpus: int,
        forgetting_config: Path,
        brats_config: Path,
        network_config: Path,
    ) -> None:
        self._protocol = task_table["protocol"]
        self._features_root = Path(task_table["assets"]["features_root"])
        self.snapshots = task_table["snapshots"]
        self._code_root = Path(code_root)
        self._num_gpus = num_gpus
        self._forgetting_config = Path(forgetting_config)
        self._brats_config = Path(brats_config)
        self._network_config = Path(network_config)

    @property
    def features_root(self) -> Path:
        return self._features_root

    def generation_command(self, snapshot: dict, forgetting_side: bool) -> list[str]:
        config = self._forgetting_config if forgetting_side else self._brats_config
        labels = FORGETTING_LABELS if forgetting_side else BRATS_GENERATION_LABELS
        output_dir = snapshot["outputs_forgetting"] if forgetting_side else snapshot["outputs_brats"]
        prefix = snapshot["generation_prefix"] if forgetting_side else snapshot["brats_generation_prefix"]

        return [
            "torchrun",
            f"--nproc_per_node={self._num_gpus}",
            "--nnodes=1",
            "--master_addr=localhost",
            f"--master_port={random.randint(20000, 39999)}",
            "-m",
            "scripts.diff_model_infer_batch",
            "-e",
            snapshot["env"],
            "-c",
            str(config),
            "-t",
            str(self._network_config),
            "-g",
            str(self._num_gpus),
            "--labels",
            *[str(label) for label in labels],
            "--seeds",
            *[str(seed) for seed in GENERATION_SEEDS],
            "--count",
            str(self._protocol["num_images"]),
            "--output-dir",
            output_dir,
            "--output-prefix",
            prefix,
            "--decoder-roi",
            self._protocol["decoder_roi"],
        ]

    def generation_complete(self, snapshot: dict, forgetting_side: bool) -> bool:
        labels = FORGETTING_LABELS if forgetting_side else BRATS_GENERATION_LABELS
        directory = Path(snapshot["outputs_forgetting"] if forgetting_side else snapshot["outputs_brats"])
        if not directory.is_dir():
            return False
        prefix = snapshot["generation_prefix"] if forgetting_side else snapshot["brats_generation_prefix"]
        expected = len(labels) * len(GENERATION_SEEDS) * self._protocol["num_images"]
        found = sum(1 for _ in directory.glob(f"{prefix}_label*_seed*_*.nii.gz"))
        return found >= expected

    def fid_command(self, comparison: dict) -> list[str]:
        if comparison["kind"] == "gen_gen":
            return self._cpu_fid_command(comparison)
        return self._gpu_fid_command(comparison)

    def _gpu_fid_command(self, comparison: dict) -> list[str]:
        return [
            "torchrun",
            f"--nproc_per_node={self._num_gpus}",
            "--nnodes=1",
            "--master_addr=localhost",
            f"--master_port={random.randint(20000, 39999)}",
            "-m",
            GPU_TOOL,
            "--real_dataset_root",
            comparison["real_dataset_root"],
            "--real_filelist",
            comparison["real_filelist"],
            "--real_features_dir",
            comparison["real_features_dir"],
            "--synth_dataset_root",
            comparison["synth_dataset_root"],
            "--synth_filelist",
            comparison["synth_filelist"],
            "--synth_features_dir",
            comparison["synth_features_dir"],
            "--num_images",
            str(comparison["num_images"]),
            "--output_root",
            str(self._features_root),
            "--model_name",
            self._protocol["model_name"],
            "--modality",
            self._protocol["modality"],
            "--target_shape",
            self._protocol["target_shape"],
            "--enable_center_slices_ratio",
            str(self._protocol["center_slices_ratio"]),
            "--enable_padding=True" if self._protocol["padding"] else "--enable_padding=False",
            "--enable_center_cropping=True" if self._protocol["center_cropping"] else "--enable_center_cropping=False",
            "--enable_resampling_spacing",
            self._protocol["resampling_spacing"],
            "--comparison_tag",
            comparison["tag"],
            "--result_json",
            comparison["result_json"],
            "--cache_only",
            "real",
        ]

    def _cpu_fid_command(self, comparison: dict) -> list[str]:
        return [
            sys.executable,
            "-m",
            CPU_TOOL,
            "--real_features_dir",
            str(self._features_root / self._protocol["modality"] / comparison["real_features_dir"]),
            "--synth_features_dir",
            str(self._features_root / self._protocol["modality"] / comparison["synth_features_dir"]),
            "--comparison_tag",
            comparison["tag"],
            "--num_images",
            str(comparison["num_images"]),
            "--real_filelist",
            comparison["real_filelist"],
            "--synth_filelist",
            comparison["synth_filelist"],
            "--modality",
            self._protocol["modality"],
            "--model_name",
            self._protocol["model_name"],
            "--target_shape",
            self._protocol["target_shape"],
            "--center_slices_ratio",
            str(self._protocol["center_slices_ratio"]),
            "--enable_padding",
            str(self._protocol["padding"]),
            "--enable_center_cropping",
            str(self._protocol["center_cropping"]),
            "--enable_resampling_spacing",
            self._protocol["resampling_spacing"],
            "--result_json",
            comparison["result_json"],
        ]


class AcceptanceOrchestrator:
    """Resumable driver: per snapshot, generate both sides then run every comparison."""

    def __init__(
        self,
        task_table: dict,
        code_root: Path,
        num_gpus: int,
        forgetting_config: Path = Path("configs/config_maisi_diff_model_rflow-mr-brain.json"),
        brats_config: Path = Path("configs/config_maisi_diff_model_rflow-mr-brain-brats.json"),
        network_config: Path = Path("configs/config_network_rflow.json"),
        executor=None,
        dry_run: bool = False,
    ) -> None:
        self._planner = AcceptanceCommandPlanner(
            task_table=task_table,
            code_root=code_root,
            num_gpus=num_gpus,
            forgetting_config=forgetting_config,
            brats_config=brats_config,
            network_config=network_config,
        )
        self._executor = executor
        self._dry_run = dry_run
        self._floor_split = Path(task_table["assets"].get("floor_split", Path(task_table["assets"]["features_root"]).parent / "floor-split"))
        self._features_root = Path(task_table["assets"]["features_root"])

    def run_all(self, only_run_ids: list[str] | None = None, generate: bool = True) -> None:
        ValFullFeatureAssembler(features_root=self._features_root, floor_split=self._floor_split).assemble_labels()
        snapshots = self._planner.snapshots
        if only_run_ids is not None:
            snapshots = [s for s in snapshots if s["run_id"] in only_run_ids]
        for snapshot in snapshots:
            self.run_snapshot(snapshot, generate=generate)

    def run_snapshot(self, snapshot: dict, generate: bool = True) -> None:
        print(f"=== snapshot {snapshot['run_id']} (tier {snapshot['tier']}, epoch {snapshot['epoch']}) ===", flush=True)
        if generate:
            for forgetting_side in (True, False):
                side_name = "forgetting" if forgetting_side else "brats"
                if self._planner.generation_complete(snapshot, forgetting_side):
                    print(f"  generation {side_name}: already complete", flush=True)
                else:
                    self._execute(self._planner.generation_command(snapshot, forgetting_side))

        for comparison in snapshot["comparisons"]:
            if Path(comparison["result_json"]).exists():
                print(f"  fid {comparison['tag']}: already done", flush=True)
                continue
            self._execute(self._planner.fid_command(comparison))

    def _execute(self, command: list[str]) -> None:
        printable = " ".join(command)
        if self._dry_run:
            print(f"  [dry-run] {printable}", flush=True)
            return
        print(f"  run: {printable}", flush=True)
        self._executor(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_table", type=Path, required=True)
    parser.add_argument("--code_root", type=Path, default=Path.cwd())
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--forgetting_config", type=Path, default=Path("configs/config_maisi_diff_model_rflow-mr-brain.json"))
    parser.add_argument("--brats_config", type=Path, default=Path("configs/config_maisi_diff_model_rflow-mr-brain-brats.json"))
    parser.add_argument("--network_config", type=Path, default=Path("configs/config_network_rflow.json"))
    parser.add_argument("--log_dir", type=Path, default=None, help="where executor logs go (default: <task_table dir>/logs)")
    parser.add_argument("--snapshots", nargs="*", default=None, help="run_ids to process (default: all, in table order)")
    parser.add_argument("--no_generate", action="store_true", help="skip generation stages (FID only)")
    parser.add_argument("--dry_run", action="store_true", help="print commands without executing")
    args = parser.parse_args()

    with open(args.task_table) as file:
        task_table = json.load(file)

    executor = None
    if not args.dry_run:
        log_dir = args.log_dir or args.task_table.parent / "logs"
        executor = SubprocessExecutor(code_root=args.code_root, log_dir=log_dir)

    orchestrator = AcceptanceOrchestrator(
        task_table=task_table,
        code_root=args.code_root,
        num_gpus=args.num_gpus,
        forgetting_config=args.forgetting_config,
        brats_config=args.brats_config,
        network_config=args.network_config,
        executor=executor,
        dry_run=args.dry_run,
    )
    orchestrator.run_all(only_run_ids=args.snapshots, generate=not args.no_generate)


if __name__ == "__main__":
    main()
