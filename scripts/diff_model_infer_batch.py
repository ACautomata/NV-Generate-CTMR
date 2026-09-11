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

"""Task-table batch generation for the rflow-mr-brain acceptance protocol (ticket T8, issue #24).

``scripts.diff_model_infer`` generates exactly one volume per invocation and reloads
the networks every time -- workable for a handful of samples, but the frozen
pretrained baseline (spec #13 section 5.1: 10 old labels x 2 seeds x 50 volumes) and
every future snapshot's acceptance run need hundreds of volumes. This script loads
the networks once per process, walks a (label, seed, count) task table sharded
across ranks, and writes deterministically named volumes so the FID stage can build
its filelists and an interrupted run can resume by file existence.

Determinism contract (the gen-gen paired-FID requirement, spec section 5.1): every
task seeds the RNG with its own seed, so two models run over the same task table
with the same sharding produce identical noise per (label, seed, index) -- output
differences then come from the weights alone.

Usage (single GPU)::

    python -m scripts.diff_model_infer_batch \
        -e env_baseline.json -c config_maisi_diff_model_rflow-mr-brain-brats.json \
        -t config_network_rflow.json -g 1 \
        --labels 9 10 11 16 20 29 30 31 32 33 --seeds 42 1337 --count 50 \
        --output-dir runs/baseline-gen/outputs --output-prefix baseline_v1

Under ``torchrun --nproc_per_node=N`` the task table is sharded across ranks
(tasks assigned by stride, so any world size reproduces the same volumes).
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from monai.utils import set_determinism

from .diff_model_infer import load_models, run_inference, save_image
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .sample import check_input_ct


@dataclass(frozen=True)
class GenerationTask:
    """One (label, seed, count) row of the task table: ``count`` volumes of one label at one seed."""

    label: int
    seed: int
    count: int


class TaskTable:
    """The ordered (label, seed) expansion plus rank sharding of a generation plan.

    Order is label-major then seed, so any sharding of the same table assigns the
    same (label, seed, index) work regardless of how many ranks run it.
    """

    def __init__(self, tasks: list[GenerationTask]) -> None:
        if not tasks:
            raise ValueError("task table is empty")
        self._tasks = tasks

    @classmethod
    def from_args(cls, labels: list[int], seeds: list[int], count: int) -> "TaskTable":
        """Expand labels x seeds into tasks; refuses empty inputs and non-positive counts."""
        if not labels:
            raise ValueError("no labels requested")
        if not seeds:
            raise ValueError("no seeds requested")
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate labels in the task table: {sorted(labels)}")
        tasks = [GenerationTask(label=label, seed=seed, count=count) for label in labels for seed in seeds]
        return cls(tasks)

    def shard(self, rank: int, world_size: int) -> list[GenerationTask]:
        """This rank's tasks: stride assignment over the ordered table."""
        return self._tasks[rank::world_size]

    def __len__(self) -> int:
        return len(self._tasks)


class SampleNamer:
    """Deterministic output names: ``{prefix}_label{L}_seed{S}_{index:03d}.nii.gz``.

    Predictable names are what let the FID stage glob its filelists and an
    interrupted run resume by file existence.
    """

    def __init__(self, output_dir: Path, prefix: str) -> None:
        self._output_dir = output_dir
        self._prefix = prefix

    def path_for(self, label: int, seed: int, index: int) -> Path:
        return self._output_dir / f"{self._prefix}_label{label}_seed{seed}_{index:03d}.nii.gz"

    def pending_indices(self, task: GenerationTask) -> list[int]:
        """Indices of ``task`` whose volume is not on disk yet (the resume set)."""
        return [index for index in range(task.count) if not self.path_for(task.label, task.seed, index).is_file()]


class ConditioningPlan:
    """The inference conditioning tensors: size/spacing conditions once per process, modality per task.

    Mirrors ``scripts.diff_model_infer.prepare_tensors`` (values x 1e2, half precision).
    """

    def __init__(self, inference_config: dict, device: torch.device) -> None:
        self._device = device
        self.output_size = tuple(inference_config["dim"])
        self.out_spacing = tuple(inference_config["spacing"])
        self._top_region_index = self._conditioning_tensor(inference_config["top_region_index"])
        self._bottom_region_index = self._conditioning_tensor(inference_config["bottom_region_index"])
        self._spacing = self._conditioning_tensor(inference_config["spacing"])

    def _conditioning_tensor(self, values: list[float]) -> torch.Tensor:
        array = torch.asarray(values, dtype=torch.float32) * 1e2
        return array.unsqueeze(0).half().to(self._device)

    def tensors(self, label: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(top_region_index, bottom_region_index, spacing, modality) for one label."""
        modality = label * torch.ones((1,), dtype=torch.long, device=self._device)
        return self._top_region_index, self._bottom_region_index, self._spacing, modality


class BaselineVolumeGenerator:
    """Runs the task table on one rank: networks loaded once, volumes saved per deterministic name."""

    def __init__(self, args: argparse.Namespace, device: torch.device, logger: logging.Logger, namer: SampleNamer) -> None:
        self._args = args
        self._device = device
        self._logger = logger
        self._namer = namer
        self._autoencoder, self._unet, self._scale_factor = load_models(args, device, logger)
        num_downsample_level = max(
            1,
            (
                len(args.diffusion_unet_def["num_channels"])
                if isinstance(args.diffusion_unet_def["num_channels"], list)
                else len(args.diffusion_unet_def["attention_levels"])
            ),
        )
        self._divisor = 2 ** (num_downsample_level - 2)

    def run_task(self, task: GenerationTask, plan: ConditioningPlan) -> int:
        """Generate the task's still-missing volumes; returns how many were written."""
        set_determinism(task.seed)
        if 1 <= task.label <= 7:
            check_input_ct(None, None, None, plan.output_size, plan.out_spacing, None)
        written = 0
        for index in self._namer.pending_indices(task):
            data = run_inference(
                self._args,
                self._device,
                self._autoencoder,
                self._unet,
                self._scale_factor,
                *plan.tensors(task.label),
                plan.output_size,
                self._divisor,
                self._logger,
            )
            save_image(data, plan.output_size, plan.out_spacing, str(self._namer.path_for(task.label, task.seed, index)), self._logger)
            written += 1
        return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config", type=str, required=True)
    parser.add_argument("-c", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--model_def", type=str, required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1, help="number of GPUs (torchrun ranks) sharding the task table")
    parser.add_argument(
        "--labels", type=int, nargs="+", required=True, help="modality labels to generate (rflow-mr-brain old labels: 9 10 11 16 20 29 30 31 32 33)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="generation seeds, e.g. 42 1337 (spec section 5.1 pairing)")
    parser.add_argument("--count", type=int, default=50, help="volumes per (label, seed) task (default 50)")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the generated volumes")
    parser.add_argument("--output-prefix", required=True, help="filename prefix of the generated volumes")
    args = parser.parse_args()

    config = load_config(args.env_config, args.model_config, args.model_def)
    local_rank, world_size, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("baseline_generation")
    config.cfg_guidance_scale = config.diffusion_unet_inference["cfg_guidance_scale"]

    table = TaskTable.from_args(labels=args.labels, seeds=args.seeds, count=args.count)
    namer = SampleNamer(args.output_dir, args.output_prefix)
    plan = ConditioningPlan(config.diffusion_unet_inference, device)
    generator = BaselineVolumeGenerator(config, device, logger, namer)

    my_tasks = table.shard(local_rank, world_size)
    logger.info(f"[rank {local_rank}/{world_size}] {len(my_tasks)}/{len(table)} tasks, {args.count} volumes each")
    for task in my_tasks:
        written = generator.run_task(task, plan)
        logger.info(f"[rank {local_rank}] task label={task.label} seed={task.seed}: wrote {written}/{task.count}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
