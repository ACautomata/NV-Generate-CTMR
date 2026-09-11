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

"""Tests for the batch generation task table and output naming (ticket T8, issue #24).

The frozen pretrained baseline needs 10 labels x 2 seeds x 50 volumes; every future
snapshot's acceptance run reuses the same machinery. The seams here are the pure
parts the GPU loop consumes: the task-table expansion and rank sharding (which must
agree across any world size so a run can be interrupted and relaunched on a
different number of free GPUs), and the deterministic naming/resume contract the
FID stage's filelists are built from. The GPU loop itself is not unit-tested.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from monai.utils import set_determinism

from scripts.diff_model_infer_batch import BaselineVolumeGenerator, ConditioningPlan, GenerationTask, SampleNamer, TaskTable

LATENT_CHANNELS = 4
INFERENCE_CONFIG = {
    "dim": [256, 256, 128],
    "spacing": [0.94, 0.94, 1.36],
    "top_region_index": [0, 1, 0, 0],
    "bottom_region_index": [0, 0, 1, 0],
}
# 8 downsample levels -> divisor 2 ** 6 -> the latent of a 256x256x128 volume.
LATENT_SHAPE = (1, LATENT_CHANNELS, 4, 4, 2)


class TestTaskTable:
    def test_expands_labels_x_seeds_in_label_major_order(self) -> None:
        table = TaskTable.from_args(labels=[9, 10], seeds=[42, 1337], count=50)

        assert len(table) == 4
        assert table.shard(0, 1) == [
            GenerationTask(label=9, seed=42, count=50),
            GenerationTask(label=9, seed=1337, count=50),
            GenerationTask(label=10, seed=42, count=50),
            GenerationTask(label=10, seed=1337, count=50),
        ]

    def test_ten_labels_x_two_seeds_is_twenty_tasks(self) -> None:
        old_labels = [9, 10, 11, 16, 20, 29, 30, 31, 32, 33]

        table = TaskTable.from_args(labels=old_labels, seeds=[42, 1337], count=50)

        assert len(table) == 20

    def test_sharding_partitions_the_table_for_any_world_size(self) -> None:
        old_labels = [9, 10, 11, 16, 20, 29, 30, 31, 32, 33]
        table = TaskTable.from_args(labels=old_labels, seeds=[42, 1337], count=50)

        shards = [table.shard(rank, 3) for rank in range(3)]

        assert [len(shard) for shard in shards] == [7, 7, 6]
        union = [task for shard in shards for task in shard]
        assert len(union) == len(table)
        assert set(union) == set(table.shard(0, 1))

    def test_sharding_is_identical_for_the_same_rank_and_world_size(self) -> None:
        table = TaskTable.from_args(labels=[9, 10], seeds=[42], count=50)

        assert table.shard(1, 3) == table.shard(1, 3)

    def test_duplicate_labels_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate labels"):
            TaskTable.from_args(labels=[9, 9], seeds=[42], count=50)

    def test_duplicate_seeds_are_refused(self) -> None:
        """A repeated seed expands to repeated tasks, and stride sharding hands them to different ranks.

        Both ranks then find the same volumes missing and write the same filenames at
        once -- the output names do not carry a rank, so nothing arbitrates between them.
        """
        with pytest.raises(ValueError, match="duplicate seeds"):
            TaskTable.from_args(labels=[9], seeds=[42, 42], count=50)

    def test_empty_labels_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no labels"):
            TaskTable.from_args(labels=[], seeds=[42], count=50)

    def test_non_positive_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            TaskTable.from_args(labels=[9], seeds=[42], count=0)


class TestSampleNamer:
    def test_names_are_deterministic_and_zero_padded(self, tmp_path: Path) -> None:
        namer = SampleNamer(tmp_path, "baseline_v1")

        path = namer.path_for(label=9, seed=42, index=7)

        assert path == tmp_path / "baseline_v1_label9_seed42_007.nii.gz"

    def test_pending_indices_lists_everything_when_nothing_exists(self, tmp_path: Path) -> None:
        namer = SampleNamer(tmp_path, "p")
        task = GenerationTask(label=9, seed=42, count=3)

        assert namer.pending_indices(task) == [0, 1, 2]

    def test_pending_indices_skips_volumes_already_on_disk(self, tmp_path: Path) -> None:
        namer = SampleNamer(tmp_path, "p")
        task = GenerationTask(label=9, seed=42, count=3)
        namer.path_for(9, 42, 1).touch()

        assert namer.pending_indices(task) == [0, 2]


class TestConditioningPlan:
    """CPU-constructible: the x1e2/half-precision contract of the conditioning tensors."""

    def test_tensors_scale_by_1e2_and_match_config(self) -> None:
        plan = ConditioningPlan(INFERENCE_CONFIG, torch.device("cpu"))

        assert plan.output_size == (256, 256, 128)
        assert plan.out_spacing == (0.94, 0.94, 1.36)

        top, bottom, spacing, modality = plan.tensors(9)

        assert top.dtype == torch.float16
        assert torch.allclose(top.float(), torch.tensor([[0.0, 100.0, 0.0, 0.0]]))
        assert torch.allclose(spacing.float(), torch.tensor([[94.0, 94.0, 136.0]]))
        assert modality.tolist() == [9]


class RecordingInference:
    """Stands in for ``diff_model_infer.run_inference``: the initial noise of each call, recorded in order.

    The signature is the real one, noise included -- so a ``run_task`` that stopped
    handing the noise over would fail here rather than quietly fall back to a noise
    of the double's own choosing.
    """

    def __init__(self) -> None:
        self.noises: list[torch.Tensor] = []

    def __call__(
        self,
        args,
        device,
        autoencoder,
        unet,
        scale_factor,
        top_region_index,
        bottom_region_index,
        spacing,
        modality,
        output_size,
        noise,
        logger,
        decoder_roi_size=None,
    ) -> np.ndarray:
        self.noises.append(noise.clone())
        return noise.detach().cpu().numpy()


class RawValueWriter:
    """Stands in for ``diff_model_infer.save_image``: the returned tensor's raw values, byte for byte."""

    def __call__(self, data, output_size, out_spacing, path, logger) -> None:
        Path(path).write_bytes(np.asarray(data, dtype=np.float32).tobytes())


def seeded_noise_stream(seed: int, count: int) -> list[torch.Tensor]:
    """The determinism contract's definition: index i carries the (i+1)-th draw of the task-seeded stream."""
    set_determinism(seed)
    return [torch.randn(LATENT_SHAPE) for _ in range(count)]


@pytest.fixture
def volume_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Builds a CPU ``BaselineVolumeGenerator`` (networks and inference doubled) per output directory."""
    monkeypatch.setattr("scripts.diff_model_infer_batch.load_models", lambda *args, **kwargs: (None, None, 1.0))
    monkeypatch.setattr("scripts.diff_model_infer_batch.save_image", RawValueWriter())

    def build(output_dir: Path) -> tuple[BaselineVolumeGenerator, RecordingInference]:
        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = RecordingInference()
        monkeypatch.setattr("scripts.diff_model_infer_batch.run_inference", recorder)
        args = SimpleNamespace(latent_channels=LATENT_CHANNELS, diffusion_unet_def={"num_channels": [1] * 8})
        return BaselineVolumeGenerator(args, torch.device("cpu"), logging.getLogger("test"), SampleNamer(output_dir, "gen")), recorder

    return build


class TestResumeDeterminism:
    """The volume under an index must depend on the index alone, never on where a previous run stopped.

    The frozen baseline is a 1000-volume run that is relaunched the same way it
    started, and the FID stage's gen-gen pairing (spec #13 section 5.1) reads the
    volumes of two models index by index -- so a resume that shifts the noise breaks
    the pairing rather than merely losing work.
    """

    def test_resumed_run_reproduces_the_uninterrupted_volumes(self, volume_generator, tmp_path: Path) -> None:
        task = GenerationTask(label=9, seed=42, count=4)
        plan = ConditioningPlan(INFERENCE_CONFIG, torch.device("cpu"))

        complete, _ = volume_generator(tmp_path / "complete")
        complete.run_task(task, plan)

        resumed_dir = tmp_path / "resumed"
        resumed, _ = volume_generator(resumed_dir)
        SampleNamer(resumed_dir, "gen").path_for(9, 42, 0).touch()
        SampleNamer(resumed_dir, "gen").path_for(9, 42, 1).touch()
        resumed.run_task(task, plan)

        for index in (2, 3):
            name = f"gen_label9_seed42_{index:03d}.nii.gz"
            assert (resumed_dir / name).read_bytes() == (tmp_path / "complete" / name).read_bytes()

    def test_the_noise_under_an_index_does_not_move_with_the_resume_point(self, volume_generator, tmp_path: Path) -> None:
        task = GenerationTask(label=9, seed=42, count=4)
        plan = ConditioningPlan(INFERENCE_CONFIG, torch.device("cpu"))
        resumed_dir = tmp_path / "resumed"
        resumed, recorder = volume_generator(resumed_dir)
        SampleNamer(resumed_dir, "gen").path_for(9, 42, 0).touch()
        SampleNamer(resumed_dir, "gen").path_for(9, 42, 1).touch()

        resumed.run_task(task, plan)

        expected = seeded_noise_stream(task.seed, task.count)
        assert len(recorder.noises) == 2
        assert torch.equal(recorder.noises[0], expected[2])
        assert torch.equal(recorder.noises[1], expected[3])

    def test_an_uninterrupted_run_keeps_the_frozen_index_to_noise_mapping(self, volume_generator, tmp_path: Path) -> None:
        """The frozen baseline's unconsumed cells were generated by this mapping -- moving it invalidates them.

        Characterisation, not aspiration: every uncontaminated (label, seed) task of
        the T8 baseline was a fresh run over all 50 indices, so its index i sits on
        the (i+1)-th draw of the task-seeded stream. A change that reseeds per index
        would be a tidier design and a broken freeze at the same time.
        """
        task = GenerationTask(label=9, seed=42, count=4)
        plan = ConditioningPlan(INFERENCE_CONFIG, torch.device("cpu"))
        generator, recorder = volume_generator(tmp_path / "outputs")

        generator.run_task(task, plan)

        expected = seeded_noise_stream(task.seed, task.count)
        assert len(recorder.noises) == task.count
        assert all(torch.equal(drawn, want) for drawn, want in zip(recorder.noises, expected))
