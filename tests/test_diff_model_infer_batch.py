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

from pathlib import Path

import pytest

from scripts.diff_model_infer_batch import GenerationTask, SampleNamer, TaskTable


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
        import torch

        from scripts.diff_model_infer_batch import ConditioningPlan

        config = {
            "dim": [256, 256, 128],
            "spacing": [0.94, 0.94, 1.36],
            "top_region_index": [0, 1, 0, 0],
            "bottom_region_index": [0, 0, 1, 0],
        }
        plan = ConditioningPlan(config, torch.device("cpu"))

        assert plan.output_size == (256, 256, 128)
        assert plan.out_spacing == (0.94, 0.94, 1.36)

        top, bottom, spacing, modality = plan.tensors(9)

        assert top.dtype == torch.float16
        assert torch.allclose(top.float(), torch.tensor([[0.0, 100.0, 0.0, 0.0]]))
        assert torch.allclose(spacing.float(), torch.tensor([[94.0, 94.0, 136.0]]))
        assert modality.tolist() == [9]
