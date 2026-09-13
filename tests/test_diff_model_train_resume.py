# Tests for resuming diff_model_train from a checkpoint written by the same run
# (T9a restart, issue #25): epoch counting continues, scale_factor is taken from
# the checkpoint, and the lr schedule spans only the remaining epochs.
import pytest
import torch

from scripts.diff_model_train import ResumeState, plan_resume_schedule, read_resume_state


class TestReadResumeState:
    def test_reads_epoch_and_scale_factor_from_ckpt(self):
        ckpt = {
            "epoch": 16,
            "loss": 0.9261,
            "num_train_timesteps": 1000,
            "scale_factor": torch.tensor(0.9954),
            "unet_state_dict": {},
        }

        state = read_resume_state(ckpt)

        assert state == ResumeState(start_epoch=16, scale_factor=torch.tensor(0.9954))

    def test_missing_epoch_key_raises(self):
        ckpt = {"scale_factor": torch.tensor(0.9954), "unet_state_dict": {}}

        with pytest.raises(ValueError, match="epoch"):
            read_resume_state(ckpt)

    def test_missing_scale_factor_key_raises(self):
        ckpt = {"epoch": 16, "unet_state_dict": {}}

        with pytest.raises(ValueError, match="scale_factor"):
            read_resume_state(ckpt)


class TestPlanResumeSchedule:
    # gauss N300 numbers: n_epochs=300, dataset 7,428, world size 2 -> per-rank
    # steps per epoch 3,714; original run died during global epoch 17 (ckpt epoch 16).
    def test_total_steps_spans_only_the_remaining_epochs(self):
        remaining, total_steps = plan_resume_schedule(n_epochs=300, start_epoch=16, dataset_size=7428, batch_size=2)

        assert remaining == 284
        assert total_steps == 284 * 3714

    def test_start_epoch_zero_reproduces_the_from_scratch_formula(self):
        _, total_steps = plan_resume_schedule(n_epochs=300, start_epoch=0, dataset_size=7428, batch_size=2)

        assert total_steps == 300 * 7428 / 2

    def test_checkpoint_at_or_past_n_epochs_raises(self):
        with pytest.raises(ValueError, match="already completed"):
            plan_resume_schedule(n_epochs=300, start_epoch=300, dataset_size=7428, batch_size=2)
