# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The generate family's assembly (ADR-0019 §2, issues #270/#272/#273/#274).

The train verbs' runtime topology: ``ctmr generate <case> train`` arrives
WITHOUT torchrun and derives the ``torchrun --nproc_per_node=<num_gpus> -m
<module> <rest-argv>`` child; with ``WORLD_SIZE`` already set the process IS
the torchrun worker and runs the train entry in-process. Both branches ride
this one assembly -- the CLI face and the torchrun worker entry reuse it --
so the spawn topology has exactly one home, the composition root. The
collaborator classes (``TorchrunLauncher`` / ``num_gpus_of``, stdlib-light)
are imported at module top and called exactly as the interface layer used
to; the train modules themselves load lazily on dispatch (they are the
production torch entries).

The per-case port assemblies land with the family migration tickets: the
modality-label (#272) and mask (#273) families' are the ``*_train_session``
assemblies and the ``*_engine`` seams below -- the engine adapter, the
distributed session + logger, the gradient executor chosen by the amp
declaration, the modality-label MONAI-checkpoint archive behind the
``CheckpointRepository`` load face, and (with the #276 ratchet-zeroing sweep)
the shell-side publication store behind the same port's save face
(ADR-0019 §2: concrete knowledge settles
nowhere else; §3: the family entries consume only domain ports). The
cross-modal family's (#274) is :class:`GenerateRuntime` -- the frozen engine
adapter behind the ``GenerationEngine`` port, the distributed session
bootstrap, the run logger, the ControlNet mounting, the pinned precision
executor and the checkpoint file identity -- resolved lazily on dispatch
(importing the composition root still pulls no third-party dependency). The
torchrun worker entry reuses the same assemblies: the family ``main``s
import them from here, so the worker process assembles through the
composition root too.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

if TYPE_CHECKING:
    from argparse import Namespace

    import torch

    from ctmr.domain.checkpoints import CheckpointRepository
    from ctmr.domain.engine import GenerationEngine
    from ctmr.domain.generation import BypassMounting, GradientExecutor
    from ctmr.domain.logging import Logger


class GenerateRuntime:
    """The generate family's production runtime (ADR-0019 §2, #272-#274).

    One method per collaborator the application entries receive at their
    seams; each resolves its concrete adapter lazily (importlib on dispatch)
    so the frozen maisi_engine functions, the precision executors, the
    ControlNet mounting, the distributed session and the run logger load only
    when a verb actually runs. The application entries see the domain port
    faces (``GenerationEngine`` / ``Logger`` / ``GradientExecutor``) and the
    injected collaborators -- never the concrete addresses.
    """

    def engine(self):
        """The frozen maisi_engine adapter behind the GenerationEngine port."""
        return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()

    def logger(self, name):
        """The run logger (the Logger port's stdlib realization)."""
        return importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting").setup_logging(name)

    def train_session(self, args):
        """Bootstrap the distributed training session: (local_rank, world_size, device)."""
        return importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting").initialize_distributed(args.num_gpus)

    def bypass_mounting(self, args, device, logger):
        """The ControlNet-only hook-up collaborator the bypass kernels drive."""
        return importlib.import_module("ctmr.infrastructure.bypass_mounting").BypassMounting(args, device, logger)

    def gradient_executor(self, amp, amp_dtype):
        """The pinned precision strategy (ADR-0016): fp16 (scaler), bf16 (DCU default) or plain."""
        executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
        if amp and amp_dtype == "fp16":
            return executors.Fp16GradientExecutor()
        if amp:
            return executors.Bf16GradientExecutor()
        return executors.PlainGradientExecutor()

    def checkpoint_repository(self, model_dir):
        """The shell-side weight store behind the CheckpointRepository port (ADR-0015 §4)."""
        checkpoints = importlib.import_module("ctmr.infrastructure.checkpoints")
        return checkpoints.CheckpointRepository(model_dir)

    def weights_ref_of_file(self):
        """The checkpoint file identity callable (path -> domain WeightsRef)."""
        return importlib.import_module("ctmr.infrastructure.weightsref").weights_ref_of_file


class TrainDispatch:
    """The one assembly behind every ``ctmr generate <case> train`` dispatch
    (ADR-0019 §2): outside torchrun it derives the torchrun child; already a
    worker, it runs the train entry in-process. Either way the return value
    is the trainer's exit code, and the child runs with the same argv
    namespace (spawn precedent #123: no fork)."""

    def __init__(self, module, argv):
        self._module = module
        self._argv = list(argv)

    def run(self):
        """Derive the topology and relay the trainer's exit code."""
        if os.environ.get("WORLD_SIZE"):
            return importlib.import_module(self._module).main(self._argv)
        return TorchrunLauncher(self._module, self._argv, num_gpus_of(self._argv)).run()


class MonaiCheckpointArchive:
    """MONAI-pickled training checkpoints behind the CheckpointRepository load
    face (ADR-0019 §3, #272).

    The P1 base checkpoint pickles MONAI meta-tensor globals: the allowlisted
    weights_only realization (``MonaiCheckpoint``) is mounted here in the
    composition root and reaches the family only as the domain port."""

    def __init__(self, device):
        self._device = device

    def load(self, path):
        bypass_mounting = importlib.import_module("ctmr.infrastructure.bypass_mounting")
        return bypass_mounting.MonaiCheckpoint(path, self._device).load()


@dataclass
class ModalityLabelTrainSession:
    """The assembled modality-label train runtime (ADR-0019 §2, #272): the
    port set the family entry consumes, constructed nowhere else. ``merged``
    is the parsed config namespace -- resolution happens inside the assembly,
    before the distributed group forms, so a bad config fails on every rank
    ahead of any collective (the pre-migration ordering)."""

    local_rank: int
    device: torch.device
    logger: Logger
    engine: GenerationEngine
    gradient_executor: GradientExecutor
    base_checkpoints: CheckpointRepository
    checkpoint_repository: CheckpointRepository
    merged: Namespace


@dataclass
class MaskTrainSession:
    """The assembled mask train runtime (ADR-0019 §2, #273): the port set
    the family entry consumes, constructed nowhere else. ``merged`` is the
    parsed config namespace -- resolution happens inside the assembly, before
    the distributed group forms, so a bad config fails on every rank ahead of
    any collective (the pre-migration ordering). The bypass mounting is the
    domain port the kernel composes the entities from."""

    local_rank: int
    device: torch.device
    logger: Logger
    engine: GenerationEngine
    gradient_executor: GradientExecutor
    mounting: BypassMounting
    checkpoint_repository: CheckpointRepository
    merged: Namespace


def modality_label_engine():
    """The modality-label family's GenerationEngine assembly (ADR-0019 §2, #272)."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()


def modality_label_validation(args, merged, session, kernel):
    """The modality-label embedded periodic validation assembly (ADR-0019 §2+§5, #278).

    The shell-side mechanics live in ``ctmr.application.shell``; here settles
    the concrete knowledge: the fixed dev cohort unfolded to 64
    (case, modality) shard items, the real reference bank (rank 0 builds and
    writes it, the barrier publishes it, every other rank loads the cache), the
    dev-eval sampling recipe pinned on the parsed config (cfg=10, 30 steps --
    the retired sidecar's identical constants), the live-weight shard sampler,
    the FID scorer and the pre-registered early-stop rule (the ADR-0005 values
    with the trainer's own n_epochs as the cap). Missing dev inputs refuse at
    assembly, on every rank, before the first epoch.
    """
    monitor = importlib.import_module("ctmr.application.generation.modality_label.monitor")
    shell = importlib.import_module("ctmr.application.shell")
    trend = importlib.import_module("ctmr.application.generation.trend")
    dist = importlib.import_module("torch.distributed")
    recipe = importlib.import_module("ctmr.domain.recipe")
    for name in ("dev_list", "raw_root", "emb_root"):
        if getattr(args, name, None) is None:
            raise ValueError(f"--val-every {args.val_every} requires --{name.replace('_', '-')} (the embedded validation's dev cohort inputs)")
    eval_root = Path(merged.model_dir) / shell.DEV_EVAL_DIR
    cohort = monitor.DevCohortBuilder(args.dev_list).build()
    items = [{**entry, "modality": modality} for entry in cohort for modality in shell.TARGET_MODALITIES]
    features = trend.MrTrendFeatures(session.device)
    rule = shell.EarlyStopRule(
        patience=recipe.P1_DEV_EARLY_STOP["patience"],
        min_epoch=recipe.P1_DEV_EARLY_STOP["min_epoch"],
        max_epoch=merged.diffusion_unet_train["n_epochs"],
    )
    if session.local_rank == 0:
        monitor.DevCohortBuilder(args.dev_list).write(eval_root / "dev_cohort.json")
        bank = trend.RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
        (eval_root / "early_stop_rule.json").write_text(
            json.dumps(
                {
                    "rule": rule.rule_text(),
                    "patience": rule.patience,
                    "min_epoch": rule.min_epoch,
                    "max_epoch": rule.max_epoch,
                },
                indent=2,
            )
            + "\n"
        )
    if dist.is_initialized():
        dist.barrier()
    if session.local_rank != 0:
        bank = trend.RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    sampler = monitor.LiveCohortSampler(
        merged, session.device, session.engine, kernel, monitor.CohortSpacingSource(args.dev_list, args.emb_root), features
    )
    return shell.ValidationPhase(
        every=args.val_every,
        validator=shell.PeriodicValidator(
            items,
            sampler,
            monitor.CohortFeatureScorer(trend.TrendFid(bank)),
            session.local_rank,
            session.device,
            cohort_file=str(eval_root / "dev_cohort.json"),
        ),
        rule=rule,
    )


def modality_label_reencode_runtime():
    """The embedding re-encode assembly (issue #251, series-② T4): the vendored
    ``diff_model_create_training_data`` execution (the clip=True recipe's
    encoding chain) and the distribution family's engine for the self-eval
    decode arm. The re-encode entry consumes only these injected callables --
    the infrastructure addresses settle here (ADR-0019 §2)."""
    create_training_data = importlib.import_module("ctmr.infrastructure.maisi_engine.create_training_data").diff_model_create_training_data
    engine = importlib.import_module("ctmr.wiring.distribution").intensity_domain_engine()
    return create_training_data, engine


def mask_engine():
    """The mask family's GenerationEngine assembly (ADR-0019 §2, #273)."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()


def modality_label_train_session(args, engine=None):
    """The modality-label train assembly (ADR-0019 §2, #272): the config
    resolution (strictly before the distributed bootstrap -- a malformed
    config must fail on every rank ahead of any collective), the session
    bootstrap, the logger, the gradient executor chosen by the amp
    declaration, the base-checkpoint archive, and the shell-side publication
    store (#276)."""
    engine = engine if engine is not None else modality_label_engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    setting = importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting")
    executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
    checkpoints = importlib.import_module("ctmr.infrastructure.checkpoints")
    # The 2h process-group timeout covers the first-run bank build: rank 0
    # preprocesses the whole dev list (RadImageNet inference) alone before the
    # peers' barrier rendezvous, the same allowance the heavy quantitative
    # path pins (fid_2d5; codex review, PR #301). Cached runs rendezvous in
    # seconds -- the timeout only bounds a genuinely stranded peer.
    local_rank, _world, device = setting.initialize_distributed(args.num_gpus, timeout=timedelta(seconds=7200))
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = executors.Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = executors.Bf16GradientExecutor()
    else:
        gradient_executor = executors.PlainGradientExecutor()
    return ModalityLabelTrainSession(
        local_rank=local_rank,
        device=device,
        logger=setting.setup_logging("modality-label-finetune"),
        engine=engine,
        gradient_executor=gradient_executor,
        base_checkpoints=MonaiCheckpointArchive(device),
        checkpoint_repository=checkpoints.CheckpointRepository(merged.model_dir),
        merged=merged,
    )


def mask_train_session(args, engine=None):
    """The mask train assembly (ADR-0019 §2, #273): the config resolution
    (strictly before the distributed bootstrap -- a malformed config must
    fail on every rank ahead of any collective), the session bootstrap, the
    logger, the gradient executor chosen by the amp declaration, the
    bypass mounting the kernel composes the domain entities from, and the
    shell-side publication store (#276)."""
    engine = engine if engine is not None else mask_engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    setting = importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting")
    executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
    mounting = importlib.import_module("ctmr.infrastructure.bypass_mounting")
    checkpoints = importlib.import_module("ctmr.infrastructure.checkpoints")
    local_rank, _world, device = setting.initialize_distributed(args.num_gpus)
    logger = setting.setup_logging("mask-finetune")
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = executors.Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = executors.Bf16GradientExecutor()
    else:
        gradient_executor = executors.PlainGradientExecutor()
    return MaskTrainSession(
        local_rank=local_rank,
        device=device,
        logger=logger,
        engine=engine,
        gradient_executor=gradient_executor,
        mounting=mounting.BypassMounting(merged, device, logger),
        checkpoint_repository=checkpoints.CheckpointRepository(merged.model_dir),
        merged=merged,
    )
