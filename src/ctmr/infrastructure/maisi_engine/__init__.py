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

"""Vendored upstream MAISI diffusion engine (issue #134, ADR-0015 §2 M3).

The four engine files are frozen copies of the upstream script set:

- ``diff_model_setting``      configuration loading / logging / DDP init / torchrun launcher
- ``diff_model_infer``        DM inference driver (the live image-only entry)
- ``create_training_data``    VAE-latency training-data encoder
- ``utils_infer``             shared inference-sampling toolkit (ReconModel, image-side model loading)

The training driver ``diff_model_train`` and the rectified-flow img2img
driver ``img2img_infer`` were deleted with issue #175 (ADR-0016 M4): the
domain ``DiffusionModel`` / ``DiffusionScheduler`` / ``ModalityLabelPerturber``
entities are their canonical replacement — git history is the reproduction
anchor.

supported by two freeze-side primitive modules:

- ``instance_definition``     config-key -> MONAI object instantiation plus modality intensity transforms
- ``inference_primitives``    stateless inference helpers and input constraint guards

Behavior stays byte-stable versus the retired originals; the observable
behavior is guarded by ``tests/infrastructure/maisi_engine/test_engine_smoke.py``
(execution smoke on synthetic config/argv). The #143 upstream-equivalence AST
gate is retired — git history is the reproduction anchor (ADR-0015 M5). All
former legacy-layer consumers switched over with their use-case-family
migration batches.

Deliberately NOT vendored here: ``sample.py`` from the retired scripts layer
(git history; the LDMSampler
backward-compat shell). Its re-exported mask/image pipelines live in
``sample_mask``/``infer_image_from_mask``, which belong to the dataio and
application layers of later migration batches, not to the frozen engine.

Process-spawn precedent (#123) is preserved by construction: neither this
package nor its smoke test touches multiprocessing context choices.
DDP/torchrun launch behavior lives in ``diff_model_setting.run_torchrun``
and reaches the server only through gpu-flagged callers; the CPU smoke test
does not execute those GPU-bound paths.
"""
