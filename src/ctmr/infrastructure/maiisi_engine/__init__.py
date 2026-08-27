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

The six engine files are frozen copies of the upstream script set:

- ``diff_model_setting``      configuration loading / logging / DDP init / torchrun launcher
- ``diff_model_train``        DM training driver
- ``diff_model_infer``        DM inference driver
- ``create_training_data``    VAE-latency training-data encoder
- ``utils_infer``             shared inference-sampling toolkit (ReconModel, model loading, ...)
- ``img2img_infer``           rectified-flow img2img sampling (ticket 08; the stage-0 baseline chain)

supported by two freeze-side primitive modules:

- ``instance_definition``     config-key -> MONAI object instantiation plus modality intensity transforms
- ``inference_primitives``    stateless inference helpers and input constraint guards

Behavior stays byte-stable versus the legacy ``scripts/`` originals; that is
machine-guarded by ``tests/infrastructure/maiisi_engine/test_vendored_parity.py``:
whole-file AST equality (import statements stripped) for the five engine files,
and function-level AST equality for the seven symbols extracted into the two
support modules. Consumers on ``scripts.*`` keep working untouched during the
expand phase; use-case-family tickets switch them over later.

Deliberately NOT vendored here: ``scripts/sample.py`` (the LDMSampler
backward-compat shell). Its re-exported mask/image pipelines live in
``sample_mask``/``infer_image_from_mask``, which belong to the dataio and
application layers of later migration batches, not to the frozen engine.

Process-spawn precedent (#123) is preserved by construction: neither this
package nor its parity guard touches multiprocessing context choices.
DDP/torchrun launch behavior lives in ``diff_model_setting.run_torchrun``
and reaches the server only through gpu-flagged callers; the CPU parity test
keeps that code byte-identical without executing it.
"""
