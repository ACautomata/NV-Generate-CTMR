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

"""Thin forwarding shim -- the phase training shell moved to
``ctmr.application.shell`` (ADR-0015 §2 application layer, ticket 08).
The scripts-side consumers (P1/P2 finetune entries still in scripts/) keep
importing from here until their migration batches relocate them; behaviour is
unchanged.
"""

from ctmr.application.shell import STOP_FILE, PhaseHarness, PhaseTrainKernel, TrainContext, TrainProvenanceWriter

__all__ = ["STOP_FILE", "PhaseHarness", "PhaseTrainKernel", "TrainContext", "TrainProvenanceWriter"]
