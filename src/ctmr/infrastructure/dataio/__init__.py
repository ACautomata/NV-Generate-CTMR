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

"""ctmr.infrastructure.dataio -- image transforms, augmentation, quality checks,
mask retrieval, plotting, and model/dataset download assembly (ADR-0015 §2).

Module map (old leaf -> module):

- ``scripts/utils.py``        (erode_one_img / dilate_one_img) -> ``morphology``
- ``scripts/transforms.py``   -> ``transforms``
- ``scripts/augmentation.py`` -> ``augmentation``
- ``scripts/quality_check.py``-> ``quality_check``
- ``scripts/find_masks.py``   -> ``find_masks``
- ``scripts/sample_mask.py``  -> ``sample_mask`` (input validation + mask filtering;
                                the DDPM sampler core stays on its old leaf until the
                                MAISI engine helpers it consumes are collected)
- ``scripts/utils_plot.py``   -> ``plotting``
- ``scripts/download_model_data.py`` -> ``download``

Every submodule carries heavy third-party deps (monai / torch / matplotlib /
huggingface_hub), so this ``__init__`` intentionally re-exports nothing:
`import ctmr.infrastructure.dataio` must stay runnable on a light stack while
consumers import submodules directly (the born-with-test gates skip themselves
on light stacks via pytest.importorskip, per ADR-0013 §4).
"""
