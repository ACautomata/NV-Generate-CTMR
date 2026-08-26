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

"""Vendoring integrity for ``ctmr.infrastructure.maiisi_engine`` (issue #134).

Text-level assertions runnable on any machine -- no torch/monai required:

- the six vendored engine bodies are byte-identical to their ``scripts/``
  originals (vendoring is adoption, not rewrite);
- each bridge module forwards exactly its documented name list from the
  documented origin;
- engine bodies carry no out-of-package absolute imports;
- the package docstring keeps the #123 spawn-context constraint registered.
"""

import ast
import filecmp
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
ENGINE = REPO_ROOT / "src" / "ctmr" / "infrastructure" / "maiisi_engine"

VENDORED_BODIES = [
    "diff_model_setting.py",
    "diff_model_infer.py",
    "diff_model_train.py",
    "diff_model_create_training_data.py",
    "sample.py",
    "utils_infer.py",
]

EXPECTED_BRIDGES = {
    "augmentation": ("scripts.augmentation", {"augmentation"}),
    "find_masks": ("scripts.find_masks", {"find_masks"}),
    "infer_image_from_mask": ("scripts.infer_image_from_mask", {"crop_img_body_mask", "ldm_conditional_sample_one_image"}),
    "quality_check": ("scripts.quality_check", {"is_outlier"}),
    "sample_mask": (
        "scripts.sample_mask",
        {"ReconModel", "check_input_ct", "check_input_mr", "filter_mask_with_organs", "initialize_noise_latents", "ldm_conditional_sample_one_mask"},
    ),
    "transforms": ("scripts.transforms", {"SUPPORT_MODALITIES", "define_fixed_intensity_transform"}),
    "utils": ("scripts.utils", {"define_instance", "dynamic_infer", "get_body_region_index_from_mask"}),
}


def _top_level_import_from(source):
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, {alias.name for alias in node.names}))
    return out


def test_bridge_modules_forward_exactly_the_documented_names():
    for bridge_name, (origin, expected_names) in EXPECTED_BRIDGES.items():
        bridge_path = ENGINE / (bridge_name + ".py")
        assert bridge_path.exists(), "missing bridge module " + bridge_name
        got = {}
        for module, names in _top_level_import_from(bridge_path.read_text()):
            got.setdefault(module, set()).update(names)
        assert got == {origin: expected_names}, "bridge " + bridge_name + " drifted: " + repr(got)


def test_engine_bodies_have_no_out_of_package_absolute_imports():
    pattern = re.compile(r"^\s*(?:from|import)\s+(?:scripts|ctmr)\b", re.MULTILINE)
    for name in VENDORED_BODIES:
        hits = pattern.findall((ENGINE / name).read_text())
        assert not hits, name + " reaches outside the package via absolute import: " + repr(hits)


def test_package_docstring_registers_spawn_constraint():
    doc = (ENGINE / "__init__.py").read_text()
    assert "#123" in doc, "spawn-context constraint (#123) must stay registered in the maiisi_engine package docstring"
    assert "spawn" in doc


def test_vendored_engine_bodies_are_byte_identical_to_scripts():
    for name in VENDORED_BODIES:
        assert filecmp.cmp(SCRIPTS / name, ENGINE / name, shallow=False), name + " diverged byte-wise from scripts/"
