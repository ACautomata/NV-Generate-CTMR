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

"""One MR-RATE acquisition and the three files the replay pipeline derives from it
(spec #13 sections 3.2 / 3.3, ticket T6 #22).

MR-RATE publishes a study zip per study with two members per series -- the defaced
native-space image under ``img/`` and its HD-BET brain mask under ``seg/``.  Nothing
in the release is skull-stripped, so the pipeline derives a third file by masking the
image (section 3.2: "数据集只发布 whole-brain + mask，无独立 skull-stripped 文件").

This module owns that layout.  Download, spine filtering, skull-stripping, encoding and
sidecar generation all name the same three files, so the naming lives in one place:
a variant knows its three paths and the two labels its training entries carry.

Layout (relative to a data root; the unzip step reproduces the official zip layout)::

    mri/batch00/<study>/img/<study>_<series>.nii.gz                  whole-brain
    mri/batch00/<study>/seg/<study>_<series>_brain-mask.nii.gz       HD-BET mask
    mri/batch00/<study>/img/<study>_<series>_skull_stripped.nii.gz   derived twin

The skull-stripped twin sits next to its source rather than in a parallel tree, so one
``data_base_dir`` reaches every replay entry the training set consumes -- the merge
generator (ticket T7 #23) rebases these relative paths onto the training env's root.
"""

from pathlib import Path

WHOLE_BRAIN_LABEL = {
    "t1w": "mri_t1",
    "t2w": "mri_t2",
    "flair": "mri_flair",
    "swi": "mri_swi",
    "mra": "mri_mra",
}
SKULL_STRIPPED_LABEL = {
    "t1w": "mri_t1_skull_stripped",
    "t2w": "mri_t2_skull_stripped",
    "flair": "mri_flair_skull_stripped",
    "swi": "mri_swi_skull_stripped",
    "mra": "mri_mra_skull_stripped",
}
NIFTI_EXTENSION = ".nii.gz"
BRAIN_MASK_SUFFIX = "_brain-mask"
SKULL_STRIPPED_SUFFIX = "_skull_stripped"
IMAGE_DIRNAME = "img"
SEGMENTATION_DIRNAME = "seg"
# The published layout ``mri/<batch>/<study>/img/<file>``: mri/ is parts[0], so the batch is 1.
BATCH_DIRNAME_PART_INDEX = 1


class MrRateVariant:
    """One acquisition seen through the names the pipeline gives it.

    ``whole_brain_path`` is the published image, ``mask_path`` the published HD-BET brain
    mask, ``skull_stripped_path`` the derived twin; ``label`` / ``skull_stripped_label`` are
    the dataset.json modality strings of the two training entries this series becomes (the
    "双产" of spec section 3.3).
    """

    def __init__(self, image_path: str) -> None:
        self._image_path = image_path
        self._stem = Path(image_path).name.removesuffix(NIFTI_EXTENSION)

    @classmethod
    def for_series(cls, batch: str, study_uid: str, series_id: str) -> "MrRateVariant":
        """Build the variant from its three identifiers, in the official unzip layout."""
        return cls(f"mri/{batch}/{study_uid}/{IMAGE_DIRNAME}/{study_uid}_{series_id}{NIFTI_EXTENSION}")

    @property
    def whole_brain_path(self) -> str:
        """The published defaced native-space image."""
        return self._image_path

    @property
    def batch(self) -> str:
        """The batch directory the study zip lives in (``mri/<batch>/<study_uid>.zip`` upstream)."""
        return Path(self._image_path).parts[BATCH_DIRNAME_PART_INDEX]

    @property
    def mask_path(self) -> str:
        """The HD-BET brain mask, same voxel grid as the image."""
        sibling = Path(self._image_path).parent.parent / SEGMENTATION_DIRNAME / f"{self._stem}{BRAIN_MASK_SUFFIX}{NIFTI_EXTENSION}"
        return sibling.as_posix()

    @property
    def skull_stripped_path(self) -> str:
        """The derived skull-stripped twin (``image * brain-mask``); same directory as the image."""
        sibling = Path(self._image_path).parent / f"{self._stem}{SKULL_STRIPPED_SUFFIX}{NIFTI_EXTENSION}"
        return sibling.as_posix()

    @property
    def label(self) -> str:
        """Whole-brain v1 label string (``mri_t1`` ...); drives the intensity dispatch."""
        return WHOLE_BRAIN_LABEL[self.modality]

    @property
    def skull_stripped_label(self) -> str:
        """Skull-stripped v1 label string (``mri_t1_skull_stripped`` ...), the dual twin's condition."""
        return SKULL_STRIPPED_LABEL[self.modality]

    @property
    def modality(self) -> str:
        """Lowercase MR-RATE modality (``t1w`` ...), recovered from the series-id prefix."""
        return self.series_id.split("-", 1)[0]

    @property
    def series_id(self) -> str:
        """The MR-RATE series id (``t1w-raw-sag``): the file stem minus the leading study uid."""
        return self._stem.removeprefix(f"{self.study_uid}_")

    @property
    def study_uid(self) -> str:
        """The study directory name, which is also the file stem's leading underscore-delimited field."""
        return Path(self._image_path).parent.parent.name
