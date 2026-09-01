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

"""Frozen sample-plan vocabulary and seed rule shared by the cross_modal family.

The four-anchor-round protocol and the deterministic per-(case, direction)
seed come from spec #51 decisions 4/5/8 and are shared by the baseline
(zero-training img2img) and candidate (ControlNet) plan builders plus the
dev-eval watch, so baseline / candidate / dev samples stay directly comparable.
Pure stdlib (no torch, no IO).
"""

from __future__ import annotations

import hashlib

MODALITIES = ("t1n", "t1c", "t2w", "t2f")

# BraTS file suffix -> (modality_mapping long key, class-label token). Pinned
# by spec #51 decision 5; t1c=34 is the modality-label-planned addition. The dev watch
# translates BraTS suffixes through these long keys when resolving list fields.
MODALITY_PAIRS = {
    "t1n": ("mri_t1_skull_stripped", 29),
    "t2w": ("mri_t2_skull_stripped", 30),
    "t2f": ("mri_flair_skull_stripped", 31),
    "t1c": ("mri_t1c_skull_stripped", 34),
}

SEED_MODULUS = 2**31 - 1


def seed_of(case, src, tgt):
    """Deterministic per (case, direction) seed: sha256('<case>|<src>-><tgt>') truncated.

    Identical rule for baseline and candidate so both share the same (case,
    src, tgt) noise schedule and are directly comparable.
    """
    digest = hashlib.sha256(f"{case}|{src}->{tgt}".encode()).hexdigest()[:8]
    return int(digest, 16) % SEED_MODULUS
