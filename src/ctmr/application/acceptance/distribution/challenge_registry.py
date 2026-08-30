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

"""L2 shared vocabulary: the frozen challenge/seed/envelope registration
(ADR-0017 decision 1, issue #229).

The five frozen challenges, their 20% final-holdout quotas, the unified
bootstrap seed band (protocol §4: B, global seed, per-challenge offsets) and
the ADR-0002 published envelope literals -- all moved verbatim out of the
terminal-acceptance judge (``final_acceptance``), which now imports this
module, as do the diagnostic readers. This is the registration *data*: the
envelope verification gate stays with the judge's ``FrozenEnvelopes`` and the
TOST bit-stream stays with the judge's judgement chain. The diagnostic seed
namespace joined the registry in the successor ticket (#232, ADR-0017
decision 5): the whole L2 seed space -- judge band + diagnostic namespace --
is registered here, and the「L2 全域种子无碰撞」invariant is a unit test on
this table (``tests/application/acceptance/distribution/`` +
``test_diagnostic_support.py``); jobs A/B draw seeds through
``diagnostic_support.DiagnosticSeedAllocator``, never from local constants.

Every value here is frozen by ADR-0002/0003/0004 and the split manifest --
literals only, no logic.

The dependency closure is third-party-free -- stdlib only, numpy/scipy/torch
unreachable -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py``.
"""

CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")

# Frozen 20% final-holdout quotas (spec #51 decision 3 / split manifest).
HOLDOUT_QUOTAS = {"GLI": 250, "SSA": 12, "MEN": 200, "METS": 48, "PED": 20}

# Bootstrap registration (protocol §4): B, global seed, per-challenge offsets.
BOOTSTRAP_B = 10_000
GLOBAL_SEED = 20260821
CHALLENGE_SEED_OFFSET = {"GLI": 1, "SSA": 2, "MEN": 3, "METS": 4, "PED": 5}

# Diagnostic seed namespace (ADR-0017 decision 5, issue #232): the diagnostic
# segment of the unified L2 seed registry -- base 900,000,000, one 1000-wide
# band per challenge (CHALLENGE_SEED_OFFSET x band), a full band above the
# judge band's reach so a diagnostic CI can never be mistaken for the
# registered TOST bit-stream. Slots are allocated here, once: job A (#206)
# holds the uncompensated block 0/1 and the compensated block 100/101 of each
# band, job B (#207) slot 200 -- the pre-#232 job-module constants reproduced
# byte-exactly. Registered slots are drawn only through
# ``diagnostic_support.DiagnosticSeedAllocator``. KNOWN DEBT: jobs C/D's
# bandless slot blocks (base+300..307 / base+300..320, overlapping on
# base+300..304) and the geometry audit's banded slot 300 are still
# job-local constants pending their follow-up registration -- registering
# them here is what extends the no-collision invariant over those draws.
DIAGNOSTIC_SEED_BASE = 900_000_000
DIAGNOSTIC_SEED_BAND = 1000
DIAGNOSTIC_SEED_SLOTS = {
    "zcrop_vol_uncomp": 0,  # job A: vol_wt_rel, uncompensated block
    "zcrop_centroid_uncomp": 1,  # job A: centroid_wt_z, uncompensated block
    "zcrop_vol_comp": 100,  # job A: vol_wt_rel, compensated block
    "zcrop_centroid_comp": 101,  # job A: centroid_wt_z, compensated block
    "et_rel_diff": 200,  # job B: per-case relative-difference CI90
}

# ADR-0002 frozen envelopes (published 4-dp literals; the authoritative source
# for every pass line -- equality against a controlled calibration summary is
# enforced to +-5e-5 so drift AND narrowing both reject).
FROZEN_ENVELOPES = {
    #            region: D_r,low   E_r,vol   E_r,centroid(mm)   R_fail Wilson 95% upper
    "GLI": {"WT": (0.8053, 0.2802, 5.38), "TC": (0.6819, 0.4373, 4.79), "ET": (0.4093, 0.5702, 4.41), "r_fail_upper": 0.0043},
    "SSA": {"WT": (0.7046, 0.7223, 12.36), "TC": (0.6578, 0.7767, 8.64), "ET": (0.7111, 0.7786, 8.42), "r_fail_upper": 0.0838},
    "MEN": {"WT": (0.7562, 0.3235, 3.92), "TC": (0.7208, 0.3576, 6.17), "ET": (0.7501, 0.3367, 5.70), "r_fail_upper": 0.0053},
    "METS": {"WT": (0.0000, 1.6510, 28.58), "TC": (0.0000, 1.0000, 35.08), "ET": (0.0000, 1.0000, 35.08), "r_fail_upper": 0.0220},
    "PED": {"WT": (0.0093, 0.9946, 17.87), "TC": (0.0105, 0.9939, 18.33), "ET": (0.0000, 1.0000, 22.65), "r_fail_upper": 0.0507},
}
