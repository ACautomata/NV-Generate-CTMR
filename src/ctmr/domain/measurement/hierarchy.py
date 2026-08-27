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

"""Mask-validity predicates: canonical hierarchy violation vs calibration usability (ADR-0010, issue #109).

Two different things used to share one name (``hier_viol``):

- ``HierarchyChecker.violates`` is the canonical containment semantics -- the
  ADR-0004 decision-4 frozen definition, converged verbatim from the
  terminal-acceptance implementation: value domain outside {0,1,2,3} _or_
  ET not inside TC _or_ TC not inside WT. One boolean expression, no
  precondition guards, and an empty superset is NOT exempted (ET present while
  TC empty is a real violation). Because the region label sets are nested
  (ET {3} ⊆ TC {1,3} ⊆ WT {1,2,3}), violation of a well-formed mask can only
  come from the value-domain term; the containment terms are kept as the
  frozen single-expression definition so the semantics cannot drift.
- ``CalibrationCaseUsability`` is the calibration-side gate that the old
  calibration mother *also* called ``hier_viol`` (needs GT: GT and pred value
  domains OK, GT WT non-empty). Different concept, different inputs; it feeds
  the calibration R_fail (ADR-0002 semantics) and does NOT enter
  ``CaseMeasurement.hierarchy_violation``.
"""

import numpy as np

from ctmr.domain.measurement.regions import LABEL_DOMAIN, REGIONS


class HierarchyChecker:
    """Canonical hierarchy violation: containment + value domain, one expression."""

    @classmethod
    def violates(cls, pred: np.ndarray) -> bool:
        """True iff the mask escapes {0,1,2,3} or violates ET⊆TC or TC⊆WT.

        No precondition guards (ADR-0010 decision 3): an empty superset is not
        exempted -- ET voxels with no TC voxels anywhere still violate.
        """
        wt = np.isin(pred, REGIONS["WT"])
        tc = np.isin(pred, REGIONS["TC"])
        et = np.isin(pred, REGIONS["ET"])
        outside_domain = not np.isin(pred, LABEL_DOMAIN).all()
        return bool(outside_domain or (et & ~tc).any() or (tc & ~wt).any())


class CalibrationCaseUsability:
    """Calibration case-usability gate (ADR-0002 semantics; NOT hierarchy_violation).

    Needs GT (meaningless for generated data): usable iff both mask value
    domains are {0,1,2,3} and the GT whole tumour is non-empty. Feeds the
    calibration R_fail, exactly as the mother's ``hier_viol`` column did.
    """

    @classmethod
    def usable(cls, gt: np.ndarray, pred: np.ndarray) -> bool:
        """True iff the (GT, pred) pair may enter envelope estimation."""
        gt_in_domain = np.isin(gt, LABEL_DOMAIN).all()
        pred_in_domain = np.isin(pred, LABEL_DOMAIN).all()
        gt_wt_present = np.isin(gt, REGIONS["WT"]).any()
        return bool(gt_in_domain and pred_in_domain and gt_wt_present)
