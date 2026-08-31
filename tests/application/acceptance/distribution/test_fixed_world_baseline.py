"""Fixed-world baseline rerun (issue #252, parent #247), observed as pytest.

The series-② write-path fix (#249) re-declares the holdout artifacts at the v1
DM's real sampling spacing, which switches the readings' world: the declared z
domain extends to [0, 174) mm (the geometry audit's 146 cases / ~431 ml above
the retired 1 mm edge z >= 128 leave the declared domain) and the centroid
axis reads in the registered crop world (comp +9, dual to the pad world's
-13). Job T5 re-pairs all 14 registered L2 quantity families per case under
the crop window on the SAME frozen masks -- zero inference -- and binds the
report to the recorded diagnostic history through two certificates: the job A
anchor (comp WT volume / centroid-z medians must reproduce the recorded 6-dp
literals, same masks, same registered seed bit-streams) and the job B anchor
(the ET discrimination re-run on the same CSV must equal the recorded
detection/pairing tallies). These tests pin, on synthetic volumes with
hand-computed expectations: the fixed-world window and its +9 mm anchor, the
region/axis-generalized window reading (WT volume and z bit-equal to job A's
own restrict-and-measure), the per-case pairing of all quantity families
against CSV and window readings, the seed table (job A bit-streams reused for
the anchors, the T5 blocks at 400/500 + judge quantity index), both anchor
certificates, the field-of-view scan under both declarations, and the report /
CLI surface.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.challenge_registry import (
    DIAGNOSTIC_SEED_BASE,
    DIAGNOSTIC_SEED_SLOTS,
)
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError, DiagnosticSeedAllocator
from ctmr.application.acceptance.distribution.et_discrimination import EtDiscrimination
from ctmr.application.acceptance.distribution.fixed_world_baseline import (
    COMPENSATION_MM,
    FIXED_WORLD_WINDOW,
    QUANTITIES,
    SEED_SLOTS,
    FixedWorldBaselineReport,
    FixedWorldFieldScan,
    FixedWorldPairedReadings,
    InstrumentInputsRepository,
    JobAAnchor,
    JobBAnchor,
    main,
)
from ctmr.application.acceptance.distribution.measurement_table import MEASUREMENT_FIELDS, MeasurementTable
from ctmr.application.acceptance.distribution.zcrop_compensation import (
    GEN_RESAMPLED_Z,
    INSTRUMENT_Z,
    ZCropCompensation,
)

ARRAY_SHAPE = (INSTRUMENT_Z, 240, 240)  # zyx, the frozen prediction shape


class InMemoryMaskRepository:
    def __init__(self, masks):
        self.masks = masks

    def wt_mask(self, challenge, obs_id):
        return self.masks.get(obs_id)


class InMemoryBrainRepository:
    def __init__(self, brains):
        self.brains = brains

    def brain_mask(self, challenge, obs_id):
        return self.brains.get(obs_id)


def _slab(slices, label=1):
    """A label volume filling the given z ranges with one label, 20x20 in xy."""
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    for z_lo, z_hi in slices:
        mask[z_lo:z_hi, 100:120, 100:120] = label
    return mask


def _labeled_tumour(slices_by_label):
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    for label, slices in slices_by_label.items():
        for z_lo, z_hi in slices:
            mask[z_lo:z_hi, 100:120, 100:120] = label
    return mask


def _rows(case, challenge="GLI", **fields):
    """A real/gen CSV row pair carrying the given measurement fields verbatim."""
    rows = []
    for side in ("real", "gen"):
        row = dict.fromkeys(MEASUREMENT_FIELDS, "")
        row.update(obs_id=f"{case}__{side}", challenge=challenge, case=case, side=side)
        for name, (real, gen) in fields.items():
            row[name] = real if side == "real" else gen
        rows.append(row)
    return rows


# ------------------------------------------------- fixed-world window geometry


def test_fixed_world_window_is_job_as_registered_crop_window():
    """The post-fix declaration makes the registered geometry the world: the
    resampled 174-layer generated domain centre-crops onto the grid (crop
    start 9), the shared window with the real side is [9, 155) mm."""
    assert GEN_RESAMPLED_Z == 174
    assert FIXED_WORLD_WINDOW == ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    assert FIXED_WORLD_WINDOW.phys_lo == 9 and FIXED_WORLD_WINDOW.phys_hi == 155
    assert COMPENSATION_MM == 9


def test_region_restriction_reproduces_job_a_bit_for_bit_on_wt():
    """The WT volume / centroid-z face of the generalized window reading must
    equal job A's own restrict-and-measure exactly (the anchor certifies this
    equivalence on the real artifacts; here it is pinned on synthetics)."""
    real, gen = _slab([(40, 60)]), _slab([(53, 73)])
    for side, mask in (("real", real), ("gen", gen)):
        job_a = ZCropCompensation.restrict_and_measure(mask, FIXED_WORLD_WINDOW, side)
        mine = FixedWorldPairedReadings.measure_region(mask, FIXED_WORLD_WINDOW, side, "WT")
        assert mine["vol_ml"] == job_a["vol_ml"]
        assert mine["cz_mm"] == job_a["centroid_z_mm"]


def test_region_restriction_carries_the_crop_offset_only_on_z():
    """Gen-side physical z = window index + crop start 9; x/y stay plain grid
    indices (the crop window adds no x/y offset)."""
    gen = _slab([(53, 73)])
    reading = FixedWorldPairedReadings.measure_region(gen, FIXED_WORLD_WINDOW, "gen", "WT")
    assert reading["cz_mm"] == pytest.approx(62.5 + 9.0)  # slab centre 62.5, shifted into physical mm
    assert reading["cx_mm"] == pytest.approx(109.5)  # x columns 100..119, no offset
    assert reading["cy_mm"] == pytest.approx(109.5)
    real = FixedWorldPairedReadings.measure_region(_slab([(40, 60)]), FIXED_WORLD_WINDOW, "real", "WT")
    assert real["cz_mm"] == pytest.approx(49.5)  # the real side carries no offset


def test_region_restriction_projections_follow_the_label_vocabulary():
    """WT = {1,2,3} superset of TC = {1,3} superset of ET = {3} on one labelled
    tumour: labels 1 (oedema) z 40..50, label 2 z 50..55, label 3 z 55..60 --
    20/15/5 layers of 400 voxels at 0.001 ml."""
    mask = _labeled_tumour({1: [(40, 50)], 2: [(50, 55)], 3: [(55, 60)]})
    wt = FixedWorldPairedReadings.measure_region(mask, FIXED_WORLD_WINDOW, "real", "WT")
    tc = FixedWorldPairedReadings.measure_region(mask, FIXED_WORLD_WINDOW, "real", "TC")
    et = FixedWorldPairedReadings.measure_region(mask, FIXED_WORLD_WINDOW, "real", "ET")
    assert wt["vol_ml"] == pytest.approx(8.0)
    assert tc["vol_ml"] == pytest.approx(6.0)  # labels {1,3}
    assert et["vol_ml"] == pytest.approx(2.0)  # label 3 only
    assert et["cz_mm"] == pytest.approx(57.0)  # z 55..59 centre
    assert wt["cz_mm"] == pytest.approx(49.5)  # z 40..59 centre


def test_empty_region_reads_as_zero_volume_with_no_centroid():
    reading = FixedWorldPairedReadings.measure_region(_slab([(40, 60)], label=1), FIXED_WORLD_WINDOW, "real", "ET")
    assert reading == {"vol_ml": 0.0, "cx_mm": None, "cy_mm": None, "cz_mm": None}


# ------------------------------------------------------------- seed registry


def test_seed_table_reuses_job_a_bit_streams_for_the_anchor_quantities():
    """vol_wt_rel and centroid_wt_z re-draw job A's registered slots exactly --
    the two bit-streams behind the recorded report the anchor certifies."""
    assert SEED_SLOTS["vol_wt_rel"] == ("zcrop_vol_uncomp", "zcrop_vol_comp")
    assert SEED_SLOTS["centroid_wt_z"] == ("zcrop_centroid_uncomp", "zcrop_centroid_comp")
    assert QUANTITIES[0].seed("GLI", "comp") == DIAGNOSTIC_SEED_BASE + 1 * 1000 + 100
    assert QUANTITIES[3].seed("SSA", "uncomp") == DIAGNOSTIC_SEED_BASE + 2 * 1000 + 1


def test_seed_table_takes_the_next_free_blocks_for_the_new_quantities():
    """The 12 non-anchor quantities draw the T5 blocks (comp 400+index,
    uncomp 500+index, anchor indices 0/3 exempt) -- the next free banded slots
    after A (0/1/100/101), B (200) and the geometry audit (300)."""
    assert DIAGNOSTIC_SEED_SLOTS["t5_comp_centroid_wt_x"] == 401
    assert DIAGNOSTIC_SEED_SLOTS["t5_comp_et_wt_rel"] == 413
    assert DIAGNOSTIC_SEED_SLOTS["t5_uncomp_vol_tc_rel"] == 504
    spec = QUANTITIES[4]  # vol_tc_rel, judge index 4
    assert spec.seed("MEN", "comp") == DIAGNOSTIC_SEED_BASE + 3 * 1000 + 404
    assert spec.seed("MEN", "uncomp") == DIAGNOSTIC_SEED_BASE + 3 * 1000 + 504
    assert QUANTITIES[0].seed("GLI", "comp") != DIAGNOSTIC_SEED_BASE + 1 * 1000 + 400  # index 0 stays job A's


def test_quantity_margins_quote_the_frozen_envelopes():
    """The judge's margin builders over the ADR-0002 literals (vol per region,
    centroid axes per region, wt_brain on the WT volume margin, et_wt on ET+WT)."""
    assert QUANTITIES[0].margin("GLI") == pytest.approx(0.2802)  # E_WT,vol
    assert QUANTITIES[1].margin("GLI") == pytest.approx(5.38)  # E_WT,centroid
    assert QUANTITIES[8].margin("GLI") == pytest.approx(0.5702)  # E_ET,vol
    assert QUANTITIES[12].margin("GLI") == pytest.approx(0.2802)  # wt_brain_rel: E_WT,vol
    assert QUANTITIES[13].margin("GLI") == pytest.approx(0.5702 + 0.2802)  # et_wt_rel: E_ET,vol + E_WT,vol


# ------------------------------------------------------------- per-case pairing


def _identity_case_rows():
    """One in-window case: the same physical tumour at real grid [40,60) and
    gen grid [53,73) (the pad layout of physical [40,60)); the CSV carries the
    instrument's uncompensated readings (+13 mm coordinate artefact on z)."""
    return _rows("CASE-A", vol_wt_ml=("8.0", "8.0"), cz_wt_mm=("49.5", "62.5"))


def test_per_case_fixed_world_centroid_z_is_uncomp_plus_nine():
    """The fixed-world re-pairing moves the in-window centroid-z diff from the
    recorded +13 (pad layout) to exactly +22 = comp_crop (job A's +9 world)."""
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    readings = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, InMemoryBrainRepository({}))
    reading = readings[0]
    assert reading["centroid_wt_z_uncomp"] == pytest.approx(13.0)  # the recorded CSV artefact
    assert reading["centroid_wt_z_comp"] == pytest.approx(22.0)  # (62.5 + 9) - 49.5
    assert reading["vol_wt_rel_uncomp"] == pytest.approx(0.0)
    assert reading["vol_wt_rel_comp"] == pytest.approx(0.0)
    assert reading["excluded"] is None
    assert reading["brain_missing"] is True  # no inputs provided -> the ratio family drops, tallied


def test_per_case_volume_window_reads_match_job_as_block():
    """The WT window volumes enter both the relative volume and the centroid
    availability: 8.0 ml per side inside the [9,155) window on this pair."""
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, InMemoryBrainRepository({}))[0]
    assert reading["win_vol_wt_real_ml"] == pytest.approx(8.0)
    assert reading["win_vol_wt_gen_ml"] == pytest.approx(8.0)


def test_gen_empty_prediction_stays_in_the_volume_family_at_minus_one():
    """A generated-side empty prediction is a measurement result (protocol §4):
    -1.0 in both arms' volume distributions, no centroid axes anywhere."""
    rows = _rows("CASE-B", vol_wt_ml=("8.0", "0.0"), cz_wt_mm=("49.5", ""))
    masks = InMemoryMaskRepository({"CASE-B__real": _slab([(40, 60)]), "CASE-B__gen": np.zeros(ARRAY_SHAPE, dtype=np.uint8)})
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, masks, InMemoryBrainRepository({}))[0]
    assert reading["vol_wt_rel_uncomp"] == pytest.approx(-1.0)
    assert reading["vol_wt_rel_comp"] == pytest.approx(-1.0)
    assert reading["centroid_wt_z_uncomp"] is None
    assert reading["centroid_wt_z_comp"] is None


def test_ratio_families_read_the_brain_union_in_the_window():
    """wt_brain_rel compares the per-side WT/brain ratios restricted to the
    window; the recorded arm reads the CSV's own ratio columns."""
    brain_real = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    brain_real[20:130, :, :] = 1  # a 110-slice brain union
    brain_real[40:60, 100:120, 100:120] = 0  # carve the real tumour out of the union
    brain_gen = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    brain_gen[20:130, :, :] = 1  # the gen side's union keeps its full slab
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    brains = InMemoryBrainRepository({"CASE-A__real": brain_real, "CASE-A__gen": brain_gen})
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, brains)[0]
    # window brain: both unions restrict to [20,130) (110 slices of 57,600 voxels);
    # the real side additionally loses the carved 8,000-voxel tumour block:
    # gen = 6336.0 ml, real = 6328.0 ml
    assert reading["win_brain_gen_ml"] == pytest.approx(6336.0)
    assert reading["win_brain_real_ml"] == pytest.approx(6328.0)
    gen_ratio = 8.0 / 6336.0
    real_ratio = 8.0 / 6328.0
    assert reading["wt_brain_rel_comp"] == pytest.approx((gen_ratio - real_ratio) / real_ratio)
    assert reading["et_wt_rel_comp"] is None  # ET empty on both sides -> ratio undefined


def test_missing_prediction_excludes_the_fixed_world_arm_but_keeps_the_recorded():
    masks = InMemoryMaskRepository({})
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, InMemoryBrainRepository({}))[0]
    assert reading["excluded"] == "missing_prediction"
    assert reading["centroid_wt_z_uncomp"] == pytest.approx(13.0)  # the CSV survives
    assert reading["centroid_wt_z_comp"] is None


def test_missing_inputs_drop_only_the_ratio_family():
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, InMemoryBrainRepository({}))[0]
    assert reading["brain_missing"] is True
    assert reading["wt_brain_rel_comp"] is None
    assert reading["et_wt_rel_comp"] is None
    assert reading["centroid_wt_z_comp"] == pytest.approx(22.0)  # the mask families survive


def test_brain_cross_check_raises_when_the_inputs_tree_does_not_back_the_csv():
    """A full-grid brain union disagreeing with the recorded ``brain_ml`` means
    the inputs tree moved -- the ratio family would be silently wrong."""
    brain = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    brain[0:100, :, :] = 1  # 100x57600 voxels = 5760.0 ml, the CSV claims something else
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    brains = InMemoryBrainRepository({"CASE-A__real": brain, "CASE-A__gen": brain})
    rows = _rows("CASE-A", vol_wt_ml=("8.0", "8.0"), brain_ml=("1446.605", "1446.605"))
    with pytest.raises(DiagnosticError, match="inputs tree does not back"):
        FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, masks, brains)


def test_tc_et_families_read_their_own_window_projections():
    """A two-label tumour: TC (labels 1+3) and ET (label 3) re-pair against
    their own relative volumes inside the window."""
    real = _labeled_tumour({1: [(40, 50)], 3: [(50, 60)]})
    gen = _labeled_tumour({1: [(53, 63)], 3: [(63, 73)]})  # the same physical layout, pad-shifted
    masks = InMemoryMaskRepository({"CASE-C__real": real, "CASE-C__gen": gen})
    rows = _rows("CASE-C", vol_tc_ml=("8.0", "8.0"), vol_et_ml=("4.0", "4.0"), cz_et_mm=("54.5", "67.5"), vol_wt_ml=("8.0", "8.0"))
    reading = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, masks, InMemoryBrainRepository({}))[0]
    assert reading["vol_tc_rel_comp"] == pytest.approx(0.0)
    assert reading["vol_et_rel_comp"] == pytest.approx(0.0)
    assert reading["centroid_et_z_uncomp"] == pytest.approx(13.0)  # gen 67.5 vs real 54.5 grid
    assert reading["centroid_et_z_comp"] == pytest.approx(22.0)  # (67.5 + 9) - 54.5


def test_cases_are_paired_in_sorted_order():
    rows = _rows("CASE-B") + _rows("CASE-A")
    masks = InMemoryMaskRepository({})
    readings = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, masks, InMemoryBrainRepository({}))
    assert [(reading["case"]) for reading in readings] == ["CASE-A", "CASE-B"]


# --------------------------------------------------------- quantity block face


def test_quantity_block_builds_both_arms_with_attribution():
    rows = _identity_case_rows() + _rows(
        "CASE-D",
        vol_wt_ml=("4.0", "8.0"),
        cz_wt_mm=("49.5", "62.5"),
    )
    masks = InMemoryMaskRepository(
        {"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)]), "CASE-D__real": _slab([(40, 50)]), "CASE-D__gen": _slab([(53, 63)])}
    )
    readings = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, masks, InMemoryBrainRepository({}))
    block = FixedWorldBaselineReport.quantity_block(readings, "GLI", QUANTITIES[3], bootstrap_b=200)  # centroid_wt_z
    assert block["margin"] == pytest.approx(5.38)
    assert block["uncomp"]["median"] == pytest.approx(13.0)
    assert block["comp"]["median"] == pytest.approx(22.0)
    assert block["comp"]["n_cases"] == 2
    assert block["uncomp_ci_within_margin"] is False
    assert block["comp_ci_within_margin"] is False
    assert block["attribution"]["classification"] == "candidate_dominant"  # the compensation moves the offset further from 0
    assert block["attribution"]["measurement_fraction"] == 0.0


def test_quantity_block_degenerate_ci_and_registered_seed_identity():
    """A single-case block's CI degenerates to the value itself, and the
    block's seed is exactly the registered T5 slot's allocator draw."""
    rows = _rows("CASE-A", vol_wt_ml=("8.0", "8.0"), cz_wt_mm=("49.5", "62.5"), cx_wt_mm=("5.0", "18.0"))
    readings = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, InMemoryMaskRepository({}), InMemoryBrainRepository({}))
    spec = QUANTITIES[1]  # centroid_wt_x
    block = FixedWorldBaselineReport.quantity_block(readings, "GLI", spec, bootstrap_b=200)
    assert block["uncomp"]["n_cases"] == 1
    assert block["uncomp"]["median"] == block["uncomp"]["ci90_low"] == block["uncomp"]["ci90_high"]
    assert spec.seed("GLI", "comp") == DiagnosticSeedAllocator.seed("GLI", DIAGNOSTIC_SEED_SLOTS["t5_comp_centroid_wt_x"])


# ------------------------------------------------------------- anchor certificates


def test_job_a_anchor_passes_on_the_recorded_medians_and_flags_drift():
    def _per_challenge(offset):
        per_challenge = {}
        for name, by_challenge in JobAAnchor.RECORDED_COMP_MEDIANS.items():
            for challenge, recorded in by_challenge.items():
                per_challenge.setdefault(challenge, {})[name] = {"comp": {"median": recorded + offset}}
        return per_challenge

    assert JobAAnchor.verify(_per_challenge(0.0))["matched"] is True
    assert JobAAnchor.verify(_per_challenge(1e-3))["matched"] is False
    assert JobAAnchor.verify(_per_challenge(0.0))["centroid_wt_z"]["GLI"]["reproduced"] == pytest.approx(2.927747)


def test_job_a_recorded_literals_are_pinned_byte_exact():
    """The recorded job A comp medians (6-dp, the geometry audit's bit-exact
    reproduction), the numbers the issue's hard acceptance criterion names."""
    assert JobAAnchor.RECORDED_COMP_MEDIANS == {
        "vol_wt_rel": {"GLI": 5.207930, "MEN": 0.513802, "METS": -0.992933, "PED": 16.009951, "SSA": 5.907041},
        "centroid_wt_z": {"GLI": 2.927747, "MEN": -9.854146, "METS": 7.199478, "PED": 22.425059, "SSA": -2.480662},
    }
    assert JobAAnchor.TOLERANCE == 5e-7


def test_job_b_anchor_passes_on_the_recorded_tallies_and_flags_drift():
    readings = []
    for challenge, recorded in JobBAnchor.RECORDED.items():
        readings.append(
            {
                "challenge": challenge,
                "gen": {"k_detected": recorded["gen_k"], "n": recorded["gen_n"]},
                "pairing": {"real_only": recorded["real_only"]},
                "empty_pred": {"gen": {"k": recorded["gen_empty_pred"]}},
            }
        )
    assert JobBAnchor.verify(readings)["matched"] is True
    readings[2]["pairing"]["real_only"] = 9  # METS drifts
    assert JobBAnchor.verify(readings)["matched"] is False


def test_job_b_recorded_tallies_are_pinned_byte_exact():
    assert JobBAnchor.RECORDED == {
        "GLI": {"gen_k": 250, "gen_n": 250, "real_only": 0, "gen_empty_pred": 0},
        "MEN": {"gen_k": 200, "gen_n": 200, "real_only": 0, "gen_empty_pred": 0},
        "METS": {"gen_k": 38, "gen_n": 48, "real_only": 10, "gen_empty_pred": 5},
        "PED": {"gen_k": 20, "gen_n": 20, "real_only": 0, "gen_empty_pred": 0},
        "SSA": {"gen_k": 12, "gen_n": 12, "real_only": 0, "gen_empty_pred": 0},
    }


def test_et_discrimination_reuse_is_the_recorded_class():
    """The ET block re-runs the recorded job B class verbatim -- the P3 reuse
    hook contract (input face: the MEASUREMENT_FIELDS CSV)."""
    assert EtDiscrimination is not None
    readings = EtDiscrimination(bootstrap_b=200).discriminate(_identity_case_rows())
    assert readings[0]["challenge"] == "GLI"


# --------------------------------------------------------- field-of-view scan


def test_field_scan_zeroes_the_fixed_declared_edge_and_keeps_the_legacy_tally():
    """The issue's hard criterion in miniature: mass at z >= 128 leaves the
    declared domain under the post-fix declaration (174), zeroing the
    out-of-domain readout the write-path fix exists to remove."""
    real = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    real[130:132, 100:120, 100:120] = 1  # 2 layers in [128, 155): legacy-declared-out, fixed-in
    real[40:42, 100:120, 100:120] = 1
    rows = _rows("CASE-A")
    scan = FixedWorldFieldScan(FIXED_WORLD_WINDOW).scan(rows, InMemoryMaskRepository({"CASE-A__real": real}))
    assert scan["GLI"]["real_above_declared_fixed_ml"] == pytest.approx(0.0)
    assert scan["GLI"]["real_over_declared_fixed_cases"] == 0
    assert scan["GLI"]["real_above_declared_legacy_ml"] == pytest.approx(0.8)
    assert scan["GLI"]["real_over_declared_legacy_cases"] == 1
    assert scan["GLI"]["worst_over_declared_legacy"]["case"] == "CASE-A"


def test_field_scan_counts_window_floor_and_gen_hygiene():
    real = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    real[5:7, 100:120, 100:120] = 1  # 2 layers below the fixed window floor z=9
    gen = _slab([(53, 73)])  # content [53,73): nothing below the floor
    gen[3, 100:120, 100:120] = 1  # a hygiene anomaly at z=3 < 9
    rows = _rows("CASE-A")
    scan = FixedWorldFieldScan(FIXED_WORLD_WINDOW).scan(rows, InMemoryMaskRepository({"CASE-A__real": real, "CASE-A__gen": gen}))
    assert scan["GLI"]["real_below_window_ml"] == pytest.approx(0.8)
    assert scan["GLI"]["gen_below_window_ml"] == pytest.approx(0.4)
    assert scan["GLI"]["gen_below_window_cases"] == 1


def test_field_scan_of_a_clean_pair_is_all_zero():
    rows = _rows("CASE-A")
    scan = FixedWorldFieldScan(FIXED_WORLD_WINDOW).scan(
        rows, InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    )
    assert scan["GLI"]["real_above_declared_fixed_ml"] == 0.0
    assert scan["GLI"]["real_over_declared_fixed_cases"] == 0
    assert scan["GLI"]["real_above_declared_legacy_ml"] == 0.0
    assert scan["GLI"]["gen_below_window_ml"] == 0.0


# ---------------------------------------------------------------- report / CLI


def _readings_for_report():
    masks = InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})
    return FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(_identity_case_rows(), masks, InMemoryBrainRepository({}))


def test_report_writes_json_and_markdown_with_the_fixed_world_blocks(tmp_path):
    report = FixedWorldBaselineReport(
        measurements_path=Path("/controlled/measurements.csv"),
        pred_root=Path("/controlled/predictions"),
        inputs_root=Path("/controlled/inputs"),
        bootstrap_b=200,
    )
    readings = _readings_for_report()
    per_challenge = {"GLI": {spec.name: FixedWorldBaselineReport.quantity_block(readings, "GLI", spec, bootstrap_b=200) for spec in QUANTITIES}}
    et_readings = EtDiscrimination(bootstrap_b=200).discriminate(_identity_case_rows())
    job_a_anchor, job_b_anchor = JobAAnchor.verify(per_challenge), JobBAnchor.verify(et_readings)
    field_of_view = FixedWorldFieldScan(FIXED_WORLD_WINDOW).scan(_identity_case_rows(), InMemoryMaskRepository({}))
    json_path, md_path = report.write(readings, per_challenge, field_of_view, et_readings, job_a_anchor, job_b_anchor, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "fixed-world-baseline-diagnostic/1"
    assert payload["issue"] == 252
    assert payload["variant"] == "diagnostic"
    assert payload["run_id"] is None
    assert payload["worlds"]["fixed_world"]["declared_domain_mm"] == [0, 174]
    assert payload["worlds"]["fixed_world"]["compensation_mm"] == 9
    assert payload["worlds"]["legacy_world"]["compensation_mm"] == -13.0
    assert payload["quantity_order"][0] == "vol_wt_rel" and payload["quantity_order"][-1] == "et_wt_rel"
    assert payload["per_challenge"]["GLI"]["centroid_wt_z"]["comp"]["median"] == pytest.approx(22.0)
    assert payload["job_a_anchor"]["matched"] is False  # the synthetic median is not job A's
    assert payload["job_b_anchor"]["matched"] is False  # the synthetic tallies are not job B's
    assert payload["field_of_view"]["GLI"]["real_over_declared_fixed_cases"] == 0
    assert payload["et_discrimination"]["GLI"]["challenge"] == "GLI"
    md = md_path.read_text()
    assert "# 序列②T5" in md and "父 #247" in md and "不产生任何验收判定" in md
    assert "锚点对账" in md and "视场缺口" in md and "作业 B 协议复用" in md
    assert "centroid_wt_z" in md and "**否**" in md  # the anchor drift is loud in markdown


def test_cli_end_to_end_writes_the_report_and_fails_loud_on_the_anchor(tmp_path):
    """The full CLI on synthetic artifacts: the report lands either way and the
    anchor drift fails the run loudly (exit 1) -- on the real holdout artifacts
    the same path must exit 0 (the certificates' production-facing face)."""
    pred_root = tmp_path / "predictions" / "GLI"
    input_root = tmp_path / "inputs" / "GLI"
    pred_root.mkdir(parents=True)
    input_root.mkdir(parents=True)
    brain = np.full(ARRAY_SHAPE, 1, dtype=np.uint8)
    brain[40:60, 100:120, 100:120] = 0  # the real union loses the tumour block
    full = np.full(ARRAY_SHAPE, 1, dtype=np.uint8)
    for obs_id, slab, union in (("CASE-A__real", [(40, 60)], brain), ("CASE-A__gen", [(53, 73)], full)):
        sitk.WriteImage(sitk.GetImageFromArray(_slab(slab)), str(pred_root / f"{obs_id}.nii.gz"))
        for suffix in ("0000", "0001", "0002", "0003"):  # the four-channel input contract
            sitk.WriteImage(sitk.GetImageFromArray(union), str(input_root / f"{obs_id}_{suffix}.nii.gz"))

    def _row(obs_id, side, vol, cz):
        row = dict.fromkeys(MEASUREMENT_FIELDS, "")
        brain_ml = 8920.0 if side == "real" else 8928.0  # the full-grid unions of the carved / full brains
        row.update(obs_id=obs_id, challenge="GLI", case="CASE-A", side=side, vol_wt_ml=vol, cz_wt_mm=cz, brain_ml=brain_ml)
        return row

    csv_path = tmp_path / "measurements.csv"
    MeasurementTable.write([_row("CASE-A__real", "real", "8.0", "49.5"), _row("CASE-A__gen", "gen", "8.0", "62.5")], csv_path)
    exit_code = main(
        [
            "--measurements",
            str(csv_path),
            "--pred-root",
            str(tmp_path / "predictions"),
            "--inputs-root",
            str(tmp_path / "inputs"),
            "--output-dir",
            str(tmp_path / "out"),
            "--bootstrap-b",
            "200",
        ]
    )
    assert exit_code == 1  # the anchor drift is loud on synthetic data
    payload = json.loads((tmp_path / "out" / "fixed_world_baseline_diagnostic.json").read_text())
    assert payload["job_a_anchor"]["centroid_wt_z"]["GLI"]["matched"] is False
    assert payload["per_challenge"]["GLI"]["centroid_wt_z"]["comp"]["median"] == pytest.approx(22.0)
    gen_ratio, real_ratio = 8.0 / 8409.6, 8.0 / 8401.6  # full vs carved window brain union
    assert payload["per_challenge"]["GLI"]["wt_brain_rel"]["comp"]["median"] == pytest.approx((gen_ratio - real_ratio) / real_ratio)
    assert (tmp_path / "out" / "fixed_world_baseline_diagnostic.md").is_file()


def test_inputs_repository_reads_the_brain_union_from_the_nifti_tree(tmp_path):
    challenge_dir = tmp_path / "GLI"
    challenge_dir.mkdir(parents=True)
    empty = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    half = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    half[0:100, :, :] = 1
    sitk.WriteImage(sitk.GetImageFromArray(empty), str(challenge_dir / "OBS_0000.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(half), str(challenge_dir / "OBS_0001.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(empty), str(challenge_dir / "OBS_0002.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(half), str(challenge_dir / "OBS_0003.nii.gz"))
    brain = InstrumentInputsRepository(tmp_path).brain_mask("GLI", "OBS")
    assert brain is not None
    assert brain.sum() == 100 * 240 * 240  # the union of the two half-brains
