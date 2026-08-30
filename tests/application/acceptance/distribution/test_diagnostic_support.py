"""Diagnostic support module and the unified seed registry (ADR-0017 decisions 5/6, #232).

``diagnostic_support`` births the diagnostic reading support pieces: the one
``DiagnosticError`` for the whole diagnostic fleet, the ``variant=diagnostic``
report writer (the P3 reuse point, #205 series-③) and the diagnostic seed
allocator. ``challenge_registry`` now hosts the whole L2 seed space -- the
judge bootstrap band and the diagnostic namespace -- so the「L2 全域种子无碰撞」
invariant is a unit test on registered data, no longer prose discipline across
job files. The pre-#232 constants (judge band, job A/B slots) are pinned
byte-exactly here: the seed streams behind the recorded diagnostic reports
reproduce bit-for-bit.
"""

import json

import pytest

import ctmr.application.acceptance.distribution.et_discrimination as et_discrimination
import ctmr.application.acceptance.distribution.intensity_domain as intensity_domain
import ctmr.application.acceptance.distribution.token_dilution as token_dilution
import ctmr.application.acceptance.distribution.zcrop_compensation as zcrop_compensation
import ctmr.application.acceptance.distribution.zcrop_geometry_audit as zcrop_geometry_audit
from ctmr.application.acceptance.distribution.challenge_registry import (
    CHALLENGE_SEED_OFFSET,
    CHALLENGES,
    DIAGNOSTIC_SEED_BAND,
    DIAGNOSTIC_SEED_BASE,
    DIAGNOSTIC_SEED_SLOTS,
    GLOBAL_SEED,
)
from ctmr.application.acceptance.distribution.diagnostic_support import (
    DiagnosticError,
    DiagnosticReportWriter,
    DiagnosticSeedAllocator,
)
from ctmr.application.acceptance.distribution.final_acceptance import QuantityRegistry
from ctmr.application.acceptance.distribution.statistics import DistributionReadout
from ctmr.domain.vocabulary import REGIONS

# ── registry pins: the pre-#232 constants, byte-exact ───────────────────


def test_diagnostic_namespace_constants_are_pinned_byte_exact():
    assert DIAGNOSTIC_SEED_BASE == 900_000_000
    assert DIAGNOSTIC_SEED_BAND == 1000


def test_diagnostic_seed_slot_table_is_pinned_byte_exact():
    """Job A (#206) holds the uncompensated block 0/1 and the compensated block
    100/101 of each challenge band; job B (#207) takes slot 200."""
    assert DIAGNOSTIC_SEED_SLOTS == {
        "zcrop_vol_uncomp": 0,
        "zcrop_centroid_uncomp": 1,
        "zcrop_vol_comp": 100,
        "zcrop_centroid_comp": 101,
        "et_rel_diff": 200,
    }


# ── allocator: the band formula, byte-exact against the legacy modules ──


def test_allocator_derives_the_registered_band_formula():
    assert DiagnosticSeedAllocator.seed("GLI", 0) == 900_001_000
    assert DiagnosticSeedAllocator.seed("SSA", 200) == 900_002_200


def test_allocator_reproduces_the_pre_232_job_module_formulas_byte_exact():
    """The recorded diagnostic reports were drawn with these exact integers
    (the legacy in-module formulas, kept here as the reproduction witness)."""
    assert DiagnosticSeedAllocator.seed("GLI", DIAGNOSTIC_SEED_SLOTS["zcrop_vol_uncomp"]) == 900_000_000 + CHALLENGE_SEED_OFFSET["GLI"] * 1000 + 0
    assert (
        DiagnosticSeedAllocator.seed("GLI", DIAGNOSTIC_SEED_SLOTS["zcrop_centroid_uncomp"]) == 900_000_000 + CHALLENGE_SEED_OFFSET["GLI"] * 1000 + 1
    )
    assert DiagnosticSeedAllocator.seed("PED", DIAGNOSTIC_SEED_SLOTS["zcrop_vol_comp"]) == 900_000_000 + CHALLENGE_SEED_OFFSET["PED"] * 1000 + 100
    assert (
        DiagnosticSeedAllocator.seed("PED", DIAGNOSTIC_SEED_SLOTS["zcrop_centroid_comp"]) == 900_000_000 + CHALLENGE_SEED_OFFSET["PED"] * 1000 + 101
    )
    assert DiagnosticSeedAllocator.seed("MEN", DIAGNOSTIC_SEED_SLOTS["et_rel_diff"]) == 900_000_000 + CHALLENGE_SEED_OFFSET["MEN"] * 1000 + 200


def test_allocator_rejects_unregistered_challenges():
    with pytest.raises(KeyError):
        DiagnosticSeedAllocator.seed("XXX", 0)


# ── the L2 global no-collision invariant (ADR-0017 decision 5) ──────────


def test_diagnostic_seeds_are_pairwise_disjoint():
    seeds = [DiagnosticSeedAllocator.seed(challenge, slot) for challenge in CHALLENGES for slot in DIAGNOSTIC_SEED_SLOTS.values()]
    assert len(seeds) == len(set(seeds)) == len(CHALLENGES) * len(DIAGNOSTIC_SEED_SLOTS)


def test_diagnostic_namespace_stays_structurally_clear_of_the_judge_band():
    """Every judge draw lives inside one band-width window above GLOBAL_SEED +
    offset (tost indices and the round-trip block stay far below 1000), so a
    full band of headroom above the judge's reach separates the namespaces."""
    for offset in CHALLENGE_SEED_OFFSET.values():
        assert GLOBAL_SEED + offset + DIAGNOSTIC_SEED_BAND <= DIAGNOSTIC_SEED_BASE


def test_no_diagnostic_seed_collides_with_any_derived_judge_seed():
    """The enumerated invariant: the judge's actual derivation (tost: base+i
    over the registered quantity list; round_trip: base+100+region) against
    every registered diagnostic (challenge, slot) draw."""
    tost_width = len(QuantityRegistry().all())
    judge = set()
    for offset in CHALLENGE_SEED_OFFSET.values():
        base = GLOBAL_SEED + offset
        judge |= {base + index for index in range(tost_width)}
        judge |= {base + 100 + region_index for region_index in range(len(REGIONS))}
    diagnostic = {DiagnosticSeedAllocator.seed(challenge, slot) for challenge in CHALLENGES for slot in DIAGNOSTIC_SEED_SLOTS.values()}
    assert diagnostic.isdisjoint(judge)
    assert max(judge) == 20260821 + 5 + 100 + 2  # the judge band's byte-exact top anchor (3 regions)
    assert min(diagnostic) == 900_001_000  # and the diagnostic namespace's floor


# ── unified DiagnosticError (decision 6) ────────────────────────────────


def test_diagnostic_error_has_a_single_definition_across_the_fleet():
    for module in (zcrop_compensation, et_discrimination, intensity_domain, token_dilution):
        assert module.DiagnosticError is DiagnosticError, module.__name__
    assert issubclass(zcrop_geometry_audit.GeometryAuditError, DiagnosticError)


# ── the variant=diagnostic report writer ────────────────────────────────


def test_writer_payload_carries_the_diagnostic_prologue_in_the_recorded_order():
    writer = DiagnosticReportWriter(
        schema="x-diagnostic/1",
        title="作业 X 读数",
        issue=206,
        job_label="作业 A",
        stem="x_diagnostic",
        inputs={"measurements": "m.csv"},
        run_id="run-1",
    )
    payload = writer.payload({"per_case": []})
    assert list(payload)[:8] == ["schema", "title", "issue", "variant", "disclaimer", "run_id", "generated_utc", "inputs"]
    assert payload["schema"] == "x-diagnostic/1"
    assert payload["variant"] == "diagnostic"
    assert payload["run_id"] == "run-1"
    assert payload["generated_utc"].endswith("Z")
    assert payload["per_case"] == []  # the job body rides after the prologue


def test_writer_disclaimer_reproduces_the_recorded_text_byte_exact():
    writer = DiagnosticReportWriter(schema="s/1", title="t", issue=206, job_label="作业 A", stem="s", inputs={}, run_id=None)
    assert writer.disclaimer == (
        "诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#205 作业 A)。" "bootstrap 种子独立于正式判定链(诊断基 900000000)。"
    )


def test_writer_markdown_preamble_matches_the_recorded_report_shape():
    writer = DiagnosticReportWriter(schema="s/1", title="作业 B 读数", issue=207, job_label="作业 B", stem="s", inputs={}, run_id=None)
    preamble = "\n".join(writer.markdown_preamble(writer.payload({})))
    assert preamble.startswith(
        "# 作业 B 读数\n"
        "\n"
        "**Issue**: [#207](https://github.com/ACautomata/NV-Generate-CTMR/issues/207)(父 #205 作业 B) · **run**: `未绑定`\n"
        "**variant: diagnostic —— 诊断读数"
    )


def test_writer_writes_the_json_markdown_artifact_pair(tmp_path):
    writer = DiagnosticReportWriter(schema="s/1", title="t", issue=1, job_label="作业 X", stem="job_x_diagnostic", inputs={}, run_id=None)
    payload = writer.payload({"body_key": 1})
    json_path, md_path = writer.write(payload, "# t\n", tmp_path)
    assert json_path == tmp_path / "job_x_diagnostic.json"
    assert json_path.read_text() == json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    assert md_path == tmp_path / "job_x_diagnostic.md"
    assert md_path.read_text() == "# t\n"


# ── the shared distribution read-out (decision 1's statistics member) ───


def test_distribution_readout_follows_the_linear_quantile_rule():
    readout = DistributionReadout.of([1.0, 2.0, 3.0, 4.0])
    assert readout["median"] == pytest.approx(2.5)
    assert readout["mean"] == pytest.approx(2.5)
    assert readout["q05"] == pytest.approx(1.15)  # q*(n-1) linear rule
    assert readout["q95"] == pytest.approx(3.85)
    assert DistributionReadout.of([]) == {"median": None, "mean": None, "q05": None, "q95": None}
    # the paired relative difference is the neighbour primitive's job:
    # statistics.RelativeDifference (single definition, #231) owns rel_diff
    assert not hasattr(DistributionReadout, "rel_diff")
