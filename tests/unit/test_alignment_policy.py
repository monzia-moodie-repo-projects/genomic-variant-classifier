"""tests/unit/test_alignment_policy.py

Author: Monzia Moodie
Written 2026-08-07. ALIGNMENT-1, part of GATE-1.

One definition of "the score<->label join looks broken", consumed by both
places that used to carry it independently.

WHY THIS FILE EXISTS. `conformal/calibrate.py` and
`scripts/forensics/verify_oof_alignment.py` each declared `0.90` with no shared
source. Two copies of a number are wrong in at least one place eventually, and
GATE-1's original census of four AUROC thresholds missed the forensic one
entirely -- recorded as a CENSUS CORRECTION, not as a fifth policy.

THE MOST IMPORTANT TEST IN THIS FILE is the one asserting that both consumers
resolve to the same default. Everything else checks the type; that one checks
the thing the type was created to prevent.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.alignment import (
    DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY,
    AlignmentVerdict,
    ScoreLabelAlignmentPolicy,
)
from genomic_variant_classifier.monitoring.model_registry import (
    PolicyEvidenceStatus,
    PolicyProvenance,
)


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #

def test_the_default_minimum_is_the_inherited_090():
    """Not because 0.90 is right, but because changing it silently would be a
    different decision wearing this one's clothes."""
    assert DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.minimum_auroc == 0.90


def test_the_default_policy_is_not_claimed_to_be_justified():
    """0.90 was inherited from two independent declarations and no record of
    what established it. Typing a threshold does not validate it."""
    provenance = DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.provenance
    assert provenance.status is (
        PolicyEvidenceStatus.LEGACY_PENDING_JUSTIFICATION)
    assert provenance.is_justified is False
    assert "verify_oof_alignment" in provenance.source
    assert "calibrate" in provenance.source


@pytest.mark.parametrize("minimum", [0.5, 0.4, 0.0, -0.1, 1.01, 2.0])
def test_a_minimum_outside_the_open_unit_half_is_refused(minimum):
    """At or below 0.5 the policy cannot distinguish a broken join from
    chance, which is the only thing it exists to detect."""
    with pytest.raises(ValueError, match=r"\(0.5, 1.0\]"):
        ScoreLabelAlignmentPolicy(minimum_auroc=minimum)


@pytest.mark.parametrize("minimum", [0.51, 0.90, 0.99, 1.0])
def test_a_minimum_inside_the_range_is_accepted(minimum):
    assert ScoreLabelAlignmentPolicy(
        minimum_auroc=minimum).minimum_auroc == minimum


def test_the_policy_is_frozen():
    """A shared default that could be mutated in place would be a global
    variable with extra steps."""
    with pytest.raises(Exception):
        DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.minimum_auroc = 0.5


def test_an_unsourced_provenance_is_refused():
    with pytest.raises(Exception, match="stated source"):
        PolicyProvenance(status=PolicyEvidenceStatus.JUSTIFIED, source="")


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("auroc,plausible", [
    (1.00, True),
    (0.9988, True),
    (0.90, True),      # the boundary is INCLUSIVE
    (0.8999, False),
    (0.51, False),
    (0.0, False),
])
def test_the_boundary_is_inclusive(auroc, plausible):
    verdict = DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.judge(auroc)
    assert verdict.plausible is plausible


def test_judging_does_not_raise():
    """`calibrate.py` refuses to proceed; the forensic script prints a flag and
    continues. Both are legitimate responses to the same finding, and a policy
    that raised would force one of them on the other."""
    verdict = DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.judge(0.10)
    assert verdict.plausible is False
    assert isinstance(verdict, AlignmentVerdict)


def test_the_verdict_carries_the_threshold_it_was_judged_against():
    """A bare boolean is what PROD-1 removed from the web service: a verdict
    detached from the standard that produced it."""
    verdict = ScoreLabelAlignmentPolicy(minimum_auroc=0.95).judge(0.93)
    assert verdict.auroc == pytest.approx(0.93)
    assert verdict.minimum_auroc == 0.95
    assert verdict.plausible is False
    assert "0.9300" in verdict.describe()
    assert "0.95" in verdict.describe()
    assert "SUSPECT" in verdict.describe()


def test_a_plausible_verdict_says_so_without_shouting():
    assert "plausible" in DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.judge(
        0.99).describe()


# --------------------------------------------------------------------------- #
# The point of the whole exercise
# --------------------------------------------------------------------------- #

def test_conformal_calibration_uses_the_shared_default():
    """`CalibrationConfig` must not carry its own copy of the number."""
    from genomic_variant_classifier.conformal.calibrate import (
        CalibrationConfig)

    config = CalibrationConfig()
    assert config.score_label_alignment_policy.minimum_auroc == (
        DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.minimum_auroc)
    assert not hasattr(config, "auroc_floor"), (
        "auroc_floor was renamed to make its MEANING impossible to confuse "
        "with a production-quality AUROC; a surviving alias would restore "
        "exactly that confusion")


def test_the_forensic_script_uses_the_shared_default():
    """The forensic checker consumes the same authority WITHOUT importing
    conformal calibration -- that would be a backwards dependency from a
    general integrity check into a specific statistical method."""
    import ast
    import pathlib

    for candidate in pathlib.Path(__file__).resolve().parents:
        script = (candidate / "scripts" / "forensics"
                  / "verify_oof_alignment.py")
        if script.is_file():
            break
    else:
        raise AssertionError("verify_oof_alignment.py was not found")

    source = script.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)

    imported_from = {node.module for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
    assert any("evaluation.alignment" in module for module in imported_from), (
        "the forensic script must consume the shared alignment policy")
    assert not any("conformal" in module for module in imported_from), (
        "the forensic script must not depend on conformal calibration")

    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, float)
                and node.value == 0.90]
    assert not literals, (
        "the forensic script still declares 0.90 itself; two copies of a "
        "number are wrong in at least one place eventually")
