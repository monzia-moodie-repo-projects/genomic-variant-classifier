"""A capability cannot claim validation it never climbed to.

WHY THIS FILE EXISTS
====================
`capabilities.py` checks that a capability's declared state is internally
consistent. It cannot check that the state was REACHED legitimately. Those are
different questions, and only the second prevents the failure that matters: a
capability flipped from "not built" to "validated" in a single commit has, by
construction, never existed in the state where the interesting defects live --
an artifact produced but wrong, a probe that runs but leaks, a metric computed
on the wrong tensor.

The instruction "do not flip all five directly from NOT_IMPLEMENTED to OK" was
written as prose. PROSE DOES NOT ENFORCE ITSELF. This module makes it
structural, and these tests make sure the structure can actually refuse.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    CapabilityState,
    MetricStatus,
    TargetState,
)
from genomic_variant_classifier.evaluation.capability_lifecycle import (
    CAPABILITY_LADDER,
    REASON_CHECKPOINT_SERIES_UNAVAILABLE,
    REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES,
    REASON_LOCAL_GEOMETRY_EVALUATOR_ABSENT,
    REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE,
    REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED,
    REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE,
    REASON_REPRESENTATION_NOT_EXPOSED,
    IllegalCapabilityTransition,
    TransitionVerdict,
    assert_transition_legal,
    panel_r_expected_ladder,
    transition_is_legal,
)

C = CapabilityState


# --------------------------------------------------------------------------- #
# 1. the ladder itself
# --------------------------------------------------------------------------- #
def test_the_ladder_is_ordered_and_excludes_deprecated():
    """DEPRECATED is reachable from anywhere and leads nowhere, so treating it
    as a rung would imply it can be climbed past."""
    assert CAPABILITY_LADDER == (
        C.NOT_IMPLEMENTED, C.IMPLEMENTED_NO_OUTPUT, C.OUTPUT_AVAILABLE, C.VALIDATED)
    assert C.DEPRECATED not in CAPABILITY_LADDER


def test_every_ladder_state_is_a_real_capability_state():
    for state in CAPABILITY_LADDER:
        assert isinstance(state, CapabilityState)


# --------------------------------------------------------------------------- #
# 2. forward moves advance exactly one rung
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("i", range(len(CAPABILITY_LADDER) - 1))
def test_each_single_rung_step_is_legal(i):
    v = transition_is_legal(CAPABILITY_LADDER[i], CAPABILITY_LADDER[i + 1])
    assert v.legal and v.direction == "forward" and v.rungs == 1


def test_the_two_rung_jump_is_refused():
    v = transition_is_legal(C.NOT_IMPLEMENTED, C.OUTPUT_AVAILABLE)
    assert not v.legal
    assert v.rungs == 2
    assert "implemented_no_output" in v.reason


def test_the_full_jump_from_nothing_to_validated_is_refused():
    """The exact move the specification forbids in prose."""
    v = transition_is_legal(C.NOT_IMPLEMENTED, C.VALIDATED)
    assert not v.legal
    assert v.rungs == 3
    assert "implemented_no_output" in v.reason
    assert "output_available" in v.reason


def test_the_refusal_explains_what_the_skipped_state_is_for():
    v = transition_is_legal(C.NOT_IMPLEMENTED, C.VALIDATED)
    assert "never been audited" in v.reason or "interesting defects" in v.reason


# --------------------------------------------------------------------------- #
# 3. backward moves and withdrawal are always available
# --------------------------------------------------------------------------- #
# Generated rather than filtered. A 4x4 cross-product with the forward and
# identity cases skipped would leave TEN permanent skips in the suite -- skips
# that exist only because the test was written lazily, and that dilute the
# signal from the seven real ones the project actually carries.
BACKWARD_PAIRS = [
    (CAPABILITY_LADDER[i], CAPABILITY_LADDER[j])
    for i in range(len(CAPABILITY_LADDER))
    for j in range(i)
]


@pytest.mark.parametrize("from_state,to_state", BACKWARD_PAIRS,
                         ids=lambda s: s.value)
def test_any_backward_distance_is_legal(from_state, to_state):
    """Discovering an artifact is wrong must never be HARDER to record than
    claiming it is right. A contract that made retraction awkward would quietly
    encourage not retracting."""
    v = transition_is_legal(from_state, to_state)
    assert v.legal and v.direction == "backward"


def test_the_backward_pairs_cover_every_descent():
    """Guards the generator: n*(n-1)/2 pairs for a ladder of n rungs. If this
    silently produced an empty list, the test above would vacuously pass."""
    n = len(CAPABILITY_LADDER)
    assert len(BACKWARD_PAIRS) == n * (n - 1) // 2 == 6


@pytest.mark.parametrize("state", list(CapabilityState))
def test_deprecation_is_reachable_from_anywhere(state):
    v = transition_is_legal(state, C.DEPRECATED)
    assert v.legal and v.direction == "withdrawal"


def test_a_deprecated_capability_cannot_be_revived_in_place():
    v = transition_is_legal(C.DEPRECATED, C.VALIDATED)
    assert not v.legal
    assert "new name" in v.reason


def test_staying_put_is_legal_and_is_not_called_forward():
    v = transition_is_legal(C.OUTPUT_AVAILABLE, C.OUTPUT_AVAILABLE)
    assert v.legal and v.direction == "none" and v.rungs == 0


# --------------------------------------------------------------------------- #
# 4. the verdict cannot lie
# --------------------------------------------------------------------------- #
def test_an_illegal_verdict_must_carry_a_reason():
    with pytest.raises(ValueError, match="must carry a reason"):
        TransitionVerdict(False, C.NOT_IMPLEMENTED, C.VALIDATED, "forward", 3, None)


def test_a_legal_verdict_must_not_carry_a_reason():
    """A reason explains a REFUSAL. Attaching one to a pass invites a reader to
    skim the reason and mistake it for a warning."""
    with pytest.raises(ValueError, match="must not carry a reason"):
        TransitionVerdict(True, C.NOT_IMPLEMENTED, C.IMPLEMENTED_NO_OUTPUT,
                          "forward", 1, "looks fine")


def test_a_verdict_is_immutable():
    v = transition_is_legal(C.NOT_IMPLEMENTED, C.VALIDATED)
    with pytest.raises(Exception):
        v.legal = True


@pytest.mark.parametrize("bad", ["validated", 3, None, MetricStatus.OK])
def test_a_non_capability_state_is_a_type_error_not_a_silent_pass(bad):
    with pytest.raises(TypeError):
        transition_is_legal(C.NOT_IMPLEMENTED, bad)
    with pytest.raises(TypeError):
        transition_is_legal(bad, C.VALIDATED)


# --------------------------------------------------------------------------- #
# 5. the raising form actually raises
# --------------------------------------------------------------------------- #
def test_assert_transition_legal_raises_on_a_skip():
    with pytest.raises(IllegalCapabilityTransition):
        assert_transition_legal(C.NOT_IMPLEMENTED, C.VALIDATED)


def test_assert_transition_legal_returns_the_verdict_on_a_pass():
    v = assert_transition_legal(C.OUTPUT_AVAILABLE, C.VALIDATED)
    assert v.legal and v.rungs == 1


def test_the_raised_error_is_not_a_bare_valueerror_by_accident():
    assert issubclass(IllegalCapabilityTransition, ValueError)


# --------------------------------------------------------------------------- #
# 6. the Panel R map
# --------------------------------------------------------------------------- #
def test_all_five_panel_r_stages_are_mapped():
    assert set(panel_r_expected_ladder()) == {
        "panel_r3_norm_angle_decomposition",
        "panel_r4_conditioning_recoverability",
        "panel_r5_hubness_local_geometry",
        "panel_r6_training_trajectory",
        "panel_r7_downstream_sensitivity",
    }


@pytest.mark.parametrize("name", sorted(panel_r_expected_ladder()))
def test_every_mapped_stage_climbs_the_ladder_in_order(name):
    states = [rung[0] for rung in panel_r_expected_ladder()[name]]
    assert states == list(CAPABILITY_LADDER)


@pytest.mark.parametrize("name", sorted(panel_r_expected_ladder()))
def test_every_step_of_every_map_is_a_legal_transition(name):
    """The map must not itself describe an illegal path."""
    rungs = panel_r_expected_ladder()[name]
    for (a, _ra, _wa), (b, _rb, _wb) in zip(rungs, rungs[1:]):
        assert transition_is_legal(a, b).legal


@pytest.mark.parametrize("name", sorted(panel_r_expected_ladder()))
def test_only_the_final_rung_has_no_blocking_reason(name):
    """Every state below VALIDATED must name what is missing. A capability that
    cannot say why it is not validated is not documented, it is merely stalled."""
    rungs = panel_r_expected_ladder()[name]
    for state, reason, _why in rungs[:-1]:
        assert reason, f"{name} at {state.value} has no blocking reason"
    assert rungs[-1][0] is C.VALIDATED
    assert rungs[-1][1] is None


def test_r3_and_r4_are_blocked_on_the_same_thing():
    """They share infrastructure, so they must not claim independent blockers."""
    lad = panel_r_expected_ladder()
    r3 = lad["panel_r3_norm_angle_decomposition"][0][1]
    r4 = lad["panel_r4_conditioning_recoverability"][0][1]
    assert r3 == r4 == REASON_REPRESENTATION_NOT_EXPOSED


def test_the_blocking_reason_names_the_actual_return_boundary():
    lad = panel_r_expected_ladder()
    why = lad["panel_r3_norm_angle_decomposition"][0][2]
    assert "gnn.py" in why and "classifier(focal_embeddings)" in why


def test_r6_and_r7_are_blocked_on_replicates_not_on_code():
    """One checkpoint is not a trajectory, and no amount of code fixes that."""
    lad = panel_r_expected_ladder()
    assert lad["panel_r6_training_trajectory"][2][1] == REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES
    assert lad["panel_r7_downstream_sensitivity"][2][1] == REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES


def test_r4_names_the_leakage_that_would_invalidate_it():
    lad = panel_r_expected_ladder()
    why = lad["panel_r4_conditioning_recoverability"][2][2]
    assert "leakage" in why and "no labels" in why


def test_r5_names_the_estimand_choices_that_would_silently_change_it():
    lad = panel_r_expected_ladder()
    why = lad["panel_r5_hubness_local_geometry"][0][2]
    for token in ("cosine", "approximate", "zero-vector"):
        assert token in why


def test_every_reason_constant_is_distinct():
    """Two capabilities blocked for different reasons must not read identically."""
    reasons = [
        REASON_REPRESENTATION_NOT_EXPOSED,
        REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE,
        REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED,
        REASON_LOCAL_GEOMETRY_EVALUATOR_ABSENT,
        REASON_CHECKPOINT_SERIES_UNAVAILABLE,
        REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE,
        REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES,
    ]
    assert len(set(reasons)) == len(reasons)
    assert all(r and not r.endswith(" ") for r in reasons)


def test_the_map_asserts_nothing_about_the_present_state():
    """It is a MAP, not a gate. The map records the ROUTE each Panel R stage must
    climb; it must not be confused with, contradict, or overwrite the stages'
    PRESENT states, which live in panel_r_capabilities() and change as code lands.

    Updated 2026-07-21: when the extraction boundary landed, R3/R4/R5 advanced to
    IMPLEMENTED_NO_OUTPUT while R6/R7 stayed NOT_IMPLEMENTED. The map is unchanged
    -- it still describes the full NOT_IMPLEMENTED -> VALIDATED route for all five
    -- which is precisely the point: a MAP of the whole journey does not move when
    a traveller takes a step. This test asserts the map still covers exactly the
    five stages and remains a static description, regardless of present state."""
    from genomic_variant_classifier.evaluation.representation_geometry import (
        panel_r_capabilities)
    live = {c.capability_name: c for c in panel_r_capabilities()}
    # the map covers exactly the stages that exist
    assert set(live) == set(panel_r_expected_ladder())
    # every stage's map still BEGINS at NOT_IMPLEMENTED and ENDS at VALIDATED,
    # independent of where the stage presently sits -- that is what makes it a map.
    for name, ladder in panel_r_expected_ladder().items():
        states = [step[0] for step in ladder]
        assert states[0] is C.NOT_IMPLEMENTED
        assert states[-1] is C.VALIDATED
    # and the present states are whatever the code has reached -- not forced to
    # match the map's starting rung.
    for c in live.values():
        assert c.target_state is TargetState.ABSENT
