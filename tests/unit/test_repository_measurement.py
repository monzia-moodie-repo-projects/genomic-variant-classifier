"""A measurement may claim no more than its corpus and completeness establish.

Created 2026-09-05. ADR-0005.

WHY
---
Four repository inspections on 2026-09-04 and 2026-09-05 produced conclusions
exceeding their evidence. Each was individually reasonable; none recorded what
population was inspected, how completely, or what the result licensed:

    a scan for `import audit_data_tree` returned ZERO while the gate was
    demonstrably invoked, because the wiring loads by path; the same scan
    counted 31 "invocations" of preflight_data_guard, every one Markdown prose

    a false-positive rate calibrated on EIGHT documents was applied to 1,637
    files and produced 2,408 noise tokens

    a `git grep` pathspec matching NO FILE exited 0 with no output, and the
    silence was read as zero matches in an existing file

    fifteen findings were called coherent on narrative heading sequence

WHAT THESE TESTS ARE
--------------------
The four sabotage cases the adopted plan requires, plus the invariants each
type carries. They are NEGATIVE-HEAVY on purpose: the value of this layer is
what it REFUSES, and a type whose refusals are untested refuses nothing in
practice.

`test_the_round_trip_is_exact` is the one positive structural test that would
catch a serializer and parser drifting apart -- the shape that let four
installers each carry a private notion of "neutral".

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.repository_measurement.claims import (
    MeasurementClaim)
from genomic_variant_classifier.repository_measurement.corpus import (
    CorpusKind, CorpusSnapshot, CorpusSpec, SelectionCoverage,
    corpus_membership_digest)
from genomic_variant_classifier.repository_measurement.evidence import (
    AnalysisCoverage, EvidenceItem, EvidenceStrength,
    IncompleteMeasurementError, require_complete_census)
from genomic_variant_classifier.repository_measurement.report import (
    MeasurementMode, MeasurementResult, Verdict)
from genomic_variant_classifier.repository_measurement.serialization import (
    MeasurementSchemaError, SCHEMA, SCHEMA_VERSION, parse_measurement,
    serialize_measurement)


def _spec(**kw):
    base = dict(kind=CorpusKind.TRACKED, selector="tests/**/test_*.py",
                enumerator="git ls-tree -r -z HEAD", minimum_members=1)
    base.update(kw)
    return CorpusSpec(**base)


def _snapshot(members=("a.py", "b.py"), **kw):
    base = dict(repository_head="f22edc5", worktree_dirty=False)
    base.update(kw)
    return CorpusSnapshot(spec=kw.pop("spec", _spec()), members=members, **base)


def _claim():
    return MeasurementClaim(
        proves=("Every selected source was parsed.",),
        does_not_prove=("Runtime reachability.",),
        method="Enumerate tracked blobs, parse each with ast.")


def _evidence():
    return (EvidenceItem(statement="2 sources selected.",
                         strength=EvidenceStrength.DIRECT,
                         basis="git ls-tree -r -z HEAD"),)


def _result(**kw):
    base = dict(corpus=_snapshot(), mode=MeasurementMode.CENSUS,
                claim=_claim(), evidence=_evidence(),
                coverage=AnalysisCoverage(selected=2, attempted=2,
                                          succeeded=2, failed=0))
    base.update(kw)
    return MeasurementResult(**base)


# ---------------------------------------------------------------------------
# 1. The four sabotage cases
# ---------------------------------------------------------------------------

def test_equal_member_counts_do_not_mean_equal_corpus():
    """|A| == |B| does not imply A == B.

    Two corpora can each hold 357 files and hold different 357 files. This is
    the same invariant SuiteTransition enforces for test node identity, and
    the reason membership is hashed rather than counted.
    """
    left = ("a.py", "b.py")
    right = ("a.py", "c.py")
    assert len(left) == len(right)
    assert corpus_membership_digest(left) != corpus_membership_digest(right)


def test_incomplete_analysis_cannot_claim_a_complete_census():
    """Parse failures must not coexist with `complete census = true`."""
    with pytest.raises(ValueError):
        _result(coverage=AnalysisCoverage(selected=100, attempted=10,
                                          succeeded=10, failed=0),
                complete_census=True)


def test_discovery_cannot_masquerade_as_a_complete_census():
    """DISCOVERY evidence cannot license an absence claim.

    This is the rule that would have prevented: grep found no
    `import audit_data_tree`, therefore runtime invocation count = 0.
    """
    with pytest.raises(ValueError):
        _result(mode=MeasurementMode.DISCOVERY, complete_census=True)


def test_a_missing_requested_root_is_not_a_zero_result():
    """`git grep -- tests/foo.py` where foo.py does not exist exits 0 SILENTLY.

    A requested root whose fate is unrecorded is exactly that failure. The
    type refuses to construct rather than reporting an empty selection.
    """
    with pytest.raises(ValueError):
        SelectionCoverage(requested_roots=("exists.py", "missing.py"),
                          resolved_roots=("exists.py",),
                          missing_roots=())
    ok = SelectionCoverage(requested_roots=("exists.py", "missing.py"),
                           resolved_roots=("exists.py",),
                           missing_roots=("missing.py",),
                           enumeration_complete=False)
    assert ok.missing_roots == ("missing.py",)


# ---------------------------------------------------------------------------
# 2. Verdicts belong to predicates, and NOT_JUDGED is not PASS
# ---------------------------------------------------------------------------

def test_a_predicate_requires_a_verdict():
    with pytest.raises(ValueError):
        _result(mode=MeasurementMode.PREDICATE)


def test_a_bare_pass_is_impossible():
    """A verdict with no evidence is an assertion, not a measurement."""
    with pytest.raises(ValueError):
        _result(mode=MeasurementMode.PREDICATE, evidence=(),
                verdict=Verdict.PASS)


def test_a_descriptive_census_may_not_carry_pass_or_fail():
    """The authority catalog is a census. It adjudicated no proposition, so a
    PASS would be fabricated."""
    for bad in (Verdict.PASS, Verdict.FAIL):
        with pytest.raises(ValueError):
            _result(verdict=bad)


def test_a_census_may_carry_not_judged():
    assert _result(verdict=Verdict.NOT_JUDGED).verdict is Verdict.NOT_JUDGED


def test_not_judged_is_not_pass_and_does_not_count_as_one():
    assert Verdict.NOT_JUDGED is not Verdict.PASS
    rows = (Verdict.PASS, Verdict.NOT_JUDGED, Verdict.FAIL)
    assert sum(v is Verdict.PASS for v in rows) == 1


# ---------------------------------------------------------------------------
# 3. Claims must not contradict themselves
# ---------------------------------------------------------------------------

def test_a_proposition_cannot_be_both_proved_and_not_proved():
    with pytest.raises(ValueError):
        MeasurementClaim(proves=("X",), does_not_prove=("X",), method="m")


def test_a_measurement_must_establish_something():
    with pytest.raises(ValueError):
        MeasurementClaim(proves=(), does_not_prove=("Y",), method="m")


def test_claim_order_is_the_authors_and_is_not_sorted():
    """proves/does_not_prove are semantic prose. Silently sorting them would
    change the emphasis the author chose."""
    c = MeasurementClaim(proves=("zebra", "apple"), does_not_prove=(),
                         method="m")
    assert c.proves == ("zebra", "apple")


# ---------------------------------------------------------------------------
# 4. Corpus identity and tracked-corpus honesty
# ---------------------------------------------------------------------------

def test_members_must_be_canonically_sorted_and_unique():
    with pytest.raises(ValueError):
        CorpusSnapshot(spec=_spec(), members=("b.py", "a.py"),
                       repository_head="f22edc5", worktree_dirty=False)
    with pytest.raises(ValueError):
        CorpusSnapshot(spec=_spec(), members=("a.py", "a.py"),
                       repository_head="f22edc5", worktree_dirty=False)


def test_a_tracked_corpus_requires_a_head_and_refuses_a_dirty_worktree():
    """Printing "measured at c18a1df" for a dirty worktree would attribute
    uncommitted bytes to that commit."""
    with pytest.raises(ValueError):
        CorpusSnapshot(spec=_spec(), members=("a.py",))
    with pytest.raises(ValueError):
        CorpusSnapshot(spec=_spec(), members=("a.py",),
                       repository_head="f22edc5", worktree_dirty=True)


def test_a_tracked_spec_cannot_include_untracked_or_ignored_artifacts():
    for kw in ({"includes_untracked": True}, {"includes_ignored": True}):
        with pytest.raises(ValueError):
            _spec(**kw)


def test_minimum_members_separates_a_legitimate_zero_from_a_broken_selector():
    with pytest.raises(ValueError):
        CorpusSnapshot(spec=_spec(minimum_members=1), members=(),
                       repository_head="f22edc5", worktree_dirty=False)
    empty = CorpusSnapshot(spec=_spec(minimum_members=0), members=(),
                           repository_head="f22edc5", worktree_dirty=False)
    assert empty.n_members == 0


def test_the_membership_digest_is_domain_separated_and_unambiguous():
    """Length-delimited encoding: two different member lists cannot collide by
    concatenation."""
    assert corpus_membership_digest(("ab", "c")) != \
        corpus_membership_digest(("a", "bc"))
    assert len(corpus_membership_digest(("a",))) == 64


# ---------------------------------------------------------------------------
# 5. Analysis coverage arithmetic
# ---------------------------------------------------------------------------

def test_coverage_arithmetic_is_enforced():
    with pytest.raises(ValueError):
        AnalysisCoverage(selected=1, attempted=2, succeeded=2, failed=0)
    with pytest.raises(ValueError):
        AnalysisCoverage(selected=9, attempted=9, succeeded=8, failed=0)


def test_selected_1637_attempted_8_is_not_complete():
    """The measured shape: a rate calibrated on eight documents, applied to
    1,637 files."""
    c = AnalysisCoverage(selected=1637, attempted=8, succeeded=8, failed=0)
    assert not c.fully_attempted
    assert not c.complete


def test_a_universal_negative_requires_complete_analysis():
    with pytest.raises(IncompleteMeasurementError):
        require_complete_census(
            AnalysisCoverage(selected=10, attempted=3, succeeded=3, failed=0))
    require_complete_census(
        AnalysisCoverage(selected=10, attempted=10, succeeded=10, failed=0))


def test_evidence_requires_a_stated_basis():
    with pytest.raises(ValueError):
        EvidenceItem(statement="s", strength=EvidenceStrength.DIRECT, basis="  ")


# ---------------------------------------------------------------------------
# 6. Transport: strict, versioned, round-tripping
# ---------------------------------------------------------------------------

def test_the_round_trip_is_exact():
    original = _result(complete_census=True)
    assert parse_measurement(serialize_measurement(original)) == original


def test_serialisation_is_deterministic():
    r = _result()
    assert serialize_measurement(r) == serialize_measurement(r)


def test_an_unknown_schema_version_is_refused():
    text = serialize_measurement(_result()).replace(
        '"schema_version":{}'.format(SCHEMA_VERSION), '"schema_version":99')
    with pytest.raises(MeasurementSchemaError):
        parse_measurement(text)


def test_an_unknown_schema_name_is_refused():
    text = serialize_measurement(_result()).replace(SCHEMA, "some.other.schema")
    with pytest.raises(MeasurementSchemaError):
        parse_measurement(text)


def test_an_unknown_key_is_refused_rather_than_ignored():
    """`member_counts` must not coexist silently beside `member_count`."""
    text = serialize_measurement(_result()).replace(
        '"mode":', '"member_counts":357,"mode":')
    with pytest.raises(MeasurementSchemaError):
        parse_measurement(text)


def test_members_disagreeing_with_their_own_identity_are_refused():
    text = serialize_measurement(_result()).replace('"b.py"', '"z.py"')
    with pytest.raises(MeasurementSchemaError):
        parse_measurement(text)


def test_a_payload_that_is_not_json_is_refused():
    with pytest.raises(MeasurementSchemaError):
        parse_measurement("not json at all")


# ---------------------------------------------------------------------------
# 7. One object, two renderings
# ---------------------------------------------------------------------------

def test_the_human_rendering_derives_from_the_same_object():
    """Independently assembled prose and payload eventually disagree, and then
    nobody can say which was the measurement."""
    r = _result(complete_census=True)
    text = r.render()
    assert r.corpus.membership_sha256[:16] in text
    assert str(r.corpus.n_members) in text
    assert r.claim.proves[0] in text
    assert r.claim.does_not_prove[0] in text
    assert "none -- descriptive measurement" in text


def test_the_rendering_names_a_dirty_worktree_rather_than_a_commit():
    spec = _spec(kind=CorpusKind.WORKTREE, minimum_members=0)
    snap = CorpusSnapshot(spec=spec, members=("a.py",),
                          repository_head="f22edc5", worktree_dirty=True)
    text = _result(corpus=snap).render()
    assert "DIRTY" in text
    assert "base head" in text
