"""Readiness is not assessment, and a capability gap is not a negative result.

Created 2026-08-24.

WHAT THIS GUARDS
----------------
`.github/workflows/drift_monitor.yml` ran a monthly cron invoking
`scripts/run_drift_monitor.py` with neither `--new-data` nor `--new-clinvar`.
MEASURED against `run_drift_monitor.py:313-351`, that combination takes the else
branch and returns `EXIT_NOT_CHECKED`: the job could not reach a verdict BY
CONSTRUCTION, every month, for the life of the workflow.

Its own comment named the older form of the same defect: "THIS USED TO
`return 0`. Exit 0 means 'checked, clean'. There was no new data; nothing was
checked ... so the monthly drift monitor took this branch EVERY TIME and
reported a clean bill of health, for its entire life."

These cases keep the repaired semantics from collapsing back.

THE DISTINCTION THAT MATTERS MOST
---------------------------------
`NOT_READY` says readiness WAS evaluated and the answer is no.
`UNDETERMINED` says readiness could not be evaluated at all.

MEASURED 2026-08-24 across 1,622 tracked files: `ObservationCohort`,
`CohortRecord`, `cohort_id`, `candidate_population`, `new_observation`,
`production_cohort`, `inference_batch` and `CandidatePopulation` occur ZERO
times. The 42 lines combining a discovery verb with a population noun all glob
data-source artifacts, sequence-window shards, or training splits. Nothing
discovers a new observation population.

So the present state is UNDETERMINED. Claiming `NO_NEW_OBSERVATION_POPULATION`
would assert a fact about the world; `CANDIDATE_DISCOVERY_NOT_IMPLEMENTED`
asserts a fact about this repository, which is the only thing measured.

WHY THESE ARE THE CASES THEY ARE
--------------------------------
Every assertion below corresponds to a mutation a future contributor or agent
could plausibly make while "simplifying": widening UNDETERMINED to NOT_READY,
promoting the capability reason to the population reason, flipping
`feature_drift_checked`, replacing the legacy `UNKNOWN` projection with `none`,
or relaxing the READY invariant. All five were exercised against this module
before these tests were written, and each broke an assertion here.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.monitoring.drift_readiness import (
    READINESS_SCHEMA_VERSION,
    DriftReadiness,
    DriftReadinessReason,
    DriftReadinessStatus,
    as_document,
    current_feature_drift_readiness,
    render_github_output_lines,
    render_json,
    validate_document,
)

MODULE = (Path(__file__).resolve().parents[2] / "src"
          / "genomic_variant_classifier" / "monitoring" / "drift_readiness.py")

#: A readiness check must run on a hosted runner to report that it has nothing
#: to measure. MEASURED 2026-08-24: drift_detector.py imports numpy, pandas,
#: scipy and scipy.spatial.distance AT MODULE LEVEL, and
#: drift_reference_profile.py imports numpy and pandas at module level.
FORBIDDEN_IMPORTS = frozenset({
    "numpy", "pandas", "scipy", "sklearn", "nannyml", "torch", "pyarrow",
    "matplotlib", "lightgbm", "xgboost", "catboost",
})


# ---------------------------------------------------------------------------
# 1. THE PRESENT STATE
# ---------------------------------------------------------------------------

def test_missing_discovery_capability_is_undetermined_not_not_ready():
    """The distinction the whole module exists for.

    NOT_READY would claim the system looked. It cannot look: there is no
    discovery authority to look with.
    """
    result = current_feature_drift_readiness()

    assert result.status is DriftReadinessStatus.UNDETERMINED, (
        f"present readiness is {result.status.value}. Candidate discovery is "
        "not implemented, so readiness was never evaluated -- and a status "
        "that says it was is a fabricated measurement."
    )
    assert result.status is not DriftReadinessStatus.NOT_READY


def test_the_reason_names_the_missing_capability_not_a_missing_population():
    """A fact about the repository, not a fact about the world.

    Only a discovery authority that looked and found nothing may claim
    NO_NEW_OBSERVATION_POPULATION. Nothing in this repository can look.
    """
    result = current_feature_drift_readiness()

    assert result.reason is (
        DriftReadinessReason.CANDIDATE_DISCOVERY_NOT_IMPLEMENTED)
    assert result.reason.value != "no_new_observation_population", (
        "the capability reason was replaced by the population reason. That "
        "asserts no new observation population EXISTS, which nothing here "
        "measured."
    )


def test_no_new_observation_population_is_in_the_vocabulary_but_unemitted():
    """The reason exists so a discovery authority can emit it. Not this layer.

    Its absence from the vocabulary would force a future migration; its
    emission here would be a claim no layer at this level owns.
    """
    values = {member.value for member in DriftReadinessReason}
    assert "no_new_observation_population" in values

    assert current_feature_drift_readiness().reason is not (
        DriftReadinessReason.NO_NEW_OBSERVATION_POPULATION)


def test_readiness_is_not_checked_today():
    assert current_feature_drift_readiness().checked is False


# ---------------------------------------------------------------------------
# 2. THE INVARIANTS
# ---------------------------------------------------------------------------

def test_ready_may_not_carry_a_refusal_reason():
    """A verdict permitting assessment cannot also explain why it does not."""
    with pytest.raises(ValueError) as exc:
        DriftReadiness(
            status=DriftReadinessStatus.READY,
            reason=DriftReadinessReason.CANDIDATE_DISCOVERY_NOT_IMPLEMENTED,
        )
    assert "cannot also explain" in str(exc.value)


@pytest.mark.parametrize(
    "status",
    [DriftReadinessStatus.NOT_READY, DriftReadinessStatus.UNDETERMINED],
)
def test_a_refusal_requires_a_reason(status):
    """A refusal without a reason is the prose this module replaces."""
    with pytest.raises(ValueError) as exc:
        DriftReadiness(status=status, reason=None)
    assert "requires a reason" in str(exc.value)


def test_ready_remains_constructible_so_the_type_survives_p1():
    """A type that must be rewritten to express success is a type that will be.

    Discovery does not exist today, so nothing in the module RETURNS READY --
    but forbidding it structurally would force P1 to replace the type rather
    than reuse it.
    """
    ready = DriftReadiness(status=DriftReadinessStatus.READY, reason=None)
    assert ready.checked is True


# ---------------------------------------------------------------------------
# 3. UNKNOWN IS A PROJECTION, NOT A DOMAIN STATE
# ---------------------------------------------------------------------------

def test_unknown_appears_in_no_domain_enumeration():
    """`UNKNOWN` conflates "no verdict" with a severity.

    That overload is why a monthly job could report a clean bill of health
    having measured nothing. It survives only as a compatibility spelling at
    the adapter boundary.
    """
    for enumeration in (DriftReadinessStatus, DriftReadinessReason):
        values = {member.value for member in enumeration}
        assert "UNKNOWN" not in values, (
            f"{enumeration.__name__} carries UNKNOWN. Epistemic state and "
            "severity are different quantities."
        )


def test_the_legacy_drift_level_is_unknown_while_nothing_is_measured():
    lines = render_github_output_lines(current_feature_drift_readiness())
    assert "drift_level=UNKNOWN" in lines, (
        "the legacy projection changed. `none` would mean CHECKED AND CLEAN, "
        "which is the exact false-green this repair removes."
    )
    assert "drift_level=none" not in lines


def test_a_ready_verdict_asserts_no_severity():
    """Severity belongs to the assessment layer, which has not run.

    READY means an assessment MAY proceed -- not that one did, and certainly
    not what it would find.
    """
    lines = render_github_output_lines(
        DriftReadiness(status=DriftReadinessStatus.READY, reason=None))
    assert "drift_level=" in lines
    assert "drift_level=UNKNOWN" not in lines
    assert "drift_level=none" not in lines


# ---------------------------------------------------------------------------
# 4. ONE RECORD, MANY PROJECTIONS
# ---------------------------------------------------------------------------

def test_the_github_projection_derives_from_the_document():
    """Three fields, one producer.

    The previous workflow authored `drift_level`, and could therefore have
    authored `checked=false` beside `drift_level=none`. Deriving all of them
    from one record makes that combination unconstructible.
    """
    result = current_feature_drift_readiness()
    document = as_document(result)
    lines = render_github_output_lines(result)

    assert f"readiness_status={document['readiness_status']}" in lines
    assert "feature_drift_checked=false" in lines
    assert f"not_checked_reason={document['not_checked_reason']}" in lines
    assert document["feature_drift_checked"] is False


def test_the_document_validates_against_its_own_schema():
    validate_document(as_document(current_feature_drift_readiness()))


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: dict(d, unexpected="x"), "differ from the schema"),
        (lambda d: {k: v for k, v in d.items() if k != "not_checked_reason"},
         "differ from the schema"),
        (lambda d: dict(d, schema_version=READINESS_SCHEMA_VERSION + 1),
         "owns version"),
    ],
    # EXPLICIT identifiers. MEASURED 2026-08-24: without them pytest generates
    # `[<lambda>-differ from the schema0]` and `...schema1`, whose numeric
    # suffixes are positional -- reordering the cases RENAMES the node, and an
    # installer that declares suite identities by node ID would refuse a
    # reorder as a removal plus an addition. A generated identity is not an
    # identity.
    ids=["extra_key", "missing_key", "future_schema_version"],
)
def test_a_malformed_document_is_refused(mutate, fragment):
    """An adapter that hand-builds "roughly the same" dictionary is how two
    producers of one record diverge."""
    document = as_document(current_feature_drift_readiness())
    with pytest.raises(ValueError) as exc:
        validate_document(mutate(document))
    assert fragment in str(exc.value)


def test_the_rendered_json_is_deterministic_and_authored():
    result = current_feature_drift_readiness()
    first = render_json(result)
    assert first == render_json(result)
    assert first.endswith("\n")
    assert not any(ord(ch) > 0x7F for ch in first)
    assert json.loads(first)["schema_version"] == READINESS_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 5. THE LAYERING BOUNDARY
# ---------------------------------------------------------------------------

def test_the_readiness_module_imports_only_the_standard_library():
    """Parsed, not grepped: a name in a docstring is not an import.

    This module runs on a hosted runner to report that it has nothing to
    measure. Importing the scientific stack to say so would make the cheapest
    honest statement in the system the most expensive.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    offenders = sorted(modules & FORBIDDEN_IMPORTS)
    assert not offenders, (
        f"drift_readiness.py imports {offenders}. Readiness answers whether an "
        "assessment may happen; drift_detector.py answers what one found."
    )


def test_the_readiness_module_does_not_import_the_assessment_stack():
    """The boundary pinned before convenience erodes it.

    Later orchestration may import both. This module may not import the thing
    it stands upstream of.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    for forbidden in (
        "genomic_variant_classifier.monitoring.drift_detector",
        "genomic_variant_classifier.monitoring.drift_reference_profile",
        "genomic_variant_classifier.training.continual_trainer",
    ):
        assert forbidden not in imported, (
            f"drift_readiness.py imports {forbidden}. That inverts the layer "
            "it exists to sit above."
        )


def test_the_module_uses_the_projects_enumeration_convention():
    """MEASURED 2026-08-24 across src/: 79 classes use `(str, Enum)`, 0 use
    `StrEnum`. A new module follows the project, not a preference."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    enums = [node for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef)
             and any("Enum" in ast.unparse(base) for base in node.bases)]

    assert enums, "the module defines no enumeration"
    for node in enums:
        bases = [ast.unparse(base) for base in node.bases]
        assert bases == ["str", "Enum"], (
            f"{node.name} declares {bases}; this project uses (str, Enum)."
        )
