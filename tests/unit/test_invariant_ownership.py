"""Completeness invariants that no longer depend on README prose.

Created 2026-08-22. INVARIANT-HANDOFF-1.

WHY THIS FILE EXISTS
--------------------
`tests/unit/test_readme_claims.py` is scheduled for decomposition: the README is
to become authoritative for public scientific identity only, and must stop being
an executable mirror of internal state. That is correct, and it is also
dangerous, because a census on 2026-08-22 over the entire tracked corpus (1,573
files, 1,565 textual) found that three of the five invariants that file enforces
have NO OTHER OWNER:

    INV-SUITE-SIZE                     owned elsewhere: tests/conftest.py
                                       implements --assert-suite-size, and
                                       test_suite_size_ratchet.py binds it.
                                       SAFE.

    INV-FEATURE-CONTRACT-CARDINALITY   owned elsewhere: the fail-loud guard runs
                                       at import in variant_ensemble.py, and
                                       test_zero_variance_guard.py imports both
                                       the constant and the list. SAFE.

    INV-MODEL-ROSTER-COMPLETENESS      NO OTHER OWNER. Nineteen test files
                                       reference `base_estimators`, but eight
                                       MUTATE it as a fixture, three enumerate
                                       it to iterate, one asserts a single
                                       conditional member, and one asserts the
                                       module docstring does NOT enumerate the
                                       roster -- the opposite binding. Nothing
                                       compared the runtime roster to an
                                       independently authored list.

    INV-AGENT-REGISTRY-COMPLETENESS    NO OTHER OWNER. Seven wiring tests each
                                       assert ONE named agent is registered;
                                       one asserts len(registry) >= 1. FIFTEEN
                                       of the twenty-two agents had no
                                       registration coverage at all.

    INV-DRIFT-EXIT-CODE                NO OTHER OWNER, and the single
                                       non-README reference in the corpus is a
                                       COMMENT inside tests/EXPECTED_SUITE_SIZE,
                                       not an assertion.

A count of files referencing a symbol is not a count of invariant owners. That
distinction is the whole reason this file was written before anything was
deleted.

    No assertion may be retired until its owned invariant has another
    PROVEN owner.   -- INVARIANT-HANDOFF-1

"Proven" means proven by a deliberate break, not by inspection. Six of the nine
tests below are negative controls that exist solely to demonstrate that the
three checks can FAIL. A checker that has never rejected anything has not been
shown to work; `test_module_docstring_is_not_a_stale_roster.py` establishes this
pattern in this repository, and three of its seven tests exist for the same
reason.

WHY A DECLARED LIST IS NOT THE DEFECT IT LOOKS LIKE
---------------------------------------------------
A completeness test needs something to compare against, and a declared list is a
second authored copy -- the exact shape that made the README load-bearing.

The resolution is that these lists are not documentation. They are LINEAGE.

`PATHOGENICITY_BENCHMARK_V1` is the immutable membership of the first
pathogenicity benchmark. Its permanence attaches to benchmark-lineage
membership, NOT to production participation, architecture, implementation, or
scientific role. A model may migrate from classifier to encoder, may leave
production entirely, and remains a member of this benchmark forever, because a
benchmark whose membership changes is not a benchmark. Adding a model to the
ensemble therefore requires a DELIBERATE amendment here, in the same commit --
which is the point.

`AGENT_REGISTRY_LINEAGE_2026_08` is weaker and is labelled so honestly. The
adopted ruling is that governance CAPABILITIES are permanent while the class
roster is mutable: an obligation such as calibration-drift monitoring must
remain covered, but whether it is met by twenty-two agent subclasses or by eight
controllers plus fourteen deterministic sentinels is an implementation choice.
Until a capability registry exists, this snapshot is the interim owner. It is a
lineage record, not a permanent contract, and the docstring says so rather than
letting a future reader assume otherwise.

WHY THE EXIT-CODE CHECK IS STRONGER THAN THE ONE IT REPLACES
------------------------------------------------------------
The retiring assertion is a substring test, `"EXIT_NOT_CHECKED = 4" in script`,
which a comment would satisfy. This one parses the module and reads the
module-level assignment's value, so a commented-out constant, an indented
rebinding, or a changed value are all detected. Relocating an invariant is an
opportunity to strengthen it, not merely to move it.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Declared lineage
# ---------------------------------------------------------------------------

#: The immutable membership of the first pathogenicity benchmark.
#:
#: Roadmap 6.6a is the defect this exists to prevent: a thirteen-model ensemble
#: SILENTLY BECAME A TWELVE-MODEL ENSEMBLE. The Kolmogorov-Arnold Network's
#: out-of-fold step raised, a bare `except Exception` swallowed it, and the model
#: vanished from `trained_models_`, from the blend, and from every
#: cross-algorithm comparison artifact -- while the run reported normal metrics
#: and looked healthy. Anyone checking twelve models against a document that said
#: twelve would have concluded the ensemble was complete.
PATHOGENICITY_BENCHMARK_V1: frozenset[str] = frozenset({
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "gradient_boosting",
    "logistic_regression",
    "svm",
    "svm_bagged_rbf",
    "kan",
    "tabular_nn",
    "cnn_1d",
    "mc_dropout",
    "deep_ensemble",
})

#: Interim lineage snapshot, NOT a permanent contract. See the module docstring.
#: To be superseded by a capability registry, at which point the invariant
#: becomes "every required governance capability has an implementation owner"
#: rather than "these exact class names are registered".
AGENT_REGISTRY_LINEAGE_2026_08: frozenset[str] = frozenset({
    "AdaptationAgent",
    "AdversarialSubmissionMonitorAgent",
    "AgentOpsMonitorAgent",
    "AnnotationPolicyMonitorAgent",
    "CalibrationDriftMonitorAgent",
    "ConceptDriftMonitorAgent",
    "DataFreshnessAgent",
    "DataReadinessAgent",
    "DatabaseFreshnessMonitorAgent",
    "FairnessSubgroupMonitorAgent",
    "FeatureCoverageSentinelMonitorAgent",
    "FinOpsAdvisorAgent",
    "InfrastructureDriftMonitorAgent",
    "InterpretabilityAgent",
    "LabelShiftMonitorAgent",
    "LiteratureScoutAgent",
    "ModelInsightsAgent",
    "ProvisioningAgent",
    "ReclassificationSentinelMonitorAgent",
    "SchemaDriftMonitorAgent",
    "TrainingLifecycleAgent",
    "VersionMonitorAgent",
})

DRIFT_MONITOR = Path("scripts/run_drift_monitor.py")


# ---------------------------------------------------------------------------
# The comparison functions. Factored out DELIBERATELY so the negative controls
# exercise the SAME code path the real assertions use -- a control that tests a
# reimplementation proves nothing about the checker that ships.
# ---------------------------------------------------------------------------

def _membership_disagreement(
    observed: frozenset[str], declared: frozenset[str]
) -> tuple[list[str], list[str]]:
    """Return (declared-but-absent, present-but-undeclared), both sorted."""
    return sorted(declared - observed), sorted(observed - declared)


def _module_level_int(source: str, name: str) -> int | None:
    """Read a module-level `name = <int>` binding, or None.

    Parses rather than searching, so a comment mentioning the constant, an
    indented rebinding inside a function, or a changed value are all
    distinguishable from a real module-level definition.
    """
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, int
                ):
                    return node.value.value
                return None
    return None


# ---------------------------------------------------------------------------
# 1. INV-MODEL-ROSTER-COMPLETENESS
# ---------------------------------------------------------------------------

def test_runtime_roster_matches_the_declared_benchmark_lineage(tmp_path):
    """The live roster must equal PATHOGENICITY_BENCHMARK_V1 exactly.

    Read from a REAL VariantEnsemble instance, not with a regular expression
    over the source: `kan` is added by dict-unpacking behind
    `_KAN_AVAILABLE and not cfg.skip_kan`, and a naive line-regex MISSES IT --
    which is precisely the mistake that produced a wrong count previously.
    """
    from genomic_variant_classifier.models.variant_ensemble import (
        EnsembleConfig,
        VariantEnsemble,
    )

    runtime = frozenset(
        VariantEnsemble(EnsembleConfig(model_dir=str(tmp_path))).base_estimators
    )
    assert runtime, "VariantEnsemble built no base estimators at all."

    missing, extra = _membership_disagreement(runtime, PATHOGENICITY_BENCHMARK_V1)
    assert not missing and not extra, (
        "THE RUNTIME ROSTER DISAGREES WITH PATHOGENICITY_BENCHMARK_V1.\n"
        f"  declared but ABSENT from the ensemble ({len(missing)}): {missing}\n"
        f"  present but UNDECLARED ({len(extra)}): {extra}\n"
        f"  runtime ({len(runtime)}): {sorted(runtime)}\n"
        f"  declared ({len(PATHOGENICITY_BENCHMARK_V1)}): "
        f"{sorted(PATHOGENICITY_BENCHMARK_V1)}\n\n"
        "A model absent from the runtime roster may have been SILENTLY DROPPED "
        "-- roadmap 6.6a. A model present but undeclared means the benchmark "
        "lineage was not amended in the same commit as the ensemble change; "
        "amend it deliberately, do not delete this assertion."
    )


def test_the_roster_check_detects_a_silently_dropped_model():
    """NEGATIVE CONTROL. Reproduce roadmap 6.6a exactly and require detection."""
    sabotaged = PATHOGENICITY_BENCHMARK_V1 - {"kan"}
    missing, extra = _membership_disagreement(sabotaged, PATHOGENICITY_BENCHMARK_V1)
    assert missing == ["kan"], (
        "the roster check FAILED TO DETECT a silently dropped model. A check "
        "that cannot reject is not a check."
    )
    assert extra == []


def test_the_roster_check_detects_an_undeclared_model():
    """NEGATIVE CONTROL. A model added without amending the lineage."""
    sabotaged = PATHOGENICITY_BENCHMARK_V1 | {"undeclared_model"}
    missing, extra = _membership_disagreement(sabotaged, PATHOGENICITY_BENCHMARK_V1)
    assert missing == []
    assert extra == ["undeclared_model"], (
        "the roster check FAILED TO DETECT an undeclared model."
    )


# ---------------------------------------------------------------------------
# 2. INV-AGENT-REGISTRY-COMPLETENESS
# ---------------------------------------------------------------------------

def test_agent_registry_matches_the_declared_lineage():
    """The orchestrator's registry must equal the declared lineage snapshot.

    Built with `__new__` plus `_register_agents()` because the registry is
    assembled from string literals and lazy wrappers only, so it needs no
    __init__ state -- and __init__ would touch shared state on disk, which a
    unit test must not do.
    """
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._register_agents()
    registry = frozenset(orch._agent_registry)

    assert registry, "Orchestrator registered NO agents -- the registry is empty."

    missing, extra = _membership_disagreement(
        registry, AGENT_REGISTRY_LINEAGE_2026_08
    )
    assert not missing and not extra, (
        "THE AGENT REGISTRY DISAGREES WITH AGENT_REGISTRY_LINEAGE_2026_08.\n"
        f"  declared but NOT registered ({len(missing)}): {missing}\n"
        f"  registered but NOT declared ({len(extra)}): {extra}\n"
        f"  registry ({len(registry)}): {sorted(registry)}\n\n"
        "This snapshot is LINEAGE, not a permanent contract: governance "
        "capabilities are permanent, the class roster is not. If the fleet is "
        "deliberately restructured, amend this set in the same commit and "
        "record which capability each removed class's obligation moved to."
    )


def test_the_agent_check_detects_an_unregistered_agent():
    """NEGATIVE CONTROL. Fifteen of twenty-two agents had no coverage at all."""
    sabotaged = AGENT_REGISTRY_LINEAGE_2026_08 - {"ProvisioningAgent"}
    missing, extra = _membership_disagreement(
        sabotaged, AGENT_REGISTRY_LINEAGE_2026_08
    )
    assert missing == ["ProvisioningAgent"], (
        "the agent check FAILED TO DETECT an agent that stopped being registered."
    )
    assert extra == []


def test_the_agent_check_detects_an_undeclared_agent():
    """NEGATIVE CONTROL. An agent added without amending the lineage."""
    sabotaged = AGENT_REGISTRY_LINEAGE_2026_08 | {"UndeclaredAgent"}
    missing, extra = _membership_disagreement(
        sabotaged, AGENT_REGISTRY_LINEAGE_2026_08
    )
    assert missing == []
    assert extra == ["UndeclaredAgent"], (
        "the agent check FAILED TO DETECT an undeclared agent."
    )


# ---------------------------------------------------------------------------
# 3. INV-DRIFT-EXIT-CODE
# ---------------------------------------------------------------------------

def test_drift_monitor_defines_the_not_checked_exit_code():
    """Exit 4 = NOT CHECKED, and it must be a real module-level constant.

    The monitor's original defect was that a run which measured NOTHING exited
    0, and 0 means "no drift" -- so it reported a clean bill of health having
    never read a row of data. The obvious repair, exiting 3, is the same defect
    in the opposite costume: 3 means urgent_retrain and would fire a severe
    alarm on a healthy model. "I could not look" therefore has its own code, and
    that distinction is the entire fix.
    """
    if not DRIFT_MONITOR.is_file():
        pytest.fail(f"{DRIFT_MONITOR} not found at {DRIFT_MONITOR.resolve()}")

    value = _module_level_int(
        DRIFT_MONITOR.read_text(encoding="utf-8"), "EXIT_NOT_CHECKED"
    )
    assert value == 4, (
        f"{DRIFT_MONITOR} does not define EXIT_NOT_CHECKED = 4 at module level "
        f"(observed: {value!r}).\n\n"
        "Without it, 'I looked and found nothing' and 'I could not look' "
        "collapse into one exit code, and a monitor that read no data reports a "
        "clean bill of health."
    )


def test_the_exit_code_check_detects_removal():
    """NEGATIVE CONTROL. The constant deleted entirely."""
    assert _module_level_int("EXIT_OK = 0\nEXIT_DRIFT = 1\n",
                             "EXIT_NOT_CHECKED") is None, (
        "the exit-code check FAILED TO DETECT the constant's removal."
    )


def test_the_exit_code_check_detects_a_comment_or_a_wrong_value():
    """NEGATIVE CONTROL, and the reason this replacement is stronger.

    The retiring assertion was a substring test, which a COMMENT satisfies. Both
    cases below would have passed it.
    """
    commented = "# EXIT_NOT_CHECKED = 4 -- removed 2026-01-01\nEXIT_OK = 0\n"
    assert _module_level_int(commented, "EXIT_NOT_CHECKED") is None, (
        "a commented-out constant was accepted as a definition."
    )
    wrong = "EXIT_NOT_CHECKED = 3\n"
    assert _module_level_int(wrong, "EXIT_NOT_CHECKED") == 3, (
        "the checker did not read the actual value, so a changed exit code "
        "would go undetected."
    )
