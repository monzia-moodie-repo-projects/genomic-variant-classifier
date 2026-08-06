"""Contract tests for `_registry_metadata_prefix` (OP-1 step 3c).

Author: Monzia Moodie
Project: genomic-variant-classifier
Written 2026-08-05.

TEN CONTRACTS. Six were owed by the step-3c ruling; F-strict added four more,
because a collision guard that is never observed to fire is not known to work.

    1  the direct contract: identity plus the whole support snapshot
    2  N_CLUSTERS propagates WITHOUT the test enumerating it
    3  every construction path carries the base mapping
    4  descriptor diagnostics survive, DERIVED from the live verdict
    5  refusal asymmetry preserved -- auroc on a single-class cohort
    6  the one-authority gate, on the ABSTRACT SYNTAX TREE
    7  insertion order preserved, with the support tail derived at runtime
    8  a support key supplied by a caller is REFUSED
    9  the metric name supplied by a caller is REFUSED
   10  a PLAIN STRING equal to a support key's value is REFUSED

Contract 10 is not redundant with 8. `MetricMetadataKey` is a `(str, Enum)`
mixin, so `hash(member) == hash(member.value)` and the two forms are the SAME
dictionary key -- measured 2026-07-27 and re-measured 2026-08-05. That
equivalence is what makes the guard sufficient and it is not obvious from
reading it, so it is pinned.

Every expectation about support is compared against `ctx.support()` AT RUNTIME.
No test in this file writes a support key literal.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pytest

import genomic_variant_classifier.evaluation.registry as reg
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.registry import (
    Applicability,
    MetricContext,
    MetricDescriptor,
    MetricInput,
    MetricMetadataKey,
    MetricStatus,
    RegistryInvariantError,
    ResultKind,
    _registry_metadata_prefix,
    by_name,
    compute,
)

_SOURCE_ID = "op1-step3c-prefix"


# --- fixtures, following test_metric_registry.py rather than coining a second
#     construction pattern (the SWEEP-1 shape applied to test scaffolding) ----

def _pop(n, scope):
    return EvaluationPopulation.full(n, scope=scope, source_id=_SOURCE_ID)


def _two_class(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(rng.uniform(0, 1, n) * 0.5 + y * 0.4, 0, 1)
    return MetricContext(y_true=y, y_score=p, y_prob=p,
                         population=_pop(n, "step3c_two_class"))


def _single_class():
    y = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([0.9, 0.8, 0.85, 0.95])
    return MetricContext(y_true=y, y_score=p, y_prob=p,
                         population=_pop(4, "step3c_single_class"))


def _clustered(n=400, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(rng.uniform(0, 1, n) * 0.5 + y * 0.4, 0, 1)
    clusters = rng.integers(0, 30, n)
    return MetricContext(y_true=y, y_score=p, y_prob=p, clusters=clusters,
                         population=_pop(n, "step3c_clustered"))


def _probe(name="probe", function=None, required=None, applicability=None):
    """A synthetic descriptor, built exactly as test_metric_registry.py does."""
    return MetricDescriptor(
        name=name,
        function=function if function is not None else (lambda ctx: 1.0),
        required_inputs=frozenset(required or {MetricInput.LABELS}),
        applicability=applicability if applicability is not None
        else (lambda ctx: reg.APPLICABLE),
        result_kind=ResultKind.PREDICTION_METRIC,
        display_name="probe", description="probe")


class _CountingContext:
    """Delegates to a real MetricContext and counts support() calls.

    A duck-typed proxy rather than a subclass: MetricContext is a FROZEN
    dataclass whose __post_init__ validates array alignment, and re-running that
    validation to count a method call would test the wrong thing.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self.support_calls = 0

    def support(self):
        self.support_calls += 1
        return self._ctx.support()

    def __getattr__(self, item):
        return getattr(self._ctx, item)


# --- 1  the direct contract ------------------------------------------------

def test_the_prefix_is_exactly_identity_plus_the_support_snapshot():
    ctx = _two_class()
    d = by_name("auroc")

    prefix = _registry_metadata_prefix(d, ctx)

    assert set(prefix) == {MetricMetadataKey.METRIC_NAME} | set(ctx.support())
    assert prefix[MetricMetadataKey.METRIC_NAME] == d.name
    for key, value in ctx.support().items():
        assert prefix[key] == value


def test_the_prefix_evaluates_the_support_snapshot_exactly_once():
    counting = _CountingContext(_two_class())

    _registry_metadata_prefix(by_name("auroc"), counting)

    assert counting.support_calls == 1, (
        "two snapshots reintroduce the time-of-check/time-of-use seam this "
        "step exists to close")


# --- 2  the conditional key, never enumerated ------------------------------

def test_a_conditional_support_key_propagates_without_being_enumerated():
    clustered = _clustered()
    unclustered = _two_class()
    d = by_name("auroc")

    live_clustered = clustered.support()
    live_unclustered = unclustered.support()

    # The DIFFERENCE between the two live snapshots is derived, not named.
    conditional = set(live_clustered) - set(live_unclustered)
    assert conditional, (
        "the clustered fixture must add at least one support key, or this "
        "test cannot distinguish wholesale expansion from a fixed list")

    prefix = _registry_metadata_prefix(d, clustered)

    assert conditional <= set(prefix)
    for key in conditional:
        assert prefix[key] == live_clustered[key]

    assert set(_registry_metadata_prefix(d, unclustered)).isdisjoint(conditional)


# --- 3  every construction path ---------------------------------------------

def _missing_inputs_case():
    return _probe(name="needs_clusters",
                  required={MetricInput.LABELS, MetricInput.CLUSTERS}), _two_class()


def _refusal_case():
    return by_name("auroc"), _single_class()


def _raising_case():
    return _probe(name="raiser", function=lambda ctx: 1 / 0), _two_class()


def _nonfinite_case():
    return _probe(name="broken", function=lambda ctx: float("inf")), _two_class()


def _ok_case():
    return by_name("auroc"), _two_class()


_ALL_PATHS = {
    "required_inputs_missing": _missing_inputs_case,
    "refusal": _refusal_case,
    "metric_computation_failed": _raising_case,
    "applicable_metric_returned_non_finite": _nonfinite_case,
    "ok": _ok_case,
}


@pytest.mark.parametrize("path", sorted(_ALL_PATHS))
def test_every_construction_path_carries_the_base_mapping(path):
    d, ctx = _ALL_PATHS[path]()

    result = compute(d, ctx)

    expected = {MetricMetadataKey.METRIC_NAME} | set(ctx.support())
    assert expected <= set(result.metadata), (
        f"{path}: missing {sorted(str(k) for k in expected - set(result.metadata))}")
    assert result.metadata[MetricMetadataKey.METRIC_NAME] == d.name


def test_the_five_paths_are_reached_and_are_distinct():
    """A guard against the parametrised test passing vacuously.

    If a fixture silently stopped exercising its branch, the base-mapping
    assertion above would still pass on whatever branch it landed on instead.
    """
    reasons = {}
    for path, build in _ALL_PATHS.items():
        d, ctx = build()
        result = compute(d, ctx)
        reasons[path] = (result.status, result.reason)

    assert reasons["required_inputs_missing"] == (
        MetricStatus.NOT_APPLICABLE, "required_inputs_missing")
    assert reasons["metric_computation_failed"] == (
        MetricStatus.FAILED, "metric_computation_failed")
    assert reasons["applicable_metric_returned_non_finite"] == (
        MetricStatus.FAILED, "applicable_metric_returned_non_finite")
    assert reasons["ok"][0] is MetricStatus.OK
    assert reasons["refusal"][0] is not MetricStatus.OK
    assert reasons["refusal"][1], "a refusal must carry a nonempty reason"


# --- 4  descriptor diagnostics survive, derived from the live verdict -------

def _maximally_populated_single_class():
    """A single-class cohort carrying EVERY input the catalogue can require.

    Labels, scores, probabilities, clusters and sample weights, so that no
    registered descriptor drops out of the loop below because an input is
    absent. `_missing_inputs` is asserted empty for every descriptor, so a
    future descriptor requiring something new FAILS here by name instead of
    quietly leaving coverage.
    """
    y = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([0.9, 0.8, 0.85, 0.95])
    return MetricContext(y_true=y, y_score=p, y_prob=p,
                         clusters=np.array([0, 0, 1, 1]),
                         sample_weight=np.array([1.0, 1.0, 1.0, 1.0]),
                         population=_pop(4, "step3c_single_class_full"))


def test_live_descriptor_diagnostics_survive_on_the_paths_that_merge_them():
    """Verdict metadata reaches the result on the merging paths -- and ONLY there.

    ONLY TWO of `compute`'s five construction sites merge `verdict.metadata`:
    the refusal branch, which merges it LAST, and the OK branch, which merges it
    FIRST. The `metric_computation_failed` and
    `applicable_metric_returned_non_finite` branches do not merge it at all, and
    `required_inputs_missing` returns before a verdict even exists. That was
    true before OP-1 step 3c and is unchanged by it.

    An earlier version of this test asserted survival on EVERY path and failed
    on `integrated_calibration_index`, which is ruled APPLICABLE while carrying
    `reference_class_support`, then returns nan, and so lands on the non-finite
    branch. The test was wrong, not the extraction.

    THAT FAILURE CLOSED A MEASUREMENT GAP. Item C2-1 recorded that no metric had
    been OBSERVED to be applicable on a single-class cohort while carrying
    `reference_class_support`; enumerating the live catalogue rather than naming
    five metrics found one.

    IT ALSO RAISED FOLLOW-UP DIAG-1: an applicable verdict's diagnostics are
    dropped when the kernel then fails or returns non-finite, so the reader of
    that FAILED result learns the metric returned nan but not the cohort fact
    that explains it. That is a production question outside step 3c's scope. It
    is PINNED here rather than tolerated: the absence is asserted, so resolving
    DIAG-1 must update this test deliberately instead of silently.

    Carriers are DERIVED from the live catalogue and its live verdicts, so this
    tracks the registry as it grows and enumerates nothing. `reg._METRICS` is
    the same collection `evaluate_registered` iterates (measured 2026-08-05: 24
    descriptors). No second list is maintained here.
    """
    ctx = _maximally_populated_single_class()

    descriptors = reg._METRICS
    assert descriptors, "the registry catalogue is empty"

    base_keys = {MetricMetadataKey.METRIC_NAME} | set(ctx.support())
    merged = []
    unmerged = []

    for descriptor in descriptors:
        missing = reg._missing_inputs(descriptor, ctx)
        assert not missing, (
            f"{descriptor.name} requires inputs this cohort does not carry "
            f"({sorted(str(item) for item in missing)}). Widen the fixture "
            "rather than let a descriptor drop silently out of coverage.")
        try:
            verdict = descriptor.applicability(ctx)
        except Exception as exc:                                # noqa: BLE001
            pytest.fail(f"{descriptor.name}: applicability raised {exc!r}; "
                        "catalogue inspection could not be completed")
        declared = dict(verdict.metadata)
        if not declared:
            continue
        result = compute(descriptor, ctx)
        if (not verdict.applicable) or result.status is MetricStatus.OK:
            merged.append((descriptor, declared, result))
        else:
            unmerged.append((descriptor, declared, result))

    assert merged, (
        "no registered descriptor reaches a verdict-merging path carrying its "
        "own metadata on this cohort, so this test would assert nothing")

    for descriptor, declared, result in merged:
        for key, value in declared.items():
            assert key in result.metadata, (
                f"{descriptor.name}: descriptor diagnostic {key!r} was lost on "
                "a path that merges verdict metadata")
            assert result.metadata[key] == value, (
                f"{descriptor.name}: diagnostic {key!r} changed value")

    for descriptor, declared, result in unmerged:
        assert result.status is MetricStatus.FAILED, (
            f"{descriptor.name}: an applicable verdict reached a non-merging "
            f"path with status {result.status!r}; only the two FAILED branches "
            "do not merge verdict metadata")
        for key in set(declared) - base_keys:
            assert key not in result.metadata, (
                f"{descriptor.name}: {key!r} appeared on a branch that does "
                "NOT merge verdict metadata. If DIAG-1 has been resolved and "
                "sites 3 and 4 now merge it, this test must be updated "
                "deliberately rather than left to pass by accident.")


# --- 5  the refusal asymmetry ----------------------------------------------

def test_auroc_refusing_a_single_class_cohort_still_reports_the_class_count():
    """The case REG-1's first attempt turned 29 tests red.

    N_CLASSES_OBSERVED is the GROUND of this refusal, so the descriptor owns it
    HERE and only here. If the extraction collapsed the two protected sets, the
    guard would call this a violation and raise.
    """
    ctx = _single_class()
    d = by_name("auroc")

    result = compute(d, ctx)          # must not raise RegistryInvariantError

    assert result.status is not MetricStatus.OK
    assert MetricMetadataKey.N_CLASSES_OBSERVED in result.metadata
    assert result.metadata[MetricMetadataKey.N_CLASSES_OBSERVED] == (
        ctx.n_classes_observed)


def test_the_two_protected_sets_are_still_different_expressions():
    """The asymmetry must be visible in the source, not merely in behaviour.

    A single shared expression would be the REG-1 v1 mistake, and a reader
    would have to run the suite to discover it.
    """
    source = inspect.getsource(compute)
    assert "_DESCRIPTOR_OWNED_ON_REFUSAL" in source
    assert "CERTIFICATION_BLOCKED_BY" in source
    tree = ast.parse(inspect.getsource(reg).encode("utf-8").decode("utf-8"))
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)
              and n.func.id == "_reject_registry_owned_keys"]
    assert len(guards) == 2, f"expected 2 guard call sites, found {len(guards)}"
    rendered = {ast.dump(g.args[3]) for g in guards}
    assert len(rendered) == 2, (
        "the two protected-set expressions are IDENTICAL; the asymmetry REG-1 "
        "established has been collapsed")


# --- 6  the one-authority gate, structural ----------------------------------

def _support_calls_owned_by(function_node):
    """Calls to `.support()` in this function, excluding nested functions."""
    nested = set()
    for inner in ast.walk(function_node):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and inner is not function_node:
            nested.update(id(node) for node in ast.walk(inner))
    return [node for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "support"
            and id(node) not in nested]


def test_exactly_one_function_in_the_registry_calls_support():
    """The invariant, measured at the highest semantic level available.

    NOT a text count: a grep for `.support()` also matches the COMMENT at
    registry.py:665, which is why the parent handoff reported eight call sites
    where the syntax tree finds seven. Line numbers are deliberately not pinned.
    """
    tree = ast.parse(pathlib.Path(reg.__file__).read_text(encoding="utf-8"))

    owners = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = _support_calls_owned_by(node)
            if calls:
                owners[node.name] = len(calls)

    assert owners == {"_registry_metadata_prefix": 1}, (
        "MetricContext.support() must have exactly ONE caller in registry.py, "
        f"and it must call it once. Found: {owners}")


def test_the_authority_gate_can_actually_fail():
    """A gate never observed to reject anything is not known to work."""
    mutated = ast.parse(
        "def compute(d, ctx):\n"
        "    a = ctx.support()\n"
        "    b = ctx.support()\n"
        "    return a, b\n"
        "def _registry_metadata_prefix(d, ctx):\n"
        "    return dict(ctx.support())\n")

    owners = {}
    for node in ast.walk(mutated):
        if isinstance(node, ast.FunctionDef):
            calls = _support_calls_owned_by(node)
            if calls:
                owners[node.name] = len(calls)

    assert owners == {"compute": 2, "_registry_metadata_prefix": 1}
    assert owners != {"_registry_metadata_prefix": 1}


def test_a_comment_is_not_counted_as_a_call():
    """Pins the specific error the parent handoff made."""
    tree = ast.parse("def f(ctx):\n"
                     "    # N_CLASSES_OBSERVED is NOT supplied: ctx.support()\n"
                     "    return 1\n")
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert _support_calls_owned_by(node) == []


# --- 7  insertion order ------------------------------------------------------

def test_the_prefix_preserves_the_order_identity_fields_support():
    ctx = _clustered()
    d = by_name("auroc")

    prefix = _registry_metadata_prefix(
        d, ctx,
        pre_support={"exception_type": "ValueError",
                     "exception_message": "example"})

    assert list(prefix) == [MetricMetadataKey.METRIC_NAME,
                            "exception_type",
                            "exception_message",
                            *ctx.support().keys()]


def test_the_ok_path_keeps_the_certification_blocker_last():
    """The blocker is attached AFTER the verdict merge and must stay last.

    THE PRECONDITION IS ASSERTED, NOT SKIPPED. `_certification_eligibility`
    refuses a single-class cohort outright, so an OK result over this cohort
    carries a certification blocker BY CONSTRUCTION. An earlier version skipped
    when the blocker was absent, which would have reported success if a future
    change stopped attaching it.
    """
    ctx = _single_class()
    result = compute(by_name("brier_score"), ctx)

    assert result.status is MetricStatus.OK
    assert MetricMetadataKey.CERTIFICATION_BLOCKED_BY in result.metadata, (
        "a single-class cohort cannot support a certified claim, so this OK "
        "result must carry a certification blocker")
    assert list(result.metadata)[-1] is (
        MetricMetadataKey.CERTIFICATION_BLOCKED_BY), (
        "the blocker is attached after the verdict merge and must remain the "
        "last key; a reordering would change the serialised artifact")


# --- 8, 9, 10  the collision guard ------------------------------------------

def test_the_prefix_refuses_a_support_key_supplied_by_the_caller():
    ctx = _two_class()
    a_support_key = next(iter(ctx.support()))

    with pytest.raises(RegistryInvariantError, match="identity/support key"):
        _registry_metadata_prefix(by_name("auroc"), ctx,
                                  pre_support={a_support_key: 999})


def test_the_prefix_refuses_the_metric_name_supplied_by_the_caller():
    with pytest.raises(RegistryInvariantError, match="identity/support key"):
        _registry_metadata_prefix(
            by_name("auroc"), _two_class(),
            pre_support={MetricMetadataKey.METRIC_NAME: "forged"})


def test_the_prefix_refuses_a_plain_string_alias_of_a_support_key():
    """MetricMetadataKey is a (str, Enum) mixin, so the string form is the SAME
    dictionary key -- `hash(member) == hash(member.value)`. A caller cannot
    evade the guard by spelling the key as a plain string.
    """
    ctx = _two_class()
    a_support_key = next(iter(ctx.support()))
    plain = a_support_key.value

    assert plain == a_support_key
    assert hash(plain) == hash(a_support_key)

    with pytest.raises(RegistryInvariantError, match="identity/support key"):
        _registry_metadata_prefix(by_name("auroc"), ctx,
                                  pre_support={plain: 999})


def test_the_prefix_accepts_branch_fields_that_collide_with_nothing():
    """The negative control. Without it the three tests above are satisfied by
    a guard that refuses everything.
    """
    ctx = _two_class()

    prefix = _registry_metadata_prefix(
        by_name("auroc"), ctx,
        pre_support={"missing_inputs": ["y_prob"], "returned": "inf"})

    assert prefix["missing_inputs"] == ["y_prob"]
    assert prefix["returned"] == "inf"


def test_a_conditional_key_is_only_reserved_when_support_supplies_it():
    """N_CLUSTERS is reserved on a clustered context and free on an
    unclustered one, because the reserved set is DERIVED from the snapshot
    rather than from a fixed list. A hard-coded list of five keys would refuse
    it in both cases; a hard-coded list of four would allow it in both.
    """
    clustered = _clustered()
    unclustered = _two_class()
    conditional = next(iter(set(clustered.support()) - set(unclustered.support())))

    with pytest.raises(RegistryInvariantError, match="identity/support key"):
        _registry_metadata_prefix(by_name("auroc"), clustered,
                                  pre_support={conditional: 1})

    allowed = _registry_metadata_prefix(by_name("auroc"), unclustered,
                                        pre_support={conditional: 1})
    assert allowed[conditional] == 1
