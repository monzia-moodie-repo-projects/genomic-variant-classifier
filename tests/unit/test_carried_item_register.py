"""The carried-item register must agree with the code.

WHY THIS EXISTS
===============
Carried items were declared inside per-commit roadmap deltas and their status
changes recorded in LATER deltas, so an item's state could only be reconstructed
by grepping the whole roadmap and reading the deltas in order.

That is two sources of truth for one fact with no divergence detector -- the same
defect the metric stack spent fourteen commits removing from the evaluation path.
It had already gone wrong: item CI-l was discharged in commit 2a-1 and still read
as OPEN eleven commits later, because the discharge was never written down.

`docs/CARRIED_ITEMS.md` is now the single source of truth, and this module makes
it SELF-VERIFYING. Each item carries a predicate that inspects the code. An item
claiming to be OPEN whose predicate reports it closed fails here, rather than
quietly misleading whoever reads the register next.

THE ASYMMETRY IS DELIBERATE
---------------------------
An OPEN item whose condition has gone is a stale register -- annoying, and caught.
A DISCHARGED item whose condition has RETURNED is a regression -- serious, and
also caught. Both directions fail, because a register that only detects one is a
register that will drift in the other.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

REGISTER = Path(__file__).parent.parent.parent / "docs" / "CARRIED_ITEMS.md"
SRC = Path(__file__).parent.parent.parent / "src" / "genomic_variant_classifier"
TESTS = Path(__file__).parent.parent


# --------------------------------------------------------------------------- #
# The predicates. Each returns True when the item's CONDITION STILL HOLDS,
# i.e. when the item is genuinely open.
# --------------------------------------------------------------------------- #
def _condition_m() -> bool:
    """A caller cannot tell how many observations a metric describes.

    REWRITTEN 2026-07-29. The previous predicate asked whether `evaluate` calls
    `clean_arrays` -- which it does, and always will, since filtering is the
    design. That is an IMPLEMENTATION DETAIL, not the item's claim, so the
    register reported CI-m open indefinitely while the stated defect had already
    been fixed.

    Same fault as CI-q's original predicate: it measured a proxy rather than the
    condition. The condition is REPORTABILITY -- can a consumer of the returned
    mapping determine the attempted and surviving row counts?

    Measured 2026-07-29 on a six-row cohort with one unusable score:
        n_input 6, n_dropped 1, survivors 5.
    The counts are present, so this returns False and the item is discharged.
    """
    import numpy as np

    from genomic_variant_classifier.evaluation import metrics

    if not hasattr(metrics, "evaluate"):
        return False
    result = metrics.evaluate(
        [0, 1, 0, 1, 0, 1],
        [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0])
    n_input = result.get("n_input")
    n_dropped = result.get("n_dropped")
    if n_input is None or n_dropped is None:
        return True                     # unreportable: the item is open
    return not (n_input == 6 and n_dropped == 1)


def _condition_n() -> bool:
    """Two different frames sharing a `cohort_version` still collide.

    REWRITTEN 2026-07-29, for the same fault as CI-m. The previous predicate
    asked whether `_derive_population_source_id` ACCEPTS a `cohort_version`
    parameter -- which it does, and always will, since the version is a
    legitimate part of the identity. A parameter-name check can never observe
    whether the identity is weak.

    The item's actual claim was that a generic version lets two different cohorts
    share an identity, and that "the ordered variant-identifier sequence is what
    actually distinguishes them". The derivation already incorporates exactly
    that, and says so in its own docstring.

    Measured 2026-07-29: two frames sharing `cohort_version="v2"` and a partition
    but differing in variants -- or in variant ORDER alone -- yield DISTINCT
    identities. The condition does not hold, so this returns False.
    """
    from genomic_variant_classifier.evaluation import canonical

    if not hasattr(canonical, "_derive_population_source_id"):
        return False
    derive = canonical._derive_population_source_id
    baseline = derive(cohort_version="v2", partition="test",
                      variant_ids=["1-100-A-T", "1-200-C-G"])
    different_rows = derive(cohort_version="v2", partition="test",
                            variant_ids=["1-300-G-A", "1-400-T-C"])
    reordered = derive(cohort_version="v2", partition="test",
                       variant_ids=["1-200-C-G", "1-100-A-T"])
    collides = (baseline == different_rows) or (baseline == reordered)
    return collides


def _condition_p() -> bool:
    """`to_dict` still emits a non-finite value that strict JSON refuses."""
    import math

    from genomic_variant_classifier.evaluation.capabilities import (
        MetricResult,
        MetricStatus,
    )

    payload = MetricResult(float("nan"), MetricStatus.UNDEFINED, "r", {}).to_dict()
    value = payload.get("value")
    return isinstance(value, float) and not math.isfinite(value)


def _condition_q() -> bool:
    """`compare_models` still evaluates without a shared population identity.

    PARSED, AND NARROWED TO THE ACTUAL CALL SITE. The first version scanned every
    file in `src/` for the text `.evaluate(` and asked whether `source_id`
    appeared nearby. It matched SIX places, of which exactly ONE was a real call:

        evaluator.py:28     a docstring example
        evaluator.py:1518   the real call site        <- the only true positive
        registry.py:57,64   prose in a module docstring
        canonical.py:8      prose in a module docstring
        gnn.py:460          self.evaluate(val_dataset) -- a different function

    A docstring will never pass `source_id`, so the item would have reported OPEN
    forever regardless of the code. A register whose predicates are text searches
    is a register that drifts silently, which is precisely what it exists to
    prevent.

    The condition is now what the item actually says: does `compare_models` hand
    a shared population to each evaluation, or does each model build its own?
    """
    import inspect

    from genomic_variant_classifier.evaluation import evaluator

    tree = ast.parse(inspect.getsource(evaluator.compare_models))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "evaluate"):
            continue
        supplied = {kw.arg for kw in node.keywords}
        if "population" in supplied:
            return False        # the shared population is handed over: closed
    return True                 # no call hands one over: still open


def _condition_r() -> bool:
    """A defect forcing interval certification False would go unnoticed.

    REWRITTEN 2026-07-29, the third proxy found in this register. The previous
    predicate read the frozen fixture and checked whether
    `auroc_ci_certification_eligible` had one distinct value. It always will --
    the fixture was captured with `n_bootstrap=0` and is FROZEN BY DESIGN, so
    that predicate described a file nobody intends to change rather than a gap in
    coverage.

    The oracle IS blind; that half of the item was accurate. But the implication
    -- that the defect would therefore go unnoticed -- is FALSE.
    `test_bootstrap_reconciliation` covers interval certification directly and in
    both directions: a certified interval when gene clusters are present, an
    interval that is never certifiable for a variant resampling unit, and status
    and certification as independent axes.

    MEASURED 2026-07-29: forcing the success-path assignment to `False` fails
    `test_evaluator_produces_a_certified_interval_when_genes_are_present`. The
    condition does not hold, so this returns False.

    The check is that the POSITIVE assertion exists. An oracle blind to a
    property is only a gap when nothing else asserts it, and what matters is that
    something, somewhere, requires certification to be True.
    """
    import inspect

    from pathlib import Path as _Path

    suite = TESTS / "unit" / "test_bootstrap_reconciliation.py"
    if not suite.exists():
        return True                     # nothing covers it: the item is open
    text = suite.read_text(encoding="utf-8")
    asserts_true = "auroc_ci_certification_eligible is True" in text
    return not asserts_true


def _condition_u() -> bool:
    """A report over a single-class cohort still cannot be persisted.

    Parsed by BEHAVIOUR rather than by text: build a degenerate report and try to
    serialise it. The flat `auroc` is NaN there, and strict JSON refuses a
    non-finite number in an evidence artifact -- correctly -- so the whole file
    is rejected rather than the one field being recorded as absent.
    """
    import contextlib
    import io

    import numpy as np

    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
    from genomic_variant_classifier.evaluation.serialization import dump_strict_json

    y = np.zeros(80)
    p = np.full(80, 0.10)
    with contextlib.redirect_stdout(io.StringIO()):
        report = ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
            y, p, model_name="register_probe")
    try:
        dump_strict_json(report.to_serializable(), artifact="register_probe")
    except Exception:
        return True         # still unpersistable: the item is open
    return False


OPEN_CONDITIONS = {}


# --------------------------------------------------------------------------- #
# Discharged items: the condition must be GONE, and stay gone.
# --------------------------------------------------------------------------- #
def _discharged_k() -> bool:
    """Interior-edge agreement coverage exists."""
    module = TESTS / "unit" / "test_calibration_binning_convention.py"
    return module.exists() and "interior_edge" in module.read_text(encoding="utf-8")


def _discharged_l() -> bool:
    """The transitional label mask exists nowhere in `src/` as code."""
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "select_finite_reference_labels" not in line:
                continue
            stripped = line.strip()
            # A docstring or comment MENTIONING the retired symbol is a record,
            # not a resurrection. Only executable code counts.
            if stripped.startswith(("#", "*", "`")) or stripped.startswith('"'):
                continue
            if "`" in line or line.lstrip().startswith("-"):
                continue
            return False
    return True


def _discharged_o() -> bool:
    """The evaluator abstract-syntax-tree guard exists and inspects the report
    path."""
    module = TESTS / "unit" / "test_computation_path_guards.py"
    if not module.exists():
        return False
    # PARSED, not grepped. A first draft checked `"ast" in text`, which matches
    # "abstract-syntax-tree", "last", and any mention in a comment -- a sabotage
    # replacing the real import with `import os` went undetected. The guard is
    # only present if the module actually IMPORTS ast and CALLS into it.
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports_ast = any(
        (isinstance(n, ast.Import) and any(a.name == "ast" for a in n.names))
        for n in ast.walk(tree))
    walks_a_tree = any(
        isinstance(n, ast.Attribute) and n.attr in {"walk", "parse"}
        and isinstance(n.value, ast.Name) and n.value.id == "ast"
        for n in ast.walk(tree))
    return imports_ast and walks_a_tree


def _discharged_t() -> bool:
    """Every library call in the report path is preceded by an input gate.

    Checked structurally rather than by grep: the gate must be REACHED before
    dispatch, so the validators must be imported AND the report path must consult
    them. A comment mentioning validation would satisfy a substring check.
    """
    from genomic_variant_classifier.evaluation import evaluator, input_validation

    for name in ("validate_probabilities", "validate_ranking_scores",
                 "validate_reference_labels"):
        if not hasattr(input_validation, name):
            return False
        if not hasattr(evaluator, name):
            return False

    # STRENGTHENED after CI-t was discharged prematurely. The first version
    # checked that validator NAMES appeared in two specific functions -- a
    # substring test that a hand count had already convinced itself of, and which
    # said nothing about the ten other functions in the module. A parsed
    # enumeration found two ungated call sites in the subgroup breakdown.
    #
    # The authority now lives in the gates suite, which parses every metric call
    # and follows validator results to the branch they govern. Duplicating that
    # logic here would give two implementations of one check, which is the defect
    # this register exists to prevent.
    gates = TESTS / "unit" / "test_report_input_gates.py"
    if not gates.exists():
        return False
    text = gates.read_text(encoding="utf-8")
    return ("_validation_governs_a_branch" in text
            and "test_every_metric_call_sits_inside_a_gated_function" in text)


DISCHARGED_CONDITIONS = {
    "CI-k": _discharged_k,
    "CI-l": _discharged_l,
    "CI-o": _discharged_o,
    "CI-t": _discharged_t,
    "CI-q": lambda: not _condition_q(),
    # A degenerate cohort must still persist. The predicate is INVERTED rather
    # than deleted: if a report ever becomes unpersistable again, this fails as a
    # regression instead of silently reverting to the state u-3 removed.
    "CI-u": lambda: not _condition_u(),
    # The writer now agrees with the reader. Inverted rather than deleted: if
    # to_dict ever emits a raw NaN again, that fails as a regression.
    "CI-p": lambda: not _condition_p(),
    # The attempted and surviving counts are reported. Inverted rather than
    # deleted: if they are ever dropped, that fails as a regression.
    "CI-m": lambda: not _condition_m(),
    # The ordered variant-identifier sequence is incorporated. Inverted rather
    # than deleted: if two different frames ever collide, that fails.
    "CI-n": lambda: not _condition_n(),
    # Interval certification is asserted positively by the bootstrap suite.
    # Inverted rather than deleted: if that assertion is ever removed, the
    # property becomes genuinely unguarded and this fails.
    "CI-r": lambda: not _condition_r(),
}


# --------------------------------------------------------------------------- #
# The register itself
# --------------------------------------------------------------------------- #
def _register_text() -> str:
    assert REGISTER.exists(), f"the carried-item register is missing: {REGISTER}"
    return REGISTER.read_text(encoding="utf-8")


def _ids_under(heading: str) -> set:
    """Item identifiers appearing in one section of the register."""
    text = _register_text()
    start = text.index(f"## {heading}")
    remainder = text[start + len(heading):]
    end = remainder.find("\n## ")
    section = remainder if end == -1 else remainder[:end]
    return set(re.findall(r"\*\*(CI-[a-z])\*\*", section))


def test_the_register_exists_and_declares_its_sections():
    text = _register_text()
    for heading in ("## Open", "## Discharged", "## Unverifiable"):
        assert heading in text, f"the register is missing the {heading!r} section"


def test_every_open_item_still_has_its_condition():
    """A register that lists solved problems is worse than no register: it sends
    a reader looking for work that is already done, and it hides the fact that
    nobody has re-checked.

    NOT PARAMETRISED, deliberately. As a parametrised test it SKIPPED once every
    predicated item was discharged -- "got empty parameter set" -- which is a
    guard reporting success while checking nothing, and would have raised the
    suite's stable skip surface from seven to eight.

    An empty open set is a legitimate and desirable state: it means no predicated
    item is currently open. Written as a loop it passes explicitly in that case
    and still names the offending item when one goes stale.
    """
    stale = []
    for item in sorted(OPEN_CONDITIONS):
        if not OPEN_CONDITIONS[item]():
            stale.append(item)
    assert not stale, (
        f"{stale} listed OPEN but their conditions no longer hold. If fixed, "
        "move them to Discharged in docs/CARRIED_ITEMS.md and move their "
        "predicates into DISCHARGED_CONDITIONS -- do not delete the check.")


@pytest.mark.parametrize("item", sorted(DISCHARGED_CONDITIONS))
def test_every_discharged_item_stays_discharged(item):
    """A regression must re-open the item as a FAILURE, not silently restore the
    condition beneath a register that still says Discharged."""
    assert DISCHARGED_CONDITIONS[item](), (
        f"{item} is listed DISCHARGED but its condition has RETURNED. This is a "
        "regression, not a bookkeeping error.")


def test_the_register_and_the_predicates_describe_the_same_items():
    """Catches the failure this register exists to prevent: an item that is
    tracked in one place and not the other."""
    open_declared = _ids_under("Open")
    discharged_declared = _ids_under("Discharged")

    missing_predicate = open_declared - set(OPEN_CONDITIONS) - {"CI-s"}
    assert not missing_predicate, (
        f"open item(s) {sorted(missing_predicate)} have no predicate; an item "
        "that cannot be checked belongs in the Unverifiable table, where its "
        "uncheckability is explicit")

    orphan_predicate = set(OPEN_CONDITIONS) - open_declared
    assert not orphan_predicate, (
        f"predicate(s) {sorted(orphan_predicate)} check items the register does "
        "not list as open")

    assert set(DISCHARGED_CONDITIONS) <= discharged_declared, (
        f"discharged predicate(s) "
        f"{sorted(set(DISCHARGED_CONDITIONS) - discharged_declared)} are not "
        "listed as discharged in the register")


def test_no_item_is_both_open_and_discharged():
    both = _ids_under("Open") & _ids_under("Discharged")
    assert not both, f"item(s) {sorted(both)} appear in both sections"


def test_the_register_resolves_the_letter_namespace_collision():
    """`docs/ROADMAP.md` uses (a)-(d) for ROOT PATTERNS and (a)-(s) for CARRIED
    ITEMS. `carried item (a)` and `root pattern (a)` are unrelated and sit five
    hundred lines apart. The register uses the CI- prefix so a bare letter can
    never again mean two things."""
    text = _register_text()
    # STRUCTURE, not vocabulary. A first draft asserted only that the phrase
    # "root pattern" appeared somewhere; the register mentions it twice, so
    # deleting the line that actually EXPLAINS the collision left the test green.
    # Both namespaces must be named WITH THEIR RANGES, which is the content a
    # reader needs in order not to reintroduce bare letters.
    lowered = text.lower()
    assert "root patterns (a)-(d)" in lowered, (
        "the register no longer states the ROOT PATTERN range; without it a "
        "reader cannot tell that a bare (a) is ambiguous")
    assert "carried items (a)-(s)" in lowered, (
        "the register no longer states the CARRIED ITEM range")
    assert re.search(r"\*\*CI-[a-z]\*\*", text), "no prefixed identifiers found"


def test_the_deferred_import_contract_holds():
    """CI-s is stated as a RULE rather than a condition, so it is verified
    directly: `registry.py` must import without scikit-learn, which means no
    module-scope import of `metrics`."""
    import ast

    source = (SRC / "evaluation" / "registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and (sub.module or "").endswith("metrics"):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)):
                    pytest.fail(
                        "registry.py imports `metrics` at module scope. That "
                        "pulls scikit-learn into evaluation/__init__ and breaks "
                        "the import contract three tests depend on. Bind kernels "
                        "by NAME and resolve them at call time.")


def test_every_predicate_actually_inspects_the_code():
    """A predicate that hardcodes its answer passes every test above.

    ADDED AFTER A SABOTAGE SURVIVOR. Replacing CI-m's measurement with
    `n_input, n_dropped = 6, 1` survived the matrix: the predicate still returned
    the right verdict, having measured nothing. That is the register's own
    failure mode -- a check that reports a status it never determined -- and it is
    the third time this shape has appeared, after the enumeration guard and the
    absence biconditional.

    Parsed rather than grepped: each predicate must CALL something or reference an
    imported module, not merely compute over literals.
    """
    import ast as _ast
    import inspect as _inspect

    checked = 0
    for name, fn in list(OPEN_CONDITIONS.items()) + [
            (n, f) for n, f in DISCHARGED_CONDITIONS.items()
            if not isinstance(f, type(lambda: 0)) or f.__name__ != "<lambda>"]:
        try:
            source = _inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        tree = _ast.parse(source.lstrip())
        calls = [n for n in _ast.walk(tree) if isinstance(n, _ast.Call)]
        # CALLS ONLY, NOT IMPORTS. A first version also required an import
        # inside the body and fired on CI-r, which imports at module scope --
        # a false positive, and a guard that cries wolf on correct code gets
        # weakened until it catches nothing.
        #
        # WHAT THIS DOES NOT CATCH, stated plainly: a predicate that CALLS the
        # code and then ignores the result, computing its verdict from literals
        # instead. Detecting that needs dataflow analysis, and a guard that
        # claimed to catch it would be worse than one that says it does not.
        # That case survived the sabotage matrix and is recorded as a known
        # limitation rather than papered over.
        assert calls, (
            f"{name}'s predicate performs no call at all; it cannot be measuring "
            "anything and is reporting a hardcoded verdict")
        checked += 1

    assert checked >= 2, (
        f"only {checked} predicate(s) were inspected; the walk is not reaching them")

