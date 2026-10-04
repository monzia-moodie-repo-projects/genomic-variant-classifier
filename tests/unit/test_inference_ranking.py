"""DANDELION with minimum-score gene aggregation and the top-k tie audit (owner ruling 2026-10-04): the owner's eight
named checks and worked example, exhaustive tie verification, and the regression suite proving that reference labels,
descriptive annotations, significance flags and row order cannot change a rank.

Author: Monzia Moodie
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction as F

import pytest

from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.ranking import (
    PINNED_COMMIT, Calibration, Implementation, MethodIdentity, Pair, State, Universe, UniverseKind, annotate,
    audit_top_k, contrast, project_scores, rank_genes, score_from_binary64, top_k)

PLAN = {("g1", "e1"): True, ("g1", "e2"): True, ("g2", "e1"): True, ("g2", "e2"): False}
ROWS = [Pair("g1", "e1", State.SCORED, F(3, 100), False), Pair("g1", "e2", State.SCORED, F(1, 100), True),
        Pair("g2", "e1", State.SCORED, F(2, 100), True), Pair("g2", "e2", State.STRUCTURAL)]


def code_of(call):
    with pytest.raises(InferenceError) as exc:
        call()
    return exc.value.code


# ------------------------------------------------------------------ the owner's eight named checks

def test_score_ordering_and_best_exposure():
    r = rank_genes(ROWS, PLAN)
    assert [x.gene for x in r] == ["g1", "g2"] and (r[0].score, r[0].best_exposure) == (F(1, 100), "e2")
    assert (r[0].tested_pairs, r[0].structural_pairs, r[1].tested_pairs, r[1].structural_pairs) == (2, 0, 1, 1)


def test_row_order_invariance():
    assert rank_genes(list(reversed(ROWS)), PLAN) == rank_genes(ROWS, PLAN)


def test_significance_flags_never_change_the_order():
    flipped = [Pair(x.gene, x.exposure, x.state, x.score, None if x.significant is None else not x.significant) for x in ROWS]
    assert [x.gene for x in rank_genes(flipped, PLAN)] == [x.gene for x in rank_genes(ROWS, PLAN)]


@pytest.mark.parametrize("rows, code", [
    (ROWS + [ROWS[0]], "duplicate_pair"),
    (ROWS[:-1], "missing_pair_record"),
    (ROWS[:3] + [Pair("g2", "e2", State.FAILED)], "incomplete_method_execution"),
    (ROWS[:3] + [Pair("g2", "e2", State.STRUCTURAL, F(1, 2))], "structural_pair_has_result"),
    (ROWS + [Pair("g3", "e1", State.SCORED, F(1, 2), False)], "unexpected_pair"),
    (ROWS[:2] + [Pair("g2", "e1", State.STRUCTURAL), ROWS[3]], "required_pair_unscored"),
])
def test_incomplete_or_inconsistent_execution_refuses_the_ranking(rows, code):
    assert code_of(lambda: rank_genes(rows, PLAN)) == code


def test_a_failure_even_on_a_structural_pair_refuses():
    assert code_of(lambda: rank_genes(ROWS[:3] + [Pair("g2", "e2", State.FAILED)], PLAN)) == "incomplete_method_execution"


def test_a_gene_with_only_structural_pairs_must_be_excluded_before_sealing():
    plan = {**PLAN, ("g3", "e1"): False}
    assert code_of(lambda: rank_genes(ROWS + [Pair("g3", "e1", State.STRUCTURAL)], plan)) == "gene_has_no_planned_score"


def test_insufficient_nomination_budget():
    assert code_of(lambda: top_k(rank_genes(ROWS, PLAN), 3)) == "nomination_budget_incomplete"


# ------------------------------------------------------------------ tie audit

def test_the_owners_worked_example():
    scores = {"a": F("0.01"), "b": F("0.02"), "c": F("0.02"), "d": F("0.02"), "e": F("0.10")}
    r = audit_top_k(scores, frozenset(scores), frozenset({"a", "c", "d"}), 3)
    assert (r["selected"], r["hits"], r["hit_lower"], r["hit_upper"]) == (("a", "b", "c"), 2, 2, 3)


def test_tie_bounds_equal_exhaustive_tie_choices():
    rng = random.Random(20261004)
    for _ in range(600):
        genes = ["g{}".format(i) for i in range(rng.randint(2, 7))]
        scores = {g: F(rng.choice([1, 2, 2, 3]), 10) for g in genes}
        reference = frozenset(g for g in genes if rng.random() < 0.5)
        k = rng.randint(1, len(genes))
        out = audit_top_k(scores, frozenset(genes), reference, k)
        better = [g for g in genes if scores[g] < out["cutoff"]]
        tied = [g for g in genes if scores[g] == out["cutoff"]]
        choices = {len((set(better) | set(c)) & reference) for c in itertools.combinations(tied, k - len(better))}
        assert (out["hit_lower"], out["hit_upper"]) == (min(choices), max(choices))
        assert out["hit_lower"] <= out["hits"] <= out["hit_upper"]


def test_contrast_bounds_combine_each_methods_ties_independently():
    a = {"hits": 3, "hit_lower": 2, "hit_upper": 4}
    b = {"hits": 1, "hit_lower": 1, "hit_upper": 2}
    assert contrast(a, b) == {"delta": 2, "tie_lower": 0, "tie_upper": 3}


@pytest.mark.parametrize("call, code", [
    (lambda: audit_top_k({"a": F(1, 2)}, frozenset({"a"}), frozenset({"z"}), 1), "reference_outside_evaluation_universe"),
    (lambda: audit_top_k({"a": F(1, 2)}, frozenset({"a", "b"}), frozenset(), 1), "evaluation_score_missing"),
    (lambda: audit_top_k({"a": 0.5}, frozenset({"a"}), frozenset(), 1), "invalid_score"),
    (lambda: audit_top_k({"a": F(1, 2)}, frozenset({"a"}), frozenset(), 2), "invalid_k"),
])
def test_audit_refusals(call, code):
    assert code_of(call) == code


# ------------------------------------------------------------------ regression: nothing downstream changes a rank

def test_reference_labels_cannot_change_the_ranking_or_the_selection():
    scores = {g: F(i + 1, 100) for i, g in enumerate("abcdefgh")}
    universe = frozenset(scores)
    selections = {audit_top_k(scores, universe, frozenset(ref), 3)["selected"]
                  for ref in (set(), {"a"}, {"h"}, {"b", "g"}, set("abcdefgh"))}
    assert selections == {("a", "b", "c")}


def test_descriptive_annotations_cannot_change_the_order():
    ranked = rank_genes(ROWS, PLAN)
    labelled = annotate(ranked, {"g2": {"cis_gene": "ZZZ", "burden_significant": True}, "g1": {"cis_gene": None}})
    assert tuple(r for r, _ in labelled) == ranked and labelled[1][1]["cis_gene"] == "ZZZ"


def test_scores_outside_the_evaluation_universe_cannot_change_the_projected_result():
    ranked = rank_genes(ROWS, PLAN)
    universe = Universe(UniverseKind.EVALUATION, "common-eligibility-v1", frozenset({"g1", "g2"}))
    plan = {**PLAN, ("g0", "e1"): True}
    wider = rank_genes(ROWS + [Pair("g0", "e1", State.SCORED, F(1, 10**6), True)], plan)   # best-scoring, but ineligible
    assert audit_top_k(project_scores(wider, universe), universe.genes, frozenset({"g1"}), 1) == \
        audit_top_k(project_scores(ranked, universe), universe.genes, frozenset({"g1"}), 1)


# ------------------------------------------------------------------ transport, universes and method identities

def test_binary64_scores_are_transported_exactly():
    assert score_from_binary64(0.1) == F.from_float(0.1) != F(1, 10)
    for bad in (0.0, 1.5, float("nan"), float("inf"), F(1, 2), 1):
        assert code_of(lambda bad=bad: score_from_binary64(bad)) == "invalid_score"


def test_universe_identity_is_its_content():
    u = Universe(UniverseKind.EVALUATION, "rule-v1", frozenset({"a", "b"}))
    assert u.identity == Universe(UniverseKind.EVALUATION, "rule-v1", frozenset({"b", "a"})).identity
    assert u.identity != Universe(UniverseKind.FITTING, "rule-v1", frozenset({"a", "b"})).identity
    assert u.identity != Universe(UniverseKind.EVALUATION, "rule-v2", frozenset({"a", "b"})).identity


def test_projection_requires_an_evaluation_universe_with_every_score():
    ranked = rank_genes(ROWS, PLAN)
    fitting = Universe(UniverseKind.FITTING, "fit-v1", frozenset({"g1"}))
    assert code_of(lambda: project_scores(ranked, fitting)) == "evaluation_universe_required"
    wide = Universe(UniverseKind.EVALUATION, "eval-v1", frozenset({"g1", "g9"}))
    assert code_of(lambda: project_scores(ranked, wide)) == "evaluation_score_missing"


def method(**kw):
    args = dict(published_method="Trans-regulatory gene mapping prioritizes disease drivers in asthma",
                repository="https://github.com/mxxptian/DANDELION", commit=PINNED_COMMIT,
                implementation=Implementation.ROOT_PACKAGE, calibration=Calibration.NONE_EXECUTED,
                qvalue_backend_policy="safe_qvalues: BH if n < 10, < 4 distinct values, or any qvalue warning/error",
                extension="minimum-score gene aggregation v1")
    args.update(kw)
    return MethodIdentity(**args)


def test_the_pinned_implementation_cannot_claim_the_published_calibration():
    assert method().calibration is Calibration.NONE_EXECUTED
    assert code_of(lambda: method(calibration=Calibration.PUBLISHED_EMPIRICAL_NULL)) == \
        "calibration_not_executed_at_pinned_commit"
    assert code_of(lambda: method(commit="f471153")) == "method_commit"


def test_the_rulings_illustration_minimum_score_not_significance_first():
    """Gene A: minimum 0.001, no significant pair; gene B: 0.004 with one. Minimum-score order is A, B; a
    significance-first tier would give B, A (ruling 2026-10-04, illustrative configuration)."""
    plan = {("A", "e1"): True, ("B", "e2"): True}
    rows = [Pair("A", "e1", State.SCORED, F(1, 1000), False), Pair("B", "e2", State.SCORED, F(4, 1000), True)]
    assert [x.gene for x in rank_genes(rows, plan)] == ["A", "B"]


def test_score_orders_before_name_when_they_disagree():
    plan = {("aaa", "e1"): True, ("zzz", "e1"): True}
    rows = [Pair("aaa", "e1", State.SCORED, F(1, 2), False), Pair("zzz", "e1", State.SCORED, F(1, 10), False)]
    assert [x.gene for x in rank_genes(rows, plan)] == ["zzz", "aaa"]
    tie = [Pair("aaa", "e1", State.SCORED, F(1, 10), False), Pair("zzz", "e1", State.SCORED, F(1, 10), False)]
    assert [x.gene for x in rank_genes(tie, plan)] == ["aaa", "zzz"]          # exact tie -> stable gene ID


# ------------------------------------------------------------------ numerical sensitivity audit (owner ruling 2026-10-04b)

from genomic_variant_classifier.inference.ranking import ScoreRange, topk_audit  # noqa: E402


def test_topk_audit_equals_exhaustive_endpoint_enumeration():
    rng = random.Random(20261005)
    grid = [F(i, 8) for i in range(9)]
    for _ in range(300):
        n = rng.randint(2, 5)
        rows = [ScoreRange("g{}".format(i), *sorted(rng.choice(grid) for _ in range(2))) for i in range(n)]
        for k in range(1, n + 1):
            out = topk_audit(rows, k)
            ranks = {r.gene: set() for r in rows}
            for choice in itertools.product(*[sorted({r.low, r.high}) for r in rows]):
                for pos, (_, gene) in enumerate(sorted(zip(choice, [r.gene for r in rows])), 1):
                    ranks[gene].add(pos)
            for gene, seen in ranks.items():
                assert (out[gene]["best_rank"], out[gene]["worst_rank"]) == (min(seen), max(seen))


def test_topk_audit_statuses():
    rows = [ScoreRange("a", F(1, 100), F(1, 100)), ScoreRange("b", F(5, 100), F(30, 100)),
            ScoreRange("c", F(10, 100), F(10, 100)), ScoreRange("z", F(90, 100), F(95, 100))]
    out = topk_audit(rows, 2)
    assert {g: out[g]["status"] for g in out} == {"a": "always_in", "b": "sensitive", "c": "sensitive", "z": "always_out"}


@pytest.mark.parametrize("call, code", [
    (lambda: ScoreRange("", F(0), F(1)), "gene_id"),
    (lambda: ScoreRange("a", 0.1, F(1)), "exact_bounds_required"),
    (lambda: ScoreRange("a", F(1, 2), F(1, 4)), "score_range"),
    (lambda: ScoreRange("a", F(0), F(3, 2)), "score_range"),
    (lambda: topk_audit([ScoreRange("a", F(0), F(1))], 2), "k_range"),
    (lambda: topk_audit([ScoreRange("a", F(0), F(1)), ScoreRange("a", F(0), F(1))], 1), "duplicate_gene"),
])
def test_topk_audit_refusals(call, code):
    assert code_of(call) == code
