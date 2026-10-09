"""DANDELION with minimum-score gene aggregation and the top-k tie audit (owner ruling 2026-10-04): the owner's eight
named checks and worked example, exhaustive tie verification, and the regression suite proving that reference labels,
descriptive annotations, significance flags and row order cannot change a rank.

Author: Monzia Moodie
"""
from __future__ import annotations

import itertools
import math
import random
import shutil
import subprocess
from fractions import Fraction as F

import pytest

from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.ranking import (
    PINNED_COMMIT, Calibration, Implementation, MethodIdentity, Pair, State, Universe, UniverseKind, annotate,
    audit_top_k, contrast, project_scores, rank_genes, score_from_binary64, score_from_hex, top_k)

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


# ------------------------------------------------------------------ exact score TRANSPORT from R (owner ruling 2026-10-07; measured hazard)

def test_hex_scores_are_transported_exactly():
    assert score_from_hex((0.1).hex()) == score_from_binary64(0.1) == F.from_float(0.1) != F(1, 10)
    assert score_from_hex("0x1.3333333333334p-2") != score_from_hex("0x1.3333333333333p-2")      # 0.1 + 0.2 vs 0.3: distinct


def test_default_r_text_would_manufacture_a_false_tie_that_hex_transport_does_not():
    """R wrote both 0.1 + 0.2 and 0.3 as "0.3" (measured). Read from that text, g1 and g2 tie at the k = 1 boundary; read from "%a" they do
    not, and the exactly smaller score (0.3, g2) is selected with no tie envelope."""
    reference, eligible = frozenset({"g1"}), frozenset({"g1", "g2", "g3"})
    from_text = {"g1": F("0.3"), "g2": F("0.3"), "g3": F("0.9")}
    from_hex = {"g1": score_from_hex("0x1.3333333333334p-2"), "g2": score_from_hex("0x1.3333333333333p-2"), "g3": score_from_hex((0.9).hex())}
    text_audit, hex_audit = audit_top_k(from_text, eligible, reference, 1), audit_top_k(from_hex, eligible, reference, 1)
    assert text_audit["tied_genes"] == ("g1", "g2") and (text_audit["hit_lower"], text_audit["hit_upper"]) == (0, 1)
    assert hex_audit["selected"] == ("g2",) and hex_audit["tied_genes"] == ("g2",) and (hex_audit["hit_lower"], hex_audit["hit_upper"]) == (0, 0)


@pytest.mark.parametrize("bad, code", [
    (0.3, "score_encoding"), (b"0x1p-1", "score_encoding"), ("0xZZ", "score_encoding"), ("", "score_encoding"),
    ("inf", "score_encoding"), ("nan", "score_encoding"), ("-0x1p-1", "invalid_score"), ("0x0p+0", "invalid_score"), ("0x1.8p+0", "invalid_score"),
])
def test_hex_transport_refusals(bad, code):
    assert code_of(lambda: score_from_hex(bad)) == code


# ------------------------------------------------------------------ the hexadecimal transport GRAMMAR (owner ruling 2026-10-07b)

@pytest.mark.parametrize("text", ["0.3", "1", "0x1", "0x1p-1\n", " 0x1p-1", "0x1p-1 ", "0x.8p0", "0x1p", "1p-1"])
def test_hex_transport_requires_the_explicit_grammar(text):
    """float.fromhex alone accepted "0.3" as 3/16 (a decimal-looking string crossing a hexadecimal boundary), "1", "0x1" and a newline."""
    assert code_of(lambda: score_from_hex(text)) == "score_encoding"


def test_hex_transport_never_rounds_extra_precision():
    assert code_of(lambda: score_from_hex("0x1.00000000000001p-1")) == "score_not_exact_binary64"


def test_hex_transport_bounds_the_exponent_and_length():
    assert code_of(lambda: score_from_hex("0x1p-5000")) == "score_exponent"
    assert code_of(lambda: score_from_hex("0x1." + "0" * 130 + "p-1")) == "score_encoding"


def test_hex_transport_allows_a_nonunit_leading_digit_and_subnormals():
    assert score_from_hex("0x8p-4") == F(1, 2)
    smallest = math.nextafter(0.0, 1.0)
    assert score_from_hex(smallest.hex()) == F.from_float(smallest)
    assert score_from_hex("0x0.0000000000001p-1022") == F.from_float(smallest)      # R's (glibc) spelling of the same value


RSCRIPT = shutil.which("Rscript")


@pytest.mark.skipif(RSCRIPT is None, reason="needs Rscript: the transport is R's own %a output on this platform")
def test_r_percent_a_output_on_this_platform_round_trips_exactly(tmp_path):
    """The boundary carries R's output, not Python's: R's %a comes from the platform C library (Windows and Linux spell some values
    differently). Every value R prints must parse to EXACTLY the double R holds (checked through R's 17-significant-digit decimal)."""
    program = tmp_path / "probe.R"
    program.write_text('x <- c(0.5, 1, 0.1 + 0.2, 0.3, 2^-1074, 2^-1022, 1 - 2^-53, 0.9, 1e-300, 3e-310, 0.1, 2^-1060)\n'
                       'cat(paste(sprintf("%a", x), sprintf("%.17g", x), sep = "\\t"), sep = "\\n")\n', encoding="ascii")
    out = subprocess.run([RSCRIPT, "--vanilla", str(program)], capture_output=True, text=True, timeout=120, check=True).stdout
    rows = [line.split("\t") for line in out.splitlines() if line.strip()]
    assert len(rows) == 12
    for hex_text, decimal_text in rows:
        assert score_from_hex(hex_text) == F.from_float(float(decimal_text)), (hex_text, decimal_text)


# ------------------------------------------------------------------ Universe identity version 2 (owner ruling 2026-10-07b)

def test_universe_identity_is_unambiguous():
    """Version 1 joined with newlines: {"a", "b"} and {"a\\nb"} produced the same bytes before hashing."""
    assert Universe(UniverseKind.EVALUATION, "r", frozenset({"a", "b"})).identity != Universe(UniverseKind.EVALUATION, "r", frozenset({"a\nb"})).identity
    assert Universe(UniverseKind.EVALUATION, "r\na", frozenset({"b"})).identity != Universe(UniverseKind.EVALUATION, "r", frozenset({"a", "b"})).identity


# ---------------------------------------------------------------------------------------------------- the score artifact (2026-10-09)
from genomic_variant_classifier.inference.ranking import read_score_matrix  # noqa: E402


def test_the_score_matrix_reads_exact_scores_and_missing_cells():
    raw = b"g1\te1\t0x1.47ae147ae147bp-7\ng1\te2\tNA\ng2\te1\t0x1p+0\n"
    assert read_score_matrix(raw) == {("g1", "e1"): F(float.fromhex("0x1.47ae147ae147bp-7")), ("g1", "e2"): None, ("g2", "e1"): F(1)}
    assert read_score_matrix(b"") == {}


@pytest.mark.parametrize("raw, code", [
    (b"g1\te1\t0x1p-1\ng1\te1\t0x1p-2\n", "score_matrix_duplicate_cell"),       # never "the last value wins"
    (b"g1\te1\t0x1p-1\ng1\te1\tNA\n", "score_matrix_duplicate_cell"),
    (b"g1\te1\t0x1p-1\textra\n", "score_matrix_line"),
    (b"g1\te1\n", "score_matrix_line"),
    (b"\te1\t0x1p-1\n", "score_matrix_line"),
    (b"g1\t\t0x1p-1\n", "score_matrix_line"),
    (b"g1\te1\t0x1p-1", "score_matrix_bytes"),
    (b"g1\te1\t0x1p-1\r\n", "score_matrix_bytes"),
    ("g1\te1\t0x1p-1\n", "score_matrix_bytes"),
    ("gö\te1\t0x1p-1\n".encode("utf-8"), "score_matrix_bytes"),
    (b"g1\te1\t0x0p+0\n", "invalid_score"),                                         # a score is in (0, 1]
    (b"g1\te1\t0x1p+1\n", "invalid_score"),
    (b"g1\te1\t0.5\n", "score_encoding"),
    (b"g1\te1\t0x1.00000000000001p-1\n", "score_not_exact_binary64"),
])
def test_the_score_matrix_is_strict(raw, code):
    assert code_of(lambda: read_score_matrix(raw)) == code
