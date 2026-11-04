"""The bootstrap replicate engine: an interval and its accounting are ONE set of replicates.

CERTIFIED-AUPRC-DEGENERATE-REPLICATES-1, RECALL-CURVE-FABRICATED-1 (2026-11-02).

WHAT WAS WRONG, MEASURED
========================
The percentile set of a gene-cluster interval and its valid/degenerate counts came
from two different loops. `cluster_bootstrap_ci` kept every replicate whose metric
value was finite; `_count_valid_replicates` replayed the draws and called a
single-class replicate degenerate. They agree only for a metric that returns a
non-finite value on one class. scikit-learn 1.8.0's `average_precision_score` -- the
metric of the evaluator's certified interval for the area under the precision-recall
curve -- returns 0.0 on an all-negative replicate (with a UserWarning) and 1.0 on an
all-positive one (with NO warning). On a 40-gene cohort with 2 positive genes, 1,000
replicates: reported [0.0000, 0.8333], status OK, "863 valid, 137 degenerate", while all
1,000 values entered the percentiles; over the two-class replicates, [0.4408, 0.8333].

The same suite's 33 warnings (recorded as expected since 2026-08-26) were scikit-learn
objecting to exactly these calls, and to the curves computed on single-class cohorts --
where the precision-recall curve on an all-negative cohort persisted recall = 1.0 at
every threshold, a value nobody measured.

Every assertion below is paired with the case that must succeed, and the oracles are
written independently of the engine they check.

Author: Monzia Moodie
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation import metrics as M
from genomic_variant_classifier.evaluation.absence import AbsenceCause
from genomic_variant_classifier.evaluation.capabilities import BootstrapUnit, MetricStatus
from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator, derive_seed

sklearn_metrics = pytest.importorskip("sklearn.metrics")
ROOT = Path(__file__).resolve().parents[2]
AP = ClinicalEvaluator._nan_safe(sklearn_metrics.average_precision_score)


def _cohort(n_genes, n_pos_genes, per_gene, seed):
    """Positives concentrated in a few genes, so whole-gene resampling can draw none."""
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.array([f"G{i:03d}" for i in range(n_genes)], dtype=object), per_gene)
    y = np.zeros(clusters.size)
    for g in rng.choice(n_genes, n_pos_genes, replace=False):
        rows = np.flatnonzero(clusters == f"G{g:03d}")
        y[rows[: max(1, per_gene // 2)]] = 1
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, y.size), 0, 1)
    return y, p, clusters


def _oracle_two_class_interval(fn, y, s, clusters, *, n_boot, seed, alpha):
    """An INDEPENDENT oracle: the documented design written out longhand.

    Whole clusters with replacement, in `np.unique` order, from one generator; a
    replicate counts only if it carries both classes and a finite value.
    """
    uniq = np.unique(clusters)
    rows = {u: np.flatnonzero(clusters == u) for u in uniq}
    rng = np.random.default_rng(seed)
    kept = []
    for _ in range(n_boot):
        i = np.concatenate([rows[u] for u in rng.choice(uniq, size=uniq.size, replace=True)])
        if np.unique(y[i]).size < 2:
            continue
        v = fn(y[i], s[i])
        if np.isfinite(v):
            kept.append(v)
    lo, hi = np.percentile(kept, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(kept)


# --------------------------------------------------------------------------- #
# 1. The defect: a value that does not exist never enters a certified interval
# --------------------------------------------------------------------------- #
def test_scikit_learns_average_precision_answers_on_one_class():
    """THE PREMISE, pinned, so the tests below fail loudly if a scikit-learn upgrade
    changes it rather than passing for a reason nobody re-examined."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert sklearn_metrics.average_precision_score([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0
        assert sklearn_metrics.average_precision_score([1, 1, 1], [0.1, 0.2, 0.3]) == 1.0
    assert [type(w.message).__name__ for w in caught] == ["UserWarning"], (
        "the all-positive case must still be SILENT: it is the one no warning filter "
        "can catch, which is why the engine refuses it by class count")


def test_a_single_class_replicate_never_enters_the_certified_interval():
    y, p, g = _cohort(40, 2, 6, seed=1)
    r = M.bootstrap_metric(AP, y, p, clusters=g, unit=BootstrapUnit.GENE, n_boot=400, seed=1)
    lo, hi, n_kept = _oracle_two_class_interval(AP, y, p, g, n_boot=400, seed=1,
                                                alpha=1.0 - 0.95)
    assert (r.lower, r.upper) == (lo, hi)
    assert r.n_valid == n_kept == 353 and r.n_degenerate == 47
    # The pre-change kernel reported lower = 0.0 here (measured 2026-11-02): the 47
    # zeros of the all-negative replicates were inside the percentile set.
    assert r.lower > 0.2, r.lower
    assert r.status is MetricStatus.OK and r.certification_eligible is True


def test_the_measured_cohort_reproduces():
    """The cohort of the finding, built exactly as the 2026-11-02 measurement built it
    (unpadded gene names, so `np.unique` orders them G0, G1, G10, ...), at its own
    replicate count: reported [0.0000, 0.8333] before the engine, [0.4408, 0.8333] now."""
    rng = np.random.default_rng(42)
    g = np.repeat([f"G{i}" for i in range(40)], 6)
    y = np.zeros(g.size)
    for gene in rng.choice(40, 2, replace=False):
        y[np.flatnonzero(g == f"G{gene}")[:3]] = 1
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, y.size), 0, 1)
    r = M.bootstrap_metric(AP, y, p, clusters=g, n_boot=1000, seed=7)
    assert (round(r.lower, 4), round(r.upper, 4)) == (0.4408, 0.8333)
    assert (r.n_valid, r.n_degenerate) == (863, 137)


def test_an_all_positive_replicate_is_degenerate_although_the_metric_answers_one():
    """The silent case. Negatives live in ONE gene of ten, so a draw that misses it is
    all-positive; average precision would call that replicate perfect."""
    rng = np.random.default_rng(3)
    g = np.repeat(np.array([f"G{i}" for i in range(10)], dtype=object), 8)
    y = np.ones(g.size)
    y[:4] = 0
    p = np.clip(0.4 + 0.3 * y + rng.normal(0, 0.2, g.size), 0, 1)
    calls = []

    def spy(yi, si):
        calls.append(np.unique(yi).size)
        return AP(yi, si)

    values, n_degenerate = M._two_class_replicates(
        spy, y, p, draw=M._cluster_draw(np.unique(g), {u: np.flatnonzero(g == u) for u in np.unique(g)}),
        n_boot=300, seed=5)
    assert n_degenerate > 50, n_degenerate      # P(miss G0 in 10 draws) = 0.9**10 ~ 0.35
    assert calls and set(calls) == {2}, "the metric was called on a single-class replicate"
    assert values.size + n_degenerate == 300
    assert len(calls) == values.size, "every two-class call here returns a finite value"


def test_the_accounting_counts_exactly_the_values_the_interval_was_taken_from():
    y, p, g = _cohort(30, 3, 6, seed=3)
    lo, hi, values, n_degenerate, n_clusters = M._cluster_bootstrap(
        AP, y, p, g, n_boot=500, alpha=0.05, seed=11, two_stage=False)
    r = M.bootstrap_metric(AP, y, p, clusters=g, n_boot=500, seed=11, confidence_level=0.95)
    assert r.n_valid == values.size and r.n_degenerate == n_degenerate
    assert values.size + n_degenerate == 500 and n_clusters == 30
    assert np.all(np.isfinite(values)) and values.min() > 0.0, (
        "an all-negative replicate's 0.0 is in the percentile set")


def test_a_two_class_replicate_whose_metric_is_not_finite_is_degenerate():
    """Found by mutation (2026-11-02): dropping this branch left every other test green.
    A replicate is valid only if the metric ANSWERS; otherwise the accounting would
    claim fewer replicates than were requested."""
    y, p, g = _cohort(30, 10, 6, seed=8)
    seen = []

    def every_third_is_nan(yi, si):
        seen.append(1)
        return float("nan") if len(seen) % 3 == 0 else AP(yi, si)

    values, n_degenerate = M._two_class_replicates(
        every_third_is_nan, y, p,
        draw=M._cluster_draw(np.unique(g), {u: np.flatnonzero(g == u) for u in np.unique(g)}),
        n_boot=300, seed=2)
    assert values.size + n_degenerate == 300
    assert values.size == len(seen) - len(seen) // 3
    r = M.bootstrap_metric(every_third_is_nan, y, p, clusters=g, n_boot=300, seed=2)
    assert r.n_valid + r.n_degenerate == 300 and r.n_degenerate >= 300 // 3 - 1


def test_the_two_stage_draw_resamples_rows_within_each_cluster():
    """Found by mutation: ignoring `two_stage` left every other test green. With two
    stages a drawn cluster's rows are themselves resampled, so some replicate repeats a
    row of a cluster drawn once; with one stage every drawn cluster contributes its
    rows exactly once each."""
    g = np.repeat(np.array([f"G{i}" for i in range(12)], dtype=object), 5)
    uniq = np.unique(g)
    index_of = {u: np.flatnonzero(g == u) for u in uniq}
    rng_one, rng_two = np.random.default_rng(4), np.random.default_rng(4)
    one = M._cluster_draw(uniq, index_of, two_stage=False)
    two = M._cluster_draw(uniq, index_of, two_stage=True)
    whole_clusters_only, within_cluster_repeats = True, False
    for _ in range(50):
        a, b = one(rng_one), two(rng_two)
        counts_a = np.bincount(a, minlength=g.size).reshape(12, 5)
        whole_clusters_only &= bool(np.all(counts_a == counts_a[:, :1]))
        counts_b = np.bincount(b, minlength=g.size).reshape(12, 5)
        within_cluster_repeats |= bool(np.any(counts_b != counts_b[:, :1]))
        assert a.size == b.size == g.size
    assert whole_clusters_only and within_cluster_repeats


def test_an_unstratified_row_bootstrap_does_not_hold_the_class_balance():
    """Found by mutation: stratifying regardless of the argument left every other test
    green. Stratified replicates keep the positive count fixed; unstratified ones vary
    it -- and only they can draw a single-class replicate, which the engine refuses."""
    y = np.r_[np.ones(3), np.zeros(37)]
    p = np.linspace(0.05, 0.95, 40)
    positives = {True: set(), False: set()}
    for stratified in (True, False):
        def spy(yi, si, _s=stratified):
            positives[_s].add(int(np.sum(yi)))
            return AP(yi, si)
        _lo, _hi, values, n_degenerate = M._row_bootstrap(
            spy, y, p, n_boot=300, alpha=0.05, seed=6, stratified=stratified)
        assert values.size + n_degenerate == 300
        if stratified:
            assert n_degenerate == 0
        else:
            assert n_degenerate > 0, "P(no positive among 40 draws) = (37/40)**40 ~ 0.044"
    assert positives[True] == {3}
    assert len(positives[False]) > 3 and 0 not in positives[False]


def test_two_stage_replicates_obey_the_same_contract():
    y, p, g = _cohort(40, 2, 6, seed=1)
    lo, hi = M.cluster_bootstrap_ci(AP, y, p, g, n_boot=400, seed=1, two_stage=True)
    assert lo > 0.0, "the two-stage design admitted a single-class replicate"


def test_an_unstratified_row_bootstrap_obeys_the_same_contract():
    y, p, _ = _cohort(40, 2, 6, seed=1)
    calls = []

    def spy(yi, si):
        calls.append(np.unique(yi).size)
        return AP(yi, si)

    M.bootstrap_ci(spy, y, p, n_boot=300, seed=1, stratified=False)
    assert calls and set(calls) == {2}


def test_a_single_class_cohort_has_no_estimate_and_no_replicate():
    calls = []

    def spy(yi, si):
        calls.append(1)
        return AP(yi, si)

    y = np.zeros(60)
    g = np.repeat(np.array([f"G{i}" for i in range(6)], dtype=object), 10)
    r = M.bootstrap_metric(spy, y, np.linspace(0, 1, 60), clusters=g, n_boot=200, seed=1)
    assert calls == [], "the metric was called on a cohort carrying one class"
    assert np.isnan(r.estimate)
    assert r.status is MetricStatus.UNDEFINED and r.finding == "cluster_bootstrap_degenerate"
    assert (r.n_valid, r.n_degenerate) == (0, 200)


# --------------------------------------------------------------------------- #
# 2. Nothing else moved: metrics that already returned NaN on one class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,gene,variant", [
    ("kernel_auroc", M.auroc,
     (0.7452910310505247, 0.9792358104858105, 398, 2), (0.8340460526315789, 0.9766173245614035, 400, 0)),
    ("kernel_auprc", M.auprc,
     (0.2557305813256676, 0.7562167636015751, 398, 2), (0.3388563052722167, 0.7742423767752713, 400, 0)),
    ("sklearn_auroc", ClinicalEvaluator._nan_safe(sklearn_metrics.roc_auc_score),
     (0.7452910310505247, 0.9792358104858104, 398, 2), (0.834046052631579, 0.9766173245614035, 400, 0)),
], ids=["kernel_auroc", "kernel_auprc", "sklearn_auroc"])
def test_intervals_of_metrics_undefined_on_one_class_are_bit_identical(name, fn, gene, variant):
    """Literals measured on the PRE-change kernel (2026-11-02, the drift-profile tree
    1d3edf23). These metrics already returned NaN on a single-class replicate, so the
    old loop excluded what the engine now refuses: their intervals must not move."""
    y, p, g = _cohort(40, 4, 6, seed=2)
    r = M.bootstrap_metric(fn, y, p, clusters=g, unit=BootstrapUnit.GENE, n_boot=400, seed=2)
    v = M.bootstrap_metric(fn, y, p, unit=BootstrapUnit.VARIANT, n_boot=400, seed=2)
    assert (r.lower, r.upper, r.n_valid, r.n_degenerate) == gene
    assert (v.lower, v.upper, v.n_valid, v.n_degenerate) == variant


# --------------------------------------------------------------------------- #
# 3. The evaluator reports the corrected interval, and nothing warns
# --------------------------------------------------------------------------- #
def test_the_evaluator_reports_the_two_class_interval():
    y, p, g = _cohort(40, 2, 6, seed=1)
    with contextlib.redirect_stdout(io.StringIO()):
        report = ClinicalEvaluator(n_bootstrap=400, random_state=42).evaluate(
            y, p, meta=pd.DataFrame({"gene_symbol": g}), model_name="probe")
    lo, hi, n_kept = _oracle_two_class_interval(
        AP, y, p, g, n_boot=400, seed=derive_seed(42, "auprc"), alpha=1.0 - 0.95)
    assert report.auprc_ci_lo == round(lo, 5) and report.auprc_ci_hi == round(hi, 5)
    assert report.auprc_ci_n_valid == n_kept
    assert report.auprc_ci_lo > 0.0


@pytest.mark.parametrize("label", ["all_negative", "all_positive", "constant_classifier",
                                   "mixed", "clustered_mixed"])
def test_no_scikit_learn_warning_reaches_an_evaluation(label):
    """The 33 suite warnings were scikit-learn objecting to undefined calls. With the
    calls gone, NOT ONE warning from scikit-learn is emitted -- recorded with every
    filter set to `always`, so nothing in this process's configuration can hide one."""
    rng = np.random.default_rng(9)
    n = 240
    mixed = rng.binomial(1, 0.3, n).astype(float)
    cohorts = {"all_negative": (np.zeros(n), np.full(n, 0.1)),
               "all_positive": (np.ones(n), np.full(n, 0.9)),
               "constant_classifier": (np.r_[np.zeros(180), np.ones(60)], np.full(n, 0.1)),
               "mixed": (mixed, np.clip(0.3 + 0.4 * mixed + rng.normal(0, 0.2, n), 0, 1))}
    y, p, g = _cohort(40, 2, 6, seed=1)
    cohorts["clustered_mixed"] = (y, p)
    y, p = cohorts[label]
    meta = pd.DataFrame({"gene_symbol": np.repeat(
        np.array([f"G{i:03d}" for i in range(40)], dtype=object), 6)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            ClinicalEvaluator(n_bootstrap=200, random_state=42).evaluate(
                y, p, meta=meta, model_name=label)
    from_sklearn = [f"{Path(w.filename).name}:{w.lineno} {w.message}" for w in caught
                    if "sklearn" in w.filename.replace("\\", "/")]
    assert from_sklearn == []


# --------------------------------------------------------------------------- #
# 4. Each curve takes its cause from the channel it was computed from
# --------------------------------------------------------------------------- #
def test_a_refused_ranking_channel_withholds_only_the_ranking_curves():
    """`scores` refused beside valid probabilities: the ranking curves are WITHHELD BY
    THE INPUT GATE, not undefined on the cohort -- the data is fine -- and the
    calibration curve, computed from the valid probabilities, is kept."""
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.4, 200).astype(float)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 200), 0, 1)
    scores = p.copy()
    scores[5] = np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        report = ClinicalEvaluator(n_bootstrap=0).evaluate(y, p, scores=scores, model_name="r")
    ranking = {"fpr_curve", "tpr_curve", "precision_curve", "recall_curve"}
    assert set(report.curve_absence) == ranking
    for name in ranking:
        assert report.curve_absence[name].cause is AbsenceCause.WITHHELD_BY_INPUT_GATE
        assert report.curve_absence[name].reason == "nonfinite_ranking_scores"
    assert report.calibration_frac_pos and report.calibration_mean_pred


def test_an_unexplained_empty_curve_is_refused_loudly():
    """A supported curve over validated input that comes back empty is a computation
    defect, and the absence vocabulary has no cause for it: refused, not recorded."""
    from genomic_variant_classifier.evaluation.evaluator import _absence_maps
    from genomic_variant_classifier.evaluation.input_validation import InputValidation

    ok = InputValidation.valid()
    with pytest.raises(RuntimeError, match="defect in the curve computation"):
        _absence_maps(legacy={}, metric_results={}, label_check=ok, probability_check=ok,
                      ranking_check=ok, curves={"fpr_curve": np.array([])},
                      n_rows=10, n_pos=4, n_neg=6)
    # the falsifiable companion: the same empty curve on a cohort with no negatives is
    # undefined there, and recorded with that cause
    _, curves = _absence_maps(legacy={}, metric_results={}, label_check=ok, probability_check=ok,
                              ranking_check=ok, curves={"fpr_curve": np.array([])},
                              n_rows=10, n_pos=10, n_neg=0)
    assert curves["fpr_curve"].cause is AbsenceCause.UNDEFINED_ON_COHORT
    assert curves["fpr_curve"].reason == "roc_curve_requires_both_classes:positives=10,negatives=0"


def test_the_kernel_imports_and_bootstraps_without_scikit_learn():
    """metrics.py imported nine scikit-learn names it never used until 2026-11-02. In a
    child interpreter with scikit-learn blocked it now imports, runs the engine on its
    own metric, and loads no part of scikit-learn. A child process, because blocking an
    import in THIS interpreter would change module identity for every later test."""
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'sklearn' or name.startswith('sklearn.'):\n"
        "            raise ModuleNotFoundError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import numpy as np\n"
        "from genomic_variant_classifier.evaluation import metrics as M\n"
        "y = np.r_[np.zeros(30), np.ones(30)]\n"
        "s = np.linspace(0, 1, 60)\n"
        "g = np.repeat(np.array(['G%d' % i for i in range(12)], dtype=object), 5)\n"
        "r = M.bootstrap_metric(M.auroc, y, s, clusters=g, n_boot=200, seed=1)\n"
        "assert r.n_valid + r.n_degenerate == 200\n"
        "assert not [k for k in sys.modules if k == 'sklearn' or k.startswith('sklearn.')]\n"
        "print('KERNEL_OK')\n")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
                         timeout=300)
    assert out.returncode == 0 and "KERNEL_OK" in out.stdout, out.stderr
    # the falsifiable companion: the blocker really blocks
    blocked = subprocess.run([sys.executable, "-B", "-c", code.replace(
        "from genomic_variant_classifier.evaluation import metrics as M\n",
        "import sklearn.metrics\n")], capture_output=True, text=True, cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, timeout=300)
    assert blocked.returncode != 0 and "blocked: sklearn" in blocked.stderr, blocked.stderr


# --------------------------------------------------------------------------- #
# 5. The configuration keeps it that way
# --------------------------------------------------------------------------- #
def test_the_projects_configuration_fails_a_test_that_computes_an_undefined_ranking_metric(
        tmp_path, child_pytest_environment):
    """pyproject.toml makes any warning from scikit-learn's ranking module an ERROR, so
    a future call on undefined input fails the test that makes it instead of joining a
    warning count. Through pytest itself, against THIS repository's configuration, in
    both directions so neither verdict is vacuous."""
    (tmp_path / "test_planted.py").write_text(
        "from sklearn.metrics import precision_recall_curve, roc_auc_score\n\n\n"
        "def test_undefined():\n    roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])\n\n\n"
        "def test_no_positive():\n    precision_recall_curve([0, 0, 0], [0.1, 0.2, 0.3])\n\n\n"
        "def test_defined():\n    assert roc_auc_score([0, 1, 1], [0.1, 0.2, 0.3]) == 1.0\n",
        encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(tmp_path),
                           str(tmp_path / "test_planted.py")],
                          cwd=str(tmp_path), env=child_pytest_environment(), capture_output=True,
                          text=True, timeout=300)
    out = done.stdout + done.stderr
    assert done.returncode == 1, out
    assert "2 failed, 1 passed" in out, out
    assert "test_planted.py::test_undefined" in out and "test_planted.py::test_no_positive" in out, out
