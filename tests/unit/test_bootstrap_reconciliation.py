"""Battery for the bootstrap inference reconciliation (Option C, commit 2).

WHAT THIS FILE PROVES
=====================
Before 2026-07-26 three bootstrap implementations existed in this repository:
`ClinicalEvaluator._bootstrap_ci`, `reports.report_generator.bootstrap_metric`,
and the kernel's `bootstrap_ci` / `cluster_bootstrap_ci`. Only the kernel
respected gene clustering, and the measured gene-cluster design effect on the
real cohort was 2.935 times (suite-size ratchet entry 2055) -- meaning every
interval the other two produced was too narrow by roughly that factor.

These tests assert that exactly one engine now exists, that the resampling unit
is an explicit typed part of every interval rather than an accident of which
caller produced it, and that no path silently substitutes a row-level interval
for a gene-cluster one.

Every guard is proven FALSIFIABLE: each refusal is paired with a case that must
succeed, so a build that refused everything would fail as loudly as one that
accepted everything.

Author: written for Monzia Moodie, 2026-07-26.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.capabilities import BootstrapUnit, MetricStatus
from genomic_variant_classifier.evaluation.evaluator import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    ClinicalEvaluator,
    EvaluationReport,
    derive_seed,
    format_ci,
)
from genomic_variant_classifier.evaluation.metrics import (
    DEFAULT_MIN_VALID_FRACTION,
    DEFAULT_MIN_VALID_REPLICATES,
    InsufficientSupportError,
    _effective_min_valid,
    auroc as kernel_auroc,
    bootstrap_metric,
)
from genomic_variant_classifier.evaluation.serialization import (
    NonFiniteArtifactValue,
    UnserializableArtifactValue,
    dump_strict_json,
    to_json_compatible,
    validate_json_finite,
)

N_BOOT = 400          # floor = max(100, 200) = 200; comfortably reachable
SEED = 42


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def clustered(n_genes=30, per_gene=40, n_inverted=6, seed=7):
    """Genes differ in ranking quality: between-cluster heterogeneity is what
    makes row-level resampling anti-conservative."""
    rng = np.random.default_rng(seed)
    y, s, g = [], [], []
    for gi in range(n_genes):
        lab = rng.integers(0, 2, per_gene)
        direction = -1.0 if gi < n_inverted else 1.0
        y.append(lab)
        s.append(0.5 + direction * (lab - 0.5) * 0.6 + rng.normal(0, 0.35, per_gene))
        g.append(np.full(per_gene, f"GENE{gi:03d}", dtype=object))
    return (np.concatenate(y), np.clip(np.concatenate(s), 0, 1), np.concatenate(g))


def independent(n=1200, n_genes=30, seed=11):
    """Gene labels carry no information: the two designs must agree."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.35, n), 0, 1)
    g = np.array([f"GENE{i:03d}" for i in rng.integers(0, n_genes, n)], dtype=object)
    return y, s, g


def quiet_evaluate(ev, *args, **kwargs) -> EvaluationReport:
    """evaluate() prints its report; suppress it so test output stays readable."""
    with redirect_stdout(io.StringIO()):
        return ev.evaluate(*args, **kwargs)


@pytest.fixture(scope="module")
def clustered_fixture():
    return clustered()


@pytest.fixture(scope="module")
def certified_report(clustered_fixture):
    y, s, g = clustered_fixture
    meta = pd.DataFrame({"gene_symbol": g, "consequence": ["missense_variant"] * len(y)})
    return quiet_evaluate(ClinicalEvaluator(n_bootstrap=N_BOOT, random_state=SEED),
                          y, s, meta=meta, model_name="certified")


@pytest.fixture(scope="module")
def withheld_report(clustered_fixture):
    y, s, _ = clustered_fixture
    return quiet_evaluate(ClinicalEvaluator(n_bootstrap=N_BOOT, random_state=SEED),
                          y, s, meta=None, model_name="withheld")


# --------------------------------------------------------------------------- #
# Group 1 -- ONE canonical engine: every path agrees when it asks for the same design
# --------------------------------------------------------------------------- #
def test_report_delegate_matches_the_kernel_exactly():
    from sklearn.metrics import roc_auc_score
    from genomic_variant_classifier.reports.report_generator import bootstrap_metric as shim
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 300), rng.uniform(0, 1, 300)
    lo, hi = shim(y, p, roc_auc_score, n_bootstrap=200, seed=SEED)
    k = bootstrap_metric(lambda a, b: float(roc_auc_score(a, b)), y, p,
                         unit=BootstrapUnit.VARIANT, n_boot=200, seed=SEED)
    assert (lo, hi) == (k.lower, k.upper), "the report layer must not resample on its own"


def test_evaluator_interval_matches_a_direct_kernel_call(clustered_fixture, certified_report):
    """The evaluator must be a caller of the engine, not a second implementation."""
    from sklearn.metrics import roc_auc_score
    y, s, g = clustered_fixture

    def nan_safe(a, b):
        try:
            return float(roc_auc_score(a, b))
        except ValueError:
            return float("nan")

    k = bootstrap_metric(nan_safe, y, s, clusters=g, unit=BootstrapUnit.GENE,
                         n_boot=N_BOOT, seed=derive_seed(SEED, "auroc"))
    assert certified_report.auroc_ci_lo == round(k.lower, 5)
    assert certified_report.auroc_ci_hi == round(k.upper, 5)
    assert certified_report.auroc_ci_n_valid == k.n_valid


def test_only_one_bootstrap_implementation_remains():
    """The two retired implementations must be gone, not merely unused."""
    import inspect
    from genomic_variant_classifier.evaluation import evaluator
    from genomic_variant_classifier.reports import report_generator

    assert not hasattr(ClinicalEvaluator, "_bootstrap_ci"), (
        "ClinicalEvaluator._bootstrap_ci still exists; a dormant second engine "
        "is one import away from being used again")
    assert not hasattr(ClinicalEvaluator, "rng"), "the shared mutable generator must be gone"
    # Inspect CODE, not text: the shim's docstring legitimately NAMES the
    # percentile call it replaced, and a substring check over raw source would
    # trip on the explanation rather than on an implementation.
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(report_generator.bootstrap_metric)))
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    code = "\n".join(ast.dump(node) for node in body)
    assert "percentile" not in code, "the report layer must not compute percentiles"
    assert "_kernel_bootstrap_metric" in code, "the report layer must delegate"
    # Again by abstract syntax tree: derive_seed's docstring deliberately NAMES
    # the attribute it replaced, so a text search would find the explanation.
    ev_tree = ast.parse(inspect.getsource(evaluator))
    rng_uses = [
        node for node in ast.walk(ev_tree)
        if isinstance(node, ast.Attribute) and node.attr == "rng"
        and isinstance(node.value, ast.Name) and node.value.id == "self"
    ]
    assert not rng_uses, (
        f"self.rng is still referenced in {len(rng_uses)} place(s); the shared "
        "mutable generator made every interval depend on call order")


# --------------------------------------------------------------------------- #
# Group 2 -- cluster power: the whole reason this commit exists
# --------------------------------------------------------------------------- #
def test_gene_clusters_widen_the_interval(clustered_fixture):
    y, s, g = clustered_fixture
    r = bootstrap_metric(kernel_auroc, y, s, clusters=g, unit=BootstrapUnit.GENE,
                         n_boot=N_BOOT, seed=SEED)
    assert r.ci_width_ratio_vs_row > 1.5, (
        f"expected a substantial design effect on a heterogeneous-gene fixture, "
        f"got {r.ci_width_ratio_vs_row}")
    assert r.cluster_ci_width > r.row_ci_width
    assert math.isclose(r.variance_ratio_vs_row, r.ci_width_ratio_vs_row ** 2, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Group 3 -- independent-row null: the test above is sensitive to REAL clustering
# --------------------------------------------------------------------------- #
def test_designs_agree_when_clustering_carries_no_signal():
    y, s, g = independent()
    r = bootstrap_metric(kernel_auroc, y, s, clusters=g, unit=BootstrapUnit.GENE,
                         n_boot=N_BOOT, seed=SEED)
    assert 0.7 <= r.ci_width_ratio_vs_row <= 1.4, (
        f"a design effect on randomly assigned clusters would mean the widening "
        f"is mechanical, not real; got {r.ci_width_ratio_vs_row}")


# --------------------------------------------------------------------------- #
# Group 4 -- missing cluster identifier: refusal at BOTH layers
# --------------------------------------------------------------------------- #
def test_kernel_refuses_gene_design_without_clusters(clustered_fixture):
    y, s, _ = clustered_fixture
    with pytest.raises(InsufficientSupportError) as exc:
        bootstrap_metric(kernel_auroc, y, s, unit=BootstrapUnit.GENE, n_boot=50)
    assert "clusters=" in str(exc.value) and "VARIANT" in str(exc.value)


def test_evaluator_withholds_the_interval_but_still_computes_point_metrics(withheld_report):
    r = withheld_report
    assert r.auroc_ci_lo is None and r.auroc_ci_hi is None
    assert r.auroc_ci_status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.auroc_ci_finding == "gene_cluster_identifier_required"
    assert r.auroc_ci_certification_eligible is False
    assert r.auroc_ci_n_requested == 0, "nothing was requested, so the count is zero"
    # falsifiable half: the point metrics are unaffected
    assert 0.0 < r.auroc < 1.0
    assert 0.0 < r.auprc < 1.0
    assert r.at_sensitivity_90 is not None
    assert math.isfinite(r.calibration_ece)


def test_evaluator_produces_a_certified_interval_when_genes_are_present(certified_report):
    """The falsifiable companion to the refusal above."""
    r = certified_report
    assert r.auroc_ci_status is MetricStatus.OK
    assert r.auroc_ci_certification_eligible is True
    assert r.auroc_ci_resampling_unit is BootstrapUnit.GENE
    assert r.auroc_ci_cluster_source == "gene_symbol"
    assert r.auroc_ci_n_requested == N_BOOT


def test_disagreeing_gene_columns_fail_rather_than_choosing_one(clustered_fixture):
    y, s, g = clustered_fixture
    bad_symbol = np.array([f"SYM{i % 7}" for i in range(len(g))], dtype=object)
    meta = pd.DataFrame({"gene_id": g, "gene_symbol": bad_symbol})
    r = quiet_evaluate(ClinicalEvaluator(n_bootstrap=N_BOOT, random_state=SEED),
                       y, s, meta=meta, model_name="conflict")
    assert r.auroc_ci_status is MetricStatus.FAILED
    assert r.auroc_ci_finding == "gene_cluster_partitions_disagree"
    assert r.auroc_ci_lo is None


def test_equivalent_gene_columns_in_different_namespaces_are_accepted(clustered_fixture):
    """Ensembl identifiers and HUGO symbols share no characters yet group identically."""
    y, s, g = clustered_fixture
    ensembl = np.array([f"ENSG{int(x[4:]):05d}" for x in g], dtype=object)
    meta = pd.DataFrame({"gene_id": ensembl, "gene_symbol": g})
    r = quiet_evaluate(ClinicalEvaluator(n_bootstrap=N_BOOT, random_state=SEED),
                       y, s, meta=meta, model_name="dual")
    assert r.auroc_ci_status is MetricStatus.OK
    assert r.auroc_ci_cluster_source == "gene_id+gene_symbol"
    assert r.auroc_ci_partition_verified is True
    assert not any(a == b for a, b in zip(ensembl, g))


# --------------------------------------------------------------------------- #
# Group 5 -- explicit exploratory mode
# --------------------------------------------------------------------------- #
def test_variant_unit_produces_an_interval_that_is_never_certifiable(clustered_fixture):
    y, s, _ = clustered_fixture
    r = bootstrap_metric(kernel_auroc, y, s, unit=BootstrapUnit.VARIANT,
                         n_boot=N_BOOT, seed=SEED)
    assert r.status is MetricStatus.OK, "the interval WAS produced"
    assert r.certification_eligible is False, "but it is not admissible"
    assert r.resampling_unit is BootstrapUnit.VARIANT
    assert r.stratified is True
    assert r.finding == "variant_level_resampling_assumes_row_independence"
    assert math.isfinite(r.lower) and math.isfinite(r.upper)


def test_status_and_certification_are_independent_axes(clustered_fixture):
    """All four combinations that can legally occur, in one place."""
    y, s, g = clustered_fixture
    gene = bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=N_BOOT, seed=SEED)
    variant = bootstrap_metric(kernel_auroc, y, s, unit=BootstrapUnit.VARIANT,
                               n_boot=N_BOOT, seed=SEED)
    thin = bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=20, seed=SEED)
    assert (gene.status, gene.certification_eligible) == (MetricStatus.OK, True)
    assert (variant.status, variant.certification_eligible) == (MetricStatus.OK, False)
    assert thin.status is MetricStatus.INSUFFICIENT_DATA
    assert thin.certification_eligible is False


# --------------------------------------------------------------------------- #
# Group 6 -- alignment sabotage
# --------------------------------------------------------------------------- #
def test_misaligned_cluster_vector_fails_before_any_resampling(clustered_fixture):
    y, s, g = clustered_fixture
    with pytest.raises(ValueError, match="clusters length"):
        bootstrap_metric(kernel_auroc, y, s, clusters=g[:-1], n_boot=10)


def test_clusters_stay_positionally_aligned_with_labels_and_scores():
    """Sabotage: permute ONLY the cluster vector. If the pipeline silently
    realigned, or cleaned the arrays independently, the interval would not move."""
    y, s, g = clustered()
    rng = np.random.default_rng(3)
    scrambled = g.copy()
    rng.shuffle(scrambled)
    intact = bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=N_BOOT, seed=SEED)
    broken = bootstrap_metric(kernel_auroc, y, s, clusters=scrambled, n_boot=N_BOOT, seed=SEED)
    assert (intact.lower, intact.upper) != (broken.lower, broken.upper)
    assert intact.ci_width_ratio_vs_row > broken.ci_width_ratio_vs_row, (
        "scrambling gene labels destroys the clustering, so the design effect "
        "must fall towards 1; if it does not, the labels were never used")


def test_missing_labels_do_not_desynchronise_the_cluster_vector():
    """A NaN label is dropped inside the metric, per replicate, never by
    pre-filtering the parallel arrays -- which would shift clusters off their rows."""
    y, s, g = clustered()
    y_f = y.astype(float)
    y_f[::37] = np.nan
    r = bootstrap_metric(kernel_auroc, y_f, s, clusters=g, n_boot=N_BOOT, seed=SEED)
    assert r.status is MetricStatus.OK
    assert r.n_observations == len(y_f)
    assert r.n_clusters == 30


# --------------------------------------------------------------------------- #
# Group 7 -- replicate accounting
# --------------------------------------------------------------------------- #
def test_replicate_accounting_balances(certified_report):
    r = certified_report
    assert r.auroc_ci_n_valid + r.auroc_ci_n_degenerate == r.auroc_ci_n_requested
    assert r.auprc_ci_n_valid + r.auprc_ci_n_degenerate == r.auprc_ci_n_requested


def test_the_two_metrics_keep_independent_replicate_counts(certified_report):
    """They can fail differently on the same cohort, so the counts are per metric."""
    r = certified_report
    for field in ("n_requested", "n_valid", "n_degenerate", "status"):
        assert hasattr(r, f"auroc_ci_{field}") and hasattr(r, f"auprc_ci_{field}")


def test_relative_floor_binds_where_the_absolute_floor_would_not():
    assert _effective_min_valid(20, 100, 0.5) == 100
    assert _effective_min_valid(1000, 100, 0.5) == 500
    assert _effective_min_valid(100000, 100, 0.5) == 50000
    assert (DEFAULT_MIN_VALID_REPLICATES, DEFAULT_MIN_VALID_FRACTION) == (100, 0.5)


def test_too_few_valid_replicates_withholds_the_interval(clustered_fixture):
    y, s, g = clustered_fixture
    r = bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=20, seed=SEED)
    assert r.status is MetricStatus.INSUFFICIENT_DATA
    assert "of 20 replicates" in r.finding
    assert r.min_valid_effective == 100


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_replicate_fraction_is_rejected(clustered_fixture, bad):
    y, s, g = clustered_fixture
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=10, min_valid_fraction=bad)


def test_single_class_cohort_is_undefined_not_unsupported():
    y = np.ones(200, dtype=int)
    s = np.linspace(0, 1, 200)
    g = np.array([f"G{i % 10}" for i in range(200)], dtype=object)
    r = bootstrap_metric(kernel_auroc, y, s, clusters=g, n_boot=200, seed=1)
    assert r.status is MetricStatus.UNDEFINED
    assert r.finding == "cluster_bootstrap_degenerate"


# --------------------------------------------------------------------------- #
# Group 8 -- reproducibility, and the shared-generator defect it replaces
# --------------------------------------------------------------------------- #
def test_repeated_evaluate_on_one_evaluator_is_identical(clustered_fixture):
    """The regression test for the shared mutable generator retired 2026-07-26:
    the second call previously inherited an advanced stream."""
    y, s, g = clustered_fixture
    meta = pd.DataFrame({"gene_symbol": g})
    ev = ClinicalEvaluator(n_bootstrap=N_BOOT, random_state=SEED)
    a = quiet_evaluate(ev, y, s, meta=meta, model_name="a")
    b = quiet_evaluate(ev, y, s, meta=meta, model_name="b")
    assert (a.auroc_ci_lo, a.auroc_ci_hi) == (b.auroc_ci_lo, b.auroc_ci_hi)
    assert (a.auprc_ci_lo, a.auprc_ci_hi) == (b.auprc_ci_lo, b.auprc_ci_hi)


def test_metric_order_does_not_affect_either_interval():
    assert derive_seed(SEED, "auroc") != derive_seed(SEED, "auprc")
    assert derive_seed(SEED, "auroc") == derive_seed(SEED, "auroc")
    assert derive_seed(1, "auroc") != derive_seed(2, "auroc")


def test_derive_seed_is_stable_across_processes():
    """hashlib, not the builtin hash(): PYTHONHASHSEED randomises string hashing
    per process, which would make a 'reproducible' seed vary between runs."""
    code = ("import sys; sys.path.insert(0, 'src');"
            "from genomic_variant_classifier.evaluation.evaluator import derive_seed;"
            "print(derive_seed(42, 'auroc'))")
    seen = set()
    for salt in ("0", "1", "random"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin",
                                             "PYTHONPATH": "src"})
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"seed varied with PYTHONHASHSEED: {seen}"


def test_derive_seed_refuses_an_unnamed_stream():
    for bad in ("", None, 7):
        with pytest.raises(ValueError):
            derive_seed(42, bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Group 9 -- calibration is untouched by this commit
# --------------------------------------------------------------------------- #
def test_calibration_paths_are_unchanged(clustered_fixture, certified_report):
    """The audit flagged calibration structurally; the 2026-07-20 census already
    reconciled it, so this commit makes NO calibration change. Its dedicated
    agreement suite runs unmodified; this asserts the evaluator still emits both
    calibration numbers finitely."""
    assert math.isfinite(certified_report.calibration_ece)
    assert math.isfinite(certified_report.calibration_mce)
    assert 0.0 <= certified_report.calibration_ece <= 1.0


# --------------------------------------------------------------------------- #
# Group 10 -- construction invariants: impossible artifacts cannot be built
# --------------------------------------------------------------------------- #
def _valid_kwargs(**overrides):
    base = dict(
        schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
        model_name="m", n_samples=10, n_pathogenic=5, n_benign=5, prevalence=0.5,
        auroc=0.9, auroc_ci_lo=0.8, auroc_ci_hi=0.95,
        auprc=0.9, auprc_ci_lo=0.8, auprc_ci_hi=0.95,
        mcc=0.5, f1=0.5, brier_score=0.1,
        calibration_ece=0.1, calibration_mce=0.2,
    )
    for metric in ("auroc", "auprc"):
        base.update({
            f"{metric}_ci_status": MetricStatus.OK,
            f"{metric}_ci_resampling_unit": BootstrapUnit.GENE,
            f"{metric}_ci_stratified": False,
            f"{metric}_ci_cluster_source": "gene_symbol",
            f"{metric}_ci_partition_verified": False,
            f"{metric}_ci_certification_eligible": True,
            f"{metric}_ci_n_requested": 400,
            f"{metric}_ci_n_valid": 400,
            f"{metric}_ci_n_degenerate": 0,
            f"{metric}_ci_finding": None,
        })
    base.update(overrides)
    return base


def test_the_valid_baseline_constructs():
    """Falsifiable anchor: if this failed, every refusal below would be vacuous."""
    assert isinstance(EvaluationReport(**_valid_kwargs()), EvaluationReport)


@pytest.mark.parametrize("overrides,match", [
    ({"auroc_ci_lo": None}, "requires both endpoints"),
    ({"auroc_ci_hi": None}, "requires both endpoints"),
    ({"auroc_ci_lo": float("nan")}, "must be finite"),
    ({"auroc_ci_lo": 0.99, "auroc_ci_hi": 0.10}, "exceeds upper"),
    ({"auroc_ci_resampling_unit": None}, "requires a resampling unit"),
    ({"auroc_ci_n_requested": 0}, "n_requested > 0"),
    ({"auroc_ci_n_valid": 0}, "n_valid > 0"),
    ({"auroc_ci_n_degenerate": 5}, "does not balance"),
    ({"auroc_ci_status": MetricStatus.INSUFFICIENT_SUPPORT},
     "must not carry endpoints"),
    ({"auroc_ci_status": MetricStatus.FAILED, "auroc_ci_lo": None,
      "auroc_ci_hi": None, "auroc_ci_certification_eligible": False,
      "auroc_ci_finding": None},
     "requires a machine-readable finding"),
    ({"auroc_ci_status": MetricStatus.FAILED, "auroc_ci_lo": None,
      "auroc_ci_hi": None, "auroc_ci_finding": "x"},
     "cannot be certification eligible"),
    ({"auroc_ci_resampling_unit": BootstrapUnit.VARIANT},
     "only gene-cluster intervals"),
    ({"auroc_ci_cluster_source": None}, "must name the column"),
    ({"auroc_ci_partition_verified": True, "auroc_ci_cluster_source": "gene_symbol",
      "auroc_ci_certification_eligible": False,
      "auroc_ci_status": MetricStatus.INSUFFICIENT_DATA,
      "auroc_ci_lo": None, "auroc_ci_hi": None, "auroc_ci_finding": "x"},
     "partition_verified is only meaningful"),
])
def test_impossible_reports_are_refused_at_construction(overrides, match):
    with pytest.raises((ValueError, TypeError), match=match):
        EvaluationReport(**_valid_kwargs(**overrides))


def test_a_bare_string_status_is_refused():
    with pytest.raises(TypeError, match="must be a MetricStatus"):
        EvaluationReport(**_valid_kwargs(auroc_ci_status="ok"))


def test_an_unavailable_interval_constructs_when_it_is_internally_consistent():
    r = EvaluationReport(**_valid_kwargs(
        auroc_ci_status=MetricStatus.INSUFFICIENT_SUPPORT,
        auroc_ci_lo=None, auroc_ci_hi=None,
        auroc_ci_resampling_unit=None, auroc_ci_stratified=None,
        auroc_ci_cluster_source=None, auroc_ci_certification_eligible=False,
        auroc_ci_n_requested=0, auroc_ci_n_valid=0, auroc_ci_n_degenerate=0,
        auroc_ci_finding="gene_cluster_identifier_required"))
    assert r.auroc_ci_lo is None


def test_an_exploratory_available_interval_may_carry_a_finding():
    """An available interval is NOT required to have finding=None: the variant
    design legitimately records the assumption it rests on."""
    r = EvaluationReport(**_valid_kwargs(
        auroc_ci_resampling_unit=BootstrapUnit.VARIANT,
        auroc_ci_stratified=True,
        auroc_ci_certification_eligible=False,
        auroc_ci_finding="variant_level_resampling_assumes_row_independence"))
    assert r.auroc_ci_finding


# --------------------------------------------------------------------------- #
# Group 11 -- strict serialization
# --------------------------------------------------------------------------- #
def test_numpy_integers_are_not_silently_stringified():
    assert to_json_compatible({"n": np.int64(7)}) == {"n": 7}
    assert json.dumps({"n": np.int64(7)}, default=str) == '{"n": "7"}'   # the old defect


def test_unknown_types_are_refused_rather_than_stringified():
    with pytest.raises(UnserializableArtifactValue, match="no defined JSON representation"):
        to_json_compatible({"m": object()})


def test_non_finite_values_are_reported_with_their_path():
    with pytest.raises(NonFiniteArtifactValue) as exc:
        validate_json_finite({"a": {"b": float("nan")}, "c": [1.0, float("inf")]},
                             artifact="eval_report.json")
    assert "$.a.b" in str(exc.value) and "$.c[1]" in str(exc.value)


def test_a_finite_payload_serializes(certified_report):
    text = dump_strict_json(asdict(certified_report), artifact="t")
    assert json.loads(text)["schema_version"] == EVALUATION_REPORT_SCHEMA_VERSION


def test_saved_reports_contain_null_never_nan(tmp_path, certified_report, withheld_report):
    ev = ClinicalEvaluator()
    for tag, rep in (("certified", certified_report), ("withheld", withheld_report)):
        path = tmp_path / f"{tag}.json"
        ev.save_report(rep, path)
        raw = path.read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw
        # a strict parser must accept it
        json.loads(raw, parse_constant=lambda c: pytest.fail(f"{tag}: bare {c} literal"))
    assert json.loads((tmp_path / "withheld.json").read_text())["auroc_ci_lo"] is None


def test_strict_serialization_rejects_a_non_finite_report(tmp_path, certified_report):
    """Falsifiable: the guard above only means something if this one fires."""
    payload = asdict(certified_report)
    payload["fpr_curve"] = [0.0, float("nan"), 1.0]
    with pytest.raises(NonFiniteArtifactValue, match=r"fpr_curve\[1\]"):
        dump_strict_json(payload, artifact="eval_report.json")


def test_enums_persist_as_their_stable_values(certified_report, tmp_path):
    path = tmp_path / "r.json"
    ClinicalEvaluator().save_report(certified_report, path)
    d = json.loads(path.read_text())
    assert d["auroc_ci_status"] == "ok"
    assert d["auroc_ci_resampling_unit"] == "gene"


# --------------------------------------------------------------------------- #
# Group 12 -- rendering
# --------------------------------------------------------------------------- #
def test_unavailable_intervals_render_as_words_not_as_nan():
    assert format_ci(None, None, status=MetricStatus.INSUFFICIENT_SUPPORT,
                     finding="gene_cluster_identifier_required") == \
        "unavailable (gene_cluster_identifier_required)"
    assert format_ci(None, None, status=MetricStatus.FAILED) == "unavailable"
    assert format_ci(0.1, 0.9, status=MetricStatus.OK) == "[0.1000, 0.9000]"


def test_format_ci_refuses_an_available_interval_with_no_endpoints():
    with pytest.raises(ValueError, match="status is OK but an endpoint is None"):
        format_ci(None, 0.9, status=MetricStatus.OK)


def test_printed_report_never_shows_nan_for_a_withheld_interval(withheld_report):
    buf = io.StringIO()
    with redirect_stdout(buf):
        ClinicalEvaluator().print_report(withheld_report)
    text = buf.getvalue()
    assert "nan" not in text.lower()
    assert "unavailable (gene_cluster_identifier_required)" in text


# --------------------------------------------------------------------------- #
# Group 13 -- schema versioning and legacy compatibility
# --------------------------------------------------------------------------- #
def test_schema_version_is_persisted(certified_report):
    assert certified_report.schema_version == EVALUATION_REPORT_SCHEMA_VERSION == 2


def test_version_two_round_trips_without_information_loss(tmp_path, certified_report):
    path = tmp_path / "r.json"
    ClinicalEvaluator().save_report(certified_report, path)
    d = json.loads(path.read_text())
    for metric in ("auroc", "auprc"):
        for suffix in ("lo", "hi", "status", "resampling_unit", "stratified",
                       "cluster_source", "partition_verified",
                       "certification_eligible", "n_requested", "n_valid",
                       "n_degenerate", "finding"):
            assert f"{metric}_ci_{suffix}" in d, f"{metric}_ci_{suffix} lost in serialization"


def test_legacy_version_one_artifacts_are_readable_and_never_certified(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "rra", Path(__file__).resolve().parents[2] / "scripts" / "read_run_artifacts.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({"model_name": "old", "prevalence": 0.5, "auroc": 0.91,
                              "auroc_ci_lo": 0.90, "auroc_ci_hi": 0.92,
                              "auprc": 0.88, "auprc_ci_lo": 0.86, "auprc_ci_hi": 0.90}))
    row = rra._read_eval_report(v1)
    assert row["schema_version"] == 1
    assert row["auroc_ci_lo"] == 0.90            # still readable
    assert row["auroc_ci_unit"] == "legacy_unknown"
    assert row["auroc_ci_certified"] is False    # never retroactively certified

    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps({"schema_version": 2, "model_name": "new", "prevalence": 0.5,
                              "auroc": 0.9, "auroc_ci_lo": 0.85, "auroc_ci_hi": 0.96,
                              "auroc_ci_status": "ok", "auroc_ci_resampling_unit": "gene",
                              "auroc_ci_cluster_source": "gene_symbol",
                              "auroc_ci_certification_eligible": True,
                              "auroc_ci_n_valid": 400, "auroc_ci_finding": None,
                              "auprc": 0.88, "auprc_ci_lo": None, "auprc_ci_hi": None,
                              "auprc_ci_status": "insufficient_support",
                              "auprc_ci_resampling_unit": None,
                              "auprc_ci_cluster_source": None,
                              "auprc_ci_certification_eligible": False,
                              "auprc_ci_n_valid": 0,
                              "auprc_ci_finding": "gene_cluster_identifier_required"}))
    row2 = rra._read_eval_report(v2)
    assert row2["auroc_ci_certified"] is True
    assert row2["auprc_ci_lo"] is None
    assert row2["auprc_ci_status"] == "insufficient_support"


# --------------------------------------------------------------------------- #
# Group 14 -- contracts this commit must not break
# --------------------------------------------------------------------------- #
def test_evaluator_still_imports_without_scikit_learn():
    """Duplicated deliberately from test_evaluator_phase5: this commit adds three
    module-level imports to evaluator.py, and this is the contract they could break."""
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, n, p=None):\n"
        "        if n == 'sklearn' or n.startswith('sklearn.'):\n"
        "            raise ModuleNotFoundError('blocked')\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import genomic_variant_classifier.evaluation.evaluator as m\n"
        "assert hasattr(m, '_ensure_sklearn')\n"
        "assert hasattr(m, 'format_ci')\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_validation_metrics_is_dead_and_therefore_deferred():
    """ValidationMetrics carries (0.0, 0.0) interval defaults -- a zero-width
    interval at zero is fabricated evidence. It is NOT corrected here because it
    is constructed nowhere, so correcting it would broaden this commit for no
    behavioural gain. This test is the evidence for that deferral: if anything
    ever constructs it, this fails and the deferral is revisited."""
    import pathlib
    import genomic_variant_classifier as pkg
    roots = [pathlib.Path(pkg.__file__).parent,
             pathlib.Path(pkg.__file__).resolve().parents[2] / "scripts"]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "ValidationMetrics(" in path.read_text(encoding="utf-8"):
                hits.append(str(path))
    assert not hits, (
        "ValidationMetrics is now constructed somewhere; its (0.0, 0.0) interval "
        f"defaults must be corrected to a typed unavailable state: {hits}")
