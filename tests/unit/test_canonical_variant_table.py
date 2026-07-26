"""Acceptance + sabotage battery for the CanonicalVariantTable seam.

Proves the nine acceptance points from the evaluation-wiring audit:

  1. it feeds metrics.evaluate;
  2. it feeds ClinicalEvaluator.evaluate;
  3. both receive identical aligned y and score arrays;
  4. missing labels are dropped through ONE structural mask (the kernel's), never coerced;
  5. invalid labels fail at construction;
  6. missing partition fails at construction;
  7. a nonexistent requested partition fails;
  8. cohort_version is preserved in every projection;
  9. importing the package without scikit-learn stays green (subprocess).

Each guard is proven falsifiable: the sabotage tests assert the failure fires.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.canonical import (
    CanonicalArrays,
    CanonicalVariantTable,
)


def _base_data(n: int = 8):
    rng = np.random.default_rng(0)
    return {
        "variant_id": [f"v{i}" for i in range(n)],
        "y_true": [i % 2 for i in range(n)],
        "y_score": rng.random(n),
        "gene_id": [f"g{i % 3}" for i in range(n)],
        "group_id": [f"grp{i % 2}" for i in range(n)],
        "partition": ["test"] * n,
    }


# -- construction + validation -------------------------------------------------

def test_constructs_from_mapping_and_reports_shape():
    t = CanonicalVariantTable(_base_data(), cohort_version="v2-abc")
    assert len(t) == 8
    assert t.cohort_version == "v2-abc"
    assert t.partitions == ("test",)


def test_constructs_from_dataframe():
    t = CanonicalVariantTable(pd.DataFrame(_base_data()), cohort_version="v2")
    assert t.n_rows == 8


def test_missing_required_column_fails():
    d = _base_data()
    del d["partition"]
    with pytest.raises(ValueError, match="missing required column"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_empty_table_fails():
    d = {k: [] for k in ("variant_id", "y_true", "partition")}
    with pytest.raises(ValueError, match="at least one row"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_duplicate_variant_id_fails():
    d = _base_data()
    d["variant_id"][1] = d["variant_id"][0]
    with pytest.raises(ValueError, match="unique"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_empty_cohort_version_fails():
    with pytest.raises(ValueError, match="cohort_version"):
        CanonicalVariantTable(_base_data(), cohort_version="   ")


# -- (5) invalid labels fail at construction -----------------------------------

def test_invalid_label_value_fails_at_construction():
    d = _base_data()
    d["y_true"] = [0, 1, 2, 1, 0, 1, 0, 1]  # 2 is not a binary label
    with pytest.raises(ValueError, match="0, 1, or missing"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_fractional_label_is_not_coerced():
    d = _base_data()
    d["y_true"] = [0, 1, 0.9, 1, 0, 1, 0, 1]  # 0.9 must NOT become 0
    with pytest.raises(ValueError, match="never coerced"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_missing_labels_are_allowed_and_represented_as_nan():
    d = _base_data()
    d["y_true"] = [0, 1, None, 1, 0, 1, 0, 1]
    t = CanonicalVariantTable(d, cohort_version="v2")
    y = t.arrays("test").y
    assert np.isnan(y[2])
    assert y[0] == 0.0 and y[1] == 1.0


# -- (6,7) partition semantics -------------------------------------------------

def test_null_partition_fails_at_construction():
    d = _base_data()
    d["partition"][3] = None
    with pytest.raises(ValueError, match="partition contains null"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_nonexistent_partition_projection_fails():
    t = CanonicalVariantTable(_base_data(), cohort_version="v2")
    with pytest.raises(ValueError, match="not present"):
        t.arrays("calibration")


def test_partition_selection_subsets_rows():
    d = _base_data(8)
    d["partition"] = ["train"] * 4 + ["test"] * 4
    t = CanonicalVariantTable(d, cohort_version="v2")
    assert t.arrays("train").n_rows == 4
    assert t.arrays("test").n_rows == 4
    assert set(t.partitions) == {"train", "test"}


# -- score/prob validation -----------------------------------------------------

def test_non_numeric_score_fails():
    d = _base_data()
    d["y_score"] = ["a", "b", "c", "d", "e", "f", "g", "h"]
    with pytest.raises(ValueError, match="numeric"):
        CanonicalVariantTable(d, cohort_version="v2")


def test_arrays_without_score_column_fails():
    d = _base_data()
    del d["y_score"]
    t = CanonicalVariantTable(d, cohort_version="v2")
    with pytest.raises(ValueError, match="requires a 'y_score'"):
        t.arrays("test")


# -- (1,3) feeds metrics.evaluate; identical aligned arrays --------------------

def test_feeds_metrics_evaluate():
    metrics = pytest.importorskip("genomic_variant_classifier.evaluation.metrics")
    t = CanonicalVariantTable(_base_data(40), cohort_version="v2")
    a = t.arrays("test")
    result = metrics.evaluate(a.y, a.score)
    assert "auroc" in result


def test_projection_arrays_are_aligned_and_stable():
    d = _base_data(10)
    t = CanonicalVariantTable(d, cohort_version="v2")
    a = t.arrays("test")
    # y and score come from the same rows in the same order -> same length, row-aligned
    assert a.y.shape == a.score.shape == a.prob.shape == (10,)
    # prob defaults to score when no prob column
    assert np.allclose(a.prob, a.score)


# -- (2,3) feeds ClinicalEvaluator.evaluate; same y/score reach both -----------

def test_feeds_clinical_evaluator_and_arrays_match():
    pytest.importorskip("sklearn")
    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

    d = _base_data(60)
    t = CanonicalVariantTable(d, cohort_version="v2")
    a = t.arrays("test")
    meta = t.as_meta("test")

    # the y/score the kernel sees and the y_true/meta the evaluator sees are the SAME rows
    assert len(meta) == a.n_rows
    assert np.array_equal(meta["y_true"].to_numpy(dtype=float), a.y, equal_nan=True)
    assert np.allclose(meta["y_score"].to_numpy(dtype=float), a.score)

    ev = ClinicalEvaluator()
    report = ev.evaluate(a.y.astype(int), a.score, meta=meta)
    assert report is not None


# -- (4) one structural mask: seam does not pre-drop; kernel's mask governs -----

def test_seam_does_not_pre_mask_missing_labels():
    d = _base_data(6)
    d["y_true"] = [0, 1, None, 1, 0, 1]
    t = CanonicalVariantTable(d, cohort_version="v2")
    a = t.arrays("test")
    # the seam keeps all 6 rows (NaN preserved); it is the KERNEL that drops row 2.
    assert a.n_rows == 6
    assert np.isnan(a.y).sum() == 1


# -- (8) cohort_version preserved in every projection --------------------------

def test_cohort_version_travels_with_meta():
    t = CanonicalVariantTable(_base_data(), cohort_version="v2-xyz")
    meta = t.as_meta("test")
    assert (meta["cohort_version"] == "v2-xyz").all()


def test_cohort_version_property_on_all_projections():
    t = CanonicalVariantTable(_base_data(), cohort_version="v1")
    assert t.cohort_version == "v1"
    assert (t.as_meta()["cohort_version"] == "v1").all()


# -- group/cluster projections -------------------------------------------------

def test_gene_clusters_projection():
    t = CanonicalVariantTable(_base_data(9), cohort_version="v2")
    clusters = t.gene_clusters("test")
    assert clusters.shape == (9,)
    assert set(clusters.tolist()) <= {"g0", "g1", "g2"}


def test_groups_projection():
    t = CanonicalVariantTable(_base_data(8), cohort_version="v2")
    groups = t.groups("test")
    assert groups.shape == (8,)


def test_gene_clusters_absent_fails():
    d = _base_data()
    del d["gene_id"]
    t = CanonicalVariantTable(d, cohort_version="v2")
    with pytest.raises(ValueError, match="requires a 'gene_id'"):
        t.gene_clusters("test")


def test_cluster_bootstrap_consumes_projection():
    metrics = pytest.importorskip("genomic_variant_classifier.evaluation.metrics")
    d = _base_data(80)
    t = CanonicalVariantTable(d, cohort_version="v2")
    a = t.arrays("test")
    clusters = t.gene_clusters("test")
    lo, hi = metrics.cluster_bootstrap_ci(metrics.auroc, a.y, a.score, clusters, n_boot=50, seed=0)
    assert lo <= hi


# -- (9) no-sklearn package import contract stays green -------------------------

def test_canonical_imports_without_sklearn():
    """canonical.py must import with scikit-learn absent (subprocess, no pollution)."""
    code = (
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def _block(n, *a, **k):\n"
        "    if n == 'sklearn' or n.startswith('sklearn.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'sklearn' (blocked)\")\n"
        "    return _real(n, *a, **k)\n"
        "builtins.__import__ = _block\n"
        "import genomic_variant_classifier.evaluation.canonical as c\n"
        "assert hasattr(c, 'CanonicalVariantTable')\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "canonical.py must import without sklearn.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "ok" in proc.stdout
