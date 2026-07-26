"""Canonical, provenance-carrying evaluation input for the metric stack.

This module defines 'CanonicalVariantTable' -- the typed seam that both live
metric entry points project from:

  * the low-level kernel 'metrics.evaluate' / 'stratified_evaluate' /
    'cluster_bootstrap_ci' (parallel '(y, score, groups, clusters)' sequences), and
  * the higher-level 'ClinicalEvaluator.evaluate(y_true, y_proba, meta=...)' whose
    'meta' docstring already asks for a "Canonical variant DataFrame aligned with
    y_true/y_proba".

The evaluator has always gestured at this contract informally (a bare, unschema'd
'pd.DataFrame'). This module formalises it: one aligned row per variant, a validated
schema, mandatory partition membership, and a mandatory cohort version, from which the
array inputs the kernel wants are trivial projections. Arrays are a projection of the
table, never the other way round, so the two entry points cannot desync.

DESIGN CONSTRAINTS (each traceable to a recorded defect or contract):

  * ALIGNMENT is structural. One row = one variant; every projection is a column of the
    same frame, so 'y' and 'score' cannot be masked apart (evaluation defect A).
  * MISSING labels are first-class and are represented as 'NaN' in the projected 'y',
    NOT coerced to 0. The kernel's 'clean_arrays' drops non-finite rows on ONE joint
    mask; the seam reuses that mask rather than inventing a second one (defect B, and
    acceptance "one structural mask").
  * PARTITION is mandatory. Calibration/selection must never be measured on data used to
    fit the model/method/threshold; carrying the partition lets a consumer refuse the
    wrong split (evaluation leakage findings).
  * COHORT_VERSION is mandatory. Option C forbids certifying any production metric against
    the superseded v1 cohort; the field makes a v1-derived result machine-refusable at the
    certification boundary.

IMPORT CONTRACT. This module imports only numpy and pandas at module level -- NEVER
scikit-learn, and NEVER 'metrics.py' or 'evaluator.py'. That keeps it safe to import
from the package while 'evaluation/__init__.py''s no-eager-sklearn contract holds
(locked by test_evaluator_phase5 / test_evaluation_metrics). It also imports nothing from
cohort construction, the P6 probe, or clean_cohort: the cohort builder produces instances
of this contract; the metric stack never reaches back into cohort construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["CanonicalVariantTable", "CanonicalArrays"]


# The columns the table validates and stores, in canonical order.
_REQUIRED_COLUMNS = ("variant_id", "y_true", "partition")
_OPTIONAL_COLUMNS = ("y_score", "prob", "gene_id", "group_id", "adjudication_reason")
_ALL_COLUMNS = _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS


@dataclass(frozen=True)
class CanonicalArrays:
    """A partition-scoped array projection for the metric kernel.

    'y' carries 'NaN' for withheld labels; the kernel's 'clean_arrays' drops those
    on its single joint mask together with any non-finite score/prob. 'clusters' and
    'groups' are 'None' when the underlying columns were not supplied.
    """

    y: np.ndarray
    score: np.ndarray
    prob: np.ndarray
    clusters: np.ndarray | None
    groups: np.ndarray | None
    n_rows: int
    partition: str


class CanonicalVariantTable:
    """A validated, aligned, provenance-carrying evaluation table.

    Construct from a mapping of column -> sequence (or a 'pandas.DataFrame') plus the
    cohort version. Validation is fail-closed and happens once, at construction; every
    projection is a cheap column selection afterwards.
    """

    def __init__(
        self,
        data: Mapping[str, Sequence] | pd.DataFrame,
        *,
        cohort_version: str,
    ) -> None:
        if not isinstance(cohort_version, str) or not cohort_version.strip():
            raise ValueError("cohort_version must be a non-empty string")
        self._cohort_version = cohort_version

        frame = pd.DataFrame(data)

        missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"CanonicalVariantTable is missing required column(s): {missing}. "
                f"Required: {list(_REQUIRED_COLUMNS)}."
            )

        n = len(frame)
        if n == 0:
            raise ValueError("CanonicalVariantTable must have at least one row")

        # variant_id: present, non-null, unique (identity is one-per-row).
        vid = frame["variant_id"]
        if vid.isna().any():
            raise ValueError("variant_id contains null values")
        if vid.duplicated().any():
            dupes = vid[vid.duplicated()].unique()[:10].tolist()
            raise ValueError(f"variant_id must be unique; duplicates include {dupes}")

        # y_true: each value must be 0, 1, or missing (NaN/None). Anything else fails NOW,
        # not at metric time -- and is never coerced (defect B).
        y_true = frame["y_true"]
        y_norm = self._validate_labels(y_true)

        # partition: present, non-null, string.
        partition = frame["partition"]
        if partition.isna().any():
            raise ValueError("partition contains null values")
        part_str = partition.astype(str)

        # y_score / prob: if present, must be numeric and length-aligned (guaranteed by the
        # frame) -- validate dtype is coercible to float. Non-finite is allowed (dropped by
        # the kernel's mask), but non-numeric strings are a construction error.
        y_score = self._validate_optional_float(frame, "y_score")
        prob = self._validate_optional_float(frame, "prob")

        gene_id = frame["gene_id"].astype("object") if "gene_id" in frame.columns else None
        group_id = frame["group_id"].astype("object") if "group_id" in frame.columns else None

        # Build the canonical internal frame with a stable column order.
        internal = pd.DataFrame({"variant_id": vid.astype(str).to_numpy()})
        internal["y_true"] = y_norm
        internal["partition"] = part_str.to_numpy()
        internal["y_score"] = y_score if y_score is not None else np.nan
        internal["prob"] = prob if prob is not None else np.nan
        internal["gene_id"] = gene_id.to_numpy() if gene_id is not None else None
        internal["group_id"] = group_id.to_numpy() if group_id is not None else None
        if "adjudication_reason" in frame.columns:
            internal["adjudication_reason"] = frame["adjudication_reason"].astype("object").to_numpy()
        else:
            internal["adjudication_reason"] = None

        self._frame = internal
        self._has_score = y_score is not None
        self._has_prob = prob is not None
        self._has_gene = gene_id is not None
        self._has_group = group_id is not None

    # -- validation helpers --------------------------------------------------

    @staticmethod
    def _validate_labels(y_true: pd.Series) -> np.ndarray:
        """Return a float array with 0.0/1.0/NaN; reject anything else (never coerce)."""
        arr = np.asarray(y_true.to_numpy(), dtype="object")
        out = np.empty(arr.shape, dtype=float)
        bad = []
        for i, v in enumerate(arr):
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == ""):
                out[i] = np.nan
            elif isinstance(v, (bool, np.bool_)):
                out[i] = 1.0 if v else 0.0
            elif isinstance(v, (int, np.integer)) and int(v) in (0, 1):
                out[i] = float(int(v))
            elif isinstance(v, (float, np.floating)) and float(v) in (0.0, 1.0):
                out[i] = float(v)
            else:
                bad.append(v)
        if bad:
            raise ValueError(
                f"y_true must be 0, 1, or missing; found invalid value(s) {bad[:10]} "
                f"in {len(bad)} row(s). Labels are validated, never coerced."
            )
        return out

    @staticmethod
    def _validate_optional_float(frame: pd.DataFrame, col: str) -> np.ndarray | None:
        if col not in frame.columns:
            return None
        try:
            return np.asarray(frame[col].to_numpy(), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {col!r} must be numeric (float-coercible): {exc}") from exc

    # -- properties ----------------------------------------------------------

    @property
    def cohort_version(self) -> str:
        return self._cohort_version

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    @property
    def partitions(self) -> tuple[str, ...]:
        return tuple(sorted(self._frame["partition"].unique().tolist()))

    def __len__(self) -> int:
        return len(self._frame)

    # -- projections ---------------------------------------------------------

    def _select(self, partition: str | None) -> pd.DataFrame:
        if partition is None:
            return self._frame
        available = set(self._frame["partition"].unique().tolist())
        if partition not in available:
            raise ValueError(
                f"partition {partition!r} not present; available: {sorted(available)}"
            )
        return self._frame[self._frame["partition"] == partition]

    def arrays(self, partition: str | None = None) -> CanonicalArrays:
        """Project to kernel arrays for one partition (or all rows if 'None').

        'y' carries 'NaN' for withheld labels; the kernel drops those on its single
        joint mask along with any non-finite score/prob. Requires a score column.
        """
        if not self._has_score:
            raise ValueError("arrays() requires a 'y_score' column; none was provided")
        sub = self._select(partition)
        y = np.asarray(sub["y_true"].to_numpy(), dtype=float)
        score = np.asarray(sub["y_score"].to_numpy(), dtype=float)
        prob = np.asarray(sub["prob"].to_numpy(), dtype=float) if self._has_prob else score.copy()
        clusters = np.asarray(sub["gene_id"].to_numpy(), dtype=object) if self._has_gene else None
        groups = np.asarray(sub["group_id"].to_numpy(), dtype=object) if self._has_group else None
        return CanonicalArrays(
            y=y, score=score, prob=prob, clusters=clusters, groups=groups,
            n_rows=len(sub), partition=partition if partition is not None else "__all__",
        )

    def gene_clusters(self, partition: str | None = None) -> np.ndarray:
        """Project the per-row gene cluster labels for 'cluster_bootstrap_ci'."""
        if not self._has_gene:
            raise ValueError("gene_clusters() requires a 'gene_id' column; none was provided")
        return np.asarray(self._select(partition)["gene_id"].to_numpy(), dtype=object)

    def groups(self, partition: str | None = None) -> np.ndarray:
        """Project the per-row group labels for 'stratified_evaluate'."""
        if not self._has_group:
            raise ValueError("groups() requires a 'group_id' column; none was provided")
        return np.asarray(self._select(partition)["group_id"].to_numpy(), dtype=object)

    def as_meta(self, partition: str | None = None) -> pd.DataFrame:
        """The aligned metadata frame 'ClinicalEvaluator.evaluate' expects as 'meta'.

        Columns mirror the canonical schema; a 'cohort_version' column is attached so the
        provenance travels with the frame. Row order matches 'arrays(partition)'.
        """
        sub = self._select(partition).copy()
        sub = sub.reset_index(drop=True)
        sub["cohort_version"] = self._cohort_version
        return sub
