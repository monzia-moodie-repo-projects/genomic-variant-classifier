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
    NOT coerced to 0. Label eligibility is a POPULATION decision and is made by
    'EvaluationPopulation': the caller restricts an attempted population to the
    label-eligible rows, and that restriction records its own reason and parent, so
    what was removed and why is carried with every number computed afterwards
    (defect B, and acceptance "one structural mask").
  * PREDICTIONS are validated, never selected. Predicted scores and probabilities are
    not silently filtered by numerical kernels. A non-finite model output is a
    validation failure: the registry refuses before dispatch and returns a FAILED
    MetricResult over the full attempted evaluation population, and the kernels raise
    rather than repair. 'metrics.evaluate' remains a legacy survivor-filtering
    compatibility interface and is not a certifiable path.
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
from typing import Iterable, Mapping, Optional, Sequence

import hashlib

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

    'y' carries 'NaN' for withheld labels; restricting to the label-eligible rows is
    the caller's explicit population step, performed through 'EvaluationPopulation'
    built from 'population_projection'. Non-finite scores and probabilities are NOT
    selected out: they are validation failures. 'clusters' and 'groups' are 'None'
    when the underlying columns were not supplied.
    """

    y: np.ndarray
    score: np.ndarray
    prob: np.ndarray
    clusters: np.ndarray | None
    groups: np.ndarray | None
    n_rows: int
    partition: str


def _derive_population_source_id(*, cohort_version: str, partition: Optional[str],
                                 variant_ids: Sequence) -> str:
    """Deterministic identity of ONE aligned evaluation frame.

    Derived from the cohort version, the selected partition, and the ORDERED
    `variant_id` sequence of that partition. Nothing else.

    WHY NOT `partition + cohort_version` ALONE. Those identify a CATEGORY of
    population, not an exact frame. Two tables can legitimately share both while
    differing in row membership, row order, filtered variant set, or corrected
    input data carrying the same human-readable version. Different frames would
    then produce identical membership fingerprints whenever their absolute
    indices happened to coincide -- and absolute indices coincide constantly,
    because `EvaluationPopulation.full` always yields `arange(n)`.

    WHAT IS DELIBERATELY EXCLUDED. Scope, label-eligibility masks, subgroup
    masks, support counts, prediction values, model names and `y_true` VALUES.
    Scope and restrictions belong to the population lineage and the membership
    fingerprint, not to the identity of the frame. Prediction identity belongs to
    model provenance: the same test population evaluated by two models must yield
    the SAME population fingerprint, or paired comparison becomes harder rather
    than safer. A label-policy change must be visible through `cohort_version`
    rather than silently embedded in an opaque row digest.

    LENGTH-PREFIXING. Every variable-length field is preceded by its byte length,
    so `["ab", "c"]` and `["a", "bc"]` cannot serialise identically. Without it
    the digest would be ambiguous under concatenation and two different frames
    could collide.

    THE `None` PARTITION. `arrays(None)` projects every row and labels itself
    `"__all__"`. That string is encoded under a DISTINCT namespace from a named
    partition, so a table containing a partition literally called `__all__` can
    never collide with the all-rows projection.
    """
    digest = hashlib.sha256()
    for namespace, value in (("cohort_version", cohort_version),
                             ("partition" if partition is not None
                              else "partition_all_rows",
                              partition if partition is not None else "")):
        encoded = value.encode("utf-8")
        digest.update(namespace.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(b"ordered_variant_ids\0")
    for variant_id in variant_ids:
        encoded = str(variant_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"canonical-variant-table:sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class CanonicalPopulationProjection:
    """One aligned evaluation frame, with a deterministic identity.

    This is the SOURCE that an `EvaluationPopulation` addresses. Indices are
    positions into THIS projection -- into the arrays `arrays(partition)`
    returns -- never into the whole multi-partition table.

    That choice is deliberate. A root population must contain `arange(n_source)`,
    so if indices addressed the whole table then a partition could not be a root
    and would have to be a derived population whose parent was the entire table.
    That would add lineage irrelevant to the metric estimand and force every
    `take()` to consume full-table arrays. Addressing the projection instead
    means `n_source` IS the attempted metric population before any label
    restriction, `take()` consumes exactly what `arrays()` produces, and no
    metric context can address a row in another partition.

    `source_indices` records where these rows sit in the full table, so the
    mapping back is available for provenance without being needed for
    projection.
    """

    population_source_id: str
    partition: str
    cohort_version: str
    source_indices: np.ndarray
    variant_ids: np.ndarray

    @property
    def n(self) -> int:
        return int(self.source_indices.size)

    def __len__(self) -> int:
        return self.n


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
        self._population_projection_cache: dict = {}

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

        'y' carries 'NaN' for withheld labels; restricting to the label-eligible rows
        is the caller's explicit population step, performed through
        'EvaluationPopulation' built from 'population_projection'. Non-finite scores
        and probabilities are validation failures, not selections. Requires a score
        column.
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

    def population_projection(self, partition: str | None = None
                              ) -> CanonicalPopulationProjection:
        """The aligned frame an `EvaluationPopulation` will address.

        Deliberately does NOT require a score column. Population identity is a
        property of which variants are being evaluated, not of what any model
        predicted for them; requiring predictions would make it impossible to
        name a population before scoring it, and would couple the identity to
        model provenance that belongs elsewhere.

        The digest is memoised per partition. Hashing the ordered `variant_id`
        sequence is O(n) and the cohort runs to roughly 1.5 million variants, so
        computing it for every partition at table construction would tax every
        caller -- including the many that never build a population. Memoising
        means each partition is hashed at most once, which is what "compute it
        once" is for.
        """
        key = partition if partition is not None else None
        cached = self._population_projection_cache.get(key)
        if cached is not None:
            return cached

        if partition is None:
            mask = np.ones(len(self._frame), dtype=bool)
        else:
            available = set(self._frame["partition"].unique().tolist())
            if partition not in available:
                raise ValueError(
                    f"partition {partition!r} not present; available: "
                    f"{sorted(available)}")
            mask = (self._frame["partition"] == partition).to_numpy()

        source_indices = np.flatnonzero(mask).astype(np.int64)
        variant_ids = self._frame["variant_id"].to_numpy(dtype=object)[source_indices]
        projection = CanonicalPopulationProjection(
            population_source_id=_derive_population_source_id(
                cohort_version=self._cohort_version, partition=partition,
                variant_ids=variant_ids),
            partition=partition if partition is not None else "__all__",
            cohort_version=self._cohort_version,
            source_indices=source_indices,
            variant_ids=variant_ids,
        )
        self._population_projection_cache[key] = projection
        return projection

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
