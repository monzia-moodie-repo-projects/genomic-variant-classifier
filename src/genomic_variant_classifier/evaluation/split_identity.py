"""src/genomic_variant_classifier/evaluation/split_identity.py

Author: Monzia Moodie

The identity of a data split and of the prediction units made on it (package 1, set 1b; gates G1 and G2).

A split's dataset identity binds WHAT the rows are, in order: the split role, the feature column names, the feature
matrix as binary64 values, the labels and the variant identities (chrom:pos:ref:alt, prediction_contract.
variant_identity_of). It is computed from the in-memory frames a run actually uses, so it does not depend on how (or
whether) a parquet file was written. A source row is `<role>_split_position:<i>` within that dataset.

Two unit contexts, so the same training row's out-of-fold prediction and a held-out prediction can never share a
prediction-unit identity:
  OOF_UNIT_CONTEXT_SHA256    one out-of-fold prediction per training-split row
  SPLIT_UNIT_CONTEXT_SHA256  one pathogenicity prediction per row of an evaluated split

What a digest here is not: evidence that the labels are right, that the split is leakage-free, or that the features
were computed correctly. It detects that two artifacts describe different rows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from genomic_variant_classifier.prediction_contract import (
    ContractError,
    Observation,
    ObservationSet,
    canonical_digest,
    require_sha256,
    variant_identity_of,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OOF_UNIT_CONTEXT_SHA256",
    "SPLIT_ROLES",
    "SPLIT_UNIT_CONTEXT_SHA256",
    "SplitIdentity",
    "binary_labels",
    "split_identity",
    "split_observations",
    "variant_identities",
]

SPLIT_ROLES = frozenset({"train", "val", "test", "holdout"})

OOF_UNIT_CONTEXT_SHA256 = canonical_digest("gvc.unit-context.v1", {
    "unit": "one_out_of_fold_prediction_per_training_split_row", "positive_class": 1})
SPLIT_UNIT_CONTEXT_SHA256 = canonical_digest("gvc.unit-context.v1", {
    "unit": "one_pathogenicity_prediction_per_evaluated_split_row", "positive_class": 1})
_UNIT_CONTEXTS = frozenset({OOF_UNIT_CONTEXT_SHA256, SPLIT_UNIT_CONTEXT_SHA256})


def binary_labels(y: Any, n: int) -> np.ndarray:
    """Labels as int64, exactly {0, 1}; booleans, floats with fractions and missing values are refused."""
    arr = np.asarray(y)
    if arr.ndim != 1 or len(arr) != n:
        raise ContractError(f"one label per row required ({n} rows, labels shape {arr.shape})")
    if arr.dtype.kind not in "iu":
        raise ContractError(f"integer labels required, not {arr.dtype}")
    values = arr.astype(np.int64)
    if not np.isin(values, (0, 1)).all():
        raise ContractError("labels must be exactly 0 or 1")
    return values


def _locus_value(value: Any) -> Any:
    """A locus field as the identity uses it: missing -> None; an integral float position -> int; never coerced
    otherwise."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
        raise ContractError(f"non-integral locus value {value!r}")
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def variant_identities(meta: pd.DataFrame) -> list[str]:
    """chrom:pos:ref:alt per row of `meta`, in row order (prediction_contract.variant_identity_of)."""
    if not isinstance(meta, pd.DataFrame):
        raise ContractError("a metadata DataFrame is required")
    missing = [c for c in ("chrom", "pos", "ref", "alt") if c not in meta.columns]
    if missing:
        raise ContractError(f"metadata lacks the locus columns {missing}")
    cols = [meta[c].tolist() for c in ("chrom", "pos", "ref", "alt")]
    return [variant_identity_of({"chrom": _locus_value(c), "pos": _locus_value(p), "ref": _locus_value(r),
                                 "alt": _locus_value(a)})
            for c, p, r, a in zip(*cols)]


def _features_sha256(X: pd.DataFrame) -> str:
    try:
        matrix = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"the feature matrix is not numeric: {exc}") from exc
    if np.isnan(matrix).any():   # one canonical quiet NaN, so a NaN payload difference is not an identity difference
        matrix = matrix.copy()
        matrix[np.isnan(matrix)] = np.nan
    return hashlib.sha256(matrix.astype("<f8", copy=False).tobytes()).hexdigest()


@dataclass(frozen=True)
class SplitIdentity:
    role: str
    n_rows: int
    columns: tuple[str, ...]
    features_sha256: str
    labels_sha256: str
    variant_identities_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if self.role not in SPLIT_ROLES:
            raise ContractError(f"unknown split role {self.role!r}")
        if type(self.n_rows) is not int or self.n_rows <= 0:
            raise ContractError("positive row count required")
        for name in ("features_sha256", "labels_sha256", "variant_identities_sha256", "dataset_sha256"):
            require_sha256(getattr(self, name), name)
        expected = canonical_digest("gvc.split-dataset.v1", self._content())
        if expected != self.dataset_sha256:
            raise ContractError("the split's dataset digest does not bind its stated content")

    def _content(self) -> dict[str, Any]:
        return {"role": self.role, "n_rows": self.n_rows, "columns": list(self.columns),
                "features_float64_sha256": self.features_sha256, "labels_int64_sha256": self.labels_sha256,
                "variant_identities_sha256": self.variant_identities_sha256}

    @property
    def prediction_split_sha256(self) -> str:
        """The `split_sha256` of a Binding for predictions made on this split."""
        return canonical_digest("gvc.prediction-split.v1", {"split": self.role, "dataset_sha256": self.dataset_sha256})

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["columns"] = list(self.columns)
        return d


def split_identity(X: pd.DataFrame, y: Any, meta: pd.DataFrame, role: str) -> tuple[SplitIdentity, list[str]]:
    """The dataset identity of one split, and its variant identities (row order)."""
    if not isinstance(X, pd.DataFrame) or not len(X):
        raise ContractError("a nonempty feature DataFrame is required")
    n = len(X)
    if len(meta) != n:
        raise ContractError(f"metadata rows {len(meta)} differ from feature rows {n}")
    columns = tuple(str(c) for c in X.columns)
    if len(set(columns)) != len(columns):
        raise ContractError("duplicate feature column names")
    labels = binary_labels(y, n)
    vids = variant_identities(meta)
    content = {"features": _features_sha256(X),
               "labels": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
               "vids": hashlib.sha256(json.dumps(vids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
                                      ).hexdigest()}
    probe = {"role": role, "n_rows": n, "columns": list(columns), "features_float64_sha256": content["features"],
             "labels_int64_sha256": content["labels"], "variant_identities_sha256": content["vids"]}
    identity = SplitIdentity(role=role, n_rows=n, columns=columns, features_sha256=content["features"],
                             labels_sha256=content["labels"], variant_identities_sha256=content["vids"],
                             dataset_sha256=canonical_digest("gvc.split-dataset.v1", probe))
    return identity, vids


def split_observations(identity: SplitIdentity, vids: list[str], *, unit_context_sha256: str,
                       positions: Any = None) -> ObservationSet:
    """Prediction units on `identity`'s rows: all rows in order, or the given split positions in the given order.

    The source row key is the position in the split; nothing is renumbered when a subset is taken.
    """
    if type(identity) is not SplitIdentity:
        raise ContractError("SplitIdentity required")
    if unit_context_sha256 not in _UNIT_CONTEXTS:
        raise ContractError("a declared unit context is required")
    if len(vids) != identity.n_rows:
        raise ContractError("one variant identity per split row required")
    if positions is None:
        selected = range(identity.n_rows)
    else:
        selected = positions
        if any(type(i) is not int or not 0 <= i < identity.n_rows for i in selected):
            raise ContractError("split positions must be Python ints inside the split")
    prefix = f"{identity.role}_split_position:"
    return ObservationSet(tuple(Observation(identity.dataset_sha256, f"{prefix}{i}", vids[i], unit_context_sha256)
                                for i in selected))
