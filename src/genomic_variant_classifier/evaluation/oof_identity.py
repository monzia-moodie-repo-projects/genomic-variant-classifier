"""src/genomic_variant_classifier/evaluation/oof_identity.py

Author: Monzia Moodie

Out-of-fold (OOF) predictions with their ORIGINAL training-split row identity (package 1, set 1b, gate G2).

The ensemble's OOF matrix is in the row order of the rows it was fitted on AFTER the calibration carve; its
`oof_fit_indices_` are positions in the input fit() received, NOT training-split row numbers. Run 15 published them
as `_train_row_idx = 0..n-1` and that file cannot be joined by its own key (measured: AUROC 0.4995 when joined as a
training row). The composition here is the only route from an OOF row to a training-split row:

    original training-split rows --(optional subsample: positions)--> fit input
    fit input --(oof_fit_indices_)--> OOF rows

and it is VERIFIED, not assumed: the composed rows' labels must equal the labels the ensemble recorded for its OOF
rows (`oof_labels_`), element by element. Any failure raises ContractError and nothing is published (gate G2.2).

The published table has identity columns and one `<member>_prob` column per fitted member, in the fitted order. It has
NO blend column: the stored weights were fitted ON these rows, so a blend of them is an in-sample training
diagnostic, not an out-of-fold prediction.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.split_identity import (
    OOF_UNIT_CONTEXT_SHA256,
    SplitIdentity,
    binary_labels,
    split_observations,
)
from genomic_variant_classifier.prediction_contract import (
    ContractError,
    canonical_digest,
    integer_positions,
)

logger = logging.getLogger(__name__)

__all__ = ["OOF_IDENTITY_COLUMNS", "OOFIdentity", "compose_oof_identity", "member_column"]

#: The identity columns of an OOF table, in order, before the member columns.
OOF_IDENTITY_COLUMNS = ("train_split_position", "source_observation_id", "prediction_unit_id", "fold", "label",
                        "gene_symbol", "variant_id", "variant_identity")


def member_column(name: str) -> str:
    return f"{name}_prob"


def _int64_sha256(values: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class OOFIdentity:
    """The OOF table and the evidence of how its identity was composed. Deliberately unhashable."""

    frame: pd.DataFrame
    evidence: dict
    members: tuple[str, ...]
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        expected = list(OOF_IDENTITY_COLUMNS) + [member_column(m) for m in self.members]
        if list(self.frame.columns) != expected:
            raise ContractError(f"OOF columns {list(self.frame.columns)} differ from the declared {expected}")


def compose_oof_identity(ensemble: Any, *, train: SplitIdentity, vids: list[str], labels: Any,
                         meta_train: pd.DataFrame, subsample_positions: Any = None) -> OOFIdentity:
    """Compose and verify the original training-split identity of every OOF row of a fitted ensemble.

    train / vids / labels / meta_train describe the FULL training split (before any subsample), row-aligned.
    subsample_positions: the training-split positions fit() received, in the order it received them, or None when
    fit() received the whole split in split order.
    """
    if type(train) is not SplitIdentity or train.role != "train":
        raise ContractError("the training split's SplitIdentity is required")
    y = binary_labels(labels, train.n_rows)
    if len(meta_train) != train.n_rows or len(vids) != train.n_rows:
        raise ContractError("metadata and variant identities must describe every training-split row")
    for attr in ("oof_predictions_", "oof_fit_indices_", "oof_model_names_", "fit_input_rows_", "oof_labels_",
                 "oof_fold_", "oof_cv_scheme_"):
        if getattr(ensemble, attr, None) is None:
            raise ContractError(f"the ensemble carries no {attr}: its out-of-fold identity cannot be composed "
                                "(an ensemble fitted before package 1, set 1b needs a separately identified derived "
                                "record, never a sequential index)")
    n_input = ensemble.fit_input_rows_
    if type(n_input) is not int or n_input <= 0:
        raise ContractError("fit_input_rows_ must be a positive Python int")

    if subsample_positions is None:
        if n_input != train.n_rows:
            raise ContractError(f"fit() received {n_input} rows but the training split has {train.n_rows}; the "
                                "subsample positions are required")
        parent = None
        subsample = {"applied": False}
    else:
        raw = subsample_positions if isinstance(subsample_positions, (list, tuple)) else np.asarray(subsample_positions)
        parent, conversion = integer_positions(raw)
        if len(parent) != n_input:
            raise ContractError(f"{len(parent)} subsample positions for {n_input} fit-input rows")
        if len(set(parent)) != len(parent) or any(not 0 <= p < train.n_rows for p in parent):
            raise ContractError("subsample positions must be distinct training-split positions")
        subsample = {"applied": True, "n": len(parent), "positions_int64_sha256": _int64_sha256(parent),
                     "conversion": conversion}

    fit_positions, fit_conversion = integer_positions(np.asarray(ensemble.oof_fit_indices_))
    if len(set(fit_positions)) != len(fit_positions) or any(not 0 <= p < n_input for p in fit_positions):
        raise ContractError("oof_fit_indices_ must be distinct positions inside the fit input")
    original = fit_positions if parent is None else [parent[p] for p in fit_positions]

    oof = np.asarray(ensemble.oof_predictions_)
    names = tuple(str(n) for n in ensemble.oof_model_names_)
    if oof.ndim != 2 or oof.shape != (len(original), len(names)) or oof.dtype.kind != "f":
        raise ContractError(f"OOF matrix shape {oof.shape} does not match {len(original)} rows x {len(names)} members")
    if not np.isfinite(oof).all() or ((oof < 0) | (oof > 1)).any():
        raise ContractError("OOF member probabilities must be finite and in [0, 1]; nothing is clipped")

    fold = np.asarray(ensemble.oof_fold_)
    if fold.shape != (len(original),) or fold.dtype.kind not in "iu" or (fold < 0).any():
        raise ContractError("one nonnegative integer fold per OOF row is required")

    composed_labels = y[np.asarray(original, dtype=np.int64)]
    recorded = np.asarray(ensemble.oof_labels_)
    if recorded.shape != composed_labels.shape or not np.array_equal(recorded.astype(np.int64), composed_labels):
        bad = int((recorded.astype(np.int64) != composed_labels).sum()) if recorded.shape == composed_labels.shape \
            else "shape"
        raise ContractError(f"identity binding FAILED: the composed training rows' labels differ from the labels the "
                            f"ensemble fitted on ({bad} rows). Nothing is published")

    rows = split_observations(train, vids, unit_context_sha256=OOF_UNIT_CONTEXT_SHA256, positions=original)
    genes = meta_train["gene_symbol"].to_numpy(dtype=object)[original] if "gene_symbol" in meta_train.columns \
        else np.full(len(original), None, dtype=object)
    variant_id = meta_train["variant_id"].to_numpy(dtype=object)[original] if "variant_id" in meta_train.columns \
        else np.full(len(original), None, dtype=object)
    data = {
        "train_split_position": np.asarray(original, dtype=np.int64),
        "source_observation_id": [r.source_observation_id for r in rows.rows],
        "prediction_unit_id": [r.prediction_unit_id for r in rows.rows],
        "fold": fold.astype(np.int64),
        "label": composed_labels,
        "gene_symbol": genes,
        "variant_id": variant_id,
        "variant_identity": [r.variant_identity for r in rows.rows],
    }
    for j, name in enumerate(names):
        data[member_column(name)] = np.asarray(oof[:, j], dtype=np.float64)
    frame = pd.DataFrame(data)

    evidence = {
        "schema": "gvc.oof-identity.v1",
        "train_split": train.as_dict(),
        "fit_input_rows": n_input,
        "subsample": subsample,
        "oof_fit_indices_int64_sha256": _int64_sha256(fit_positions),
        "oof_fit_indices_conversion": fit_conversion,
        "train_split_positions_int64_sha256": _int64_sha256(original),
        "folds_int64_sha256": _int64_sha256(fold),
        "cv_scheme": dict(ensemble.oof_cv_scheme_),
        "calibration_carve": getattr(ensemble, "calibration_carve_used_", None),
        "labels_verified_against_fit_labels": len(original),
        "rows_sha256": rows.digest,
        "unit_context_sha256": OOF_UNIT_CONTEXT_SHA256,
        "members": list(names),
        "oof_predictions_float64_sha256": hashlib.sha256(
            np.ascontiguousarray(oof.astype("<f8", copy=False)).tobytes()).hexdigest(),
        "fit_input_stage_sha256": ensemble.fit_input_stage_sha256(),
        "blend_column": "omitted: the stored weights were fitted on these rows (in-sample)",
    }
    evidence["identity_sha256"] = canonical_digest("gvc.oof-identity-evidence.v1", evidence)
    logger.info("OOF identity composed and verified: %d rows, %d members, labels equal on every row.",
                len(original), len(names))
    return OOFIdentity(frame=frame, evidence=evidence, members=names)
