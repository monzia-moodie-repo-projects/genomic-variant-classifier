"""src/genomic_variant_classifier/evaluation/historical_oof_identity.py

Author: Monzia Moodie

A separately identified DERIVED record of a historical run's out-of-fold (OOF) row identity (package 1, set 1b, gate
G2.3). Written for Run 15, whose `oof_predictions.parquet` carries `_train_row_idx = 0..n-1` -- positions over the
ensemble's fit subset, not training-split rows.

The record is derived FROM THE ARTIFACT, never assumed, and never rewrites the original OOF file:
  1. the orchestrator (`ensemble.joblib`) is read with the INERT reader (genomic_variant_classifier.inert_pickle): no
     model is deserialized, no class the file names is imported, and the admission route is not bypassed because
     nothing is admitted -- only the stored arrays `oof_fit_indices_`, `oof_predictions_` and `oof_model_names_` and the
     recorded configuration (`random_state`, `n_folds`) are recovered;
  2. the legacy calibration carve is RECONSTRUCTED from the code the run executed
     (train_test_split(arange(n), test_size=0.15, stratify=y, random_state)) and must equal the stored
     `oof_fit_indices_` exactly, element by element;
  3. the OOF file must hold exactly the stored OOF matrix, in the stored member order, with `_train_row_idx` equal to
     0..n-1 (the property that makes it unjoinable by its own key);
  4. the legacy per-member sidecars (`<member>_oof_indices.npy`) are checked against the reconstructed fold
     concatenation, and the folds of every OOF row are reconstructed from the run's declared splitter;
  5. AUROC is measured both ways -- joined through the derived mapping and joined naively as a training row -- as a
     consistency check of the mapping (not as evidence of model quality).

Every input file is bound by SHA-256 and size. Any mismatch raises; nothing is published partially.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from genomic_variant_classifier.inert_pickle import InertObject, read_inert
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest

logger = logging.getLogger(__name__)

__all__ = ["derive_legacy_oof_identity", "file_identity"]

_CHUNK = 1 << 20


def file_identity(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": n}


def _int64_sha256(values: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()).hexdigest()


def derive_legacy_oof_identity(run_dir: Path, *, mapping_out: Optional[Path] = None,
                               code_provenance: Optional[dict] = None) -> dict:
    """Derive and verify the OOF-row -> training-split-row mapping of a pre-set-1b run directory.

    run_dir holds models/ensemble.joblib, models/ensemble.manifest.json, oof_predictions.parquet,
    splits/y_train.parquet and splits/meta_train.parquet (Run 15's layout). Returns the record (a dict of JSON
    values). When mapping_out is given, the derived mapping table is written there as parquet and bound in the record.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import sklearn

    run_dir = Path(run_dir)
    paths = {"orchestrator": run_dir / "models" / "ensemble.joblib",
             "orchestrator_manifest": run_dir / "models" / "ensemble.manifest.json",
             "oof_predictions": run_dir / "oof_predictions.parquet",
             "y_train": run_dir / "splits" / "y_train.parquet",
             "meta_train": run_dir / "splits" / "meta_train.parquet"}
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise ContractError(f"{run_dir}: missing inputs {missing}")
    inputs = {k: {"path": p.relative_to(run_dir).as_posix(), **file_identity(p)} for k, p in paths.items()}

    orchestrator = read_inert(paths["orchestrator"])
    if not isinstance(orchestrator, dict) or orchestrator.get("format_version") != 2:
        raise ContractError("not a format_version=2 orchestrator")
    config = orchestrator.get("config")
    if not isinstance(config, InertObject) or not isinstance(config.state, dict):
        raise ContractError("the orchestrator records no EnsembleConfig state")
    random_state, n_folds = config.state.get("random_state"), config.state.get("n_folds")
    if type(random_state) is not int or type(n_folds) is not int:
        raise ContractError("random_state and n_folds must be recorded integers")
    stored = orchestrator.get("oof_fit_indices_")
    oof_matrix = orchestrator.get("oof_predictions_")
    names = orchestrator.get("oof_model_names_")
    if not (isinstance(stored, np.ndarray) and isinstance(oof_matrix, np.ndarray) and isinstance(names, list)):
        raise ContractError("the orchestrator lacks oof_fit_indices_, oof_predictions_ or oof_model_names_")

    weights = orchestrator.get("blend_weights_")
    if not (isinstance(weights, np.ndarray) and weights.dtype == np.float64 and weights.shape == (len(names),)
            and np.isfinite(weights).all()):
        raise ContractError("the orchestrator stores no finite binary64 blend weight per member")
    stored_blend = {"present": True, "n": int(len(weights)),
                    "weights_binary64_hex": [float(w).hex() for w in weights.tolist()],
                    "fsum": math.fsum(weights.tolist()),
                    "meaning": ("non-null stored weights: at the run's code, predict_proba returned this blend "
                                "and evaluate() reported it as ENSEMBLE_STACKER (see code_provenance)")}

    y = pd.read_parquet(paths["y_train"])
    if list(y.columns) != ["label"]:
        raise ContractError(f"y_train columns {list(y.columns)}; expected ['label']")
    y = y["label"].to_numpy()
    n = len(y)
    fit_idx, cal_idx = train_test_split(np.arange(n), test_size=0.15, stratify=y, random_state=random_state)
    equal = bool(stored.dtype.kind in "iu" and stored.shape == fit_idx.shape and np.array_equal(stored, fit_idx))
    if not equal:
        raise ContractError("the stored oof_fit_indices_ differ from the reconstructed legacy carve; no mapping is "
                            "derived")

    oof = pd.read_parquet(paths["oof_predictions"])
    member_cols = [c for c in oof.columns if c != "_train_row_idx"]
    if member_cols != list(names):
        raise ContractError(f"OOF file columns {member_cols} differ from oof_model_names_ {names}")
    train_row_idx = oof["_train_row_idx"].to_numpy()
    if not np.array_equal(train_row_idx, np.arange(len(oof))):
        raise ContractError("the historical _train_row_idx is not 0..n-1; this derivation's premise does not hold")
    if oof_matrix.shape != (len(oof), len(names)) or not all(
            np.array_equal(oof[c].to_numpy(), oof_matrix[:, j]) for j, c in enumerate(names)):
        raise ContractError("the OOF file does not hold exactly the stored OOF matrix")

    y_fit = y[fit_idx]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = np.empty(len(fit_idx), dtype=np.int64)
    concatenated = []
    for k, (_tr, va) in enumerate(splitter.split(np.zeros(len(fit_idx)), y_fit)):
        folds[va] = k
        concatenated.append(va)
    concatenated = np.concatenate(concatenated)

    members: dict[str, dict] = {}
    for j, name in enumerate(names):
        p = oof_matrix[:, j]
        entry = {"auroc_through_derived_mapping": float(roc_auc_score(y_fit, p)),
                 "auroc_joined_naively_as_training_row": float(roc_auc_score(y[train_row_idx], p))}
        sidecar = run_dir / "models" / f"{name}_oof_indices.npy"
        if sidecar.is_file():
            side = np.load(sidecar, allow_pickle=False)
            entry["legacy_sidecar"] = {
                "path": sidecar.relative_to(run_dir).as_posix(), **file_identity(sidecar),
                "equals_reconstructed_fold_concatenation": bool(np.array_equal(side, concatenated)),
                "auroc_if_paired_element_wise": float(roc_auc_score(y_fit[side], p)) if len(side) == len(p)
                else None}
        members[name] = entry

    mapping = pd.DataFrame({"historical_oof_row": np.arange(len(fit_idx), dtype=np.int64),
                            "train_split_position": fit_idx.astype(np.int64), "fold": folds,
                            "label": y_fit.astype(np.int64)})
    meta = pd.read_parquet(paths["meta_train"], columns=["variant_id", "gene_symbol"])
    if len(meta) != n:
        raise ContractError("meta_train rows differ from y_train rows")
    mapping["variant_id"] = meta["variant_id"].to_numpy()[fit_idx]
    mapping["gene_symbol"] = meta["gene_symbol"].to_numpy()[fit_idx]

    fit_libraries = json.loads(paths["orchestrator_manifest"].read_text(encoding="utf-8")).get("libraries", {})
    record = {
        "schema": "gvc.historical-oof-identity.v1",
        "status": "derived_and_verified",
        "statement": ("The historical OOF file is NOT rewritten. Row k of its member columns is the out-of-fold "
                      "prediction for training-split row train_split_position[k] = oof_fit_indices_[k], in the fold "
                      "given; `_train_row_idx` is a position over the fit subset and must never be joined as a "
                      "training row."),
        "inputs": inputs,
        "orchestrator_reader": "genomic_variant_classifier.inert_pickle (no class imported, no model deserialized)",
        "recorded_configuration": {"random_state": random_state, "n_folds": n_folds},
        "legacy_carve_rule": {
            "function": "sklearn.model_selection.train_test_split",
            "arguments": {"arrays": "numpy.arange(n_train)", "test_size": 0.15, "stratify": "y_train label",
                          "random_state": random_state},
            "scikit_learn_at_fit": fit_libraries.get("scikit-learn"),
            "scikit_learn_at_derivation": sklearn.__version__},
        "n_train": int(n), "n_fit": int(len(fit_idx)), "n_calibration": int(len(cal_idx)),
        "stored_oof_fit_indices_int64_sha256": _int64_sha256(stored),
        "reconstructed_fit_indices_int64_sha256": _int64_sha256(fit_idx),
        "stored_equals_reconstruction": equal,
        "calibration_indices_int64_sha256": _int64_sha256(cal_idx),
        "stored_blend_weights": stored_blend,
        "oof_file_member_order": list(names),
        "oof_file_train_row_idx_is_0_to_n_minus_1": True,
        "oof_file_columns_equal_stored_matrix": True,
        "folds": {"splitter": {"kind": "StratifiedKFold", "n_splits": n_folds, "shuffle": True,
                               "random_state": random_state, "stratified_on": "label of the fit rows"},
                  "sizes": np.bincount(folds).tolist(), "folds_int64_sha256": _int64_sha256(folds),
                  "gene_disjoint": False},
        "members": members,
        "calibration_note": ("Run 15's isotonic calibrators were fitted on the calibration rows of this LEGACY "
                             "label-stratified carve, which shares genes with the fit rows: its calibrated "
                             "probabilities are not evidence about calibration on unseen genes."),
    }
    if code_provenance is not None:
        record["code_provenance"] = code_provenance
    if mapping_out is not None:
        mapping_out = Path(mapping_out)
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_parquet(mapping_out, index=False)
        record["derived_mapping"] = {"file": mapping_out.name, **file_identity(mapping_out),
                                     "columns": list(mapping.columns), "rows": int(len(mapping))}
    record["record_sha256"] = canonical_digest("gvc.historical-oof-identity-record.v1", record)
    logger.info("Derived and verified the historical OOF identity of %s: %d rows.", run_dir, len(fit_idx))
    return record
