"""
src/genomic_variant_classifier/evaluation/prediction_artifacts.py
=======================================
RunArtifactWriter -- one-stop shop for emitting the artefact set of one run (or one ablation of a run), and the ONE
place that decides whether that run is COMPLETE evidence.

Design principles:
  - Local-first: every write lands on disk before any upload.
  - Atomic: writes go to a temporary sibling, are fsynced, then renamed; a failed write never corrupts an existing
    artefact. The digest of the bytes is measured at write time.
  - Strict: JSON through serialization.dump_strict_json (no `default=str`, no NaN literals); tables as parquet.
  - No logging configuration in this library module; `from __future__ import annotations`.

PACKAGE 1, SET 1b (2026-11-08) -- what changed and why:
  * COMPLETION MANIFEST (gate G8). `complete()` writes `completion_manifest.json` LAST, only after every required
    output exists with the bytes this writer measured when it wrote them, together with the selected predictor's
    identity, the bound prediction-block digests and the out-of-fold identity evidence. A run without it stays fully
    inspectable but is refused as completed evidence (`verify_completed_run`). A manifest left by an earlier run in
    the same directory is RETIRED (renamed, never deleted) before anything is written, so a crashed rerun can never
    inherit a previous run's completion.
  * Predictions consume BOUND captures (models.variant_ensemble.ScoredSplit): bare probability arrays are refused.
  * Out-of-fold predictions require their composed original training-split identity (evaluation.oof_identity).
  * The explanation writer explains the SELECTED predictor: exact member-score accounting with a recorded closure
    residual (evaluation.blend_attribution). The old SHAP writer explained the logistic research candidate; it is
    retired, and a feature-level explanation of the blend is reported unavailable with its reason.
  * Calibration: reliability points with their REAL bin counts and edges, separate from the per-consequence
    performance table (the old single table put AUROC in a `mean_pred` column and prevalence in `frac_pos`).
  * The run-level ablation aggregate is appended atomically, and only for a completed run.

Integration (scripts/run_phase2_eval.py, scripts/run9_ablations.py):
    writer = RunArtifactWriter(run_id="run9", ablation="full", output_dir=path)
    writer.save_manifest(git_sha=..., versions=..., config=...)
    scored_test = ensemble.score_split(X_test, seq_test, rows_test, split_sha256=...)
    writer.save_split_predictions(scored_test, y_test, meta_test, role="test")
    writer.save_oof_predictions(compose_oof_identity(ensemble, ...))
    writer.save_eval_report(report)
    writer.save_calibration(scored_test, y_test, report=report, meta=meta_test)
    writer.save_member_score_attributions(scored_test, background=scored_val)
    writer.complete(required=[...], deliverables={...})
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.serialization import dump_strict_json
from genomic_variant_classifier.prediction_contract import ContractError

logger = logging.getLogger(__name__)

__all__ = [
    "COMPLETION_MANIFEST",
    "BoundSplitPredictions",
    "IncompleteRunError",
    "RunArtifactWriter",
    "SPLIT_PREDICTION_IDENTITY_COLUMNS",
    "calibration_points",
    "load_completed_split_predictions",
    "selected_predictor_evaluation_evidence",
    "verify_completed_run",
]

COMPLETION_MANIFEST = "completion_manifest.json"
COMPLETION_SCHEMA = "gvc.run-completion.v1"
_NAME = re.compile(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*")
_READ_CHUNK = 1 << 20


class IncompleteRunError(ContractError):
    """A run directory that is not complete, verified evidence: inspect it, never consume it as a result."""


def _file_digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _require_name(name: str) -> str:
    if type(name) is not str or not _NAME.fullmatch(name) or ".." in name.split("/"):
        raise ContractError(f"artefact names are relative POSIX paths without '..': {name!r}")
    return name


def _require_capture(scored: Any) -> None:
    for attr in ("result", "members", "combiner", "split_sha256"):
        if not hasattr(scored, attr):
            raise ContractError("a bound split capture (VariantEnsemble.score_split) is required; bare probability "
                                "arrays are refused (package 1, set 1b)")


def calibration_points(p: Any, y: Any, *, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Reliability-diagram points WITH their bin edges and real counts.

    The binning is scikit-learn's calibration_curve rule, reproduced line for line (quantile edges by
    numpy.percentile, bin = searchsorted over the inner edges), so the non-empty rows equal what the evaluator
    reports; unlike calibration_curve, empty bins are kept (n = 0, means null) instead of silently dropped.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    if p.ndim != 1 or y.shape != p.shape or not len(p):
        raise ContractError("one label per probability required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("probabilities must be finite and in [0, 1]")
    if strategy == "quantile":
        bins = np.percentile(p, np.linspace(0, 1, n_bins + 1) * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ContractError(f"unknown calibration strategy {strategy!r}")
    positive = y == 1
    binids = np.searchsorted(bins[1:-1], p)
    bin_sums = np.bincount(binids, weights=p, minlength=len(bins))
    bin_true = np.bincount(binids, weights=positive, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    rows = []
    for b in range(n_bins):
        n = int(bin_total[b])
        rows.append({"bin": b, "lower_edge": float(bins[b]), "upper_edge": float(bins[b + 1]), "n": n,
                     "n_positive": int(bin_true[b]),
                     "mean_predicted": float(bin_sums[b] / bin_total[b]) if n else None,
                     "fraction_positive": float(bin_true[b] / bin_total[b]) if n else None})
    if int(bin_total[n_bins:].sum()):
        raise ContractError("a probability fell outside the declared bins")
    return pd.DataFrame(rows)


def verify_completed_run(run_dir: Path | str) -> dict:
    """The completion manifest of `run_dir`, after re-measuring every artefact it lists; else IncompleteRunError."""
    run_dir = Path(run_dir)
    path = run_dir / COMPLETION_MANIFEST
    if not path.is_file():
        raise IncompleteRunError(f"{run_dir}: no {COMPLETION_MANIFEST}; the run is not complete evidence")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncompleteRunError(f"{path}: not valid JSON ({exc})") from exc
    if manifest.get("schema") != COMPLETION_SCHEMA or manifest.get("status") != "complete":
        raise IncompleteRunError(f"{path}: not a {COMPLETION_SCHEMA} manifest with status 'complete'")
    artefacts = manifest.get("artefacts")
    if not isinstance(artefacts, dict) or not artefacts:
        raise IncompleteRunError(f"{path}: lists no artefacts")
    problems = []
    for name, entry in artefacts.items():
        target = run_dir / _require_name(name)
        if not target.is_file():
            problems.append(f"{name}: missing")
            continue
        digest, size = _file_digest(target)
        if digest != entry.get("sha256") or size != entry.get("bytes"):
            problems.append(f"{name}: bytes changed since completion")
    if problems:
        raise IncompleteRunError(f"{run_dir}: the completion manifest does not describe these files: {problems}")
    return manifest


# ---------------------------------------------------------------------------
# Bound reader of a completed run's split predictions (package 1, set 1c)
# ---------------------------------------------------------------------------
#: The identity and value columns save_split_predictions writes first, in this order; one `<member>_prob` column per
#: fitted member follows, in the predictor's member order.
SPLIT_PREDICTION_IDENTITY_COLUMNS = ("source_observation_id", "prediction_unit_id", "variant_identity", "label",
                                     "selected_score")


def _verified_bytes(run_dir: Path, name: str, manifest: Mapping[str, Any]) -> bytes:
    """The bytes of one listed artefact, read ONCE and compared with the manifest entry. Parsing these bytes (never a
    second read of the path) means what is parsed is exactly what was verified."""
    entry = (manifest.get("artefacts") or {}).get(name)
    if not isinstance(entry, dict):
        raise IncompleteRunError(f"{run_dir}: the completion manifest does not list {name}")
    data = (run_dir / _require_name(name)).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("bytes"):
        raise IncompleteRunError(f"{run_dir}: {name} changed after the run was completed")
    return data


@dataclass(frozen=True, eq=False)
class BoundSplitPredictions:
    """One split's predictions from a COMPLETED run, verified against its completion manifest and block descriptor.

    `frame` holds the identity columns, the integer labels, the selected score (the stored convex blend, full
    precision) and one column per member. `identity()` is what a consumer records so its own output names the exact
    predictor, run, file and rows it was computed from.
    """

    run_dir: Path
    role: str
    frame: pd.DataFrame
    predictor: Mapping[str, Any]
    blocks: Mapping[str, Any]
    completion_manifest_sha256: str
    predictions_sha256: str
    run_id: str
    ablation: str

    @property
    def scores(self) -> np.ndarray:
        return self.frame["selected_score"].to_numpy(dtype=np.float64)

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy(dtype=np.int64)

    def identity(self) -> dict[str, Any]:
        return {"run_dir_name": self.run_dir.name, "run_id": self.run_id, "ablation": self.ablation,
                "completion_manifest_sha256": self.completion_manifest_sha256, "role": self.role,
                "predictions_file": f"{self.role}_predictions.parquet",
                "predictions_sha256": self.predictions_sha256,
                "combiner_sha256": self.predictor["combiner_sha256"],
                "split_sha256": self.blocks["split_sha256"], "rows_sha256": self.blocks["rows_sha256"],
                "result_block_sha256": self.blocks["result_block_sha256"], "n_rows": int(self.blocks["n_rows"]),
                "labels_int64_sha256": self.blocks["labels_int64_sha256"], "score_column": "selected_score"}


def load_completed_split_predictions(run_dir: Path | str, role: str) -> BoundSplitPredictions:
    """`<role>_predictions.parquet` of a completed run, refused unless every binding holds.

    The run must verify (verify_completed_run re-measures every listed artefact). The predictions file, its block
    descriptor and the manifest itself are then read once each and compared with what the manifest recorded; the
    descriptor must agree with the manifest's prediction-block entry (split, rows, row count, result block, member
    blocks); the columns must be the identity columns followed by exactly the predictor's members; the labels must
    reproduce the recorded label digest; and the selected score must be float64, finite and in [0, 1]. Nothing is
    repaired, re-ordered or coerced.
    """
    from genomic_variant_classifier.evaluation.split_identity import SPLIT_ROLES, binary_labels
    from genomic_variant_classifier.prediction_contract import require_sha256

    run_dir = Path(run_dir)
    if role not in SPLIT_ROLES:
        raise ContractError(f"unknown split role {role!r}")
    manifest = verify_completed_run(run_dir)
    manifest_bytes = (run_dir / COMPLETION_MANIFEST).read_bytes()
    if json.loads(manifest_bytes.decode("utf-8")) != manifest:
        raise IncompleteRunError(f"{run_dir}: the completion manifest changed while it was being read")
    predictor = manifest.get("predictor")
    if not isinstance(predictor, dict):
        raise IncompleteRunError(f"{run_dir}: the completion manifest names no selected predictor")
    require_sha256(predictor.get("combiner_sha256"), "combiner_sha256")
    recorded = (manifest.get("prediction_blocks") or {}).get(role)
    if not isinstance(recorded, dict):
        raise IncompleteRunError(f"{run_dir}: the completed run holds no {role} predictions")
    name = f"{role}_predictions.parquet"
    blocks = json.loads(_verified_bytes(run_dir, f"{role}_prediction_blocks.json", manifest).decode("utf-8"))
    for key in ("split_sha256", "rows_sha256", "n_rows", "result_block_sha256"):
        if blocks.get(key) != recorded.get(key):
            raise IncompleteRunError(f"{run_dir}: {role} block descriptor and completion manifest disagree on {key}")
    member_digests = {m: d.get("sha256") for m, d in (blocks.get("member_blocks") or {}).items()}
    if member_digests != recorded.get("member_block_sha256"):
        raise IncompleteRunError(f"{run_dir}: {role} member blocks differ from the completion manifest")
    members = list(predictor.get("members") or [])
    if list(member_digests) != members:
        raise IncompleteRunError(f"{run_dir}: {role} member blocks {list(member_digests)} are not the predictor's "
                                 f"members {members} in order")
    data = _verified_bytes(run_dir, name, manifest)
    frame = pd.read_parquet(io.BytesIO(data))
    head = list(frame.columns[:len(SPLIT_PREDICTION_IDENTITY_COLUMNS)])
    members_present = list(frame.columns[len(SPLIT_PREDICTION_IDENTITY_COLUMNS):
                                         len(SPLIT_PREDICTION_IDENTITY_COLUMNS) + len(members)])
    if head != list(SPLIT_PREDICTION_IDENTITY_COLUMNS) or members_present != [f"{m}_prob" for m in members]:
        raise IncompleteRunError(f"{run_dir}: {name} columns {list(frame.columns)} are not the identity columns "
                                 f"followed by the predictor's members")
    if len(frame) != int(blocks["n_rows"]):
        raise IncompleteRunError(f"{run_dir}: {name} holds {len(frame)} rows; its blocks declare {blocks['n_rows']}")
    labels = binary_labels(frame["label"].to_numpy(), len(frame))
    if hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest() != blocks.get("labels_int64_sha256"):
        raise IncompleteRunError(f"{run_dir}: {name} labels do not reproduce the recorded label digest")
    if frame["selected_score"].dtype != np.float64:
        raise IncompleteRunError(f"{run_dir}: {name} selected_score is {frame['selected_score'].dtype}, not float64")
    scores = frame["selected_score"].to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise IncompleteRunError(f"{run_dir}: {name} selected_score is not finite in [0, 1]")
    return BoundSplitPredictions(
        run_dir=run_dir, role=role, frame=frame, predictor=dict(predictor), blocks=blocks,
        completion_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        predictions_sha256=hashlib.sha256(data).hexdigest(),
        run_id=str(manifest.get("run_id")), ablation=str(manifest.get("ablation")))


def selected_predictor_evaluation_evidence(run_dir: Path | str, combiner_sha256: str) -> dict:
    """Evaluation evidence of ONE predictor, computed from a completed run's bound split predictions (gate G7.2).

    Refused unless the run verifies and its completion manifest names `combiner_sha256` as the selected predictor:
    a number another combiner earned is never carried. For each evaluated split the run holds (val, test, holdout),
    the AUROC is recomputed from the bound `selected_score` with the repository's kernel
    (evaluation.metrics.auroc) and recorded with the identities of the file and rows it describes.
    """
    from genomic_variant_classifier.evaluation.metrics import auroc

    manifest = verify_completed_run(run_dir)
    named = (manifest.get("predictor") or {}).get("combiner_sha256")
    if named != combiner_sha256:
        raise ContractError(f"{run_dir}: the completed run's selected predictor is {named}, not {combiner_sha256}; "
                            "its evaluation is not evidence about this predictor")
    splits = {}
    manifest_digests = set()
    for role in ("val", "test", "holdout"):
        if role not in (manifest.get("prediction_blocks") or {}):
            continue
        bound = load_completed_split_predictions(run_dir, role)
        identity = bound.identity()
        manifest_digests.add(identity["completion_manifest_sha256"])
        splits[role] = {**{k: identity[k] for k in ("predictions_sha256", "result_block_sha256", "rows_sha256",
                                                     "split_sha256", "n_rows", "labels_int64_sha256")},
                        "n_positive": int(bound.labels.sum()), "auroc": float(auroc(bound.labels, bound.scores))}
    if not splits:
        raise IncompleteRunError(f"{run_dir}: the completed run holds no evaluated split predictions")
    if len(manifest_digests) != 1:
        raise IncompleteRunError(f"{run_dir}: the completion manifest changed while the evidence was read")
    return {"status": "bound", "schema": "gvc.selected-predictor-evaluation-evidence.v1",
            "combiner_sha256": combiner_sha256, "run_id": manifest.get("run_id"), "ablation": manifest.get("ablation"),
            "completion_manifest_sha256": manifest_digests.pop(),
            "metric_kernel": "genomic_variant_classifier.evaluation.metrics.auroc", "splits": splits,
            "qualification": "the selected predictor evaluated on its own run's splits; not an external, temporal or "
                             "held-out-gene validation"}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class RunArtifactWriter:
    """
    Writes the artefact set of a single ablation of a single run under `output_dir/` (local-only; copied back by the
    operator per INCIDENT_2026-04-29 -- no upload step), and decides completion.
    """

    def __init__(
        self,
        run_id: str,
        ablation: str,
        output_dir: Path,
    ) -> None:
        if not run_id or not ablation:
            raise ValueError("run_id and ablation are required")
        self.run_id = run_id
        self.ablation = ablation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._artefacts: list[str] = []
        self._written: dict[str, tuple[str, int]] = {}
        self._external: list[str] = []
        self._blocks: dict[str, dict] = {}
        self._predictor: Optional[dict] = None
        self._oof_evidence: Optional[dict] = None
        self._completed: Optional[Path] = None
        self.retired_manifest: Optional[str] = None
        stale = self.output_dir / COMPLETION_MANIFEST
        if stale.exists():
            # Retired BEFORE anything is rewritten: a rerun that crashes must not leave an earlier run's completion
            # standing over a mixture of old and new files. Renamed, never deleted (the evidence is kept), and never ONTO
            # an earlier retirement: a timestamp can repeat (Python 3.12's clock on Windows advances in steps of about
            # 15.6 ms), and os.replace would then silently destroy the earlier retired manifest.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            retired = self.output_dir / f"completion_manifest.superseded-{stamp}.json"
            n = 1
            while retired.exists():
                n += 1
                retired = self.output_dir / f"completion_manifest.superseded-{stamp}-{n}.json"
            os.replace(stale, retired)
            self.retired_manifest = retired.name
            logger.warning("An earlier completion manifest in %s was retired to %s before this run writes anything.",
                           self.output_dir, retired.name)

    # ── Atomic write helper ────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self._completed is not None:
            raise ContractError("this run is complete; its artefact set is sealed")

    def _atomic_write(
        self,
        filename: str,
        write_fn,  # callable(Path) -> None that writes the content
    ) -> Path:
        """
        Write to a temporary sibling, fsync, measure the digest, then rename. Crash-safe: either the old file remains
        (or no file), or the new file is complete.
        """
        self._require_open()
        _require_name(filename)
        dst = self.output_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dst.name}.",
            suffix=".tmp",
            dir=str(dst.parent),
        )
        os.close(tmp_fd)
        tmp = Path(tmp_name)
        try:
            write_fn(tmp)
            # Ensure data is on disk before rename
            # Windows: os.fsync requires a writable fd (_commit requires write access)
            with open(tmp, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            measured = _file_digest(tmp)
            os.replace(tmp, dst)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        if filename not in self._written:
            self._artefacts.append(filename)
        self._written[filename] = measured
        return dst

    def _write_json(self, filename: str, payload: Any) -> Path:
        def _write(path: Path) -> None:
            path.write_text(dump_strict_json(payload, artifact=filename), encoding="utf-8", newline="\n")

        return self._atomic_write(filename, _write)

    def _write_parquet(self, filename: str, frame: pd.DataFrame) -> Path:
        def _write(path: Path) -> None:
            frame.to_parquet(path, index=False, compression="zstd")

        return self._atomic_write(filename, _write)

    # ── Generic artefacts ──────────────────────────────────────────────────

    def save_json(self, filename: str, payload: Any) -> Path:
        """Strict JSON (no stringified numbers, no NaN literals)."""
        return self._write_json(filename, payload)

    def save_table_csv(self, filename: str, frame: pd.DataFrame, *, index: bool = True) -> Path:
        def _write(path: Path) -> None:
            frame.to_csv(path, index=index, lineterminator="\n")

        return self._atomic_write(filename, _write)

    def save_table_parquet(self, filename: str, frame: pd.DataFrame) -> Path:
        return self._write_parquet(filename, frame)

    def register_external(self, relative_path: str) -> None:
        """Declare a file written by another component inside output_dir (a saved model, a split parquet); it is
        measured at completion. Nothing outside output_dir can be part of the run."""
        self._require_open()
        _require_name(relative_path)
        if not (self.output_dir / relative_path).is_file():
            raise ContractError(f"{relative_path}: not a file inside {self.output_dir}")
        if relative_path not in self._external and relative_path not in self._written:
            self._external.append(relative_path)

    # ── Manifest ───────────────────────────────────────────────────────────

    def save_manifest(
        self,
        git_sha: str,
        versions: dict[str, str],
        config: dict[str, Any],
    ) -> Path:
        """The run's configuration record. STRICT since set 1b: it used `json.dumps(default=str)`, which turned NumPy
        integers into strings and anything unforeseen into plausible text."""
        manifest = {
            "run_id": self.run_id,
            "ablation": self.ablation,
            "git_sha": git_sha,
            "versions": versions,
            "config": config,
        }
        dst = self._write_json("manifest.json", manifest)
        logger.info("Manifest written: %s", dst)
        return dst

    # ── Out-of-fold predictions (training) ────────────────────────────────

    def save_oof_predictions(self, oof: Any) -> Path:
        """The out-of-fold table WITH its composed and verified original training-split identity.

        `oof` must be evaluation.oof_identity.OOFIdentity (compose_oof_identity). A bare DataFrame is refused: a frame
        with `variant_id`, `fold` and `label` columns proved nothing about which training rows its predictions belong
        to (Run 15's `_train_row_idx` was a position, joinable only through the calibration carve).
        """
        from genomic_variant_classifier.evaluation.oof_identity import OOFIdentity

        if type(oof) is not OOFIdentity:
            raise ContractError("an OOFIdentity (evaluation.oof_identity.compose_oof_identity) is required; a bare "
                                "DataFrame carries no verified training-row identity")
        dst = self._write_parquet("oof_predictions.parquet", oof.frame)
        self._write_json("oof_identity.json", oof.evidence)
        self._oof_evidence = oof.evidence
        logger.info("OOF predictions: %d rows, %d members -> %s", len(oof.frame), len(oof.members), dst)
        return dst

    # ── Split predictions ──────────────────────────────────────────────────

    def save_split_predictions(self, scored: Any, labels: Any, meta: Optional[pd.DataFrame], *, role: str) -> Path:
        """`<role>_predictions.parquet` from ONE bound capture: identities, label, the selected score and every member
        score, all from the same blocks; the block descriptors go to `<role>_prediction_blocks.json`."""
        from genomic_variant_classifier.evaluation.split_identity import SPLIT_ROLES, binary_labels

        _require_capture(scored)
        if role not in SPLIT_ROLES:
            raise ContractError(f"unknown split role {role!r}")
        rows = scored.result.rows
        y = binary_labels(labels, len(rows))
        out = pd.DataFrame({
            "source_observation_id": [r.source_observation_id for r in rows.rows],
            "prediction_unit_id": [r.prediction_unit_id for r in rows.rows],
            "variant_identity": [r.variant_identity for r in rows.rows],
            "label": y,
            "selected_score": np.asarray(scored.result.positive_probability(), dtype=np.float64),
        })
        for name, block in zip(scored.member_names, scored.members):
            out[f"{name}_prob"] = np.asarray(block.positive_probability(), dtype=np.float64)
        if meta is not None:
            if len(meta) != len(out):
                raise ContractError(f"metadata rows {len(meta)} differ from scored rows {len(out)}")
            for col in ("variant_id", "gene_symbol", "consequence", "chrom", "pos", "ref", "alt"):
                if col in meta.columns:
                    out[col] = meta[col].to_numpy()
        descriptor = {
            "role": role, "split_sha256": scored.split_sha256, "n_rows": len(rows), "rows_sha256": rows.digest,
            "result_block": scored.result.descriptor(), "result_block_sha256": scored.result.digest,
            "member_blocks": {name: {"sha256": b.digest, "descriptor": b.descriptor()}
                              for name, b in zip(scored.member_names, scored.members)},
            "labels_int64_sha256": hashlib.sha256(y.astype("<i8").tobytes()).hexdigest(),
        }
        self._bind_predictor(scored)
        dst = self._write_parquet(f"{role}_predictions.parquet", out)
        self._write_json(f"{role}_prediction_blocks.json", descriptor)
        self._blocks[role] = {"split_sha256": scored.split_sha256, "rows_sha256": rows.digest,
                              "n_rows": len(rows), "result_block_sha256": scored.result.digest,
                              "member_block_sha256": {n: b.digest for n, b in zip(scored.member_names,
                                                                                  scored.members)}}
        logger.info("%s predictions: %d rows x %d cols -> %s", role, len(out), out.shape[1], dst)
        return dst

    def save_test_predictions(self, scored: Any = None, labels: Any = None, meta: Optional[pd.DataFrame] = None,
                              *args: Any, **legacy: Any) -> Path:
        """The test split's predictions from a bound capture. The pre-1b signature (y_test, proba, base_probs, meta)
        took bare arrays whose identity and producer nobody could check; it is refused, positionally or by keyword."""
        if args or legacy or not hasattr(scored, "result"):
            raise ContractError("save_test_predictions(y_test, proba, base_probs, meta) took bare arrays and is "
                                "refused since package 1, set 1b; pass the bound capture: "
                                "save_test_predictions(ensemble.score_split(...), y_test, meta_test)")
        return self.save_split_predictions(scored, labels, meta, role="test")

    def _bind_predictor(self, scored: Any) -> None:
        described = dict(scored.combiner.describe())
        if self._predictor is None:
            self._predictor = described
        elif self._predictor["combiner_sha256"] != described["combiner_sha256"]:
            raise ContractError("two captures in one run come from different predictors")

    # ── Evaluation report (JSON) ───────────────────────────────────────────

    def save_eval_report(self, report: Any) -> Path:
        """
        Accepts a ClinicalEvaluator EvaluationReport dataclass OR a plain dict.
        Both are serialised to JSON.
        """
        # ONE SERIALISER, NOT TWO (2026-07-28, CI-u-1).
        #
        # This used `asdict`, which walks the dataclass and BYPASSES the report's
        # own `to_serializable`. Commit 3a introduced that method precisely
        # because `asdict` cannot carry `result_kind` and does not normalise a
        # refused result's non-finite value -- and this writer was never updated.
        #
        # Measured 2026-07-28 on one healthy report: the two encodings differ in
        # `metric_results`, where `asdict` omits `result_kind` entirely and leaves
        # `status` as a MetricStatus enum object rather than its value. The
        # comment below claimed they were byte-identical. That was true when
        # written on 2026-07-26 and became false the moment `to_serializable`
        # existed -- a claim in a comment is not a check, and this one silently
        # stopped being true.
        if hasattr(report, "to_serializable"):
            payload = report.to_serializable()
        elif is_dataclass(report):
            payload = asdict(report)
        elif isinstance(report, dict):
            payload = report
        else:
            raise TypeError(f"Unsupported report type: {type(report)}")

        def _write(path: Path) -> None:
            # STRICT since 2026-07-26: no `default=str` (it rendered NumPy
            # integers as JSON strings) and no bare NaN literals.
            path.write_text(dump_strict_json(payload, artifact=str(path)),
                            encoding="utf-8", newline="\n")

        dst = self._atomic_write("eval_report.json", _write)
        logger.info("Eval report: %s", dst)
        return dst

    # ── Calibration ────────────────────────────────────────────────────────

    def save_calibration(self, scored: Any, labels: Any, *, report: Any = None,
                         meta: Optional[pd.DataFrame] = None, n_bins: int = 10) -> Path:
        """Reliability points of the selected score (global and per consequence) with real bin counts, and -- in a
        SEPARATE table -- the per-consequence performance rows the evaluator reported.

        Until set 1b one `calibration.parquet` mixed both, writing a consequence's AUROC into `mean_pred`, its
        prevalence into `frac_pos` and n = -1 for every global bin. When `report` is given, its global calibration
        curve must equal the points computed here from the same bound scores (non-empty bins), or nothing is written.
        """
        from genomic_variant_classifier.evaluation.split_identity import binary_labels

        _require_capture(scored)
        p = np.asarray(scored.result.positive_probability(), dtype=np.float64)
        y = binary_labels(labels, len(p))
        frames = []
        glob = calibration_points(p, y, n_bins=n_bins)
        glob.insert(0, "scope", "global")
        frames.append(glob)
        if report is not None:
            nonempty = glob[glob["n"] > 0]
            reported_pred = list(getattr(report, "calibration_mean_pred", None) or [])
            reported_true = list(getattr(report, "calibration_frac_pos", None) or [])
            if (reported_pred or reported_true) and (
                    [float(v) for v in nonempty["mean_predicted"]] != [float(v) for v in reported_pred]
                    or [float(v) for v in nonempty["fraction_positive"]] != [float(v) for v in reported_true]):
                raise ContractError("the report's calibration curve differs from the points of the bound scores; "
                                    "the report was computed on other scores, and nothing is written")
        if meta is not None and "consequence" in meta.columns:
            if len(meta) != len(p):
                raise ContractError("metadata rows differ from scored rows")
            consequence = meta["consequence"].fillna("<missing>").astype(str).to_numpy()
            for value in sorted(set(consequence.tolist())):
                mask = consequence == value
                pts = calibration_points(p[mask], y[mask], n_bins=n_bins)
                pts.insert(0, "scope", f"consequence:{value}")
                frames.append(pts)
        points = pd.concat(frames, ignore_index=True)
        points["strategy"] = "quantile"
        points["score"] = "selected_convex_blend"
        points["result_block_sha256"] = scored.result.digest
        dst = self._write_parquet("calibration_points.parquet", points)

        if report is not None:
            rows = []
            for c in getattr(report, "consequence_breakdown", None) or []:
                d = asdict(c) if is_dataclass(c) else dict(c)
                auroc = d.get("auroc")
                auprc = d.get("auprc")
                rows.append({"consequence": str(d.get("consequence")), "n_total": int(d.get("n_total")),
                             "n_pathogenic": int(d.get("n_pathogenic")),
                             "auroc": float(auroc) if auroc is not None and np.isfinite(auroc) else None,
                             "auprc": float(auprc) if auprc is not None and np.isfinite(auprc) else None,
                             "prevalence": float(d.get("prevalence"))})
            perf = pd.DataFrame(rows, columns=["consequence", "n_total", "n_pathogenic", "auroc", "auprc",
                                               "prevalence"])
            self._write_parquet("consequence_performance.parquet", perf)
        logger.info("Calibration points: %d rows -> %s", len(points), dst)
        return dst

    # ── Explanation of the selected predictor ──────────────────────────────

    def save_member_score_attributions(self, scored: Any, *, background: Any) -> Path:
        """Exact member-score accounting of the selected blend (evaluation.blend_attribution), with the closure
        residual against the published score and the declared background capture."""
        from genomic_variant_classifier.evaluation.blend_attribution import member_score_attribution

        _require_capture(scored)
        _require_capture(background)
        attribution = member_score_attribution(scored, background)
        dst = self._write_parquet("member_score_attributions.parquet", attribution.frame)
        self._write_json("member_score_attributions.json", attribution.descriptor)
        logger.info("Member-score attributions: %d rows -> %s", len(attribution.frame), dst)
        return dst

    def save_shap_values(self, *args: Any, **kwargs: Any) -> Path:
        """RETIRED in package 1, set 1b. It explained ensemble.meta_learner -- the logistic research candidate, not the
        selected predictor -- with SHAP's LinearExplainer. Its replacement is save_member_score_attributions; a
        feature-level explanation of the blend is unavailable (blend_attribution.FEATURE_LEVEL_EXPLANATION_UNAVAILABLE).
        Refused before anything is imported, computed or written."""
        from genomic_variant_classifier.evaluation.blend_attribution import FEATURE_LEVEL_EXPLANATION_UNAVAILABLE

        raise ContractError("RunArtifactWriter.save_shap_values is retired (it explained the logistic research "
                            "candidate). " + FEATURE_LEVEL_EXPLANATION_UNAVAILABLE)

    # ── Permutation importance ─────────────────────────────────────────────

    def save_permutation_importance(
        self,
        ensemble: Any,
        X_tab_test: pd.DataFrame,
        X_seq_test: "SequenceWindows | None",
        y_test: pd.Series | np.ndarray,
        n_repeats: int = 5,
        sample_size: int = 50_000,
        seed: int = 42,
    ) -> Path:
        """
        Permutation importance of each tabular feature for the SELECTED predictor (the declared convex blend,
        VariantEnsemble.predict_proba): the drop in AUROC when one column is permuted. (Until set 1b this docstring
        said "the stacker's ensemble output"; the method has always scored predict_proba, which since 2026-11-07 is
        only the blend.) Subsamples for tractability.

        Schema: feature, mean_drop, std_drop, n_repeats, baseline
        """
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y_test)
        rng = np.random.default_rng(seed)
        n = len(y)
        if n > sample_size:
            idx = rng.choice(n, size=sample_size, replace=False)
            X_tab_sub = X_tab_test.iloc[idx].reset_index(drop=True)
            # A slice of an attachment is still an attachment: same provenance, sliced
            # masks, recomputed counts. `.iloc` would have returned a bare frame, which
            # the ensemble's gate refuses as provenance-less.
            X_seq_sub = None if X_seq_test is None else X_seq_test.subset(idx)
            y_sub = y[idx]
        else:
            X_tab_sub, X_seq_sub, y_sub = X_tab_test, X_seq_test, y

        baseline = roc_auc_score(
            y_sub, ensemble.predict_proba(X_tab_sub, X_seq_sub)[:, 1]
        )
        feature_cols = list(X_tab_sub.columns)
        rows: list[dict[str, Any]] = []
        for col in feature_cols:
            drops = []
            for _ in range(n_repeats):
                X_perm = X_tab_sub.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                auc_perm = roc_auc_score(
                    y_sub, ensemble.predict_proba(X_perm, X_seq_sub)[:, 1]
                )
                drops.append(baseline - auc_perm)
            rows.append(
                {
                    "feature": col,
                    "mean_drop": float(np.mean(drops)),
                    "std_drop": float(np.std(drops)),
                    "n_repeats": n_repeats,
                    "baseline": float(baseline),
                }
            )

        df = pd.DataFrame(rows).sort_values("mean_drop", ascending=False)
        dst = self._write_parquet("feature_importance.parquet", df)
        logger.info("Permutation importance: %d features -> %s", len(df), dst)
        return dst

    # ── Graph stats (GNN only) ─────────────────────────────────────────────

    def save_graph_stats(self, stats: dict[str, Any]) -> Path:
        """Writes the graph_stats.json required for any GNN-live run (strict JSON)."""
        required = {"node_count", "edge_count"}
        missing = required - set(stats.keys())
        if missing:
            raise ValueError(f"graph_stats missing required keys: {missing}")

        dst = self._write_json("graph_stats.json", stats)
        logger.info(
            "Graph stats: %d nodes / %d edges -> %s",
            stats["node_count"],
            stats["edge_count"],
            dst,
        )
        return dst

    # ── Completion (gate G8) ────────────────────────────────────────────────

    def complete(self, required: Iterable[str], *, deliverables: Optional[Mapping[str, Any]] = None,
                 extra: Optional[Mapping[str, Any]] = None) -> Path:
        """Validate every required output and publish `completion_manifest.json` LAST.

        Required names must have been written by THIS writer (and still hold the bytes it measured) or registered as
        external files; every written and registered file is listed with its digest. The selected predictor and the
        prediction-block digests come from the captures this writer saved; a run that saved none is not complete.
        After this call the writer is sealed.
        """
        self._require_open()
        required = [_require_name(r) for r in required]
        if not required:
            raise ContractError("a completion must name its required outputs")
        missing = [r for r in required if r not in self._written and r not in self._external]
        if missing:
            raise IncompleteRunError(f"required outputs were not produced by this run: {missing}")
        if self._predictor is None or not self._blocks:
            raise IncompleteRunError("no bound prediction capture was saved; the run has no selected-predictor result")
        entries: dict[str, dict] = {}
        for name in self._artefacts:
            digest, size = _file_digest(self.output_dir / name)
            if (digest, size) != self._written[name]:
                raise IncompleteRunError(f"{name} changed after this writer wrote it")
            entries[name] = {"sha256": digest, "bytes": size, "origin": "writer"}
        for name in self._external:
            path = self.output_dir / name
            if not path.is_file():
                raise IncompleteRunError(f"registered file {name} is missing")
            digest, size = _file_digest(path)
            entries[name] = {"sha256": digest, "bytes": size, "origin": "external"}
        manifest = {
            "schema": COMPLETION_SCHEMA,
            "status": "complete",
            "run_id": self.run_id,
            "ablation": self.ablation,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "required": sorted(required),
            "artefacts": entries,
            "predictor": self._predictor,
            "prediction_blocks": self._blocks,
            "oof_identity": self._oof_evidence,
            "deliverables": dict(deliverables or {}),
            "retired_manifest": self.retired_manifest,
            "extra": dict(extra or {}),
        }
        payload = dump_strict_json(manifest, artifact=COMPLETION_MANIFEST)

        def _write(path: Path) -> None:
            path.write_text(payload, encoding="utf-8", newline="\n")

        dst = self._atomic_write(COMPLETION_MANIFEST, _write)
        self._completed = dst
        logger.info("Run COMPLETE: %d artefacts bound in %s", len(entries), dst)
        return dst

    @property
    def completion_manifest_sha256(self) -> str:
        if self._completed is None:
            raise IncompleteRunError("the run has not been completed")
        return self._written[COMPLETION_MANIFEST][0]

    # ── Ablation results aggregator (written by top-level driver) ─────────

    def append_ablation_row(
        self,
        master_path: Path,
        row: dict[str, Any],
    ) -> None:
        """
        Appends a COMPLETED ablation's headline metrics to the shared ablation_results.parquet at the RUN level (one
        level up from the ablation-specific output_dir). Idempotent: re-running the same ablation replaces the row for
        that ablation name. Since set 1b the replacement is atomic (temporary sibling, fsync, rename) and the row
        carries the completion manifest's digest; an incomplete run is refused.
        """
        manifest_sha = self.completion_manifest_sha256
        master_path = Path(master_path)
        row = dict(row)
        row.setdefault("ablation", self.ablation)
        if row["ablation"] != self.ablation:
            raise ContractError(f"row ablation {row['ablation']!r} is not this writer's {self.ablation!r}")
        row["completion_manifest_sha256"] = manifest_sha
        new_df = pd.DataFrame([row])
        if master_path.exists():
            existing = pd.read_parquet(master_path)
            existing = existing[existing["ablation"] != self.ablation]
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        master_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{master_path.name}.", suffix=".tmp", dir=str(master_path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            merged.to_parquet(tmp, index=False, compression="zstd")
            with open(tmp, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, master_path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        logger.info(
            "Ablation results appended: %s rows -> %s", len(merged), master_path
        )

    # ── Convenience ────────────────────────────────────────────────────────

    @property
    def artefacts(self) -> list[str]:
        """Filenames (relative to output_dir) written so far."""
        return list(self._artefacts)
