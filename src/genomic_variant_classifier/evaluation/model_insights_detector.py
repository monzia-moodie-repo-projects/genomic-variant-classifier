"""
model_insights_detector.py -- Monzia Moodie

Pure model-output analysis over a COMPLETED run's oof_predictions.parquet. Since package 1, set 1b (2026-11-08)
that file carries the composed original training-split identity (evaluation.oof_identity.OOF_IDENTITY_COLUMNS:
train_split_position, source_observation_id, prediction_unit_id, fold, label, gene_symbol, variant_id,
variant_identity) and one `<member>_prob` column per fitted member -- and NO blend column (the stored weights were
fitted on these rows). Only a run whose completion manifest verifies (prediction_artifacts.verify_completed_run) is
analysed; an OOF file without one is reported as incomplete, never read as a result. (Run 15's real OOF file lacks
`label` and keys its rows by a position, so the old reader could not have analysed it correctly.)

Computes per-model metrics with the SAME sklearn functions as evaluation/evaluator.py so the numbers are consistent,
flags integrity risks (leakage-suspicion via near-perfect AUROC, degenerate OOF, AUROC/AUPRC optimism gap,
non-gene-disjoint folds), and ranks models by MCC (balanced) -- deliberately NOT by AUROC.

GUARDRAIL (scientific integrity > metrics): this module reports DIAGNOSTICS and integrity FLAGS only. It never
recommends hyperparameters or tuning toward a higher metric -- a near-perfect AUROC is surfaced as a leakage
SUSPICION (cross-ref the n_pathogenic_in_gene gene-prevalence-memorization lesson), not a trophy. No BaseAgent /
no SharedState -> unit-testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from genomic_variant_classifier.evaluation.oof_identity import OOF_IDENTITY_COLUMNS, member_column
from genomic_variant_classifier.evaluation.prediction_artifacts import IncompleteRunError, verify_completed_run

logger = logging.getLogger(__name__)

LEAKAGE_AUROC = 0.99    # at/above -> leakage-suspicion (run a gene-disjoint ablation before trusting)
DEGENERATE_STD = 1e-6   # prob-column std at/below -> degenerate (model not discriminating)
GAP_THRESHOLD = 0.15    # (auroc - auprc) at/above -> class-imbalance optimism


@dataclass
class ModelMetric:
    model: str
    auroc: float | None
    auprc: float | None
    mcc: float | None
    brier: float | None
    n: int
    n_pos: int
    note: str = ""


def prob_columns(oof: pd.DataFrame) -> list[str]:
    return [c for c in oof.columns if c.endswith("_prob")]


def per_model_metrics(oof: pd.DataFrame, members: list[str] | None = None) -> list[ModelMetric]:
    """One row per member column. With `members` (from the run's OOF identity) exactly those columns are read, in
    that order, and a missing one is an error; without it every `*_prob` column is read."""
    if "label" not in oof.columns:
        raise ValueError("oof missing required 'label' column")
    y = oof["label"].to_numpy()
    n, n_pos = len(y), int(np.asarray(y).sum())
    out: list[ModelMetric] = []
    if members is not None:
        columns = [member_column(m) for m in members]
        absent = [c for c in columns if c not in oof.columns]
        if absent:
            raise ValueError(f"oof lacks the member columns {absent}")
    else:
        columns = prob_columns(oof)
    for col in columns:
        model = col[: -len("_prob")]
        p = oof[col].to_numpy(dtype=float)
        if np.std(p) <= DEGENERATE_STD or n_pos == 0 or n_pos == n:
            out.append(ModelMetric(model, None, None, None, None, n, n_pos,
                                   "degenerate: constant probs or single-class labels"))
            continue
        out.append(ModelMetric(
            model,
            float(roc_auc_score(y, p)),
            float(average_precision_score(y, p)),
            float(matthews_corrcoef(y, (p >= 0.5).astype(int))),
            float(np.mean((p - y) ** 2)),
            n, n_pos,
        ))
    return out


def integrity_flags(metrics: list[ModelMetric]) -> list[str]:
    flags: list[str] = []
    for m in metrics:
        if m.auroc is None:
            flags.append(f"DEGENERATE_OOF[{m.model}]: {m.note}")
            continue
        if m.auroc >= LEAKAGE_AUROC:
            flags.append(
                f"LEAKAGE_SUSPICION[{m.model}]: AUROC={m.auroc:.4f} >= {LEAKAGE_AUROC} -- near-perfect on a hard "
                f"biological task; run a gene-disjoint / n_pathogenic_in_gene ablation before trusting it.")
        if m.auprc is not None and (m.auroc - m.auprc) >= GAP_THRESHOLD:
            flags.append(
                f"AUROC_AUPRC_GAP[{m.model}]: AUROC={m.auroc:.4f} vs AUPRC={m.auprc:.4f} -- class-imbalance "
                f"optimism; prefer AUPRC/MCC for ranking.")
    return flags


def gene_disjoint_check(oof: pd.DataFrame) -> tuple[bool, str]:
    if "gene_symbol" not in oof.columns or "fold" not in oof.columns:
        return True, "no gene_symbol/fold columns -- gene-disjoint check skipped"
    fold_genes = {f: set(g["gene_symbol"]) for f, g in oof.groupby("fold")}
    folds = sorted(fold_genes)
    overlaps = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            shared = fold_genes[folds[i]] & fold_genes[folds[j]]
            if shared:
                overlaps.append((folds[i], folds[j], len(shared)))
    if overlaps:
        return False, (f"NOT gene-disjoint: {len(overlaps)} fold-pair(s) share genes "
                       f"(e.g. folds {overlaps[0][0]}/{overlaps[0][1]} share {overlaps[0][2]}) -- LEAKAGE RISK.")
    return True, "folds are gene-disjoint"


def rank_by_balanced(metrics: list[ModelMetric]) -> list[str]:
    ranked = [m for m in metrics if m.mcc is not None]
    ranked.sort(key=lambda m: m.mcc, reverse=True)  # MCC, not AUROC -- guardrail
    return [m.model for m in ranked]


def discover_runs(outputs_root: str = "outputs") -> tuple[Path | None, list[dict]]:
    """(the most recently COMPLETED run directory holding an OOF table, or None; every OOF-bearing directory that is
    NOT complete evidence, with the reason). Completion order is the manifest's own `completed_utc`, never a file
    modification time."""
    root = Path(outputs_root)
    if not root.exists():
        return None, []
    complete: list[tuple[str, str, Path]] = []
    incomplete: list[dict] = []
    for oof_path in sorted(root.rglob("oof_predictions.parquet")):
        run_dir = oof_path.parent
        try:
            manifest = verify_completed_run(run_dir)
        except IncompleteRunError as exc:
            incomplete.append({"run_dir": str(run_dir), "reason": str(exc)})
            continue
        if "oof_predictions.parquet" not in manifest["artefacts"]:
            incomplete.append({"run_dir": str(run_dir), "reason": "the completion manifest does not list the OOF table"})
            continue
        complete.append((str(manifest["completed_utc"]), str(run_dir), run_dir))
    if not complete:
        return None, incomplete
    complete.sort()
    return complete[-1][2], incomplete


def discover_latest_run(outputs_root: str = "outputs") -> Path | None:
    """The most recently completed run directory holding an OOF table, or None (incomplete runs never qualify)."""
    return discover_runs(outputs_root)[0]


def load_completed_oof(run_dir: Path | str) -> tuple[pd.DataFrame, dict]:
    """The OOF table of a VERIFIED completed run and its manifest; the identity schema and member list are checked."""
    run_dir = Path(run_dir)
    manifest = verify_completed_run(run_dir)
    evidence = manifest.get("oof_identity")
    if not isinstance(evidence, dict) or "members" not in evidence:
        raise IncompleteRunError(f"{run_dir}: the completion manifest carries no OOF identity evidence")
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    expected = list(OOF_IDENTITY_COLUMNS) + [member_column(m) for m in evidence["members"]]
    if list(oof.columns) != expected:
        raise IncompleteRunError(f"{run_dir}: OOF columns {list(oof.columns)} differ from the declared {expected}")
    return oof, manifest


def analyze(oof: pd.DataFrame, members: list[str] | None = None) -> dict:
    metrics = per_model_metrics(oof, members)
    flags = integrity_flags(metrics)
    disjoint, msg = gene_disjoint_check(oof)
    if not disjoint:
        flags.insert(0, f"GENE_DISJOINT_VIOLATION: {msg}")
    return {
        "metrics": metrics,
        "flags": flags,
        "ranking_by_mcc": rank_by_balanced(metrics),
        "gene_disjoint": disjoint,
        "gene_disjoint_msg": msg,
    }
