"""
model_insights_detector.py -- Monzia Moodie

Pure model-output analysis over a run's oof_predictions.parquet (RunArtifactWriter schema: variant_id,
gene_symbol, fold, label, <model>_prob..., ensemble_prob). Computes per-model metrics with the SAME sklearn
functions as evaluation/evaluator.py so the numbers are consistent, flags integrity risks (leakage-suspicion
via near-perfect AUROC, degenerate OOF, AUROC/AUPRC optimism gap, non-gene-disjoint folds), and ranks models
by MCC (balanced) -- deliberately NOT by AUROC.

GUARDRAIL (scientific integrity > metrics): this module reports DIAGNOSTICS and integrity FLAGS only. It never
recommends hyperparameters or tuning toward a higher metric -- a near-perfect AUROC is surfaced as a leakage
SUSPICION (cross-ref the n_pathogenic_in_gene gene-prevalence-memorization lesson), not a trophy. No BaseAgent /
no SharedState -> unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

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


def per_model_metrics(oof: pd.DataFrame) -> list[ModelMetric]:
    if "label" not in oof.columns:
        raise ValueError("oof missing required 'label' column")
    y = oof["label"].to_numpy()
    n, n_pos = len(y), int(np.asarray(y).sum())
    out: list[ModelMetric] = []
    for col in prob_columns(oof):
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


def discover_latest_run(outputs_root: str = "outputs") -> Path | None:
    root = Path(outputs_root)
    if not root.exists():
        return None
    cands = sorted(root.rglob("oof_predictions.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent if cands else None


def analyze(oof: pd.DataFrame) -> dict:
    metrics = per_model_metrics(oof)
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
