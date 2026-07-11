"""End-to-end conformal calibration on a verified real substrate (from-scratch numpy).

Wires the proven conformal modules (scores/split/mondrian/grouped/coverage) to a real
prediction substrate whose score<->label join has been INDEPENDENTLY VERIFIED. The verification
gate is a NON-BYPASSABLE precondition: if the per-score AUROC is at/near chance (a broken join,
as happened with run15_baseline), calibration ABORTS. This makes silent calibration on mis-joined
labels impossible by construction.

Scientific framing: for the ablation_run15 substrate the base-model AUROC is ~0.998, reflecting a
known feature-leakage concern. This module is a METHODS DEMONSTRATION of the conformal layer's
coverage guarantee; it is explicitly NOT a clinical performance claim. Every artifact is labeled
'pre-correction, methodological only, not clinically final'.

Gene-disjoint calibration: the calibration and evaluation splits share NO gene, matching how the
model is deployed (gene-disjoint holdout). Coverage is therefore an honest estimate of coverage
on unseen genes, not on unseen variants of seen genes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .split import SplitConformalClassifier
from .mondrian import MondrianConformalClassifier
from . import coverage as _cov
from ..data.splits import _gene_hash

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

DISCLAIMER = "pre-correction, methodological only, not clinically final"


class AlignmentError(RuntimeError):
    """Raised when the substrate's score<->label join fails the AUROC gate."""


@dataclass
class CalibrationConfig:
    score_col: str = "ensemble_prob"
    label_col: str = "label"
    group_col: str = "gene_symbol"
    stratum_col: str = "consequence"
    alpha: float = 0.1
    cal_frac: float = 0.5           # fraction of GENES assigned to calibration
    seed: int = 42
    auroc_floor: float = 0.90       # the gate; below this -> AlignmentError


@dataclass
class CalibrationResult:
    n_total: int
    n_cal: int
    n_eval: int
    auroc: float
    q_hat_lac: float
    coverage: dict = field(default_factory=dict)
    mondrian_coverage: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def _gene_disjoint_mask(genes: np.ndarray, cal_frac: float, seed: int) -> np.ndarray:
    """Deterministic hash-based assignment of each GENE to calibration (True) or eval (False).
    A gene's fate depends only on (gene, seed), so no gene appears in both splits. Uses the
    project-canonical gene hash (data.splits._gene_hash, full 128-bit MD5 / 2**128) so that the
    conformal calibration split and split_protocol_v2 share one bucketing rule rather than drifting.
    """
    out = np.zeros(len(genes), dtype=bool)
    for i, g in enumerate(genes):
        out[i] = _gene_hash(str(g), seed) < cal_frac
    return out


def _binary_prob_matrix(p: np.ndarray) -> np.ndarray:
    """Turn a 1-D positive-class probability into the (n,2) matrix the conformal API expects."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return np.column_stack([1.0 - p, p])


def load_and_verify(path: str | Path, cfg: CalibrationConfig) -> pd.DataFrame:
    """Load the substrate, assert required columns, and RUN THE GATE. Raises on failure."""
    df = pd.read_parquet(path)
    required = [cfg.score_col, cfg.label_col, cfg.group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"substrate missing required columns: {missing}")
    y = pd.to_numeric(df[cfg.label_col], errors="coerce").values
    p = pd.to_numeric(df[cfg.score_col], errors="coerce").values
    ok = ~(np.isnan(y) | np.isnan(p))
    if ok.sum() < 100 or len(np.unique(y[ok])) != 2:
        raise ValueError("label/score not usable (need >=100 rows and a binary label)")
    # THE GATE
    if roc_auc_score is None:
        raise RuntimeError("sklearn required for the alignment gate")
    auroc = roc_auc_score(y[ok], p[ok])
    if auroc < cfg.auroc_floor:
        raise AlignmentError(
            f"ALIGNMENT GATE FAILED: AUROC({cfg.score_col}, {cfg.label_col}) = {auroc:.4f} "
            f"< floor {cfg.auroc_floor}. The score<->label join is broken; refusing to calibrate.")
    return df.loc[ok].reset_index(drop=True)


def calibrate(path: str | Path, cfg: CalibrationConfig | None = None) -> CalibrationResult:
    cfg = cfg or CalibrationConfig()
    df = load_and_verify(path, cfg)

    y = pd.to_numeric(df[cfg.label_col], errors="coerce").astype(int).values
    p = pd.to_numeric(df[cfg.score_col], errors="coerce").values
    genes = df[cfg.group_col].astype(str).values
    P = _binary_prob_matrix(p)
    auroc = roc_auc_score(y, p)

    # gene-disjoint calibration / evaluation split
    cal_mask = _gene_disjoint_mask(genes, cfg.cal_frac, cfg.seed)
    eval_mask = ~cal_mask
    # guard: both splits non-empty and gene-disjoint
    cal_genes, eval_genes = set(genes[cal_mask]), set(genes[eval_mask])
    if cal_mask.sum() == 0 or eval_mask.sum() == 0:
        raise RuntimeError("gene-disjoint split produced an empty side; adjust cal_frac/seed")
    if cal_genes & eval_genes:
        raise RuntimeError("gene-disjoint invariant violated: gene(s) in both splits")

    # calibrate split-conformal (LAC) on the calibration split
    scc = SplitConformalClassifier(alpha=cfg.alpha, score="lac", seed=cfg.seed)
    scc.fit(P[cal_mask], y[cal_mask])
    sets_eval = scc.predict_set(P[eval_mask])

    # diagnostics on the evaluation split
    strata = (df[cfg.stratum_col].values[eval_mask]
              if cfg.stratum_col in df.columns else None)
    rep = _cov.coverage_report(sets_eval, y[eval_mask], alpha=cfg.alpha,
                               strata=strata, groups=genes[eval_mask])

    # Mondrian class-conditional for comparison (per-class coverage on the rare class)
    mond = MondrianConformalClassifier(alpha=cfg.alpha, score="lac",
                                       group_mode="class", seed=cfg.seed)
    mond.fit(P[cal_mask], y[cal_mask])
    msets = mond.predict_set(P[eval_mask])
    mond_rep = {
        "marginal": _cov.marginal_coverage(msets, y[eval_mask]),
        "per_class": _cov.per_class_coverage(msets, y[eval_mask]).to_dict(),
    }

    return CalibrationResult(
        n_total=len(df), n_cal=int(cal_mask.sum()), n_eval=int(eval_mask.sum()),
        auroc=float(auroc), q_hat_lac=float(scc.q_hat_),
        coverage=rep, mondrian_coverage=mond_rep,
    )
