"""
Ensemble Model Framework for Genomic Variant Classification
============================================================
Implements 8 base classifiers + 1 stacking meta-learner.

Base classifiers:
  1. Random Forest         (sklearn)
  2. XGBoost               (xgboost)
  3. LightGBM              (lightgbm)
  4. SVM (RBF kernel)      (sklearn)
  5. Logistic Regression   (sklearn)
  6. Gradient Boosting     (sklearn)
  7. 1D-CNN                (TensorFlow/Keras)  -- sequence-based
  8. Feedforward NN        (TensorFlow/Keras)  -- tabular features

Meta-learner:
  Logistic Regression stacker trained on OOF predictions

CHANGES FROM PHASE 1:
  - Consolidated src/models/ensemble.py + src/models/variant_ensemble.py
    into this single file (Issue A).
  - Removed unused `k` parameter from encode_sequence; docstring fixed (Bug 7).
  - codon_position removed from TABULAR_FEATURES -- it was always 0 (Issue P).
  - base_estimators cleared after fitting to avoid double-memory (Issue H).
  - from __future__ import annotations added (Issue N).
  - Module-level logging.basicConfig removed (Issue L).
  - VariantEnsemble.fit() / evaluate() now robust when CNN/NN are excluded.

CHANGES -- LOVD integration:
  - lovd_variant_class added to TABULAR_FEATURES (ordinal 0-4)
  - lovd_variant_class added to engineer_features()
  - skip_svm added to EnsembleConfig
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

from genomic_variant_classifier.models.scalable_svm import ScalableSVM

import re as _re
import joblib
import json
from datetime import datetime

_HGVSP_CODON_RE = _re.compile(r"p\.[A-Za-z]{3}(\d+)")


def _parse_codon_position(hgvsp: object) -> int:
    """Extract codon number from HGVSp string. Returns 0 if unparseable."""
    if not hgvsp:
        return 0
    m = _HGVSP_CODON_RE.search(str(hgvsp))
    return int(m.group(1)) if m else 0


logger = logging.getLogger(__name__)

# Run 11 I3: GPU GBDT auto-detection
try:
    import torch as _torch
    _GPU_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _GPU_AVAILABLE = False

try:
    from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier as _CatBoostVC

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    logger.debug("catboost not installed -- catboost base model will be skipped.")

try:
    from genomic_variant_classifier.models.kan import KANClassifier as _KANClassifier

    _KAN_AVAILABLE = True
except ImportError:
    _KAN_AVAILABLE = False
    logger.debug("pykan not installed -- kan base model will be skipped.")

try:
    from genomic_variant_classifier.models.mc_dropout import MCDropoutWrapper as _MCDropoutWrapper
    from genomic_variant_classifier.models.mc_dropout import DeepEnsembleWrapper as _DeepEnsembleWrapper

    _MC_DROPOUT_AVAILABLE = True
except ImportError:
    _MC_DROPOUT_AVAILABLE = False
    logger.debug(
        "mc_dropout deps not available -- mc_dropout/deep_ensemble models will be skipped."
    )
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature definitions (65 features -- must match DataPrepPipeline._engineer_features)
# ---------------------------------------------------------------------------

CONSEQUENCE_SEVERITY: dict[str, int] = {
    "transcript_ablation": 10,
    "splice_acceptor_variant": 9,
    "splice_donor_variant": 9,
    "stop_gained": 9,
    "frameshift_variant": 8,
    "stop_lost": 8,
    "start_lost": 8,
    "transcript_amplification": 7,
    "inframe_insertion": 6,
    "inframe_deletion": 6,
    "missense_variant": 5,
    "protein_altering_variant": 5,
    "splice_region_variant": 4,
    "incomplete_terminal_codon_variant": 3,
    "start_retained_variant": 3,
    "stop_retained_variant": 3,
    "synonymous_variant": 2,
    "coding_sequence_variant": 2,
    "5_prime_UTR_variant": 2,
    "3_prime_UTR_variant": 2,
    "non_coding_transcript_exon_variant": 1,
    "intron_variant": 1,
    "NMD_transcript_variant": 1,
    "upstream_gene_variant": 0,
    "downstream_gene_variant": 0,
    "intergenic_variant": 0,
}

TABULAR_FEATURES = [
    # Allele frequency (6)
    "af_raw",
    "af_log10",
    "af_is_absent",
    "af_is_ultra_rare",
    "af_is_rare",
    "af_is_common",
    # Variant type (7)
    "ref_len",
    "alt_len",
    "len_diff",
    "is_snv",
    "is_insertion",
    "is_deletion",
    "is_indel",
    # Consequence (6)
    "consequence_severity",
    "is_loss_of_function",
    "is_missense",
    "is_synonymous",
    "is_splice",
    "in_coding",
    # Functional scores (9)
    "cadd_phred",
    "sift_score",
    "polyphen2_score",
    "revel_score",
    "phylop_score",
    "gerp_score",
    "alphamissense_score",
    "splice_ai_score",
    "eve_score",
    # Binary flags + meta-score (5)
    "cadd_high",
    "sift_deleterious",
    "polyphen_probably_damaging",
    "revel_pathogenic",
    "n_tools_pathogenic",
    # Gene-level (4)
    "gene_constraint_oe",
    "gene_is_constrained",
    "n_pathogenic_in_gene",
    "gene_has_known_disease",
    # Protein features (2)
    "has_uniprot_annotation",
    "n_known_pathogenic_protein_variants",
    # GTEx (6)
    "gtex_max_tpm",
    "gtex_n_tissues_expressed",
    "gtex_tissue_specificity",
    "gtex_is_eqtl",
    "gtex_min_eqtl_pval",
    "gtex_max_abs_effect",
    # Variant coding context (2)
    "codon_position",
    "dbsnp_af",
    # Gene-disease annotation (3)
    "omim_n_diseases",
    "omim_is_autosomal_dominant",
    "clingen_validity_score",
    # HGMD (2)
    "hgmd_is_disease_mutation",
    "hgmd_n_reports",
    # LOVD (1)
    "lovd_variant_class",
    # Chromosome context (3)
    "is_autosome",
    "is_sex_chrom",
    "is_mitochondrial",
    # GNN-derived (1)
    "gnn_score",
    # RNA splice-context (4)
    "maxentscan_score",
    "dist_to_splice_site",
    "exon_number",
    "is_canonical_splice",
    # Protein structure (4)
    "alphafold_plddt",
    "solvent_accessibility",
    "secondary_structure_context",
    "dist_to_active_site",
    # 1KGP population AF (5)
    "af_1kg_afr",
    "af_1kg_eur",
    "af_1kg_eas",
    "af_1kg_sas",
    "af_1kg_amr",
    # FinnGen (3)
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
    # ESM-2 (1)
    "esm2_delta_norm",
    # gnomAD v4.1 constraint (4)
    "pli_score",
    "loeuf",
    "syn_z",
    "mis_z",
]

PHASE_2_FEATURES: list[str] = []  # All Phase 2 features now active; Phase 3 adds GWAS

PHASE_4_FEATURES: list[str] = [
    "esm2_delta_norm",
    "uncertainty_epistemic",
    "uncertainty_aleatoric",
    "population_1kg_af",
]

SEQUENCE_FEATURES = ["fasta_seq"]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    n_folds: int = 5
    random_state: int = 42
    calibrate: bool = True
    class_weight: str = "balanced"
    n_jobs: int = -1
    model_dir: Path = Path("models/ensemble")
    skip_catboost: bool = False
    skip_svm: bool = False
    skip_kan: bool = False
    skip_mc_dropout: bool = False

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 65 tabular features from a raw variant DataFrame.
    Mirrors DataPrepPipeline._engineer_features() in src/genomic_variant_classifier/data/real_data_prep.py.
    All missing columns are filled with safe defaults.
    """
    feats = pd.DataFrame(index=df.index)

    # Allele frequency (6)
    af = (
        df.get("allele_freq", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0)
    )
    feats["af_raw"] = af
    feats["af_log10"] = np.log10(af + 1e-8)
    feats["af_is_absent"] = (af == 0).astype(int)
    feats["af_is_ultra_rare"] = (af < 0.0001).astype(int)
    feats["af_is_rare"] = ((af >= 0.0001) & (af < 0.001)).astype(int)
    feats["af_is_common"] = (af >= 0.01).astype(int)

    # Variant type (7)
    ref = df.get("ref", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    alt = df.get("alt", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    ref_len = ref.str.len().clip(lower=1)
    alt_len = alt.str.len().clip(lower=1)
    feats["ref_len"] = ref_len
    feats["alt_len"] = alt_len
    feats["len_diff"] = (alt_len - ref_len).abs()
    feats["is_snv"] = ((ref_len == 1) & (alt_len == 1)).astype(int)
    feats["is_insertion"] = (alt_len > ref_len).astype(int)
    feats["is_deletion"] = (ref_len > alt_len).astype(int)
    feats["is_indel"] = (feats["is_insertion"] | feats["is_deletion"]).astype(int)

    # Consequence (6)
    consequence = df.get(
        "consequence", pd.Series([""] * len(df), index=df.index)
    ).fillna("")
    feats["consequence_severity"] = consequence.map(
        lambda c: max(
            (CONSEQUENCE_SEVERITY.get(term, 0) for term in str(c).split("&")),
            default=0,
        )
    )
    feats["is_loss_of_function"] = consequence.str.contains(
        "stop_gained|frameshift|splice_donor|splice_acceptor|start_lost|stop_lost",
        case=False,
        na=False,
    ).astype(int)
    feats["is_missense"] = consequence.str.contains(
        "missense", case=False, na=False
    ).astype(int)
    feats["is_synonymous"] = consequence.str.contains(
        "synonymous", case=False, na=False
    ).astype(int)
    feats["is_splice"] = consequence.str.contains(
        "splice", case=False, na=False
    ).astype(int)
    feats["in_coding"] = consequence.str.contains(
        "missense|synonymous|stop|frameshift|inframe|splice",
        case=False,
        na=False,
    ).astype(int)

    # Functional scores (9)
    score_defaults = {
        "cadd_phred": 15.0,
        "sift_score": 0.5,
        "polyphen2_score": 0.5,
        "revel_score": 0.5,
        "phylop_score": 0.0,
        "gerp_score": 0.0,
        "alphamissense_score": 0.5,
        "splice_ai_score": 0.0,
        "eve_score": 0.5,
    }
    for col, default in score_defaults.items():
        feats[col] = (
            df.get(col, pd.Series([default] * len(df), index=df.index))
            .fillna(default)
            .astype(float)
        )

    # Binary flags + meta-score (5)
    feats["cadd_high"] = (feats["cadd_phred"] >= 20).astype(int)
    feats["sift_deleterious"] = (feats["sift_score"] < 0.05).astype(int)
    feats["polyphen_probably_damaging"] = (feats["polyphen2_score"] >= 0.908).astype(
        int
    )
    feats["revel_pathogenic"] = (feats["revel_score"] >= 0.5).astype(int)
    feats["n_tools_pathogenic"] = (
        feats["cadd_high"]
        + feats["sift_deleterious"]
        + feats["polyphen_probably_damaging"]
        + feats["revel_pathogenic"]
    )

    # Gene-level (4)
    feats["gene_constraint_oe"] = df.get(
        "gene_constraint_oe", pd.Series([1.0] * len(df), index=df.index)
    ).fillna(1.0)
    feats["gene_is_constrained"] = (feats["gene_constraint_oe"] < 0.35).astype(int)
    feats["n_pathogenic_in_gene"] = (
        df.get("n_pathogenic_in_gene", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["gene_has_known_disease"] = (feats["n_pathogenic_in_gene"] > 0).astype(int)

    # Protein features (2)
    feats["has_uniprot_annotation"] = (
        df.get("has_uniprot_annotation", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["n_known_pathogenic_protein_variants"] = (
        df.get(
            "n_known_pathogenic_protein_variants",
            pd.Series([0] * len(df), index=df.index),
        )
        .fillna(0)
        .astype(int)
    )

    # GTEx (6)
    gtex_defaults = {
        "gtex_max_tpm": 0.0,
        "gtex_n_tissues_expressed": 0,
        "gtex_tissue_specificity": 0.0,
        "gtex_is_eqtl": 0,
        "gtex_min_eqtl_pval": 0.0,
        "gtex_max_abs_effect": 0.0,
    }
    for col, default in gtex_defaults.items():
        feats[col] = df.get(col, pd.Series([default] * len(df), index=df.index)).fillna(
            default
        )
    for col in ["gtex_n_tissues_expressed", "gtex_is_eqtl"]:
        feats[col] = feats[col].astype(int)

    # Variant coding context (2)
    feats["codon_position"] = (
        df.get("codon_position", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["dbsnp_af"] = (
        df.get("dbsnp_af", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0)
    )

    # Gene-disease annotation (3)
    feats["omim_n_diseases"] = (
        df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["omim_is_autosomal_dominant"] = (
        df.get("omim_is_autosomal_dominant", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["clingen_validity_score"] = (
        df.get("clingen_validity_score", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(float)
    )

    # HGMD (2)
    feats["hgmd_is_disease_mutation"] = (
        df.get("hgmd_is_disease_mutation", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["hgmd_n_reports"] = (
        df.get("hgmd_n_reports", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )

    # LOVD classification (1) -- ordinal 0-4; 0 = not in LOVD
    feats["lovd_variant_class"] = (
        df.get("lovd_variant_class", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0, upper=4)
    )

    # Chromosome context (3)
    chrom = (
        df.get("chrom", pd.Series(["0"] * len(df), index=df.index))
        .fillna("0")
        .astype(str)
    )
    feats["is_autosome"] = chrom.isin([str(i) for i in range(1, 23)]).astype(int)
    feats["is_sex_chrom"] = chrom.isin(["X", "Y"]).astype(int)
    feats["is_mitochondrial"] = chrom.isin(["MT", "M"]).astype(int)

    # GNN-derived score (1)
    feats["gnn_score"] = (
        df.get("gnn_score", pd.Series([0.5] * len(df), index=df.index))
        .fillna(0.5)
        .astype(float)
        .clip(0.0, 1.0)
    )

    # RNA splice-context features (4)
    feats["maxentscan_score"] = (
        df.get("maxentscan_score", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )
    feats["dist_to_splice_site"] = (
        df.get("dist_to_splice_site", pd.Series([50] * len(df), index=df.index))
        .fillna(50)
        .astype(int)
    )
    feats["exon_number"] = (
        df.get("exon_number", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    feats["is_canonical_splice"] = (
        df.get("is_canonical_splice", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )

    # Protein structure features (4)
    feats["alphafold_plddt"] = (
        df.get("alphafold_plddt", pd.Series([50.0] * len(df), index=df.index))
        .fillna(50.0)
        .astype(float)
        .clip(0.0, 100.0)
    )
    feats["solvent_accessibility"] = (
        df.get("solvent_accessibility", pd.Series([0.5] * len(df), index=df.index))
        .fillna(0.5)
        .astype(float)
        .clip(0.0, 1.0)
    )
    feats["secondary_structure_context"] = (
        df.get("secondary_structure_context", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
        .clip(0, 2)
    )
    feats["dist_to_active_site"] = (
        df.get("dist_to_active_site", pd.Series([100.0] * len(df), index=df.index))
        .fillna(100.0)
        .astype(float)
        .clip(lower=0.0)
    )

    # 1KGP population AF (5)
    for col in ("af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr"):
        feats[col] = (
            df.get(col, pd.Series([0.0] * len(df), index=df.index))
            .fillna(0.0)
            .astype(float)
            .clip(lower=0)
        )

    n_nan = feats.isnull().sum().sum()
    if n_nan > 0:
        logger.warning("%d NaN values in feature matrix -- filling with 0.", n_nan)
        feats = feats.fillna(0.0)

    # FinnGen R10 population AF (three columns)
    for _col, _default in [
        ("finngen_af_fin", 0.0),
        ("finngen_af_nfsee", 0.0),
        ("finngen_enrichment", 1.0),
    ]:
        feats[_col] = (
            df.get(_col, pd.Series([_default] * len(df), index=df.index))
            .fillna(_default)
            .astype(float)
        )

    # ESM-2 delta norm (1) — 0.0 default when model unavailable or non-missense
    feats["esm2_delta_norm"] = (
        df.get("esm2_delta_norm", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )

    # gnomAD v4.1 gene constraint (4) — safe defaults when connector absent
    feats["pli_score"] = (
        df.get("pli_score", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(0.0, 1.0)
    )
    feats["loeuf"] = (
        df.get("loeuf", pd.Series([1.0] * len(df), index=df.index))
        .fillna(1.0)
        .astype(float)
        .clip(0.0, 5.0)
    )
    feats["syn_z"] = (
        df.get("syn_z", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )
    feats["mis_z"] = (
        df.get("mis_z", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )

    return feats.reset_index(drop=True)


def encode_sequence(seq: str, window: int = 101) -> np.ndarray:
    BASES = "ACGT"
    base_map = {b: i for i, b in enumerate(BASES)}
    seq = seq.upper()[:window].ljust(window, "A")
    one_hot = np.zeros((window, len(BASES)), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in base_map:
            one_hot[i, base_map[nuc]] = 1.0
    return one_hot



# ---------------------------------------------------------------------------
# Module-level _CNN1DModule (Run 10 fix for INCIDENT_2026-05-12_cnn1d-pickle-
# nested-class.md). Defined lazily so that `import variant_ensemble` does not
# require torch, but bound to module globals on first use so pickle can resolve
# the class by qualname `genomic_variant_classifier.models.variant_ensemble.
# _CNN1DModule`.
#
# Why not just `import torch.nn as nn` at module top? Per memory: ESM-2 runs
# in stub mode when torch is absent; we preserve that graceful-degradation
# property for downstream connectors that may not need the CNN branch.
#
# Why not nested in _build_model? Run 9 crash: pickle of
# CNN1DClassifier._build_model.<locals>._CNN1D fails because nested local
# classes have no stable qualified name across processes.
# ---------------------------------------------------------------------------
_CNN1DModule = None  # populated by _ensure_cnn1d_module_class() on first use


REF_WIN_COL = "fasta_seq_ref"
ALT_WIN_COL = "fasta_seq_alt"


def _ensure_cnn1d_module_class():
    """Define the siamese-delta _CNN1DModule at module level on first call; idempotent.

    Evolved from the single-window net to a *siamese* encoder shared across the
    reference and alternate windows, with the head fed [e_alt, e_alt - e_ref].
    The delta term forces the model to key on how the variant changes local
    sequence rather than on reference context alone (anti-memorization). Built
    lazily for the same reasons as before: graceful degradation without torch
    and a stable pickle qualname.
    """
    global _CNN1DModule
    if _CNN1DModule is not None:
        return _CNN1DModule

    import torch
    import torch.nn as nn

    class _CNN1DModule(nn.Module):  # noqa: F811 -- intentional global shadow
        def __init__(self, filters, kernel_size, dropout, embed=128):
            super().__init__()
            pad = kernel_size // 2
            self.encoder = nn.Sequential(
                nn.Conv1d(4, filters, kernel_size, padding=pad),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(filters, filters * 2, kernel_size, padding=pad),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Linear(filters * 2, embed),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, ref, alt):
            e_ref = self.encoder(ref)
            e_alt = self.encoder(alt)
            feats = torch.cat([e_alt, e_alt - e_ref], dim=1)
            return self.head(feats).squeeze(-1)  # logits (BCEWithLogitsLoss)

    _CNN1DModule.__name__ = "_CNN1DModule"
    _CNN1DModule.__qualname__ = "_CNN1DModule"
    _CNN1DModule.__module__ = __name__
    globals()["_CNN1DModule"] = _CNN1DModule
    return _CNN1DModule


# ---------------------------------------------------------------------------
# Sklearn-compatible siamese-delta 1D-CNN wrapper
# ---------------------------------------------------------------------------
class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """Siamese delta CNN over (ref, alt) windows.

    X may be a DataFrame with [fasta_seq_ref, fasta_seq_alt] (delta mode) or a
    Series / single 'fasta_seq' column (back-compat: ref == alt, zero delta).
    """

    def __init__(
        self,
        window=101,
        filters=64,
        kernel_size=7,
        dropout=0.3,
        epochs=30,
        batch_size=256,
        learning_rate=1e-3,
        random_state=42,
        embed=128,
        val_fraction=0.1,
        patience=5,
    ):
        self.window = window
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.embed = embed
        self.val_fraction = val_fraction
        self.patience = patience
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _build_model(self):
        import torch  # noqa: F401
        torch.manual_seed(self.random_state)
        cls = _ensure_cnn1d_module_class()
        return cls(self.filters, self.kernel_size, self.dropout, self.embed)

    def _pair_arrays(self, X):
        win = "A" * self.window
        if isinstance(X, pd.DataFrame):
            if REF_WIN_COL in X.columns and ALT_WIN_COL in X.columns:
                ref, alt = X[REF_WIN_COL], X[ALT_WIN_COL]
            elif "fasta_seq" in X.columns:
                ref = alt = X["fasta_seq"]
            else:
                ref = alt = X.iloc[:, 0]
            ref, alt = ref.fillna(win), alt.fillna(win)
        elif isinstance(X, pd.Series):
            ref = alt = X.fillna(win)
        else:
            ref = alt = pd.Series(X).fillna(win)
        r = np.stack([encode_sequence(s, window=self.window) for s in ref]).transpose(0, 2, 1)
        a = np.stack([encode_sequence(s, window=self.window) for s in alt]).transpose(0, 2, 1)
        return r, a  # each (N, 4, window)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        r, a = self._pair_arrays(X)
        ref_t = torch.tensor(r, dtype=torch.float32)
        alt_t = torch.tensor(a, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        n_val = max(1, int(self.val_fraction * len(y_t)))
        idx = torch.randperm(len(y_t))
        v, t = idx[:n_val], idx[n_val:]
        ref_val, alt_val, y_val = ref_t[v].to(device), alt_t[v].to(device), y_t[v].to(device)

        loader = DataLoader(
            TensorDataset(ref_t[t], alt_t[t], y_t[t]),
            batch_size=self.batch_size, shuffle=True,
        )
        self.model_ = self._build_model().to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val, best_state, patience_ctr = float("inf"), None, 0
        for _epoch in range(self.epochs):
            self.model_.train()
            for rb, ab, yb in loader:
                rb, ab, yb = rb.to(device), ab.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self.model_(rb, ab), yb).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(ref_val, alt_val), y_val).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: vv.cpu().clone() for k, vv in self.model_.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu")
        return self

    def predict_proba(self, X):
        import torch

        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        self.model_.eval()
        r, a = self._pair_arrays(X)
        out = np.empty(len(r), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(r), bs):
                rb = torch.tensor(r[i:i + bs], dtype=torch.float32)
                ab = torch.tensor(a[i:i + bs], dtype=torch.float32)
                out[i:i + len(rb)] = torch.sigmoid(self.model_(rb, ab)).numpy()
        return np.column_stack([1.0 - out, out])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # Portable state_dict pickling: rebuilds via the factory on load, so it does
    # not depend on the module-global _CNN1DModule being pre-populated in a fresh
    # process (more robust than relying on qualname alone across machines).
    def __getstate__(self):
        try:
            st = dict(super().__getstate__())
        except AttributeError:
            st = self.__dict__.copy()
        m = st.pop("model_", None)
        st["_model_state"] = None if m is None else {k: v.cpu() for k, v in m.state_dict().items()}
        return st

    def __setstate__(self, st):
        st = dict(st)
        ms = st.pop("_model_state", None)
        try:
            super().__setstate__(st)
        except AttributeError:
            self.__dict__.update(st)
        self.model_ = None
        if ms is not None:
            self.model_ = self._build_model()
            self.model_.load_state_dict(ms)


# ---------------------------------------------------------------------------
# Sklearn-compatible feedforward NN
# ---------------------------------------------------------------------------
class TabularNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.classes_ = np.array([0, 1])

    def _build_model(self, input_dim):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        layers_list = []
        in_dim = input_dim
        for dim in self.hidden_dims:
            layers_list += [
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = dim
        layers_list += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        return nn.Sequential(*layers_list)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_scaled = self.scaler_.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        n_val = max(1, int(0.1 * len(X_t)))
        idx = torch.randperm(len(X_t))
        X_val, y_val = X_t[idx[:n_val]].to(device), y_t[idx[:n_val]].to(device)
        X_tr, y_tr = X_t[idx[n_val:]].to(device), y_t[idx[n_val:]].to(device)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True
        )
        self.model_ = self._build_model(X_scaled.shape[1]).to(device)
        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_fn = nn.BCELoss()

        best_val, best_state, patience_ctr = float("inf"), None, 0
        for _epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb).squeeze(-1), yb).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val).squeeze(-1), y_val).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 8:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu")
        return self

    def predict_proba(self, X):
        import torch

        self.model_.eval()
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            proba = self.model_(X_t).squeeze(-1).numpy()
        return np.column_stack([1 - proba, proba])

    def _predict_proba_single_pass(self, X, seed=None):
        """Single stochastic forward pass with dropout ACTIVE.

        Used by MCDropoutWrapper for MC-dropout uncertainty estimation.
        Unlike predict_proba() which disables dropout via .eval(), this
        method keeps Dropout layers active (.train()) but selectively
        keeps BatchNorm layers in eval (uses running stats, not per-batch
        stats). Without this BatchNorm split, per-batch statistics on the
        inference set would corrupt forward passes for small batches or
        distribution-shifted batches.

        Closes A2 (mc_dropout uncertainty degenerate) per RUN_15_PLAN.md
        B.O3 / C.2.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features. Scaled via self.scaler_ before forward pass.
        seed : int, optional
            Seed for PyTorch's global RNG (controls dropout mask). When
            called by MCDropoutWrapper, this is unique per pass so that
            the n_passes stochastic samples are reproducible but distinct.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0: P(class=0) = 1 - sigmoid output.
            Column 1: P(class=1) = sigmoid output.
        """
        import torch

        if self.model_ is None:
            raise ValueError(
                "TabularNNClassifier not fitted; call .fit() before "
                "_predict_proba_single_pass()."
            )

        if seed is not None:
            torch.manual_seed(int(seed))

        # Selective dropout activation: BatchNorm stays in eval (uses
        # running stats); only Dropout layers get .train() so they
        # produce a stochastic mask.
        self.model_.eval()
        for module in self.model_.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        try:
            X_scaled = self.scaler_.transform(X)
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                proba = self.model_(X_t).squeeze(-1).numpy()
            return np.column_stack([1 - proba, proba])
        finally:
            # Restore full eval mode so subsequent predict_proba() calls
            # are not left with dropout still active.
            self.model_.eval()

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Ensemble orchestrator
# ---------------------------------------------------------------------------


class _IsotonicCalibrator:
    """
    Wraps a pre-fitted base model with isotonic regression calibration.

    Replaces CalibratedClassifierCV(cv="prefit") which was removed from
    sklearn's valid parameter set in versions shipped with Python 3.11 CI
    runners. IsotonicRegression is stable across all sklearn versions.
    """

    def __init__(self, base_model) -> None:
        from sklearn.isotonic import IsotonicRegression

        self._base = base_model
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "_IsotonicCalibrator":
        raw = self._base.predict_proba(X_cal)[:, 1]
        self._iso.fit(raw, y_cal)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self._base.predict_proba(X)[:, 1]
        p = self._iso.predict(raw)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _write_model_manifest(artifact_path):
    import json
    import platform
    import importlib.metadata
    from datetime import datetime, timezone

    artifact_path = Path(artifact_path)
    libraries = [
        "numpy",
        "scikit-learn",
        "catboost",
        "lightgbm",
        "xgboost",
        "joblib",
        "pandas",
        "scipy",
    ]
    manifest = {
        "artifact": artifact_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {lib: importlib.metadata.version(lib) for lib in libraries},
    }
    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


class VariantEnsemble:
    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()
        self._build_estimators()

    def _build_estimators(self) -> None:
        cfg = self.config
        self.base_estimators: dict = {
            "random_forest": RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10,
                eval_metric="auc",
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbosity=0,
                # Run 11 I3: GPU acceleration (auto-detected)
                **({"device": "cuda", "tree_method": "hist"} if _GPU_AVAILABLE else {}),
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbose=-1,
                # Run 11 I3: GPU acceleration (auto-detected)
                # LightGBM: CPU mode (PyPI binary lacks CUDA/OpenCL; Run 11+12 lesson)
            ),
            **(
                {}
                if cfg.skip_svm
                else {
                    "svm": ScalableSVM(
                        mode="nystrom", n_components=1024, gamma="scale",
                        C=1.0, class_weight=cfg.class_weight,
                        random_state=cfg.random_state,
                    ),
                    "svm_bagged_rbf": ScalableSVM(
                        mode="bagged_rbf", svm_max_subsample=15_000, svm_n_bags=25,
                        gamma="scale", C=1.0, class_weight=cfg.class_weight,
                        random_state=cfg.random_state,
                    ),
                }
            ),
            "logistic_regression": LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight=cfg.class_weight,
                random_state=cfg.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=cfg.random_state,
            ),
            **(
                {
                    "catboost": _CatBoostVC(
                        iterations=1000,
                        learning_rate=0.05,
                        depth=6,
                        l2_leaf_reg=3.0,
                        auto_class_weights="Balanced",
                        task_type="GPU" if _GPU_AVAILABLE else "CPU",  # Run 11 I3
                        cat_feature_names=[
                            "gene_symbol",
                            "consequence",
                            "chrom",
                            "review_status",
                        ],
                        random_seed=cfg.random_state,
                        verbose=0,
                    )
                }
                if _CATBOOST_AVAILABLE and not cfg.skip_catboost
                else {}
            ),
            "tabular_nn": TabularNNClassifier(random_state=cfg.random_state),
            "cnn_1d": CNN1DClassifier(random_state=cfg.random_state),
            **(
                {"kan": _KANClassifier(random_state=cfg.random_state)}
                if _KAN_AVAILABLE and not cfg.skip_kan
                else {}
            ),
            **(
                {
                    "mc_dropout": _MCDropoutWrapper(
                        base_estimator=TabularNNClassifier(
                            random_state=cfg.random_state
                        ),
                        random_state=cfg.random_state,
                    )
                }
                if _MC_DROPOUT_AVAILABLE and not cfg.skip_mc_dropout
                else {}
            ),
            **(
                {
                    "deep_ensemble": _DeepEnsembleWrapper(
                        base_estimator=TabularNNClassifier(
                            random_state=cfg.random_state
                        ),
                        random_state=cfg.random_state,
                    )
                }
                if _MC_DROPOUT_AVAILABLE and not cfg.skip_mc_dropout
                else {}
            ),
        }
        self.meta_learner = LogisticRegression(
            C=0.1, max_iter=1000, random_state=cfg.random_state
        )
        self.trained_models_: dict = {}
        self.blend_weights_: Optional[np.ndarray] = None

    @staticmethod
    def _find_blend_weights(oof_preds: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Nelder-Mead convex blend search over OOF predictions.

        Finds non-negative weights w (summing to 1) that maximise validation
        AUROC of the weighted blend oof_preds @ w.  Outperforms a logistic
        regression meta-learner when base model scores are highly correlated
        (the typical case when all models are trained on the same features).

        Falls back gracefully to equal weights if scipy is unavailable or
        optimisation fails to converge.
        """
        from scipy.optimize import minimize

        n_models = oof_preds.shape[1]
        w0 = np.ones(n_models) / n_models

        def neg_auroc(w: np.ndarray) -> float:
            w_abs = np.abs(w)
            total = w_abs.sum()
            if total == 0:
                return 0.0
            blend = oof_preds @ (w_abs / total)
            return -roc_auc_score(y, blend)

        result = minimize(
            neg_auroc,
            w0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5},
        )
        w = np.abs(result.x)
        w /= w.sum()
        return w

    def fit(
        self, X_tab: pd.DataFrame, X_seq: pd.Series, y: pd.Series
    ) -> "VariantEnsemble":
        from sklearn.model_selection import train_test_split as _tts

        y_arr = np.asarray(y)
        logger.info(
            "Training ensemble: %d samples, %d pathogenic.",
            len(y_arr),
            int(y_arr.sum()),
        )

        # Carve out 15% calibration split using index-based split so that
        # X_tab stays a DataFrame (required for CatBoost column-name dispatch).
        idx = np.arange(len(y_arr))
        idx_fit, idx_cal = _tts(
            idx,
            test_size=0.15,
            stratify=y_arr,
            random_state=self.config.random_state,
        )
        X_tab_fit = X_tab.iloc[idx_fit].reset_index(drop=True)
        X_tab_cal = X_tab.iloc[idx_cal].reset_index(drop=True)
        X_seq_fit = X_seq.iloc[idx_fit].reset_index(drop=True)
        X_seq_cal = X_seq.iloc[idx_cal].reset_index(drop=True)
        y_fit = y_arr[idx_fit]
        y_cal = y_arr[idx_cal]

        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        oof_preds = np.zeros((len(y_fit), len(self.base_estimators)))
        # Models that receive post-hoc isotonic calibration.
        _RECALIBRATE = {"xgboost", "lightgbm", "random_forest"}

        for model_idx, (name, model) in enumerate(self.base_estimators.items()):
            logger.info("  Training %s ...", name)

            # Mirror the same 3-way dispatch used in predict_proba().
            if name == "cnn_1d":
                X_input_fit = X_seq_fit
                X_input_cal = X_seq_cal
            elif name == "catboost":
                # Always pass DataFrame — CatBoost needs column names for
                # categorical feature resolution. Handles numeric-only
                # DataFrames correctly when no cat columns are present.
                X_input_fit = X_tab_fit
                X_input_cal = X_tab_cal
            else:
                X_input_fit = X_tab_fit.values
                X_input_cal = X_tab_cal.values

            try:
                oof = cross_val_predict(
                    model,
                    X_input_fit,
                    y_fit,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=1,
                )[:, 1]
            except Exception as exc:
                logger.error("  %s OOF failed: %s — skipping.", name, exc)
                oof_preds[:, model_idx] = 0.5
                continue

            oof_preds[:, model_idx] = oof
            model.fit(X_input_fit, y_fit)

            if name in _RECALIBRATE:
                logger.info("  %s — applying isotonic calibration ...", name)
                cal_model = _IsotonicCalibrator(model)
                cal_model.fit(X_input_cal, y_cal)
                self.trained_models_[name] = cal_model
            else:
                self.trained_models_[name] = model

            logger.info("  %s OOF AUROC: %.4f", name, roc_auc_score(y_fit, oof))

            # === incremental checkpoint patch (INCIDENT_2026-05-23) ===
            try:
                _ckpt_dir = self.config.model_dir
                _ckpt_dir.mkdir(parents=True, exist_ok=True)
                _model_path = _ckpt_dir / f"{name}.joblib"
                _oof_path = _ckpt_dir / f"{name}_oof.npy"
                _meta_path = _ckpt_dir / f"{name}_meta.json"
                joblib.dump(self.trained_models_[name], _model_path, compress=3)
                np.save(_oof_path, oof)
                # Run 11 carried-forward 3.2: OOF row-index sidecar
                # Saves the per-fold prediction-to-row mapping so meta-learner
                # can be reconstructed from saved OOF arrays in disaster recovery.
                _oof_idx_path = _ckpt_dir / f"{name}_oof_indices.npy"
                _fold_indices = [test_idx for _, test_idx in cv.split(X_input_fit, y_fit)]
                np.save(_oof_idx_path, np.concatenate(_fold_indices))
                with open(_meta_path, "w") as _f:
                    json.dump({
                        "name": name,
                        "oof_auroc": float(roc_auc_score(y_fit, oof)),
                        "saved_at_utc": datetime.utcnow().isoformat(),
                        "n_samples": int(len(y_fit)),
                    }, _f, indent=2)
                _size_mb = _model_path.stat().st_size / 1e6
                logger.info("    %s checkpoint saved: %s (%.1f MB)", name, _model_path.name, _size_mb)
            except Exception as _save_exc:
                logger.error("    %s checkpoint FAILED to save: %s", name, _save_exc, exc_info=True)
            # === end incremental checkpoint patch ===
        # Drop columns for any model that failed and was skipped.
        valid_cols = [
            i for i, n in enumerate(self.base_estimators) if n in self.trained_models_
        ]
        oof_preds = oof_preds[:, valid_cols]

        # Expose OOF matrix for Rule-5 artefacts (Run 9+). Downstream
        # writers (scripts/run9_ablations.py) read these attributes.
        self.oof_predictions_ = oof_preds.copy()
        self.oof_fit_indices_ = idx_fit
        self.oof_model_names_ = [
            n for n in self.base_estimators if n in self.trained_models_
        ]

        logger.info(
            "Training meta-learner on %d base-model OOF columns ...", len(valid_cols)
        )
        self.meta_learner.fit(oof_preds, y_fit)

        logger.info("Running Nelder-Mead blend weight search ...")
        self.blend_weights_ = self._find_blend_weights(oof_preds, y_fit)
        self.feature_names_ = list(self.trained_models_.keys())
        logger.info(
            "Blend weights: %s",
            {
                n: round(float(w), 4)
                for n, w in zip(self.feature_names_, self.blend_weights_)
            },
        )
        blend_auroc = roc_auc_score(y_fit, oof_preds @ self.blend_weights_)
        lr_auroc = roc_auc_score(
            y_fit, self.meta_learner.predict_proba(oof_preds)[:, 1]
        )
        logger.info(
            "OOF blend AUROC: %.4f  (LR stacker: %.4f  Δ=%.4f)",
            blend_auroc,
            lr_auroc,
            blend_auroc - lr_auroc,
        )

        # Free unfitted base_estimators from memory (Issue H).
        self.base_estimators.clear()
        return self

    def predict_proba(self, X_tab: pd.DataFrame, X_seq: pd.Series) -> np.ndarray:
        if not self.trained_models_:
            raise RuntimeError("Call fit() before predict_proba().")
        base_preds = np.zeros((len(X_tab), len(self.trained_models_)))
        for i, (name, model) in enumerate(self.trained_models_.items()):
            if name == "cnn_1d":
                X_input = X_seq
            elif name == "catboost":
                X_input = X_tab
            else:
                X_input = X_tab.values
            base_preds[:, i] = model.predict_proba(X_input)[:, 1]

        # Prefer Nelder-Mead convex blend; fall back to LR stacker for
        # models loaded from disk before this change was introduced.
        if self.blend_weights_ is not None:
            blend = base_preds @ self.blend_weights_
            return np.column_stack([1.0 - blend, blend])
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X_tab: pd.DataFrame, X_seq: pd.Series) -> np.ndarray:
        return (self.predict_proba(X_tab, X_seq)[:, 1] > 0.5).astype(int)

    def evaluate(
        self, X_tab: pd.DataFrame, X_seq: pd.Series, y: pd.Series
    ) -> pd.DataFrame:
        y_arr = np.asarray(y)
        results: dict[str, dict] = {}
        for name, model in self.trained_models_.items():
            # Run 10 fix: mirror the same 3-way dispatch used in
            # fit() and predict_proba(). CatBoost requires a DataFrame
            # (column names drive categorical-feature resolution); the
            # previous code called .values for catboost too and would
            # crash on the cb_wrapper's pandas-only input contract.
            if name == "cnn_1d":
                X_input = X_seq
            elif name == "catboost":
                X_input = X_tab
            else:
                X_input = X_tab.values
            proba = model.predict_proba(X_input)[:, 1]
            preds = (proba > 0.5).astype(int)
            results[name] = {
                "auroc": roc_auc_score(y_arr, proba),
                "auprc": average_precision_score(y_arr, proba),
                "f1_macro": f1_score(y_arr, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(
                    y_arr, preds, average="weighted", zero_division=0
                ),
                "mcc": matthews_corrcoef(y_arr, preds),
                "brier": brier_score_loss(y_arr, proba),
            }
        ens_proba = self.predict_proba(X_tab, X_seq)[:, 1]
        ens_preds = (ens_proba > 0.5).astype(int)
        results["ENSEMBLE_STACKER"] = {
            "auroc": roc_auc_score(y_arr, ens_proba),
            "auprc": average_precision_score(y_arr, ens_proba),
            "f1_macro": f1_score(y_arr, ens_preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_arr, ens_preds, average="weighted", zero_division=0
            ),
            "mcc": matthews_corrcoef(y_arr, ens_preds),
            "brier": brier_score_loss(y_arr, ens_proba),
        }
        df = pd.DataFrame(results).T.round(4)
        df = df.sort_values("auroc", ascending=False)
        logger.info("\n%s", df.to_string())
        return df

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the ensemble.

        Run 10 refactor: each base model is pickled into its own joblib
        first; a thin orchestrator joblib then references them by name.
        A single-model pickle failure (e.g. Run 9's CNN1D nested-class
        crash) now degrades gracefully instead of poisoning the whole
        save. The orchestrator records save_errors so downstream load()
        can warn about missing models without crashing.
        """
        import joblib

        path = Path(path or self.config.model_dir / "ensemble.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Per-model checkpoints sit in <path>_models/ next to the orchestrator.
        models_dir = path.parent / f"{path.stem}_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        saved_model_paths: dict = {}
        save_errors: dict = {}
        for name, model in self.trained_models_.items():
            model_path = models_dir / f"{name}.joblib"
            try:
                joblib.dump(model, model_path)
                saved_model_paths[name] = model_path.name
                logger.info("  Saved base model %s -> %s", name, model_path)
            except Exception as exc:
                save_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.error("  FAILED to save base model %s: %s", name, exc)

        orchestrator = {
            "format_version": 2,
            "config": self.config,
            "meta_learner": self.meta_learner,
            "blend_weights_": self.blend_weights_,
            "feature_names_": getattr(self, "feature_names_", None),
            "oof_predictions_": getattr(self, "oof_predictions_", None),
            "oof_fit_indices_": getattr(self, "oof_fit_indices_", None),
            "oof_model_names_": getattr(self, "oof_model_names_", None),
            "saved_model_paths": saved_model_paths,
            "save_errors": save_errors,
            "models_dir_name": models_dir.name,
        }
        try:
            joblib.dump(orchestrator, path)
        except Exception as exc:
            logger.error(
                "Orchestrator save FAILED but %d/%d base models survived at %s. "
                "Error: %s", len(saved_model_paths), len(self.trained_models_),
                models_dir, exc,
            )
            raise

        _write_model_manifest(path)
        if save_errors:
            logger.warning(
                "Ensemble persisted with %d/%d models failing to pickle: %s",
                len(save_errors), len(self.trained_models_),
                list(save_errors.keys()),
            )
        logger.info(
            "Ensemble saved to %s (orchestrator + %d base models in %s/)",
            path, len(saved_model_paths), models_dir.name,
        )

    @classmethod
    def load(cls, path: Path) -> "VariantEnsemble":
        """Load an ensemble.

        Back-compatible with the pre-Run-10 single-joblib format AND
        with the new format_version=2 orchestrator + per-model layout.
        """
        import joblib

        path = Path(path)
        obj = joblib.load(path)

        # Legacy: single joblib containing a pickled VariantEnsemble.
        if isinstance(obj, cls):
            return obj

        if not isinstance(obj, dict) or obj.get("format_version") != 2:
            raise ValueError(
                f"Unrecognised ensemble joblib format at {path}: "
                f"expected VariantEnsemble or format_version=2 dict, "
                f"got {type(obj).__name__}"
            )

        ens = cls.__new__(cls)
        ens.config = obj["config"]
        ens.meta_learner = obj["meta_learner"]
        ens.blend_weights_ = obj["blend_weights_"]
        ens.feature_names_ = obj.get("feature_names_")
        ens.oof_predictions_ = obj.get("oof_predictions_")
        ens.oof_fit_indices_ = obj.get("oof_fit_indices_")
        ens.oof_model_names_ = obj.get("oof_model_names_")
        ens.base_estimators = {}
        ens.trained_models_ = {}

        models_dir = path.parent / obj["models_dir_name"]
        load_errors = {}
        for name, model_filename in obj["saved_model_paths"].items():
            model_path = models_dir / model_filename
            try:
                ens.trained_models_[name] = joblib.load(model_path)
            except Exception as exc:
                load_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Failed to reload base model %s from %s: %s",
                    name, model_path, exc,
                )

        if obj.get("save_errors"):
            logger.warning(
                "Ensemble was saved with %d models that failed to pickle: %s. "
                "Predictions will use whatever base models DID survive.",
                len(obj["save_errors"]), list(obj["save_errors"].keys()),
            )
        if load_errors:
            logger.warning(
                "Failed to reload %d/%d base models: %s",
                len(load_errors), len(obj["saved_model_paths"]),
                list(load_errors.keys()),
            )

        return ens
