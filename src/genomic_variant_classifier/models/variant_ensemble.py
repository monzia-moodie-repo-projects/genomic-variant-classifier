"""Ensemble model framework for genomic variant classification.

WHAT THIS MODULE DEFINES
------------------------
    TABULAR_FEATURES / PHASE_2_FEATURES
        The tabular feature contract. Guarded by EXPECTED_TABULAR_FEATURE_COUNT, which
        fails loud on drift, and by the zero-variance census in
        VariantEnsemble._assert_no_dead_features.
    SEQUENCE_MODELS
        The subset of the roster whose input is the sequence branch rather than the
        tabular matrix. Consulted by _require_x_seq, which refuses before any fit when a
        sequence model is active without windows.
    EnsembleConfig
        Run configuration, including the deliberate escape hatches
        (allow_base_model_dropout, allow_zero_variance_features) and the row-count
        threshold that lets a guard be armed in production without reddening fixtures.
    engineer_features
        The single source of truth for feature engineering (since 2026-07-11).
    VariantEnsemble
        The base-model roster, leak-free out-of-fold stacking, Nelder-Mead blending,
        calibration, per-model checkpointing, persistence and evaluation.

THE ROSTER IS NOT ENUMERATED HERE, DELIBERATELY
------------------------------------------------
It used to be. Until 2026-07-19 this header stated a fixed count of base classifiers, listed
them by name, and named a machine-learning framework for two of them. The roster had grown
past that list, and the framework named was one this module does not import -- but the header
still said otherwise, and it is the first text a reader of this file sees.

The old wording is deliberately not quoted here. A quoted claim reads as a current one to
anyone skimming, which is the failure being removed; the exact prior text is in git and in
tests/unit/test_module_docstring_is_not_a_stale_roster.py, where it serves as the negative
control proving each check would have caught it.

A docstring that duplicates a fact defined below it will eventually contradict that fact,
because nothing forces the copy to move when the original does. This project has recorded
the same failure four times: WindowAttachment.__iter__'s self-maintained todo list (stale on
all four entries), tests/EXPECTED_SUITE_SIZE (numbers gone stale before the ratchet was
armed), the README test badge (which disagreed with that ratchet), and this header.

So the fact is stated once and pointed at, never copied:

    the roster           VariantEnsemble._build_estimators
    at runtime           VariantEnsemble(config).base_estimators
    which need sequence  SEQUENCE_MODELS
    the feature contract TABULAR_FEATURES, EXPECTED_TABULAR_FEATURE_COUNT

Ask the code. tests/unit/test_module_docstring_is_not_a_stale_roster.py fails if an
enumeration, a model count, or a framework attribution returns to this docstring.

Change history belongs in git, which records it permanently and with dates. A block of
per-issue changelog entries was removed from this header on 2026-07-19: it described edits
from a phase the project left long ago, and its content survives in the commits that made
them. The section name is not reproduced here for the same reason the old wording is not --
a stale heading quoted in a live header still reads as a live heading.
"""

from __future__ import annotations
from datetime import timezone

import logging
import warnings
import functools
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
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
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
# Feature definitions. THE contract: TABULAR_FEATURES + EXPECTED_TABULAR_FEATURE_COUNT
# below are the single source of truth for the matrix width and column ORDER.
#
# Do NOT write the feature count into this comment. It used to read "(65 features --
# must match DataPrepPipeline._engineer_features)" while the real contract held 97, and
# it mandated a hand-sync with a SECOND feature builder in real_data_prep.py that the
# correctness harness never validated. That second builder was proved equivalent and
# deleted on 2026-07-11; a count in a comment is a fact that rots, so it is gone too.
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

# Single source of truth for the per-variant tabular feature count.
# Bump by +/-1 whenever you add or remove an entry in TABULAR_FEATURES below.
# Enforced by tests/unit/test_feature_count_contract.py against both the list
# length and INFERENCE_FEATURE_COLUMNS; that test is the deliberate-bump tripwire.
# 2026-07-13: 97 -> 95. HGMD's two features (hgmd_is_disease_mutation, hgmd_n_reports) were
# REMOVED -- no license (procurement-blocked) and, independently, variant-level label leakage
# against a ClinVar-Pathogenic target. They were CONSTANT ZERO for the life of the project.
# See the HGMD block inside TABULAR_FEATURES for the full reasoning and the safe (gene-level,
# leave-one-out) way to reintroduce the signal if access is obtained.
EXPECTED_TABULAR_FEATURE_COUNT = 95

FEATURE_SOURCE = {
    # feature-name prefix / exact name  ->  (data source, the CLI flag that populates it)
    #
    # Built 2026-07-13 after the Run-15 census found 36 of 78 features CONSTANT ZERO (46% of
    # the feature space, 1,038,974 variants, AUROC 0.998 reported on the 38 that were real).
    # Every abort gate in scripts/launch_run17_*.sh checks that a FILE EXISTS. None checked
    # that a FEATURE POPULATED. A present-but-empty file, a schema change, or a failed
    # gene-symbol join all sail through a file check and still deliver a column of zeros.
    #
    # This map exists so that when a feature IS dead, the operator is told which source and
    # which flag to fix -- instead of being handed a list of column names and a shrug.
    "phylop_score":                       ("PhyloP (BigWig)",        "--phylop-path (+ pybigtools)"),
    "eve_score":                          ("EVE",                    "--eve-path / --eve-entry-map"),
    "gene_constraint_oe":                 ("gnomAD constraint",      "--gnomad-constraint"),
    "gene_is_constrained":                ("gnomAD constraint",      "--gnomad-constraint"),
    "pli_score":                          ("gnomAD constraint",      "--gnomad-constraint"),
    "loeuf":                              ("gnomAD constraint",      "--gnomad-constraint"),
    "syn_z":                              ("gnomAD constraint",      "--gnomad-constraint"),
    "mis_z":                              ("gnomAD constraint",      "--gnomad-constraint"),
    "has_uniprot_annotation":             ("UniProt",                "--esm2-uniprot-index"),
    "n_known_pathogenic_protein_variants":("UniProt",                "--esm2-uniprot-index"),
    "gtex_max_tpm":                       ("GTEx",                   "--gtex-path"),
    "gtex_n_tissues_expressed":           ("GTEx",                   "--gtex-path"),
    "gtex_tissue_specificity":            ("GTEx",                   "--gtex-path"),
    "gtex_is_eqtl":                       ("GTEx",                   "--gtex-path"),
    "gtex_min_eqtl_pval":                 ("GTEx",                   "--gtex-path"),
    "gtex_max_abs_effect":                ("GTEx",                   "--gtex-path"),
    "codon_position":                     ("Ensembl/VEP",            "--clinvar (VEP annotation)"),
    "dbsnp_af":                           ("dbSNP",                  "--dbsnp-path"),
    "omim_n_diseases":                    ("OMIM genemap2",          "--omim-genemap2-path"),
    "omim_n_diseases_molecular":          ("OMIM genemap2",          "--omim-genemap2-path"),
    "omim_is_autosomal_dominant":         ("OMIM genemap2",          "--omim-genemap2-path"),
    "clingen_validity_score":             ("ClinGen",                "--clingen-path"),
    "lovd_variant_class":                 ("LOVD",                   "--lovd-path"),
    "maxentscan_score":                   ("MaxEntScan (splice)",    "--spliceai / splice module"),
    "dist_to_splice_site":                ("splice annotation",      "--spliceai / splice module"),
    "exon_number":                        ("Ensembl/VEP",            "--clinvar (VEP annotation)"),
    "is_canonical_splice":                ("splice annotation",      "--spliceai / splice module"),
    "alphafold_plddt":                    ("AlphaFold",              "--alphafold-path"),
    "solvent_accessibility":              ("protein structure",      "--alphafold-path"),
    "secondary_structure_context":        ("protein structure",      "--alphafold-path"),
    "dist_to_active_site":                ("protein structure",      "--alphafold-path / UniProt"),
    "af_1kg_afr":                         ("1000 Genomes",           "--kg"),
    "af_1kg_eur":                         ("1000 Genomes",           "--kg"),
    "af_1kg_eas":                         ("1000 Genomes",           "--kg"),
    "af_1kg_sas":                         ("1000 Genomes",           "--kg"),
    "af_1kg_amr":                         ("1000 Genomes",           "--kg"),
    "finngen_af_fin":                     ("FinnGen",                "--finngen-path"),
    "finngen_af_nfsee":                   ("FinnGen",                "--finngen-path"),
    "finngen_enrichment":                 ("FinnGen",                "--finngen-path / --finngen-r13-path"),
    "esm2_delta_norm":                    ("ESM-2",                  "--esm2-uniprot-index"),
    "gnn_score":                          ("STRING-DB GNN",          "--string-db"),
    "cadd_phred":                         ("dbNSFP",                 "--dbnsfp-path"),
    "sift_score":                         ("dbNSFP",                 "--dbnsfp-path"),
    "polyphen2_score":                    ("dbNSFP",                 "--dbnsfp-path"),
    "revel_score":                        ("dbNSFP",                 "--dbnsfp-path"),
    "gerp_score":                         ("dbNSFP",                 "--dbnsfp-path"),
    "alphamissense_score":                ("AlphaMissense",          "--alphamissense"),
    "splice_ai_score":                    ("SpliceAI",               "--spliceai"),
}


def feature_census(X_tab: "pd.DataFrame") -> dict:
    """Which declared-real features actually carry information? Measured, not assumed.

    Returns {"live": [...], "dead": [...], "drift_blind": [...]}.

    dead        : nunique == 1 across the cohort. ZERO information. The model cannot split on
                  it, it contributes nothing, and it inflates the feature contract with
                  something that does not exist.
    drift_blind : varies, but the 1st and 99th percentile coincide -- so DriftDetector._psi
                  returns 0.0 unconditionally and this feature can NEVER signal drift, no
                  matter how far it moves.
    live        : varies AND is drift-detectable.

    In Run 15 this returned 36 dead / 4 drift-blind / 38 live, out of 78. Nobody ran it,
    because it did not exist.
    """
    declared = [c for c in TABULAR_FEATURES if c in X_tab.columns]
    dead: list[str] = []
    drift_blind: list[str] = []
    live: list[str] = []

    for col in declared:
        s = X_tab[col]
        if not pd.api.types.is_numeric_dtype(s):
            live.append(col)
            continue
        arr = s.to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0 or len(np.unique(finite)) == 1:
            dead.append(col)
        elif np.percentile(finite, 1) == np.percentile(finite, 99):
            drift_blind.append(col)
        else:
            live.append(col)

    return {"live": live, "dead": dead, "drift_blind": drift_blind,
            "declared": declared, "missing": [c for c in TABULAR_FEATURES
                                              if c not in X_tab.columns]}


def format_feature_census(census: dict, n_rows: int) -> str:
    """A census the operator can ACT on: names the source and the flag for every dead feature."""
    lines = [
        "",
        "=" * 78,
        f"  FEATURE CENSUS  --  {n_rows:,} variants, {len(census['declared'])} declared features",
        "=" * 78,
        f"  LIVE (varying, drift-detectable) : {len(census['live']):3d}",
        f"  DRIFT-BLIND (p01 == p99)         : {len(census['drift_blind']):3d}",
        f"  DEAD (zero information)          : {len(census['dead']):3d}",
    ]
    if census["missing"]:
        lines.append(f"  MISSING FROM THE MATRIX ENTIRELY : {len(census['missing']):3d}")

    if census["dead"]:
        lines += ["", "  DEAD FEATURES -- and the source/flag responsible for each:", ""]
        by_source: dict[tuple, list[str]] = {}
        for f in census["dead"]:
            by_source.setdefault(FEATURE_SOURCE.get(f, ("UNMAPPED", "unknown")), []).append(f)
        for (source, flag), feats in sorted(by_source.items()):
            lines.append(f"    {source:24s}  {flag}")
            for f in feats:
                lines.append(f"        - {f}")
    if census["drift_blind"]:
        lines += ["", "  DRIFT-BLIND (real signal, but PSI is identically 0.0 forever):",
                  f"    {', '.join(census['drift_blind'])}"]
    lines.append("=" * 78)
    return "\n".join(lines)


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
    # Gene-disease annotation (4)
    "omim_n_diseases",
    "omim_n_diseases_molecular",
    "omim_is_autosomal_dominant",
    "clingen_validity_score",
    # ---- HGMD: REMOVED 2026-07-13. Was 2 features; roster dropped 97 -> 95. ------------
    #
    #   "hgmd_is_disease_mutation",   # 1 if classified DM in HGMD
    #   "hgmd_n_reports",             # number of HGMD records for this variant
    #
    # TWO independent reasons, either sufficient:
    #
    # 1. NO ACCESS. HGMD Professional is a paid QIAGEN license and is not held
    #    (ROADMAP.md:71 -- "HGMD | hgmd_* (2) | PAID, blocked"; RUN_11_MASTER_PLAN sec. 2.5).
    #    The connector was never wired, so both columns were CONSTANT ZERO through Run 15 --
    #    contributing nothing, while occupying two slots in the feature contract and making
    #    the roster overstate the science by two. They are not "pending"; they are absent.
    #    Carrying an absent feature as a real one is the lie that root pattern (a) is about.
    #
    # 2. LABEL LEAKAGE -- and this one survives the license arriving.
    #    HGMD "DM" means *disease-causing mutation*. The label here is ClinVar Pathogenic
    #    (real_data_prep.py:512). Those are the same quantity under two vendors' names, and
    #    HGMD-DM overlaps ClinVar-P heavily. As a VARIANT-LEVEL feature it is an answer key:
    #    the gene-aware split cannot help, because the leak sits inside every fold at the
    #    variant level.
    #
    #    The deployment failure is the damning part. A novel variant of uncertain
    #    significance -- precisely what this classifier exists to score -- has no HGMD entry,
    #    so hgmd_is_disease_mutation = 0, and the model reads "not a disease mutation" and
    #    leans benign. You would publish a superb AUROC on a test set of catalogued variants
    #    and systematically under-call the VUS that matter.
    #
    #    This project already draws exactly this line elsewhere: real_data_prep.py:1169 stubs
    #    COSMIC to 0.0 and names the reason `feature-not-label`.
    #
    # WHEN HGMD ACCESS ARRIVES, DO NOT SIMPLY RESTORE THESE TWO LINES.
    # Wire it GENE-LEVEL and LEAVE-ONE-OUT -- e.g. `n_hgmd_dm_in_gene`, counting HGMD-DM
    # variants in the gene while EXCLUDING the variant being scored -- mirroring the existing
    # `n_pathogenic_in_gene`. Same biological signal, no answer key. Then bump
    # EXPECTED_TABULAR_FEATURE_COUNT in the same commit.
    # LOVD (1)
    "lovd_variant_class",
    # Chromosome context (3)
    "is_autosome",
    "is_sex_chrom",
    "is_mitochondrial",
    # GNN-derived (1)
    "gnn_score",
    # Hetero-KG GNN-derived (1) - Phase D
    "hetero_gnn_score",
    # RNA splice-context (5)
    "maxentscan_score",
    "maxentscan_delta",
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
    # FinnGen R12 (3)
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
    # FinnGen R13 (3) -- these three were appended under the "FinnGen (3)" header without
    # giving them one of their own, so the header said 3 while SIX features sat beneath it.
    # Every other group in this list carries its own count; this one silently didn't. Harmless
    # here -- EXPECTED_TABULAR_FEATURE_COUNT is derived from the list, not from these comments,
    # so nothing broke -- but it is the same shape as the defect that has cost this project the
    # most: a number written down once and never re-derived. Fixed 2026-07-13.
    "finngen_r13_af_fin",
    "finngen_r13_af_nfsee",
    "finngen_r13_enrichment",
    # ESM-2 (2)
    "esm2_delta_norm",
    "esm2_llr",
    # Nucleotide Transformer DNA-LM (2)
    "genomiclm_delta_norm",
    "genomiclm_llr",
    # COSMIC CMC (2)
    "cosmic_recurrence",
    "cosmic_sig_tier",
    # KEGG (2)
    "kegg_pathway_count",
    "kegg_disease_pathway_flag",
    # gnomAD v4.1 constraint (4)
    "pli_score",
    "loeuf",
    "syn_z",
    "mis_z",
    # Reactome pathway membership (1) - Phase D
    "reactome_pathway_count",
    # RNA-seq gene expression (5) - Phase D
    "rnaseq_mean_log_tpm",
    "rnaseq_detection_rate",
    "rnaseq_log2_cv",
    "rnaseq_log2fc",
    "rnaseq_de_neglog10p",
]

PHASE_2_FEATURES: list[str] = []  # AF features (alphafold_plddt/solvent_accessibility/secondary_structure_context/dist_to_active_site) are locked TABULAR_FEATURES; real once the AlphaFold parquet is built and --alphafold-path is wired, else sentinel stubs. Phase 3 adds GWAS.

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

    # If a base model's out-of-fold (OOF) step raises, may the ensemble carry on WITHOUT it?
    # DEFAULT: NO (2026-07-13).
    #
    # It used to. `fit()` caught EVERY exception from the OOF block, set that model's OOF
    # column to a constant 0.5, and `continue`d -- which also skipped the `model.fit(...)`
    # immediately below it.
    #
    # To be precise about the blast radius, because it was twice mis-stated during the
    # 2026-07-13 investigation: the constant 0.5 column is NOT fed to the stacking
    # meta-learner. The `valid_cols` filter further down drops the columns of any model
    # missing from `trained_models_`, so the meta-learner only ever sees real columns. That
    # part was always correct.
    #
    # The harm is that the model is ERASED FROM THE RUN, in silence:
    #   * it is never fitted, and no checkpoint is written for it;
    #   * it is absent from trained_models_, oof_model_names_, the blend, and every
    #     downstream comparison artifact;
    #   * a 13-model ensemble quietly becomes a 12-model ensemble;
    #   * the surviving models report entirely normal metrics, so the run LOOKS healthy;
    #   * the only trace is a single logger.error line in a multi-hour log.
    #
    # That collides head-on with a first-class goal of this project -- to measure and compare
    # the performance of every machine-learning algorithm in the roster. A silently dropped
    # algorithm does not appear in the report as a FAILURE. It appears as an algorithm that
    # was never a candidate, indistinguishable from one that was never configured.
    #
    # And `except Exception` is broad enough to swallow an out-of-memory error, a transient
    # data fault, or -- as actually observed on 2026-07-13 -- a merely SPURIOUS library
    # warning (LightGBM 4.6.0 populates `feature_names_in_` even when fitted on a bare
    # ndarray, which makes scikit-learn warn) escalated to an exception by a strict warning
    # filter. Noise was sufficient to delete a model from a paid training run.
    #
    # Set True ONLY to deliberately tolerate a known-failing model. Even then the dropout is
    # LOUD: it logs at ERROR with a full traceback and is recorded in
    # VariantEnsemble.dropped_models_, so the run's artifacts carry the fact -- and the
    # reason -- that the ensemble was incomplete.
    allow_base_model_dropout: bool = False

    # ---- THE ZERO-VARIANCE (SILENT-ZERO) GUARD ---------------------------------------
    # Added 2026-07-13 (roadmap 6.21).
    #
    # A connector whose source file is absent does not crash. It returns zeros -- see
    # omim.py:105 (`if gene_table.empty: result[...] = DEFAULT_N_DISEASES; return result`)
    # and variant_ensemble.py's own `df.get("omim_n_diseases", pd.Series([0] * len(df)))`.
    # No warning. No error. The column arrives full of zeros and TRAINS.
    #
    # This has already happened, at scale. In Run 15, THIRTY-SIX of the 78 features were
    # constant zero -- 46% of the feature space, across 1,038,974 variants. Whole sources were
    # silently stubbed: GTEx (6), 1000 Genomes (5), FinnGen (3), AlphaFold/protein structure
    # (4), splice/MaxEntScan (4), UniProt (2), OMIM (2), ESM-2, EVE, dbSNP, PhyloP, ClinGen,
    # codon_position, gene constraint. The run completed, reported, and PUBLISHED AN AUROC OF
    # 0.998 -- produced by the 38 features that were real -- with no indication that half its
    # feature space did not exist.
    #
    # Those launcher gates are the right instinct but the wrong LAYER: they check that a FILE
    # is present, which is a PROXY for the feature being populated. A present-but-empty file,
    # a schema change, a failed join, a bad gene-symbol merge -- all sail straight through a
    # file-existence check and still deliver a column of zeros. That is root pattern (c): a
    # gate that checks a proxy instead of the thing it protects is not a gate.
    #
    # This guard asserts THE THING ITSELF: after feature engineering, at the moment of fit,
    # every declared-real feature must actually vary. A feature that is constant across a
    # million variants carries zero information, cannot be split on, cannot signal drift
    # (p01 == p99 => Population Stability Index is identically 0.0, forever), and is a lie in
    # the 97-feature contract.
    #
    # If a feature is genuinely not computed yet, it belongs in PHASE_2_FEATURES. That is what
    # PHASE_2_FEATURES is FOR, and it is currently EMPTY.
    allow_zero_variance_features: bool = False

    # Below this row count, a constant column is plausibly just sampling (a small synthetic
    # fixture can easily produce an all-zero binary flag). The guard WARNS there and RAISES
    # above it. A constant column in a million-row cohort is not sampling; it is a dead
    # feature. This threshold is what lets the guard be armed by default in the real run
    # without turning every unit-test fixture red.
    zero_variance_min_rows: int = 10_000

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _suppress_fillna_downcast(_fn):
    """Opt into pandas' future no-silent-downcasting inside the wrapped builder.

    Silences the pandas 2.x .fillna object-downcast FutureWarning with no value
    or dtype change: the explicit .astype() calls still set each column's dtype
    and still raise on genuinely non-numeric input. No-op on pandas >= 3.
    """
    @functools.wraps(_fn)
    def _wrapper(*args, **kwargs):
        with pd.option_context("future.no_silent_downcasting", True):
            return _fn(*args, **kwargs)

    return _wrapper


@_suppress_fillna_downcast
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the tabular feature matrix from a raw variant DataFrame.

    THE SINGLE SOURCE OF TRUTH for feature engineering (since 2026-07-11).

    The contract is TABULAR_FEATURES / EXPECTED_TABULAR_FEATURE_COUNT -- never a number
    written into a comment. All missing columns are filled with safe defaults (df.get),
    and those defaults are part of the contract: e.g. an absent sift_score fills to the
    NEUTRAL 0.5, never to the 0.05 deleterious threshold, or every unannotated variant
    would be silently called deleterious (tests/unit/test_core.py::
    test_sift_score_fill_is_not_threshold).

    HISTORY -- read before adding a second implementation.
    This docstring used to read "Derive the 65 tabular features ... Mirrors
    DataPrepPipeline._engineer_features()". There were genuinely TWO implementations,
    hand-kept in sync by that comment. They had already drifted in documentation (the
    comment said 65; the contract held 97), and, far worse, the five-stage correctness
    harness imports THIS function -- so the gate validated a code path the training
    pipeline never executed. A silent zero or a truncating cast in the pipeline's copy
    was structurally invisible to the gate built to catch exactly that.

    The two were proved equivalent (117 adversarial comparisons -- exact on column set,
    ORDER, dtype and values; forcing every df.get default and every integral-input
    truncation path) and collapsed: DataPrepPipeline._engineer_features now delegates
    here. See docs/status/REMEDIATION_2026-07-11_test-suite-red.md and
    scripts/prove_engineer_features_equivalence.py.

    Do not create a second feature builder. If you need a variant, parameterise this one.
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
        "gene_constraint_oe", df.get("loeuf", pd.Series([1.0] * len(df), index=df.index))
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
    feats["omim_n_diseases_molecular"] = (
        df.get("omim_n_diseases_molecular", pd.Series([0] * len(df), index=df.index))
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

    # HGMD: REMOVED 2026-07-13 (no license + variant-level label leakage). See the block in
    # TABULAR_FEATURES above for the full reasoning, and for how to wire it SAFELY
    # (gene-level, leave-one-out) if access is ever obtained. Do not restore these two
    # `df.get(..., 0)` lines: that pattern is precisely what silently zeroed them for the
    # entire life of the project without anyone noticing.

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

    # Hetero-KG GNN-derived score (1)
    feats["hetero_gnn_score"] = (
        df.get("hetero_gnn_score", pd.Series([0.5] * len(df), index=df.index))
        .fillna(0.5)
        .astype(float)
        .clip(0.0, 1.0)
    )

    # RNA splice-context features (5)
    feats["maxentscan_score"] = (
        df.get("maxentscan_score", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )
    feats["maxentscan_delta"] = (
        df.get("maxentscan_delta", pd.Series([0.0] * len(df), index=df.index))
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
        ("finngen_r13_af_fin", 0.0),
        ("finngen_r13_af_nfsee", 0.0),
        ("finngen_r13_enrichment", 1.0),
    ]:
        feats[_col] = (
            df.get(_col, pd.Series([_default] * len(df), index=df.index))
            .fillna(_default)
            .astype(float)
        )

    # ESM-2 delta norm (1) - 0.0 default when model unavailable or non-missense
    feats["esm2_delta_norm"] = (
        df.get("esm2_delta_norm", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )

    # ESM-2 LLR (1) -- SIGNED feature (negative => damaging); NO clip
    feats["esm2_llr"] = (
        df.get("esm2_llr", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )

    # Nucleotide Transformer DNA-LM (2) - 0.0 default when model/window unavailable
    feats["genomiclm_delta_norm"] = (
        df.get("genomiclm_delta_norm", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    feats["genomiclm_llr"] = (  # SIGNED feature; NO clip
        df.get("genomiclm_llr", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
    )

    # COSMIC CMC (2) - 0.0 default when --cosmic-path absent / non-substitution
    feats["cosmic_recurrence"] = (
        df.get("cosmic_recurrence", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    feats["cosmic_sig_tier"] = (
        df.get("cosmic_sig_tier", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )

    # KEGG pathway membership (2) - 0.0 default when --kegg-path absent
    feats["kegg_pathway_count"] = (
        df.get("kegg_pathway_count", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    feats["kegg_disease_pathway_flag"] = (
        df.get("kegg_disease_pathway_flag", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )

    # gnomAD v4.1 gene constraint (4) - safe defaults when connector absent
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

    # Reactome pathway membership (1) - Phase D
    feats["reactome_pathway_count"] = (
        df.get("reactome_pathway_count", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    # RNA-seq gene expression (5) - Phase D
    for _rc in (
        "rnaseq_mean_log_tpm",
        "rnaseq_detection_rate",
        "rnaseq_log2_cv",
        "rnaseq_log2fc",
        "rnaseq_de_neglog10p",
    ):
        feats[_rc] = (
            df.get(_rc, pd.Series([0.0] * len(df), index=df.index))
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

# Default dilation schedule for the residual tower. A tuple (not a list) so it is
# hashable and round-trips cleanly through sklearn get_params/set_params and the
# CNN1DClassifier.__getstate__ pickle path.
CNN1D_DEFAULT_DILATIONS = (1, 2, 4, 8)


def _build_single_channels(
    seq, window: int, use_positional: bool, positional_sigma: float,
) -> np.ndarray:
    """(N, C, window) for single_sequence_mode: one-hot(4) [+ positional(1)].

    2026-07-15, roadmap 6.28. This exists so the reference-only mode is a REAL
    architecture rather than the delta architecture with its heart cut out.

    The old accidental path handed a lone Series to _build_delta_channels as
    `ref = alt = seq`, producing 13 channels in which 4:8 was byte-identical to 0:4 and
    8:12 was identically zero. The model then spent its capacity on eight channels
    carrying four channels of information, plus four channels of nothing -- and reported
    the result in the algorithm comparison under the name `cnn_1d`, whose entire premise
    is the delta it no longer had.

    Five channels that mean something beats thirteen that do not.
    """
    oh = np.stack([encode_sequence(str(s), window) for s in seq])   # (N, W, 4)
    chans = [oh]
    if use_positional:
        pos = np.arange(window, dtype=np.float32)
        centre = window // 2
        bump = np.exp(-0.5 * ((pos - centre) / max(positional_sigma, 1e-6)) ** 2)
        chans.append(np.repeat(bump[None, :, None], len(oh), axis=0))   # (N, W, 1)
    stacked = np.concatenate(chans, axis=2)                             # (N, W, C)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1), dtype=np.float32)


def _build_delta_channels(
    ref_seqs,
    alt_seqs,
    window: int,
    use_delta: bool,
    use_positional: bool,
    positional_sigma: float,
) -> np.ndarray:
    """Build the fused CNN input tensor, shape (N, C, window), float32.

    Channels, in fixed order (so state_dict keys stay stable across runs):
      * ref  one-hot             (4)   -- always
      * alt  one-hot             (4)   -- always
      * alt - ref  signed delta  (4)   -- iff use_delta   (non-zero only where the
                                           variant changes the base -> simultaneously
                                           the substitution identity AND an implicit
                                           locator of the edit)
      * positional Gaussian bump (1)   -- iff use_positional (fixed prior centred on
                                           the variant position = window // 2)

    The delta and positional channels are the Tier-1 change: they hand the network
    the variant signal at the *input* rather than forcing it to recover the signal
    from an embedding-space subtraction, which is what left the previous siamese
    net's probabilities compressed into a narrow band.
    """
    ref_list = list(ref_seqs)
    alt_list = list(alt_seqs)
    n = len(ref_list)
    oh_ref = np.stack([encode_sequence(s, window=window) for s in ref_list])  # (N, W, 4)
    oh_alt = np.stack([encode_sequence(s, window=window) for s in alt_list])  # (N, W, 4)
    chans = [oh_ref, oh_alt]
    if use_delta:
        chans.append(oh_alt - oh_ref)  # (N, W, 4) signed
    if use_positional:
        centre = window // 2
        pos = np.arange(window, dtype=np.float32)
        bump = np.exp(-0.5 * ((pos - centre) / max(positional_sigma, 1e-6)) ** 2)
        bump = np.broadcast_to(bump.reshape(1, window, 1), (n, window, 1)).astype(np.float32)
        chans.append(bump)
    stacked = np.concatenate(chans, axis=2)  # (N, W, C)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1), dtype=np.float32)  # (N, C, W)


def _focal_loss_with_logits(logits, targets, gamma: float, alpha: float):
    """Binary focal loss from logits (numerically stable).

    FL = alpha_t * (1 - p_t)**gamma * BCE(logits, targets), with p_t the probability
    of the true class and alpha_t the class-balanced weight. gamma down-weights easy,
    well-classified examples so the optimiser spends capacity on the hard variants --
    the lever that pushes the output distribution apart (decompressing the previous
    [0.106, 0.185] band). Reduces to alpha-weighted BCE when gamma == 0.
    """
    import torch
    import torch.nn.functional as F

    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    modulator = (1.0 - p_t).clamp(min=1e-6) ** gamma
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * modulator * ce).mean()


# ---------------------------------------------------------------------------
# Module-level _CNN1DModule (Run 10 fix for INCIDENT_2026-05-12_cnn1d-pickle-
# nested-class.md, preserved). Defined lazily so that `import variant_ensemble`
# does not require torch, but bound to module globals on first use so pickle can
# resolve the class by qualname `genomic_variant_classifier.models.
# variant_ensemble._CNN1DModule`. The fitted estimator pickles only a state_dict
# (see CNN1DClassifier.__getstate__), so the architecture is rebuilt from stored
# hyperparameters on load -- every architectural knob below MUST therefore be a
# constructor argument threaded through _build_model().
# ---------------------------------------------------------------------------
_CNN1DModule = None  # populated by _ensure_cnn1d_module_class() on first use


def _ensure_cnn1d_module_class():
    """Define the dilated-residual _CNN1DModule at module level on first call; idempotent.

    Tier-1 architecture (2026-07-05), replacing the siamese-delta encoder:
      * input = fused [ref, alt, alt-ref, positional] channels (built by
        _build_delta_channels) so the variant delta is available at the input;
      * stem Conv1d -> GroupNorm -> GELU;
      * a stack of residual blocks with GROWING DILATION (default 1,2,4,8) to widen
        the receptive field across the 101 bp window without pooling the variant
        site away early (padding keeps length constant);
      * dual global pooling (avg + max) so the sharp single-base activation (max)
        and the sequence context (avg) both reach the head;
      * MLP head -> a single logit (BCE/focal applied outside).
    GroupNorm (not BatchNorm) is used so the net is robust to singleton CPU batches
    and behaves identically in train/eval. Built lazily for the same reasons as
    before: graceful degradation without torch and a stable pickle qualname.
    """
    global _CNN1DModule
    if _CNN1DModule is not None:
        return _CNN1DModule

    import torch
    import torch.nn as nn

    def _norm(channels: int) -> "nn.Module":
        g = 8
        while channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, channels)

    class _ResidualBlock(nn.Module):
        def __init__(self, channels, kernel_size, dilation, dropout):
            super().__init__()
            pad = (dilation * (kernel_size - 1)) // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            self.norm1 = _norm(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            self.norm2 = _norm(channels)
            self.drop = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            h = self.act(self.norm1(self.conv1(x)))
            h = self.drop(h)
            h = self.norm2(self.conv2(h))
            return self.act(x + h)  # residual; padding keeps length so shapes match

    class _CNN1DModule(nn.Module):  # noqa: F811 -- intentional global shadow
        def __init__(self, in_channels, filters, kernel_size, dropout, dilations, embed):
            super().__init__()
            pad = kernel_size // 2
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=pad),
                _norm(filters),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                [_ResidualBlock(filters, kernel_size, int(d), dropout) for d in dilations]
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.maxpool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(filters * 2, embed),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed, 1),
            )

        def forward(self, x):
            h = self.stem(x)
            for blk in self.blocks:
                h = blk(h)
            pooled = torch.cat([self.avgpool(h), self.maxpool(h)], dim=1)  # (N, 2*filters, 1)
            return self.head(pooled).squeeze(-1)  # logits -> focal/BCE outside

    _CNN1DModule.__name__ = "_CNN1DModule"
    _CNN1DModule.__qualname__ = "_CNN1DModule"
    _CNN1DModule.__module__ = __name__
    globals()["_CNN1DModule"] = _CNN1DModule
    return _CNN1DModule


# ---------------------------------------------------------------------------
# Sklearn-compatible dilated-residual delta 1D-CNN wrapper (Tier-1, 2026-07-05)
# ---------------------------------------------------------------------------
class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """Dilated-residual 1D-CNN over a fused (ref, alt, delta, positional) window.

    X may be a DataFrame with [fasta_seq_ref, fasta_seq_alt] (delta mode) or a
    Series / single 'fasta_seq' column (back-compat: ref == alt -> the delta channel
    is all-zero and the net degrades to ref-only context).

    Tier-1 design (2026-07-05): the variant signal is fed at the INPUT as an explicit
    alt-ref delta channel plus a fixed positional marker at the variant site
    (window // 2); the tower is dilated + residual; training uses focal loss with a
    warmup->cosine learning-rate schedule and AdamW weight decay. This replaces the
    previous siamese-delta encoder whose outputs collapsed into a narrow probability
    band (0.5419 AUROC / MCC 0.0 at threshold 0.5 in the 2026-07-04 run). No model is
    dropped: this is the same `cnn_1d` estimator, re-architected in place.

    NOTE: the state_dict layout differs from the pre-Tier-1 CNN, so cnn_1d checkpoints
    pickled before 2026-07-05 will NOT load into this class and must be retrained (the
    corrected re-run does this).
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
        dilations=CNN1D_DEFAULT_DILATIONS,
        weight_decay=1e-4,
        warmup_epochs=3,
        lr_min=1e-5,
        focal_gamma=2.0,
        focal_alpha=None,
        use_delta_channels=True,
        use_positional=True,
        positional_sigma=3.0,
        single_sequence_mode=False,
    ):
        """
        single_sequence_mode: OPT-IN, 2026-07-15 (roadmap 6.28). Default False.

        False (default) -- DELTA MODE. `_encode_batch` requires a 2-column
            [fasta_seq_ref, fasta_seq_alt] DataFrame and RAISES on anything else:
            a Series, a bare column, a NaN, or the legacy `fasta_seq` column.
            13 channels: ref(4) + alt(4) + delta(4) + positional(1).

        True -- SINGLE-SEQUENCE MODE. Accepts a Series (or a 1-column frame) of one
            sequence per row. 5 channels: one-hot(4) + positional(1). No alt, no delta.

        WHY THIS IS A CONSTRUCTOR FLAG AND NOT A TYPE SNIFF
        ---------------------------------------------------
        Until today `_encode_batch` chose the mode from the SHAPE of whatever arrived.
        A Series silently became `ref = alt`, which makes `oh_alt - oh_ref` identically
        zero. The measured consequence, on the 13-channel default:

            channels 0:4   ref one-hot          real
            channels 4:8   alt one-hot          BYTE-IDENTICAL to 0:4
            channels 8:12  delta = alt - ref    IDENTICALLY ZERO
            channel  12    positional           constant across every row

        Four unique channels, four duplicated, four dead, one constant -- reported in the
        algorithm comparison under the name `cnn_1d`, a model whose architecture is NAMED
        for the delta it had just deleted. It fit, it converged, it produced a number.

        The mode was never the problem. CHOOSING IT BY ACCIDENT was. Since the choice
        arrived as a type rather than a decision, nothing recorded it and nothing could:

          * scripts/train.py has always passed a DataFrame (`_att_train.windows`);
          * every test has always passed a Series, because the signature said
            `X_seq: pd.Series` -- an annotation that was FALSE for the production path;
          * `_encode_batch` accepted both, so the two never had to agree, and for years
            the suite was green on a code path the run has never executed.

        As a constructor argument the mode is get_params()-visible, survives the pickle
        (see __getstate__: architecture is rebuilt from hyperparameters), and lands in
        `ensemble_completeness_` and the run artifact. "The model was in single-sequence
        mode" becomes a RECORDED FACT rather than an inference about a Series someone
        passed two years ago. CLAUDE.md 2a: if a rule can be forgotten, it will be --
        make forgetting FAIL.

        And the mode is now HONEST rather than degenerate: 5 real channels, not 13 of
        which 9 are redundant. A reference-only model that says it is one.
        """
        self.single_sequence_mode = single_sequence_mode
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
        self.dilations = dilations
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_delta_channels = use_delta_channels
        self.use_positional = use_positional
        self.positional_sigma = positional_sigma
        self.model_ = None
        self.alpha_ = None            # resolved focal alpha (set during fit)
        self.classes_ = np.array([0, 1])

    # -- architecture bookkeeping -------------------------------------------
    def _in_channels(self) -> int:
        """Channel count MUST match _build_delta_channels exactly (contract).

        single_sequence_mode collapses the roster to what actually carries information:
        one-hot(4) + positional(1) = 5. It does NOT emit 13 channels of which 8 are a
        duplicated pair and 4 are an identically-zero delta -- that was the old
        accidental behaviour, and it is the difference between a reference-only model
        and a delta model that has been quietly hollowed out.
        """
        if self.single_sequence_mode:
            c = 4  # one-hot of the single sequence; no alt, so no ref/alt pair
            if self.use_positional:
                c += 1
            return c
        c = 8  # ref (4) + alt (4)
        if self.use_delta_channels:
            c += 4
        if self.use_positional:
            c += 1
        return c

    def _build_model(self):
        import torch  # noqa: F401
        torch.manual_seed(self.random_state)
        cls = _ensure_cnn1d_module_class()
        return cls(
            self._in_channels(),
            self.filters,
            self.kernel_size,
            self.dropout,
            tuple(int(d) for d in self.dilations),
            self.embed,
        )

    # -- encoding ------------------------------------------------------------
    def _encode_batch(self, X) -> np.ndarray:
        """Return fused inputs (N, C, window) from a [fasta_seq_ref, fasta_seq_alt] frame.

        THE FIFTH FABRICATOR -- 2026-07-15, roadmap 6.28.
        =================================================
        This method used to open with `win = "A" * self.window` and then `.fillna(win)`
        every input it was handed. It was the FIFTH content-based poly-A site in the
        repository and the only one INSIDE a model: the other four merely mis-detected
        fabricated windows, while this one MANUFACTURED them, silently, at fit time.

        It was found by tests/unit/test_no_content_based_poly_detection.py -- the
        repo-wide ban -- after this author had personally read this function earlier the
        same day, named `win = "A" * self.window` as a defect in writing, and then moved
        on without fixing it. The gate caught what the reader had already seen and let go.
        That is the entire argument for gates over attention.

        THREE SILENT DEGRADATIONS ARE NOW THREE RAISES:

        1. `.fillna(win)` -- a NaN window became a full poly-A window. Legitimate callers
           cannot produce this: `attach_delta_windows` returns a WindowAttachment whose
           frame is already filled with PLACEHOLDER_BASE and whose `usable` mask says
           which rows are real. A NaN arriving here means the caller bypassed the join,
           and fabricating 101 bases to paper over that is not robustness.

        2. `elif "fasta_seq" in X.columns: ref = alt = X["fasta_seq"]` -- the legacy
           single-window column. MEASURED 2026-07-15: `fasta_seq` is **100% NULL across
           all 4,420,180 cohort rows** (null: 4,420,180 | non-null: 0). So this branch
           took a column of nothing, filled it with poly-A, and set ref == alt -- which
           makes the delta channels identically zero. Four of thirteen channels dead,
           eight duplicated, and the model trains without complaint. SEQUENCE_FEATURES
           (line 477) is a live constant still pointing at that empty column: a loaded
           gun with the safety off.

        3. `else: ref = alt = X.iloc[:, 0]` -- same ref == alt delta collapse, reached by
           handing this method any frame at all.

        The "tolerant adapter" was tolerant of exactly the inputs that destroy the model's
        only reason to exist. CLAUDE.md 4: nothing fails silently. A sequence model given
        no sequence must say so.
        """
        if self.single_sequence_mode:
            # OPT-IN reference-only mode. Reached only because a caller CONSTRUCTED this
            # estimator with single_sequence_mode=True -- never because of the shape of
            # whatever arrived. One sequence per row; no alt; no delta.
            if isinstance(X, pd.DataFrame):
                if X.shape[1] != 1:
                    raise ValueError(
                        f"single_sequence_mode expects ONE sequence column; got "
                        f"{X.shape[1]}: {list(X.columns)}. If you meant delta mode, "
                        f"construct with single_sequence_mode=False (the default) and "
                        f"pass [{REF_WIN_COL}, {ALT_WIN_COL}]."
                    )
                seq = X.iloc[:, 0]
            elif isinstance(X, pd.Series):
                seq = X
            else:
                seq = pd.Series(list(X))
            self._assert_no_null_windows(seq, "sequence")
            return _build_single_channels(
                seq, self.window, self.use_positional, self.positional_sigma,
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"cnn_1d requires a pandas DataFrame with [{REF_WIN_COL}, {ALT_WIN_COL}]; "
                f"got {type(X).__name__}. A Series or raw iterable cannot carry a ref/alt "
                f"pair, so it could only ever be encoded by setting ref == alt -- which "
                f"makes the delta channels identically zero and hollows out the model "
                f"whose architecture is named for them.\n"
                f"Build the frame with "
                f"seq_window_join.attach_delta_windows(meta, seq_windows_path=...).windows\n"
                f"If you genuinely want a reference-only model, ASK FOR IT explicitly: "
                f"CNN1DClassifier(single_sequence_mode=True). It is then 5 honest channels, "
                f"recorded in get_params() and in the run artifact -- not 13 channels of "
                f"which 9 are redundant, silently selected by an argument's type."
            )

        missing = {REF_WIN_COL, ALT_WIN_COL} - set(X.columns)
        if missing:
            extra = ""
            if "fasta_seq" in X.columns:
                extra = (
                    "\nThis frame carries the legacy 'fasta_seq' column. MEASURED "
                    "2026-07-15: that column is 100% NULL across all 4,420,180 cohort "
                    "rows (null: 4,420,180 | non-null: 0). It used to be accepted here "
                    "and filled with fabricated poly-A. SEQUENCE_FEATURES (line ~477) "
                    "still names it; do not route it to this model."
                )
            raise ValueError(
                f"cnn_1d requires a 2-column [{REF_WIN_COL}, {ALT_WIN_COL}] frame; "
                f"missing {sorted(missing)}. Got columns: {list(X.columns)}.\n"
                f"Build it with "
                f"seq_window_join.attach_delta_windows(meta, seq_windows_path=...).windows"
                f"{extra}"
            )

        ref, alt = X[REF_WIN_COL], X[ALT_WIN_COL]
        self._assert_no_null_windows(ref, REF_WIN_COL)
        self._assert_no_null_windows(alt, ALT_WIN_COL)

        return _build_delta_channels(
            ref, alt, self.window,
            self.use_delta_channels, self.use_positional, self.positional_sigma,
        )

    def _assert_no_null_windows(self, s: pd.Series, label: str) -> None:
        """Nulls used to become fabricated poly-A. Now they stop the run.

        `attach_delta_windows` NEVER emits nulls -- unresolvable rows carry
        PLACEHOLDER_BASE and are marked usable=False, so provenance travels with the
        data. A null arriving here means the join was bypassed, and inventing 101 bases
        to cover for that is not robustness; it is manufacturing reference sequence and
        calling it evidence.
        """
        n_null = int(s.isna().sum())
        if n_null:
            raise ValueError(
                f"cnn_1d received {n_null}/{len(s)} null window(s) in '{label}'. Until "
                f"2026-07-15 these were silently filled with 'A' * {self.window} -- "
                f"fabricated sequence, indistinguishable from a real poly-A tract, fed "
                f"to the model as data.\n"
                f"attach_delta_windows() never emits nulls. Fix the caller."
            )

    # -- training ------------------------------------------------------------
    def _resolve_alpha(self, y) -> float:
        if self.focal_alpha is not None:
            return float(self.focal_alpha)
        y = np.asarray(y, dtype=np.float64)
        pos_rate = float(y.mean()) if y.size else 0.5
        # up-weight the minority class; clamp so neither class is ignored.
        return float(np.clip(1.0 - pos_rate, 0.1, 0.9))

    def _epoch_lr(self, epoch: int) -> float:
        base, lo = self.learning_rate, self.lr_min
        w = max(1, int(self.warmup_epochs))
        if epoch < w:
            return base * (epoch + 1) / w
        span = max(1, self.epochs - w)
        progress = min(1.0, (epoch - w) / span)
        return lo + 0.5 * (base - lo) * (1.0 + np.cos(np.pi * progress))

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_all = self._encode_batch(X)                       # (N, C, W)
        x_t = torch.tensor(x_all, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.alpha_ = self._resolve_alpha(y)

        n = len(y_t)
        n_val = max(1, int(self.val_fraction * n)) if n > 1 else 0
        gen = torch.Generator().manual_seed(self.random_state)
        idx = torch.randperm(n, generator=gen)
        v, t = idx[:n_val], idx[n_val:]
        if len(t) == 0:                                     # tiny-N guard
            t = idx
        x_val = x_t[v].to(device) if n_val else x_t[t].to(device)
        y_val = y_t[v].to(device) if n_val else y_t[t].to(device)

        loader = DataLoader(
            TensorDataset(x_t[t], y_t[t]),
            batch_size=min(self.batch_size, max(1, len(t))),
            shuffle=True,
        )
        self.model_ = self._build_model().to(device)
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_val, best_state, patience_ctr = float("inf"), None, 0
        for epoch in range(self.epochs):
            for g in opt.param_groups:
                g["lr"] = self._epoch_lr(epoch)
            self.model_.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = self.model_(xb)
                loss = _focal_loss_with_logits(logits, yb, self.focal_gamma, self.alpha_)
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = _focal_loss_with_logits(
                    self.model_(x_val), y_val, self.focal_gamma, self.alpha_
                ).item()
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
        x_all = self._encode_batch(X)
        out = np.empty(len(x_all), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(x_all), bs):
                xb = torch.tensor(x_all[i:i + bs], dtype=torch.float32)
                out[i:i + len(xb)] = torch.sigmoid(self.model_(xb)).numpy()
        return np.column_stack([1.0 - out, out])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # Portable state_dict pickling: rebuilds via the factory on load, so it does not
    # depend on the module-global _CNN1DModule being pre-populated in a fresh process
    # (more robust than relying on qualname alone across machines).
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

    def _apply_feature_mask(self, X):
        """Select the columns kept at fit time (mirrors self.scaler_).

        Returns X unchanged when feature_mask_ is absent, so estimators
        pickled before the mask existed still score correctly.
        """
        X = np.asarray(X, dtype=float)
        mask = getattr(self, "feature_mask_", None)
        if mask is None:
            return X
        n_in = getattr(self, "n_features_in_", X.shape[1])
        if X.shape[1] != n_in:
            raise ValueError(
                f"TabularNNClassifier got {X.shape[1]} columns at predict time "
                f"but saw {n_in} at fit time."
            )
        return X[:, mask]

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_mask_ = X.var(axis=0) > 0.0
        if not self.feature_mask_.any():            # degenerate: keep all, never 0-width
            self.feature_mask_ = np.ones(X.shape[1], dtype=bool)
        X = X[:, self.feature_mask_]
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
        X_scaled = self.scaler_.transform(self._apply_feature_mask(X))
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
            X_scaled = self.scaler_.transform(self._apply_feature_mask(X))
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


# Models whose input is the SEQUENCE branch rather than the tabular matrix.
#
# Declared once, in one place, so that adding a second sequence model does not require
# finding every dispatch site by hand. `_require_x_seq` is the only consumer; the three
# `if name == "cnn_1d"` dispatches in fit/predict_proba/evaluate are deliberately left
# alone, because changing them is a separate refactor with its own risk.
SEQUENCE_MODELS: frozenset[str] = frozenset({"cnn_1d"})


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
            # SCALED (2026-07-12). This was a BARE LogisticRegression, fit on the raw
            # tabular matrix -- where `pos` runs to 1,000,000 alongside `allele_freq` at
            # 1e-6. It did not converge, and said so in every test run and every Continuous
            # Integration run for weeks:
            #
            #     ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s)
            #
            # That warning was never noise. VariantEnsemble.fit dispatches this model to
            # `X_tab_fit.values` -- raw, unscaled -- so a NON-CONVERGED logistic regression
            # was being fit, and its out-of-fold predictions fed the stacking meta-learner.
            #
            # It was the ONLY scale-sensitive model in the roster without a scaler. Audited
            # 2026-07-12, every other one already had its own:
            #     svm / svm_bagged_rbf  -> ScalableSVM: make_pipeline(StandardScaler(), ...)
            #     tabular_nn            -> TabularNNClassifier: self.scaler_ + BatchNorm1d
            #     mc_dropout / deep_ensemble -> wrap TabularNNClassifier (inherit its scaler)
            #     kan                   -> StandardScaler
            #     cnn_1d                -> consumes the ONE-HOT DNA sequence (values in {0,1});
            #                              scaling it would DESTROY the encoding. Correctly bare.
            #     trees (rf/xgb/lgbm/gbm/catboost) -> scale-invariant by construction.
            # So this was an oversight, not a design choice.
            #
            # WHY IT MATTERS BEYOND CONVERGENCE. A stated first-class goal of this project is
            # to "empirically measure/compare/validate ML algorithms ... even at small
            # performance differences". Comparing this model against XGBoost while it alone is
            # handicapped by unscaled inputs measures the DEFECT, not the algorithm. Any
            # linear-vs-tree conclusion drawn before this fix is confounded.
            #
            # MODEL CHANGE: logistic_regression's predictions WILL differ from Run 15's. That
            # is a correction, not a regression. See RUN_17_PLAN.
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    class_weight=cfg.class_weight,
                    random_state=cfg.random_state,
                ),
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
        # name -> "ExceptionType: message" for any base model that failed its out-of-fold
        # step and was dropped under allow_base_model_dropout=True. EMPTY IS THE ONLY
        # HEALTHY STATE: a non-empty dict means the ensemble is incomplete and any
        # cross-algorithm comparison drawn from this run is missing a candidate. Written to
        # the run artifacts so the incompleteness cannot be lost. See
        # EnsembleConfig.allow_base_model_dropout.
        self.dropped_models_: dict[str, str] = {}
        # Set at the end of fit(): roster / trained / dropped / complete. Written into the run
        # artifacts so that "the ensemble was complete" is a CHECKED, RECORDED fact rather
        # than an assumption. See the block at the end of fit().
        self.ensemble_completeness_: dict = {}
        #: Features that were declared REAL but arrived constant. Empty unless
        #: allow_zero_variance_features=True let the run proceed anyway -- in which case the
        #: fact is RECORDED here and in the run artifacts, not merely logged and forgotten.
        self.zero_variance_features_: list[str] = []

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

    def _leakfree_oof(self, name, model, X_tab_fit, X_seq_fit, y_fit,
                      gene_fit, cv_splits):
        """Leak-free OOF for the stacking meta-learner (Level 2). Per
        gene-disjoint inner fold: recompute n_pathogenic_in_gene (+
        gene_has_known_disease) from fold-train rows only (unseen fold-val genes
        -> 0), clone the estimator, fit on fold-train, predict fold-val. Mirrors
        the cnn_1d/catboost/else dispatch used in fit()."""
        from sklearn.base import clone

        y_fit = np.asarray(y_fit)
        oof = np.zeros(len(y_fit))
        npig, ghkd = "n_pathogenic_in_gene", "gene_has_known_disease"
        has_npig = npig in X_tab_fit.columns
        for tr, va in cv_splits:
            if has_npig and name != "cnn_1d":
                Xtr = X_tab_fit.iloc[tr].copy()
                Xva = X_tab_fit.iloc[va].copy()
                g_tr = gene_fit.iloc[tr]
                counts = pd.Series((y_fit[tr] == 1).astype(int)).groupby(g_tr.values).sum()
                tr_map = gene_fit.iloc[tr].map(counts).fillna(0).astype(int).to_numpy()
                va_map = gene_fit.iloc[va].map(counts).fillna(0).astype(int).to_numpy()
                Xtr[npig] = tr_map
                Xva[npig] = va_map
                if ghkd in Xtr.columns:
                    Xtr[ghkd] = (tr_map > 0).astype(int)
                    Xva[ghkd] = (va_map > 0).astype(int)
            else:
                Xtr = X_tab_fit.iloc[tr]
                Xva = X_tab_fit.iloc[va]
            if name == "cnn_1d":
                # _require_x_seq guarantees X_seq_fit is not None whenever a
                # SEQUENCE_MODELS member is active. Asserted rather than assumed:
                # if the guarantee ever breaks, fail here with a clear cause
                # instead of an AttributeError on None.
                assert X_seq_fit is not None, (
                    "cnn_1d reached _leakfree_oof with X_seq_fit=None; "
                    "_require_x_seq should have refused in fit()."
                )
                Xtr_in, Xva_in = X_seq_fit.iloc[tr], X_seq_fit.iloc[va]
            elif name == "catboost":
                Xtr_in, Xva_in = Xtr, Xva
            else:
                Xtr_in, Xva_in = Xtr.values, Xva.values
            m = clone(model)
            m.fit(Xtr_in, y_fit[tr])
            oof[va] = m.predict_proba(Xva_in)[:, 1]
        return oof

    def _assert_no_dead_features(self, X_tab: pd.DataFrame) -> None:
        """Every feature declared REAL must actually vary. Silent-zeros die here.

        A connector with a missing source file returns zeros rather than raising (omim.py:105
        is the canonical example). The column then trains, contributes nothing, and inflates
        the feature contract with something that does not exist. Run 15 shipped five such
        columns and nobody knew until the drift work forced someone to look at the data.

        The launcher's file-existence checks cannot catch this -- a present-but-empty file, a
        schema change, or a failed gene-symbol join all produce zeros with the file sitting
        right there. So the assertion is made against the FEATURE, at the moment of fit.

        Raises
        ------
        ValueError
            If any TABULAR_FEATURES column is constant across a cohort large enough for that
            to be meaningful. Set EnsembleConfig.allow_zero_variance_features=True to
            downgrade to a warning -- but the honest fix is PHASE_2_FEATURES.
        """
        # ONE definition of "dead", shared with scripts/run_phase2_eval.py's pre-flight census.
        # Two copies of this rule would disagree within a month -- root pattern (a).
        census = feature_census(X_tab)
        declared = census["declared"]
        dead = census["dead"]
        n_rows = len(X_tab)

        logger.info(format_feature_census(census, n_rows))

        if not dead:
            logger.info(
                "Zero-variance guard: all %d declared tabular features vary. No silent-zeros.",
                len(declared),
            )
            return

        msg = (
            f"{len(dead)} declared-real feature(s) are CONSTANT across {n_rows:,} variants "
            f"and carry ZERO information: {dead}\n"
            f"\n"
            f"A constant feature cannot be split on, contributes nothing to any model, and "
            f"can NEVER signal drift (p01 == p99 => Population Stability Index is identically "
            f"0.0, forever). It is also a lie in the {EXPECTED_TABULAR_FEATURE_COUNT}-feature "
            f"contract: the roster counts it, the science does not have it.\n"
            f"\n"
            f"This is almost always a connector that silently stubbed to zeros because its "
            f"source file was absent, unreadable, or failed to join (see omim.py:105 -- "
            f"'if gene_table.empty: result[...] = 0; return result', with no log and no "
            f"raise). Check the connector's source path and its join keys.\n"
            f"\n"
            f"If the feature is genuinely NOT YET COMPUTED, it belongs in PHASE_2_FEATURES -- "
            f"that is exactly what PHASE_2_FEATURES is for, and it is currently empty. Move "
            f"it there and drop EXPECTED_TABULAR_FEATURE_COUNT accordingly.\n"
            f"\n"
            f"To proceed anyway, set EnsembleConfig.allow_zero_variance_features=True. Doing "
            f"so trains models on columns that do not exist."
        )

        if n_rows < self.config.zero_variance_min_rows:
            # Too few rows to distinguish a dead feature from an unlucky draw (a small
            # synthetic fixture can easily produce an all-zero binary flag). Warn, don't raise.
            logger.warning(
                "Zero-variance guard: %d constant feature(s) in only %d rows (< %d) -- too "
                "small to be conclusive, so this is a WARNING, not a failure: %s",
                len(dead), n_rows, self.config.zero_variance_min_rows, dead,
            )
            return

        if self.config.allow_zero_variance_features:
            logger.error("ZERO-VARIANCE FEATURES TOLERATED BY CONFIG.\n%s", msg)
            self.zero_variance_features_ = list(dead)
            return

        raise ValueError(f"ZERO-VARIANCE FEATURES (roadmap 6.21)\n\n{msg}")

    def _require_x_seq(self, X_seq, models, method: str) -> None:
        """Refuse to run a sequence model without sequence. Raises BEFORE any fit.

        X_seq is optional (2026-07-19, Part 3) so that callers with no sequence windows can
        say so instead of manufacturing a placeholder frame to satisfy the signature --
        `scripts/train.py:523-525` documents exactly that workaround.

        Optional does not mean tolerated. If a model in SEQUENCE_MODELS is active and X_seq
        is None, there is no honest way to proceed: fabricating a placeholder is what roadmap
        6.28 recorded as training cnn_1d on invented sequence and reporting a number. So this
        raises, names the model, and states the remedy.

        Placed before any estimator is fitted, so a misconfigured run costs a second rather
        than hours of paid compute.
        """
        if X_seq is not None:
            return
        active = sorted(n for n in models if n in SEQUENCE_MODELS)
        if not active:
            return
        raise ValueError(
            "{}() received X_seq=None, but these models take the sequence branch: {}.\n"
            "\n"
            "A sequence model cannot run without sequence windows, and a placeholder frame "
            "is not a substitute -- it trains the model on fabricated sequence and reports a "
            "number (roadmap 6.28).\n"
            "\n"
            "Either pass a 2-column [fasta_seq_ref, fasta_seq_alt] DataFrame, built with\n"
            "    seq_window_join.attach_delta_windows(...).windows\n"
            "or remove the sequence model(s) first:\n"
            "    ensemble.base_estimators.pop({!r}, None)\n"
            "Launchers expose this as --skip-cnn.\n"
            "\n"
            "Refused before any estimator was fitted; no compute was spent.".format(
                method, ", ".join(active), active[0]
            )
        )

    def fit(
        self, X_tab: pd.DataFrame, X_seq: "pd.DataFrame | None", y: pd.Series,
        gene_symbol: "pd.Series | None" = None,
        X_tab_cal_ext: "pd.DataFrame | None" = None,
        X_seq_cal_ext: "pd.DataFrame | None" = None,
        y_cal_ext: "pd.Series | None" = None,
        gene_symbol_cal_ext: "pd.Series | None" = None,
    ) -> "VariantEnsemble":
        """
        X_seq: A 2-COLUMN [fasta_seq_ref, fasta_seq_alt] DataFrame, row-aligned to X_tab.
               Build it with seq_window_join.attach_delta_windows(...).windows.

        THE ANNOTATION USED TO SAY `X_seq: pd.Series`. IT WAS FALSE (fixed 2026-07-15,
        roadmap 6.28) -- and its falsity is why five tests could exercise a code path
        production has never executed.

        scripts/train.py has always passed a DataFrame: `X_seq_train = _att_train.windows`.
        Every test passed a Series, because the signature told them to. `_encode_batch`
        accepted both, and that "tolerance" is where the two realities diverged:

            DataFrame -> ref and alt are distinct -> the delta channels carry signal.
            Series    -> ref = alt = the one column -> `oh_alt - oh_ref` is IDENTICALLY
                         ZERO. 4 of 13 channels dead, 8 duplicated. cnn_1d degenerates to
                         a one-hot sequence classifier with no variant information at all,
                         fits happily, and reports a number.

        So the suite was green on a mode the run never uses, for the sequence model's only
        input, and the type hint was the instruction manual for getting there. The
        fixture in test_ensemble_save_load_with_cnn1d is `Name: fasta_seq, dtype: object`
        -- a Series named after the legacy column that is 100% NULL across all 4,420,180
        cohort rows (measured 2026-07-15).

        Roadmap 7c: a gate that checks a PROXY instead of the thing it protects is not a
        gate. A test that exercises a shape production never sends is testing a proxy.
        """
        from sklearn.model_selection import train_test_split as _tts

        y_arr = np.asarray(y)
        logger.info(
            "Training ensemble: %d samples, %d pathogenic.",
            len(y_arr),
            int(y_arr.sum()),
        )

        # Assert every declared-real feature actually exists in the data (roadmap 6.21).
        # This runs BEFORE any model is fitted, so a silent-zero costs a second, not eleven
        # hours of paid compute followed by a published algorithm comparison built on a
        # feature space that was partly imaginary.
        self._assert_no_dead_features(X_tab)

        # X_seq may be None (Part 3, 2026-07-19). Refuse loudly if a sequence model is
        # active without it, before a single estimator is fitted.
        self._require_x_seq(X_seq, self.base_estimators, "fit")

        # Calibration fold selection (W2 PATH-1, 2026-07-11).
        # If an EXTERNAL calibration partition is supplied (v2 gene-disjoint
        # 'tune' partition), use the ENTIRE incoming data as the fit fold and
        # the external partition as the calibration fold, so the post-hoc
        # isotonic calibration is fit on genes the models never trained on
        # (honest, gene-generalizing probabilities). Otherwise fall back to the
        # legacy self-carve (15% label-stratified split of the incoming data);
        # that path is byte-for-byte unchanged for backward compatibility.
        if X_tab_cal_ext is not None:
            idx_fit = np.arange(len(y_arr))
            X_tab_fit = X_tab.reset_index(drop=True)
            X_tab_cal = X_tab_cal_ext.reset_index(drop=True)
            X_seq_fit = None if X_seq is None else X_seq.reset_index(drop=True)
            X_seq_cal = X_seq_cal_ext.reset_index(drop=True)
            y_fit = y_arr
            y_cal = np.asarray(y_cal_ext)
            logger.info(
                "Calibrating on EXTERNAL gene-disjoint partition: fit=%d, cal=%d.",
                len(y_fit), len(y_cal),
            )
        else:
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
            X_seq_fit = (None if X_seq is None
                         else X_seq.iloc[idx_fit].reset_index(drop=True))
            X_seq_cal = (None if X_seq is None
                         else X_seq.iloc[idx_cal].reset_index(drop=True))
            y_fit = y_arr[idx_fit]
            y_cal = y_arr[idx_cal]

        # Level 2 (INCIDENT_2026-06-13): gene-disjoint inner CV + per-fold
        # train-only n_pathogenic_in_gene recompute when gene labels are
        # available; otherwise the legacy StratifiedKFold path (unchanged).
        gene_fit = None
        _gf = (pd.Series(np.asarray(gene_symbol)[idx_fit]).reset_index(drop=True)
               if gene_symbol is not None else None)
        if _gf is not None and _gf.nunique() >= self.config.n_folds:
            gene_fit = _gf
            cv = GroupKFold(n_splits=self.config.n_folds)
            cv_splits = list(cv.split(X_tab_fit, y_fit, groups=gene_fit))
            logger.info(
                "Level 2: gene-disjoint inner CV + per-fold train-only "
                "n_pathogenic_in_gene recompute (%d genes in fit set).",
                gene_fit.nunique(),
            )
        else:
            if _gf is not None:
                logger.warning(
                    "Level 2: only %d genes in fit set (< n_folds=%d) -- using "
                    "StratifiedKFold inner CV.", _gf.nunique(), self.config.n_folds
                )
            cv = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            cv_splits = list(cv.split(X_tab_fit, y_fit))

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
                # Always pass DataFrame - CatBoost needs column names for
                # categorical feature resolution. Handles numeric-only
                # DataFrames correctly when no cat columns are present.
                X_input_fit = X_tab_fit
                X_input_cal = X_tab_cal
            else:
                X_input_fit = X_tab_fit.values
                X_input_cal = X_tab_cal.values

            try:
                if gene_fit is not None:
                    oof = self._leakfree_oof(
                        name, model, X_tab_fit, X_seq_fit, y_fit, gene_fit, cv_splits
                    )
                else:
                    oof = cross_val_predict(
                        model,
                        X_input_fit,
                        y_fit,
                        cv=cv,
                        method="predict_proba",
                        n_jobs=1,
                    )[:, 1]
            except Exception as exc:
                # FAIL LOUD (2026-07-13). See EnsembleConfig.allow_base_model_dropout for
                # the full rationale. Previously this swallowed the exception, and the model
                # vanished from the ensemble with nothing but a log line to show for it.
                if not self.config.allow_base_model_dropout:
                    raise RuntimeError(
                        f"Base model {name!r} FAILED during out-of-fold (OOF) prediction, so "
                        f"it could not be fitted and would have been silently dropped from "
                        f"the ensemble.\n"
                        f"  underlying error: {type(exc).__name__}: {exc}\n"
                        f"This is now a hard stop. A dropped base model does not appear in "
                        f"the run report as a failure -- it appears as an algorithm that was "
                        f"never a candidate, which corrupts the model-comparison results this "
                        f"project exists to produce.\n"
                        f"Fix the underlying error, or -- if this model is knowingly expected "
                        f"to fail -- set EnsembleConfig(allow_base_model_dropout=True) to "
                        f"proceed with an explicitly incomplete, explicitly recorded ensemble."
                    ) from exc

                # Opt-in dropout: permitted, but never quiet.
                logger.error(
                    "  %s OOF FAILED: %s - DROPPING this model from the ensemble "
                    "(allow_base_model_dropout=True). The ensemble is now INCOMPLETE.",
                    name,
                    exc,
                    exc_info=True,
                )
                self.dropped_models_[name] = f"{type(exc).__name__}: {exc}"
                # NOTE: this column is discarded by the `valid_cols` filter below (the model
                # never enters trained_models_), so its value is immaterial. Kept explicit
                # rather than left at the np.zeros default so a future refactor that removes
                # the filter degrades to an uninformative prior rather than to p=0.
                oof_preds[:, model_idx] = 0.5
                continue

            oof_preds[:, model_idx] = oof
            model.fit(X_input_fit, y_fit)

            if name in _RECALIBRATE:
                logger.info("  %s - applying isotonic calibration ...", name)
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
                _fold_indices = [test_idx for _, test_idx in cv_splits]
                np.save(_oof_idx_path, np.concatenate(_fold_indices))
                with open(_meta_path, "w") as _f:
                    json.dump({
                        "name": name,
                        "oof_auroc": float(roc_auc_score(y_fit, oof)),
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
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

        # An ensemble with no surviving base models cannot be stacked. Without this, the
        # meta-learner would be handed a (n, 0) matrix and fail somewhere deep inside
        # scikit-learn with an error that says nothing about the real cause.
        if not valid_cols:
            raise RuntimeError(
                "EVERY base model failed its out-of-fold step; there is nothing to stack. "
                "Dropped models and their causes:\n  "
                + "\n  ".join(f"{n}: {e}" for n, e in self.dropped_models_.items())
            )

        # The ensemble is smaller than the roster. Say so, unmissably, in the run log --
        # a run that quietly compares 12 algorithms when 13 were configured is a corrupt
        # comparison, and this is the last point at which that fact is still visible.
        if self.dropped_models_:
            logger.error(
                "ENSEMBLE IS INCOMPLETE: %d of %d base models were DROPPED and are absent "
                "from this run's results: %s. Any cross-algorithm comparison from this run "
                "is missing those candidates.",
                len(self.dropped_models_),
                len(self.base_estimators),
                ", ".join(
                    f"{n} ({e})" for n, e in sorted(self.dropped_models_.items())
                ),
            )

        # -------------------------------------------------------------------------------
        # ENSEMBLE COMPLETENESS -- recorded, not assumed. (2026-07-13)
        #
        # "The run trained 13 models" and "the roster has 13 models" were DIFFERENT
        # STATEMENTS, and nothing ever checked the first one. On 2026-07-13 it emerged that
        # the Kolmogorov-Arnold Network had been raising NameError inside imodelsx 1.0.13 --
        # and being silently swallowed -- in EVERY Continuous Integration run since May. The
        # ensemble trained twelve models, reported normal metrics, and no artifact anywhere
        # said a model was missing.
        #
        # This writes the roster, the models actually trained, and any dropouts onto the
        # fitted object, so that completeness is a RECORDED FACT that downstream reporting
        # can assert on -- not an assumption inherited from the config.
        # -------------------------------------------------------------------------------
        self.ensemble_completeness_ = {
            "roster": sorted(self.base_estimators),
            "trained": sorted(self.trained_models_),
            "dropped": dict(sorted(self.dropped_models_.items())),
            "n_roster": len(self.base_estimators),
            "n_trained": len(self.trained_models_),
            "complete": len(self.trained_models_) == len(self.base_estimators),
        }
        if self.ensemble_completeness_["complete"]:
            logger.info(
                "Ensemble COMPLETE: all %d configured base models trained (%s).",
                len(self.base_estimators),
                ", ".join(sorted(self.trained_models_)),
            )

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
            "OOF blend AUROC: %.4f  (LR stacker: %.4f  delta=%.4f)",
            blend_auroc,
            lr_auroc,
            blend_auroc - lr_auroc,
        )

        # Free unfitted base_estimators from memory (Issue H).
        self.base_estimators.clear()
        return self

    def predict_proba(
        self, X_tab: pd.DataFrame, X_seq: "pd.DataFrame | None" = None
    ) -> np.ndarray:
        """X_seq: 2-column [fasta_seq_ref, fasta_seq_alt] frame. See fit() -- the
        `pd.Series` annotation here was false for the production path (2026-07-15)."""
        if not self.trained_models_:
            raise RuntimeError("Call fit() before predict_proba().")
        self._require_x_seq(X_seq, self.trained_models_, "predict_proba")
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

    def predict(
        self, X_tab: pd.DataFrame, X_seq: "pd.DataFrame | None" = None
    ) -> np.ndarray:
        return (self.predict_proba(X_tab, X_seq)[:, 1] > 0.5).astype(int)

    def evaluate(
        self, X_tab: pd.DataFrame, X_seq: "pd.DataFrame | None", y: pd.Series
    ) -> pd.DataFrame:
        self._require_x_seq(X_seq, self.trained_models_, "evaluate")
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
