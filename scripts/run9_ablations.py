"""
scripts/run9_ablations.py
==========================
Leave-one-class-out (LOCO) ablation harness for Run 9+.

NOT a replacement for scripts/run_phase2_eval.py. Coexists with it. The
mainline script runs the expensive data-prep once and persists scaled
feature frames to <o>/splits/; this harness reads those, applies an
ablation mask by zeroing matching-prefix columns, retrains the ensemble,
and writes the Rule-5 artefact set to its own output directory.

Why zeroing works as an ablation: the splits written by DataPrepPipeline
are already StandardScaler-transformed (mean 0, std 1). Setting a column
to 0 means every row has the training-set mean for that feature, which
is the correct "ignore this feature" semantic — tree models see zero
information gain, linear models see zero contribution.

SCHEMA CALIBRATION — 2026-04-19
--------------------------------
ABLATION_MASKS in this file was calibrated against the 78-column schema
confirmed by direct `_engineer_features` probe on 2026-04-19. The tabular
contract has since changed; its one definition is
variant_ensemble.EXPECTED_TABULAR_FEATURE_COUNT (91 since the 2026-09-22 quarantine).

COVERAGE GAP measured 2026-07-18: 32 of the 95 contract features match no
ablation prefix. Most are core variant descriptors with no external source to
ablate (ref_len, is_snv, consequence_severity and similar). But six external
annotation families have NO mask at all, so their contribution cannot currently
be measured by this harness:

    RNA-seq        rnaseq_mean_log_tpm, rnaseq_detection_rate, rnaseq_log2_cv,
                   rnaseq_log2fc, rnaseq_de_neglog10p
    COSMIC         cosmic_recurrence, cosmic_sig_tier
    KEGG           kegg_pathway_count, kegg_disease_pathway_flag
    GenomicLM/NT   genomiclm_delta_norm, genomiclm_llr
    Reactome       reactome_pathway_count
    hetero-GNN     hetero_gnn_score   (no_gnn covers gnn_score, not this one)

Adding masks for these is experimental design, not a bug fix, and is left open
deliberately rather than guessed at.

Do NOT assume your on-disk split parquet matches this schema — check it:

    python -c "import pandas as pd; \\
        print(pd.read_parquet('outputs/<run>/splits/X_train.parquet').shape)"

If your persisted parquet has 46 columns, it is stale (pre-4/19) and
needs regeneration via `scripts/run_phase2_eval.py` before any ablation
will produce meaningful results. The harness logs a WARNING when an
ablation matches zero columns, so stale parquet produces loud evidence
of the mismatch rather than silent non-ablation.

GNN CAVEAT (see PATCHES_AND_INTEGRATION.md Patch 6)
---------------------------------------------------
The persisted `gnn_score` column is likely a DEFAULT constant (0.0 for
most rows), not real GNN predictions — because DataPrepPipeline writes
splits BEFORE run_phase2_eval.py trains the GNN. The in-memory
overwrite at run_phase2_eval.py:271 never reaches disk.

Consequences for `no_gnn` ablation:
  - If you run ablations against splits written by unpatched mainline:
    the column is already ≈0, so ablation is a no-op. Harness logs it.
  - If you apply Patch 6a (re-inject real GNN scores in the harness):
    the ablation compares real-GNN vs zeroed-GNN. Correct.
  - If you apply Patch 6b (train GNN inside DataPrepPipeline): cleanest
    — splits contain real GNN scores from the start.

Typical Run 9 workflow
----------------------
# Step 1: produce the full baseline + persist splits (the slow part)
python scripts/run_phase2_eval.py `
    --clinvar data/processed/clinvar_grch38.parquet `
    --gnomad  data/processed/gnomad_v4_exomes.parquet `
    --spliceai      data/external/spliceai/spliceai_index.parquet `
    --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz `
    --gtex-genes BRCA1 BRCA2 TP53 PTEN ATM `
    --string-db  auto `
    --output outputs/run9/full

# Step 2: run the five LOCO ablations against the same splits
foreach ($A in "no_spliceai","no_gnn","no_alphamissense",
               "no_conservation","no_population_af") {
    python scripts/run9_ablations.py `
        --splits-dir outputs/run9/full/splits `
        --ablation $A `
        --run-id run9 `
        --output-dir outputs/run9/$A
}

# Step 3: aggregate
python scripts/run9_aggregate_ablations.py --run-dir outputs/run9/

Ablation mask invariants
------------------------
ABLATION_MASKS is the single source of truth for feature-class groupings.
Every new feature class added to DataPrepPipeline MUST have an entry here
or Rule 6 of docs/RUN9_SCIENTIFIC_DESIGN.md is violated silently. The
prefix-match logic warns (does not fail) when a mask matches no columns,
so schema drift produces a log line rather than a silent miscalibration.
"""

from __future__ import annotations

# PACKAGE 1, SET 1b (2026-11-08): the set-1a transition refusal (ABLATION_WRITER_MIGRATION_REQUIRED) is removed together
# with its migration. Each evaluated split is scored through ONE bound capture (VariantEnsemble.score_split); test and
# validation predictions, the out-of-fold rows (with their composed training-split identity), calibration points and
# the selected predictor's member-score attributions are all reduced from those captures; capabilities are checked
# before any model is fitted; every requested deliverable either exists or the run fails, and only then is the
# completion manifest written and the ablation row appended.
import argparse
import os
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("run9_ablations")


# ---------------------------------------------------------------------------
# Ablation masks — calibrated against the 78-column schema
# ---------------------------------------------------------------------------
#
# Columns verified present on 2026-04-19 via DataPrepPipeline._engineer_features
# probe (50-row synthetic input; all 78 columns produced regardless of
# annotation config because _engineer_features uses df.get(col, default)):
#
#   Allele frequency (6):   af_raw, af_log10, af_is_absent,
#                           af_is_ultra_rare, af_is_rare, af_is_common
#   Length/type (7):        ref_len, alt_len, len_diff, is_snv,
#                           is_insertion, is_deletion, is_indel
#   Consequence (6):        consequence_severity, is_loss_of_function,
#                           is_missense, is_synonymous, is_splice, in_coding
#   Functional scores (9):  cadd_phred, sift_score, polyphen2_score,
#                           revel_score, phylop_score, gerp_score,
#                           alphamissense_score, splice_ai_score, eve_score
#   Binary flags (5):       cadd_high, sift_deleterious,
#                           polyphen_probably_damaging, revel_pathogenic,
#                           n_tools_pathogenic
#   Gene-level (4):         gene_constraint_oe, gene_is_constrained,
#                           n_pathogenic_in_gene, gene_has_known_disease
#   UniProt (2):            has_uniprot_annotation,
#                           n_known_pathogenic_protein_variants
#   GTEx (6):               gtex_max_tpm, gtex_n_tissues_expressed,
#                           gtex_tissue_specificity, gtex_is_eqtl,
#                           gtex_min_eqtl_pval, gtex_max_abs_effect
#   Coding position (1):    codon_position
#   dbSNP (1):              dbsnp_af
#   OMIM (2):               omim_n_diseases, omim_is_autosomal_dominant
#   ClinGen (1):            clingen_validity_score
#   HGMD (2):               hgmd_is_disease_mutation, hgmd_n_reports
#   LOVD (1):               lovd_variant_class
#   Chromosome (3):         is_autosome, is_sex_chrom, is_mitochondrial
#   GNN (1):                gnn_score        [SEE GNN CAVEAT ABOVE]
#   Splice mechanics (4):   maxentscan_score, dist_to_splice_site,
#                           exon_number, is_canonical_splice
#   Structure + active site: QUARANTINED 2026-09-22 (quarantine_policy.py) -- in no model, no ablation group
#   1KG pop AF (5):         af_1kg_afr, af_1kg_eur, af_1kg_eas,
#                           af_1kg_sas, af_1kg_amr
#   FinnGen (3):            finngen_af_fin, finngen_af_nfsee,
#                           finngen_enrichment
#   ESM-2 (1):              esm2_delta_norm  [stub until HGVSp parser]
#   gnomAD constraint (4):  pli_score, loeuf, syn_z, mis_z
#
# Total: 78 columns.
#
# RUN 9 CORE (6 ablations):
#   full, no_spliceai, no_gnn, no_alphamissense,
#   no_conservation, no_population_af
#
# RUN 10+ EXTENSIONS (8 additional):
#   no_esm2, no_eve, no_constraint_scores, no_gtex,
#   no_disease_dbs, no_splice_mechanics, no_individual_predictors
#
ABLATION_MASKS: dict[str, list[str]] = {
    # ── Baseline ──────────────────────────────────────────────────────────
    "full": [],
    # -- Run 15: gene-prevalence memorization probes ----------------------
    "no_gene_prevalence": [
        # n_pathogenic_in_gene is computed from the gene's ClinVar pathogenic
        # counts -> directly circular with the label. Strict single-column
        # ablation (top feature every run; importance 391 in Run 15).
        "n_pathogenic_in_gene",
    ],
    "no_gene_level": [
        # Broader gene-identity test. Run this if no_gene_prevalence barely
        # moves AUROC, to check proxy-recovery via the disease-gene flag.
        # gene_constraint_oe / gene_is_constrained are currently dead but are
        # included for completeness. Does NOT include pli/loeuf/syn_z/mis_z,
        # which are legitimate constraint biology, not label-derived.
        "n_pathogenic_in_gene",
        "gene_has_known_disease",
        "gene_constraint_oe",
        "gene_is_constrained",
    ],
    # ── Run 9 core LOCO (6) ───────────────────────────────────────────────
    "no_spliceai": [
        # SpliceAI delta-score column only. `is_splice` stays live because
        # it's a CONSEQUENCE-based flag (derived from VEP consequence
        # strings), not a SpliceAI score. `maxentscan_score`,
        # `dist_to_splice_site`, etc. also stay live (mechanical splice
        # features, not SpliceAI). To ablate ALL splice signals, use
        # no_spliceai + no_splice_mechanics in sequence.
        "splice_ai_",
    ],
    "no_gnn": [
        # See GNN caveat in module docstring. Without Patch 6, this column
        # is a default constant on disk and the ablation is a no-op.
        "gnn_score",
        "gnn_embed_",
        "gnn_",
    ],
    "no_kan": [
        # KAN is a MODEL-LEVEL ablation, not a feature-level one. KAN uses
        # the same 78 input features as every other base estimator, so there
        # are no feature columns to zero. Instead, this ablation is handled
        # by the CLI flag --skip-kan, which pops KAN from the ensemble's
        # base_estimators before fit(). An empty prefix list here documents
        # that intent and causes apply_ablation() to log "matched zero columns",
        # which is the correct signal: no feature zeroing, model removal only.
        #
        # To run the no_kan ablation:
        #   python scripts/run9_ablations.py --ablation no_kan --skip-kan ...
        #
        # The harness enforces this pairing at runtime (see main()).
    ],
    "no_alphamissense": [
        "alphamissense_",
    ],
    "no_conservation": [
        # Inter-species conservation stack. phylop_score and gerp_score in
        # the current schema; prefix covers future additions
        # (phastcons_*, siphy_*).
        "phylop_",
        "gerp_",
        "phastcons_",
        "siphy_",
    ],
    "no_population_af": [
        # ALL allele-frequency signals: primary AF features (af_*),
        # 1000 Genomes population AFs (af_1kg_*), dbSNP AF, and FinnGen
        # (Finnish isolate) AFs. Tests the "rare = pathogenic" baseline
        # that AUROC is thought to depend on heavily.
        #
        # Note: `af_1kg_*` share the `af_` prefix so a single `af_` entry
        # covers all 6 primary + 5 1KG columns. Explicit `finngen_` and
        # `dbsnp_af` entries cover the rest.
        "af_",
        "finngen_",
        "dbsnp_af",
    ],
    # ── Run 10+ extensions ────────────────────────────────────────────────
    "no_esm2": [
        # Currently stub (returns 0.0 until HGVSp parser lands; see
        # INCIDENT_2026-04-17_esm2-hgvsp-parser.md). Ablation is a no-op
        # until ESM-2 becomes non-stub in Run 10+.
        "esm2_",
    ],
    "no_eve": [
        # Similar plumbing-gap issue to ESM-2. eve_score column exists but
        # may contain defaults unless EVE annotation is plumbed.
        "eve_",
    ],
    # "no_alphafold" was REMOVED 2026-09-23: its four features are quarantined (quarantine_policy.py) and in no
    # model, so apply_ablation would match zero columns, only warn, and report a ~0 delta as if it were a result.
    "no_constraint_scores": [
        # gnomAD gene-level constraint scores: pLI (probability of LoF
        # intolerance), LOEUF (loss-of-function observed/expected upper
        # fraction), syn_z (synonymous z-score), mis_z (missense z-score).
        # Ablating these tests whether gene-level intolerance is a
        # dominant predictor vs. per-variant features.
        "pli_score",
        "loeuf",
        "syn_z",
        "mis_z",
    ],
    "no_gtex": [
        # Tissue expression and eQTL features from GTEx.
        "gtex_",
    ],
    "no_disease_dbs": [
        # LABEL-LEAKAGE CHECK. HGMD, OMIM, LOVD, ClinGen are curated
        # disease-variant databases. If these features strongly predict
        # ClinVar labels, it may be because some variants appear in both
        # sources — not because of independent biological signal.
        # A large `no_disease_dbs` Δ-AUROC should be investigated for
        # leakage before being claimed as a biological result.
        "omim_",
        "hgmd_",
        "lovd_",
        "clingen_",
    ],
    "no_splice_mechanics": [
        # Mechanical splice features (NOT SpliceAI). Distance to splice
        # sites, exon numbering, canonical vs non-canonical splice sites,
        # MaxEntScan splice-site scores.
        "maxentscan_",
        "dist_to_splice_site",
        "exon_number",
        "is_canonical_splice",
    ],
    "no_individual_predictors": [
        # Raw outputs and derived binary flags from the individual in-silico
        # predictors (CADD, SIFT, PolyPhen-2, REVEL) PLUS the ensemble
        # n_tools_pathogenic flag. Tests whether the modern predictors
        # (AlphaMissense, ESM-2, EVE) capture signal independently of the
        # legacy predictor stack.
        "cadd_phred",
        "cadd_high",
        "sift_score",
        "sift_deleterious",
        "polyphen2_score",
        "polyphen_probably_damaging",
        "revel_score",
        "revel_pathogenic",
        "n_tools_pathogenic",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parent.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions() -> dict[str, str]:
    v: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "torch_geometric",
        "transformers",
        "shap",
    ):
        try:
            mod = __import__(pkg.replace("-", "_"))
            v[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            v[pkg] = "not_installed"
    try:
        import torch

        v["cuda"] = torch.version.cuda or "cpu_only"
        v["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        pass
    return v


def _hash_df(df: pd.DataFrame) -> str:
    """Content hash of a dataframe's values (order-sensitive)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def apply_ablation(
    X: pd.DataFrame,
    ablation: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Zero matching-prefix columns. Returns (X_ablated, zeroed_col_names).

    The 'full' ablation returns X unchanged. For every other ablation,
    a COPY of X is returned — the caller's X is never mutated.
    """
    if ablation not in ABLATION_MASKS:
        raise ValueError(
            f"Unknown ablation {ablation!r}. " f"Valid: {sorted(ABLATION_MASKS)}"
        )
    prefixes = ABLATION_MASKS[ablation]
    if not prefixes:
        return X, []

    X_abl = X.copy()
    zeroed: list[str] = []
    for col in X_abl.columns:
        if any(col.startswith(p) or col == p for p in prefixes):
            X_abl[col] = 0.0
            zeroed.append(col)

    if not zeroed:
        logger.warning(
            "Ablation %r matched zero columns. Either the feature class "
            "is not present in this pipeline, or ABLATION_MASKS prefixes "
            "are stale. Columns seen (first 25 of %d): %s",
            ablation,
            X.shape[1],
            list(X.columns)[:25],
        )
    else:
        # Also warn if a named feature is EXPECTED to be present but the
        # resulting column count is surprisingly low (common signal of
        # stale-parquet mismatch).
        logger.info(
            "Ablation %r zeroed %d columns: %s",
            ablation,
            len(zeroed),
            zeroed[:10] + (["...trimmed"] if len(zeroed) > 10 else []),
        )

        # Schema drift check: if the total column count is well below 78,
        # warn that the parquet may be stale.
        if X.shape[1] < 70:
            logger.warning(
                "Input has only %d columns — expected ~78 per the "
                "2026-04-19 schema probe. The split parquet may be "
                "stale (pre-4/19). Regenerate via scripts/run_phase2_eval.py "
                "before drawing conclusions from this ablation.",
                X.shape[1],
            )
    return X_abl, zeroed


# ---------------------------------------------------------------------------
# Splits loader
# ---------------------------------------------------------------------------
def load_splits(
    splits_dir: Path,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load the 8-tuple produced by DataPrepPipeline.run() from a directory
    written by DataPrepPipeline._save_splits (called via the mainline
    run_phase2_eval.py).

    Required files:
        X_train.parquet, X_val.parquet, X_test.parquet
        y_train.parquet, y_val.parquet, y_test.parquet
        meta_train.parquet, meta_val.parquet, meta_test.parquet

    meta_train is required since package 1, set 1b: the out-of-fold rows are published with their training-split
    identity (variant identity, gene) and the calibration carve and inner folds are gene-disjoint like mainline.

    The y_*.parquet files use column name "label" (per DataPrepPipeline
    _save_splits line 1207: `y_train.to_frame("label").to_parquet(...)`).
    """
    required = [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
        "meta_train.parquet",
        "meta_val.parquet",
        "meta_test.parquet",
    ]
    missing = [f for f in required if not (splits_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required split files in {splits_dir}: {missing}. "
            f"Run scripts/run_phase2_eval.py first and point --splits-dir "
            f"at its <output>/splits/ directory."
        )

    out: dict[str, pd.DataFrame | pd.Series] = {
        "X_train": pd.read_parquet(splits_dir / "X_train.parquet"),
        "X_val": pd.read_parquet(splits_dir / "X_val.parquet"),
        "X_test": pd.read_parquet(splits_dir / "X_test.parquet"),
        "meta_train": pd.read_parquet(splits_dir / "meta_train.parquet"),
        "meta_val": pd.read_parquet(splits_dir / "meta_val.parquet"),
        "meta_test": pd.read_parquet(splits_dir / "meta_test.parquet"),
    }
    for split in ("train", "val", "test"):
        y_df = pd.read_parquet(splits_dir / f"y_{split}.parquet")
        # DataPrepPipeline writes the Series as a one-column frame named
        # "label" (real_data_prep.py line 1207). Handle both that and the
        # generic one-column case for forward-compat.
        col = "label" if "label" in y_df.columns else y_df.columns[0]
        out[f"y_{split}"] = y_df[col].astype(int).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Preflight (package 1, set 1b): everything an expensive run needs, checked BEFORE any model is fitted
# ---------------------------------------------------------------------------
_LOCUS_COLUMNS = ("chrom", "pos", "ref", "alt")


def preflight(args: argparse.Namespace) -> list[str]:
    """Problems that would make this run unable to finish as complete evidence; empty when it can start.

    Owner ruling 2026-11-07 (decision 38f15c7a): check capabilities before an expensive run, rather than discovering a
    missing input, an unwritable directory or an impossible deliverable after hours of training.
    """
    problems: list[str] = []
    if args.ablation == "no_kan" and not args.skip_kan:
        problems.append("--ablation no_kan requires --skip-kan: no_kan is a model-level removal of KAN, not a "
                        "feature-column zero-out, and without --skip-kan KAN would still participate")
    splits_dir = Path(args.splits_dir)
    names = [f"{kind}_{split}.parquet" for kind in ("X", "y", "meta") for split in ("train", "val", "test")]
    absent = [n for n in names if not (splits_dir / n).is_file()]
    if absent:
        problems.append(f"missing split files in {splits_dir}: {absent}")
    else:
        import pyarrow.parquet as pq

        for split in ("train", "val", "test"):
            counts = {kind: pq.ParquetFile(splits_dir / f"{kind}_{split}.parquet").metadata.num_rows
                      for kind in ("X", "y", "meta")}
            if len(set(counts.values())) != 1:
                problems.append(f"{split}: X/y/meta row counts differ {counts}")
            columns = set(pq.read_schema(splits_dir / f"meta_{split}.parquet").names)
            lacking = [c for c in (*_LOCUS_COLUMNS, "gene_symbol") if c not in columns]
            if lacking:
                problems.append(f"meta_{split}.parquet lacks {lacking} (needed for prediction-unit identity and the "
                                "gene-disjoint carve)")
    out = Path(args.output_dir).resolve()
    for directory in (out, out.parent):   # the ablation's own directory and the run-level aggregate's directory
        probe = directory
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent          # the nearest existing ancestor is where the directory would be created
        if not (probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)):
            problems.append(f"not writable: {directory} (nearest existing ancestor {probe})")
    if getattr(args, "max_train", None) is not None and args.max_train <= 0:
        problems.append("--max-train must be positive")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="LOCO ablation harness for Run 9+. ABLATION_MASKS was calibrated "
                    "against a 78-column schema on 2026-04-19; the tabular contract "
                    "is defined by EXPECTED_TABULAR_FEATURE_COUNT. See the module docstring "
                    "for the coverage gap."
    )
    p.add_argument(
        "--splits-dir",
        required=True,
        type=Path,
        help="Directory holding X_{train,val,test}.parquet, y_*.parquet, "
        "meta_{train,val,test}.parquet from a prior mainline run.",
    )
    p.add_argument(
        "--ablation",
        default="full",
        choices=sorted(ABLATION_MASKS.keys()),
        help="Feature-class ablation. 'full' zeros no columns (baseline).",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Logical run identifier (e.g. 'run9'). Written to manifest.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Artefact output root for THIS ablation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Use the SAME value across ablations in one run.",
    )
    p.add_argument(
        "--skip-nn",
        action="store_true",
        help="Exclude tabular_nn (smoke tests only). cnn_1d is ALWAYS excluded from "
             "ablations: the persisted splits carry no sequence windows.",
    )
    p.add_argument("--skip-svm", action="store_true")
    p.add_argument(
        "--skip-kan",
        action="store_true",
        help="Exclude KAN base estimator. KANClassifier already caps memory "
        "via a 100K-sample stratified subsample gate; this flag is "
        "provided as an opt-in override.",
    )
    p.add_argument(
        "--skip-mc-dropout",
        action="store_true",
        help="Exclude mc_dropout and deep_ensemble base estimators (the "
        "two neural-ensemble wrappers). Useful for CPU smoke tests "
        "where these dominate wall-clock.",
    )
    p.add_argument("--skip-attributions", action="store_true",
                   help="Do not write the selected predictor's member-score attributions (recorded as not requested).")
    p.add_argument("--skip-shap", action="store_true",
                   help="Deprecated alias of --skip-attributions (the SHAP writer explained the logistic research "
                        "candidate and is retired).")
    p.add_argument("--skip-permutation", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Subsample the TRAIN split to this many rows (val/test untouched) "
        "for fast local ablation probes; deterministic given --seed. "
        "Omit for the full split.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    t0 = time.perf_counter()
    logger.info("=" * 72)
    logger.info(
        "run9_ablations.py  run_id=%s  ablation=%s  seed=%d",
        args.run_id,
        args.ablation,
        args.seed,
    )
    logger.info("  splits_dir = %s", args.splits_dir)
    logger.info("  output_dir = %s", args.output_dir)
    logger.info("=" * 72)

    problems = preflight(args)
    if problems:
        for problem in problems:
            logger.error("PREFLIGHT: %s", problem)
        logger.error("Refused before loading any model library or fitting anything (%d problem(s)).", len(problems))
        return 2
    skip_attributions = bool(args.skip_attributions or args.skip_shap)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Delayed heavy imports (keeps --help responsive) ───────────────────
    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
    from genomic_variant_classifier.evaluation.oof_identity import compose_oof_identity
    from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter
    from genomic_variant_classifier.evaluation.split_identity import (
        SPLIT_UNIT_CONTEXT_SHA256,
        split_identity,
        split_observations,
    )
    from genomic_variant_classifier.models.variant_ensemble import (
        SELECTED_ENSEMBLE_ROW,
        EnsembleConfig,
        VariantEnsemble,
    )

    # The writer first: it retires any completion manifest an earlier run left in this directory BEFORE anything
    # is rewritten, so a crash below can never leave an old completion standing over new files.
    writer = RunArtifactWriter(
        run_id=args.run_id,
        ablation=args.ablation,
        output_dir=args.output_dir,
    )

    # ── Load splits ───────────────────────────────────────────────────────
    splits = load_splits(args.splits_dir)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    meta_train, meta_val, meta_test = splits["meta_train"], splits["meta_val"], splits["meta_test"]

    # Split identities of the splits AS LOADED (before the train subsample and before ablation), so every ablation arm
    # shares one source-row identity per variant; the ablated matrix each member actually received is bound in its
    # prediction blocks (ordered-input digest) instead.
    train_identity, train_vids = split_identity(X_train, y_train, meta_train, "train")
    val_identity, val_vids = split_identity(X_val, y_val, meta_val, "val")
    test_identity, test_vids = split_identity(X_test, y_test, meta_test, "test")

    # -- Run 15: optional TRAIN subsample for fast local probes -----------
    # Deterministic given --seed so both ablation arms share the same rows. The positions are KEPT: they are the first
    # link of the out-of-fold identity composition (training-split row -> fit input -> OOF row).
    subsample_positions = None
    X_fit, y_fit, meta_fit = X_train, y_train, meta_train
    if getattr(args, "max_train", None) and len(y_train) > args.max_train:
        rng = np.random.RandomState(args.seed)
        subsample_positions = np.sort(rng.choice(len(y_train), size=args.max_train, replace=False))
        X_fit = X_train.iloc[subsample_positions].reset_index(drop=True)
        y_fit = y_train.iloc[subsample_positions].reset_index(drop=True)
        meta_fit = meta_train.iloc[subsample_positions].reset_index(drop=True)
        logger.info(
            "Subsampled TRAIN to %d rows (seed=%d) for fast probe", args.max_train, args.seed
        )

    logger.info(
        "Splits loaded: train=%d (fit %d) val=%d test=%d features=%d",
        len(X_train),
        len(X_fit),
        len(X_val),
        len(X_test),
        X_train.shape[1],
    )

    # Hash the splits BEFORE ablation so the manifest records the canonical
    # split that all ablations share (cross-ablation comparability).
    split_hashes = {
        "X_train_full": _hash_df(X_train),
        "X_val_full": _hash_df(X_val),
        "X_test_full": _hash_df(X_test),
    }

    X_fit_abl, zeroed_train = apply_ablation(X_fit, args.ablation)
    X_val_abl, zeroed_val = apply_ablation(X_val, args.ablation)
    X_test_abl, zeroed_test = apply_ablation(X_test, args.ablation)

    if not (zeroed_train == zeroed_val == zeroed_test):
        raise ValueError(
            "Inconsistent ablation across splits -- column schema differs "
            "between train/val/test. This is a Rule-6 violation; check "
            "DataPrepPipeline for schema-drift across splits."
        )

    # ── Ensemble ──────────────────────────────────────────────────────────
    ens_cfg = EnsembleConfig(
        n_folds=args.n_folds,
        random_state=args.seed,
        model_dir=args.output_dir / "models",
        skip_kan=args.skip_kan,
        skip_mc_dropout=args.skip_mc_dropout,
    )
    ensemble = VariantEnsemble(ens_cfg)

    # Mirror mainline skip rules. KAN and mc_dropout/deep_ensemble are
    # controlled via EnsembleConfig above, not popped here.
    # cnn_1d is ALWAYS excluded from ablations (2026-07-18): the persisted splits carry no sequence windows, and a
    # convolutional branch fed a constant would produce a per-model number for a model that measured nothing.
    ensemble.base_estimators.pop("cnn_1d", None)
    logger.info("cnn_1d excluded from ablation: the persisted splits carry no sequence windows")
    if args.skip_nn:
        ensemble.base_estimators.pop("tabular_nn", None)
    if args.skip_svm or len(y_fit) > 100_000:
        ensemble.base_estimators.pop("svm", None)
        logger.info("SVM skipped: O(n^2) infeasible at n=%d", len(y_fit))

    # ── Fit ───────────────────────────────────────────────────────────────
    # gene_symbol is passed (since set 1b): the calibration carve and the inner folds are gene-disjoint, as in the
    # mainline run, instead of the legacy label-stratified carve this harness used to fall back to.
    fit_t0 = time.perf_counter()
    ensemble.fit(X_fit_abl, None, y_fit, gene_symbol=meta_fit["gene_symbol"])
    fit_sec = time.perf_counter() - fit_t0
    logger.info("Fit complete in %.1f min", fit_sec / 60)
    ensemble.save(args.output_dir / "models" / "ensemble.joblib")
    writer.register_external("models/ensemble.joblib")
    for name in ensemble.trained_models_:
        writer.register_external(f"models/ensemble_models/{name}.joblib")

    # ── ONE bound capture per evaluated split ──────────────────────────────
    rows_test = split_observations(test_identity, test_vids, unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256)
    rows_val = split_observations(val_identity, val_vids, unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256)
    scored_test = ensemble.score_split(X_test_abl, None, rows_test, split_sha256=test_identity.prediction_split_sha256)
    scored_val = ensemble.score_split(X_val_abl, None, rows_val, split_sha256=val_identity.prediction_split_sha256)
    test_results = ensemble.evaluate_scored(scored_test, y_test)
    val_results = ensemble.evaluate_scored(scored_val, y_val)

    evaluator = ClinicalEvaluator(n_bootstrap=1000, random_state=args.seed)
    report = evaluator.evaluate(
        y_true=y_test,
        y_proba=scored_test.scores(),
        meta=meta_test,
        model_name=f"{args.run_id}_{args.ablation}",
    )

    # ── Rule-5 artefacts ──────────────────────────────────────────────────
    writer.save_manifest(
        git_sha=_git_sha(),
        versions=_versions(),
        config={
            "ablation": args.ablation,
            "zeroed_columns": zeroed_train,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "max_train": args.max_train,
            "splits_dir": str(args.splits_dir),
            "split_hashes": split_hashes,
            "split_identities": {"train": train_identity.as_dict(), "val": val_identity.as_dict(),
                                 "test": test_identity.as_dict()},
            "n_train": int(len(y_train)),
            "n_fit": int(len(y_fit)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            "feature_columns": list(X_train.columns),
            "ensemble_models": list(ensemble.trained_models_.keys()),
            "calibration_carve": ensemble.calibration_carve_used_,
            "fit_seconds": round(fit_sec, 1),
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    writer.save_split_predictions(scored_test, y_test, meta_test, role="test")
    writer.save_split_predictions(scored_val, y_val, meta_val, role="val")
    writer.save_eval_report(report)
    writer.save_calibration(scored_test, y_test, report=report, meta=meta_test)
    writer.save_oof_predictions(compose_oof_identity(
        ensemble, train=train_identity, vids=train_vids, labels=y_train, meta_train=meta_train,
        subsample_positions=subsample_positions))
    writer.save_table_csv("per_model_metrics.csv", test_results)
    writer.save_table_csv("per_model_metrics_val.csv", val_results)

    required = ["manifest.json", "test_predictions.parquet", "test_prediction_blocks.json", "val_predictions.parquet",
                "val_prediction_blocks.json", "eval_report.json", "calibration_points.parquet",
                "consequence_performance.parquet", "oof_predictions.parquet", "oof_identity.json",
                "per_model_metrics.csv", "per_model_metrics_val.csv", "models/ensemble.joblib"]
    deliverables: dict[str, dict] = {}
    # Requested deliverables are NOT wrapped in broad handlers (owner ruling 2026-11-07): a failure propagates, no
    # completion manifest is written, and the run stays inspectable but is refused as completed evidence.
    if skip_attributions:
        deliverables["member_score_attributions"] = {"status": "not_requested"}
    else:
        writer.save_member_score_attributions(scored_test, background=scored_val)
        required += ["member_score_attributions.parquet", "member_score_attributions.json"]
        deliverables["member_score_attributions"] = {"status": "complete", "explained": "test",
                                                     "background": "val"}
    if args.skip_permutation:
        deliverables["permutation_importance"] = {"status": "not_requested"}
    else:
        writer.save_permutation_importance(
            ensemble,
            X_test_abl,
            None,
            y_test,
            n_repeats=5,
            sample_size=min(50_000, len(X_test_abl)),
            seed=args.seed,
        )
        required.append("feature_importance.parquet")
        deliverables["permutation_importance"] = {"status": "complete"}
    deliverables["feature_level_explanation"] = {
        "status": "unavailable", "reason_code": "FEATURE_LEVEL_EXPLANATION_UNAVAILABLE"}

    writer.complete(required, deliverables=deliverables,
                    extra={"selected_row": SELECTED_ENSEMBLE_ROW, "zeroed_columns": zeroed_train})

    # Headline ablation row (for the aggregator) -- only for a COMPLETED ablation, atomically.
    aggregator_path = args.output_dir.parent / "ablation_results.parquet"
    ens_row = test_results.loc[SELECTED_ENSEMBLE_ROW]
    ens_val = val_results.loc[SELECTED_ENSEMBLE_ROW]
    writer.append_ablation_row(
        aggregator_path,
        {
            "ablation": args.ablation,
            "auroc": float(ens_row["auroc"]),
            "auprc": float(ens_row["auprc"]),
            "mcc": float(ens_row["mcc"]),
            "brier": float(ens_row["brier"]),
            "val_auroc": float(ens_val["auroc"]),
            "val_auprc": float(ens_val["auprc"]),
            "val_mcc": float(ens_val["mcc"]),
            "n_zeroed": len(zeroed_train),
            "fit_seconds": round(fit_sec, 1),
        },
    )

    total = time.perf_counter() - t0
    logger.info("=" * 72)
    logger.info(
        "DONE  run_id=%s  ablation=%s  AUROC(test)=%.5f  AUROC(val)=%.5f  %.1f min",
        args.run_id,
        args.ablation,
        float(ens_row["auroc"]),
        float(ens_val["auroc"]),
        total / 60,
    )
    logger.info("  artefacts: %s", args.output_dir)
    logger.info("  aggregator: %s", aggregator_path)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
