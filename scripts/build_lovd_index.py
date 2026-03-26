"""
scripts/build_lovd_index.py
============================
Download and parse the LOVD (Leiden Open Variation Database) full variant
export for use as an independent external validation cohort -- Step 7B.

LOVD provides curated variants from European clinical genetics centres.
It overlaps minimally with ClinVar in terms of submitting labs, making it
a genuinely independent validation source.

Data source (free, no registration required)
---------------------------------------------
  LOVD whole-genome variant export:
  https://databases.lovd.nl/whole_genome/variants?format=tab

  Alternatively, per-gene exports are available at:
  https://databases.lovd.nl/shared/variants/{GENE}?format=tab&search_pathogenicity=

Output
------
  data/external/lovd/lovd_variants.parquet
  Columns: variant_id, chrom, pos, ref, alt, label (1=pathogenic, 0=benign),
           gene_symbol, classification_raw, lovd_id

Usage
-----
  # Download + parse
  python scripts/build_lovd_index.py \\
      --output data/external/lovd/lovd_variants.parquet

  # Validate against model
  python scripts/build_lovd_index.py \\
      --output data/external/lovd/lovd_variants.parquet \\
      --model  models/phase2_pipeline.joblib \\
      --eval-output outputs/lovd_validation

  # Download specific genes only (faster for initial testing)
  python scripts/build_lovd_index.py \\
      --genes BRCA1 BRCA2 TP53 PTEN ATM \\
      --output data/external/lovd/lovd_brca_tp53.parquet
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_lovd_index")

# LOVD classification strings -> binary label
_PATH_TERMS = {
    "pathogenic", "likely pathogenic", "definitely pathogenic",
    "probably pathogenic", "class 5", "class 4",
}
_BENIGN_TERMS = {
    "benign", "likely benign", "probably not pathogenic",
    "not pathogenic", "class 1", "class 2",
}

_LOVD_WHOLE_GENOME_URL = (
    "https://databases.lovd.nl/whole_genome/variants?format=tab"
)
_LOVD_GENE_URL = (
    "https://databases.lovd.nl/shared/variants/{gene}?format=tab"
    "&search_pathogenicity=1"
)
_REQUEST_TIMEOUT = 60
_RETRY_DELAY = 5


def _lovd_label(classification: str) -> Optional[int]:
    """Map LOVD classification string to binary label or None (VUS/unknown)."""
    c = str(classification).lower().strip()
    if any(t in c for t in _PATH_TERMS):
        return 1
    if any(t in c for t in _BENIGN_TERMS):
        return 0
    return None  # VUS, not provided, etc.


def _download_tsv(url: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Download a LOVD TSV export and return as DataFrame."""
    for attempt in range(retries):
        try:
            logger.info("Downloading %s ...", url)
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            content = resp.content.decode("utf-8", errors="replace")
            # LOVD TSV files start with comment lines beginning with ##
            lines = [l for l in content.splitlines() if not l.startswith("##")]
            if not lines:
                return None
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)
            logger.info("  Downloaded: %d rows, %d columns", len(df), len(df.columns))
            return df
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY)
    return None


def _parse_lovd_row(row: pd.Series, gene: Optional[str] = None) -> Optional[dict]:
    """
    Extract normalised variant info from a LOVD export row.

    LOVD column names vary across exports; this function tries multiple
    common name variants.
    """
    def get(*names):
        for n in names:
            if n in row.index and pd.notna(row[n]) and str(row[n]).strip():
                return str(row[n]).strip()
        return None

    chrom = get("chromosome", "Chromosome", "chrom", "VariantOnGenome/DNA")
    pos   = get("position", "Position", "pos", "VariantOnGenome/Position")
    ref   = get("ref", "Ref", "ReferenceAllele")
    alt   = get("alt", "Alt", "AlternativeAllele")

    # LOVD often encodes position as "chr17:g.43094692C>T" in VariantOnGenome/DNA
    dna_field = get("VariantOnGenome/DNA")
    if (not chrom or not pos or not ref or not alt) and dna_field:
        import re
        m = re.match(r"(?:chr)?(\w+):g\.(\d+)([ACGT]+)>([ACGT]+)", str(dna_field))
        if m:
            chrom = chrom or m.group(1)
            pos   = pos   or m.group(2)
            ref   = ref   or m.group(3)
            alt   = alt   or m.group(4)

    if not all([chrom, pos, ref, alt]):
        return None

    chrom = str(chrom).replace("chr", "").replace("Chr", "")
    if chrom == "M":
        chrom = "MT"

    try:
        pos_int = int(float(str(pos)))
    except (ValueError, TypeError):
        return None

    classification = get(
        "pathogenicity", "Pathogenicity", "classification", "Classification",
        "VariantOnGenome/ClinVar", "VariantOnGenome/Pathogenicity",
    ) or ""

    label = _lovd_label(classification)
    if label is None:
        return None  # Skip VUS and unclassified variants

    gene_symbol = gene or get("gene", "Gene", "gene_symbol", "GeneSymbol") or ""
    lovd_id = get("id", "ID", "VariantOnGenome/DBID") or ""

    return {
        "variant_id": f"{chrom}:{pos_int}:{ref}:{alt}",
        "chrom": chrom,
        "pos": pos_int,
        "ref": ref.upper(),
        "alt": alt.upper(),
        "label": label,
        "gene_symbol": gene_symbol,
        "classification_raw": classification,
        "lovd_id": lovd_id,
    }


def download_whole_genome(output_path: Path) -> pd.DataFrame:
    """Download the full LOVD whole-genome export."""
    df_raw = _download_tsv(_LOVD_WHOLE_GENOME_URL)
    if df_raw is None or df_raw.empty:
        logger.error("LOVD whole-genome download failed or returned no data.")
        raise SystemExit(1)
    return _parse_and_save(df_raw, None, output_path)


def download_gene_list(genes: list[str], output_path: Path) -> pd.DataFrame:
    """Download LOVD variants for a list of specific genes."""
    all_rows = []
    for gene in genes:
        url = _LOVD_GENE_URL.format(gene=gene)
        df_raw = _download_tsv(url)
        if df_raw is None or df_raw.empty:
            logger.warning("No data for gene %s.", gene)
            continue
        for _, row in df_raw.iterrows():
            parsed = _parse_lovd_row(row, gene=gene)
            if parsed:
                all_rows.append(parsed)
        logger.info("Gene %s: %d usable variants.", gene, sum(1 for r in all_rows if r.get("gene_symbol") == gene))
        time.sleep(0.5)  # polite rate limiting

    if not all_rows:
        logger.error("No parseable variants found for any gene.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d variants to %s", len(df), output_path)
    return df


def _parse_and_save(df_raw: pd.DataFrame, gene: Optional[str], output_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in df_raw.iterrows():
        parsed = _parse_lovd_row(row, gene=gene)
        if parsed:
            rows.append(parsed)

    if not rows:
        logger.error("No parseable classified variants in LOVD export.")
        raise SystemExit(1)

    df = pd.DataFrame(rows).drop_duplicates(subset=["variant_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d variants (%d pathogenic, %d benign) to %s",
        len(df), int(df["label"].sum()), int((df["label"] == 0).sum()), output_path,
    )
    return df


def evaluate_against_model(
    lovd_df: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
) -> None:
    """Score the LOVD variants and compute validation metrics."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score, brier_score_loss,
        roc_auc_score, roc_curve,
    )
    from src.api.pipeline import InferencePipeline

    pipeline = InferencePipeline.load(model_path)
    logger.info("Loaded model: val_auroc=%.4f", pipeline.metadata.val_auroc)

    y_true  = lovd_df["label"].values
    y_proba = pipeline.predict_proba(lovd_df)

    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    bins   = np.linspace(0, 1, 11)
    counts = np.histogram(y_proba, bins=bins)[0]
    ece    = float(sum((c / len(y_true)) * abs(fp - mp)
                       for fp, mp, c in zip(frac_pos, mean_pred, counts)))

    logger.info("LOVD external validation: AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                auroc, auprc, brier, ece)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "validation_type": "lovd_external",
        "n_variants": int(len(lovd_df)),
        "n_pathogenic": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "model_val_auroc": pipeline.metadata.val_auroc,
    }
    (output_dir / "lovd_metrics.json").write_text(json.dumps(metrics, indent=2))

    out_df = lovd_df.copy()
    out_df["predicted_proba"] = y_proba
    out_df.to_parquet(output_dir / "lovd_predictions.parquet", index=False)
    logger.info("LOVD validation results saved to %s", output_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Download and parse LOVD for external validation.")
    p.add_argument("--output", type=Path, default=Path("data/external/lovd/lovd_variants.parquet"))
    p.add_argument("--genes", nargs="*", default=None,
                   help="Gene list for targeted download (default: full genome export).")
    p.add_argument("--model", type=Path, default=None,
                   help="Path to InferencePipeline joblib; if set, evaluates after download.")
    p.add_argument("--eval-output", type=Path, default=Path("outputs/lovd_validation"))
    args = p.parse_args()

    if args.output.exists():
        logger.info("Loading existing LOVD parquet from %s ...", args.output)
        lovd_df = pd.read_parquet(args.output)
    elif args.genes:
        lovd_df = download_gene_list(args.genes, args.output)
    else:
        lovd_df = download_whole_genome(args.output)

    logger.info(
        "LOVD dataset: %d variants, %d pathogenic, %d benign",
        len(lovd_df),
        int(lovd_df["label"].sum()),
        int((lovd_df["label"] == 0).sum()),
    )

    if args.model:
        evaluate_against_model(lovd_df, args.model, args.eval_output)


if __name__ == "__main__":
    main()
