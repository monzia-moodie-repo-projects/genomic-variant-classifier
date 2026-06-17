#!/usr/bin/env python3
"""
scripts/build_rnaseq_parquet.py
===============================
Ingest a quantified RNA-seq expression matrix into the per-gene parquet that
RNASeqConnector consumes. Handles BOTH input forms robustly:

  * gene x sample matrix         rows already gene-level (salmon/kallisto after
                                 a tximport gene collapse, recount3, ARCHS4,
                                 featureCounts).
  * transcript x sample matrix   rows are transcripts; pass --tx-map to collapse
      + --tx-map                 to gene by SUMMING transcript TPM per gene
                                 (tximport-style gene aggregation).

Per-gene features written (all gene-level; defaults 0):
  rnaseq_mean_log_tpm    mean of log1p(TPM) across samples
  rnaseq_detection_rate  fraction of samples with value >= --min-detect
  rnaseq_log2_cv         log2(1 + CV) of TPM across samples (expression dispersion)
  rnaseq_log2fc          log2 fold-change case/control     (0 without --sample-meta)
  rnaseq_de_neglog10p    -log10(p) Welch t-test on log1p(TPM) (0 without --sample-meta)

Differential expression is OPTIONAL. It is computed only when --sample-meta
supplies a two-level group column; otherwise rnaseq_log2fc / rnaseq_de_neglog10p
are 0 and a notice is logged (never a silent failure).

LEAKAGE WARNING: when --sample-meta is given the DE features MUST come from an
RNA-seq cohort INDEPENDENT of the variant-pathogenicity label cohort, or the DE
signal leaks the label. A loud warning is printed in that case.

Inputs: TSV/CSV/parquet (text inputs are gzip-aware). This script does not
download anything; point --matrix at your own quantified matrix.

Usage
-----
  # gene-level matrix, expression-summary only
  python scripts/build_rnaseq_parquet.py --matrix expr_gene_tpm.tsv \\
      --out data/external/rnaseq_gene_expression.parquet

  # transcript-level matrix collapsed to gene, with differential expression
  python scripts/build_rnaseq_parquet.py --matrix expr_tx_tpm.tsv \\
      --tx-map tx2gene.tsv --sample-meta samples.tsv \\
      --out data/external/rnaseq_gene_expression.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNASEQ_FEATURES = [
    "rnaseq_mean_log_tpm",
    "rnaseq_detection_rate",
    "rnaseq_log2_cv",
    "rnaseq_log2fc",
    "rnaseq_de_neglog10p",
]
_LOG2FC_PSEUDOCOUNT = 1.0


def _read_table(path: Path) -> pd.DataFrame:
    """Read TSV/CSV/parquet (gzip-aware for text). Fails LOUD on unknown types."""
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet") or suf.endswith(".pq"):
        return pd.read_parquet(path)
    if ".csv" in suf:
        return pd.read_csv(path, sep=",")
    if ".tsv" in suf or ".txt" in suf or ".gz" in suf:
        # default RNA-seq matrices are tab-separated
        return pd.read_csv(path, sep="\t")
    raise SystemExit(
        f"{path.name}: unrecognised matrix extension (use .tsv/.csv/.parquet[.gz])."
    )


def _split_id_samples(mat: pd.DataFrame, id_col: str | None):
    """Return (id_series, sample_df). id_col defaults to the first column; sample
    columns are all remaining columns coerced to numeric (non-numeric dropped LOUD)."""
    if id_col is None:
        id_col = mat.columns[0]
    if id_col not in mat.columns:
        raise SystemExit(f"--id-col '{id_col}' not in matrix columns {list(mat.columns)[:5]}...")
    ids = mat[id_col].astype(str).str.strip()
    sample_cols = [c for c in mat.columns if c != id_col]
    samples = mat[sample_cols].apply(pd.to_numeric, errors="coerce")
    all_nan = samples.columns[samples.isna().all()].tolist()
    if all_nan:
        # a non-numeric extra column (e.g. a gene Description) -> drop it, LOUD
        print(f"[note] dropping {len(all_nan)} non-numeric column(s) from samples: {all_nan[:5]}")
        samples = samples.drop(columns=all_nan)
    if samples.shape[1] == 0:
        raise SystemExit("no numeric sample columns found in --matrix.")
    return ids, samples


def _collapse_tx_to_gene(ids: pd.Series, samples: pd.DataFrame, tx_map_path: Path):
    """Sum transcript TPM per gene (tximport-style). tx_map: (transcript_id, gene_symbol)."""
    tmap = _read_table(tx_map_path)
    cols = {c.lower(): c for c in tmap.columns}
    tx_c = cols.get("transcript_id") or cols.get("tx_id") or cols.get("transcript")
    gn_c = cols.get("gene_symbol") or cols.get("gene") or cols.get("symbol")
    if not tx_c or not gn_c:
        raise SystemExit(
            f"--tx-map must have (transcript_id, gene_symbol); found {list(tmap.columns)}"
        )
    tmap = tmap[[tx_c, gn_c]].rename(columns={tx_c: "transcript_id", gn_c: "gene_symbol"})
    tmap["transcript_id"] = tmap["transcript_id"].astype(str).str.strip()
    tmap["gene_symbol"] = tmap["gene_symbol"].astype(str).str.strip()

    df = samples.copy()
    df.insert(0, "transcript_id", ids.values)
    # strip version suffix (ENST0000xxx.3 -> ENST0000xxx) on both sides for robust join
    df["_tx"] = df["transcript_id"].str.replace(r"\.\d+$", "", regex=True)
    tmap["_tx"] = tmap["transcript_id"].str.replace(r"\.\d+$", "", regex=True)
    merged = df.merge(tmap[["_tx", "gene_symbol"]], on="_tx", how="inner")
    n_unmapped = len(df) - merged["_tx"].nunique()
    if merged.empty:
        raise SystemExit("--tx-map collapsed to 0 rows (no transcript ids matched).")
    sample_cols = [c for c in samples.columns]
    gene_mat = merged.groupby("gene_symbol")[sample_cols].sum()
    print(
        f"[tx->gene] collapsed {len(df)} transcripts -> {gene_mat.shape[0]} genes "
        f"(summed TPM per gene; {n_unmapped} transcript ids unmapped/dropped)."
    )
    return gene_mat


def _read_sample_meta(path: Path, sample_cols: list[str]):
    """Return (case_cols, control_cols) from a (sample_id, group) table with EXACTLY
    two groups. Samples not present in --matrix are ignored (LOUD)."""
    meta = _read_table(path)
    cols = {c.lower(): c for c in meta.columns}
    sid = cols.get("sample_id") or cols.get("sample") or cols.get("id")
    grp = cols.get("group") or cols.get("condition") or cols.get("label")
    if not sid or not grp:
        raise SystemExit(
            f"--sample-meta must have (sample_id, group); found {list(meta.columns)}"
        )
    meta = meta[[sid, grp]].rename(columns={sid: "sample_id", grp: "group"})
    meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
    meta = meta[meta["sample_id"].isin(sample_cols)]
    groups = sorted(meta["group"].astype(str).unique())
    if len(groups) != 2:
        raise SystemExit(
            f"--sample-meta must define EXACTLY two groups for DE; found {groups}. "
            "Drop extra groups or omit --sample-meta to skip DE."
        )
    # convention: the lexicographically-LAST group is 'case' unless a 'case' label exists
    case_label = next((g for g in groups if str(g).lower() in ("case", "1", "disease", "tumor")), groups[1])
    control_label = [g for g in groups if g != case_label][0]
    case_cols = meta.loc[meta["group"] == case_label, "sample_id"].tolist()
    control_cols = meta.loc[meta["group"] == control_label, "sample_id"].tolist()
    if len(case_cols) < 2 or len(control_cols) < 2:
        raise SystemExit(
            f"DE needs >=2 samples per group (case={len(case_cols)}, control={len(control_cols)})."
        )
    print(
        f"[de] case='{case_label}' (n={len(case_cols)})  control='{control_label}' "
        f"(n={len(control_cols)}).  *** LEAKAGE: this RNA-seq cohort must be "
        f"INDEPENDENT of your variant-label cohort. ***"
    )
    return case_cols, control_cols


def summarise_matrix(
    gene_mat: pd.DataFrame,
    case_cols: list[str] | None = None,
    control_cols: list[str] | None = None,
    min_detect: float = 1.0,
) -> pd.DataFrame:
    """gene x sample TPM matrix -> per-gene feature DataFrame (RNASEQ_FEATURES)."""
    tpm = gene_mat.astype(float)
    log_tpm = np.log1p(tpm)

    mean_log_tpm = log_tpm.mean(axis=1)
    detection_rate = (tpm >= min_detect).mean(axis=1)
    mean_tpm = tpm.mean(axis=1)
    std_tpm = tpm.std(axis=1, ddof=0)
    cv = (std_tpm / mean_tpm).where(mean_tpm > 0, 0.0)
    log2_cv = np.log2(1.0 + cv)

    out = pd.DataFrame(
        {
            "gene_symbol": tpm.index.astype(str),
            "rnaseq_mean_log_tpm": mean_log_tpm.round(6).values,
            "rnaseq_detection_rate": detection_rate.round(6).values,
            "rnaseq_log2_cv": log2_cv.round(6).values,
            "rnaseq_log2fc": 0.0,
            "rnaseq_de_neglog10p": 0.0,
        }
    )

    if case_cols and control_cols:
        from scipy import stats

        a = log_tpm[case_cols].to_numpy()
        b = log_tpm[control_cols].to_numpy()
        # vectorised Welch t-test per gene (row-wise)
        t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        p = np.clip(p, 1e-300, 1.0)
        neglog10p = -np.log10(p)
        case_tpm = tpm[case_cols].mean(axis=1).to_numpy()
        ctrl_tpm = tpm[control_cols].mean(axis=1).to_numpy()
        log2fc = np.log2((case_tpm + _LOG2FC_PSEUDOCOUNT) / (ctrl_tpm + _LOG2FC_PSEUDOCOUNT))
        out["rnaseq_log2fc"] = np.round(log2fc, 6)
        out["rnaseq_de_neglog10p"] = np.round(np.where(np.isfinite(neglog10p), neglog10p, 0.0), 6)

    out = out[out["gene_symbol"].str.len() > 0]
    out = out.sort_values("gene_symbol").reset_index(drop=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True, type=Path,
                    help="gene x sample OR transcript x sample TPM/counts matrix (TSV/CSV/parquet[.gz]).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tx-map", type=Path, default=None,
                    help="(transcript_id, gene_symbol) map; pass when --matrix is transcript-level.")
    ap.add_argument("--sample-meta", type=Path, default=None,
                    help="(sample_id, group) table with EXACTLY two groups -> enables DE features.")
    ap.add_argument("--id-col", default=None,
                    help="name of the gene/transcript id column (default: first column).")
    ap.add_argument("--min-detect", type=float, default=1.0,
                    help="value >= this counts a sample as expressing the gene (default 1.0, TPM).")
    args = ap.parse_args(argv)

    if not args.matrix.exists():
        raise SystemExit(f"--matrix not found: {args.matrix}")

    mat = _read_table(args.matrix)
    ids, samples = _split_id_samples(mat, args.id_col)

    if args.tx_map is not None:
        if not args.tx_map.exists():
            raise SystemExit(f"--tx-map not found: {args.tx_map}")
        gene_mat = _collapse_tx_to_gene(ids, samples, args.tx_map)
    else:
        # gene-level: id IS the gene symbol; sum duplicate symbols defensively
        gene_mat = samples.copy()
        gene_mat.insert(0, "gene_symbol", ids.values)
        gene_mat = gene_mat[gene_mat["gene_symbol"].str.len() > 0]
        gene_mat = gene_mat.groupby("gene_symbol").sum()

    case_cols = control_cols = None
    if args.sample_meta is not None:
        if not args.sample_meta.exists():
            raise SystemExit(f"--sample-meta not found: {args.sample_meta}")
        case_cols, control_cols = _read_sample_meta(args.sample_meta, list(gene_mat.columns))
    else:
        print("[de] no --sample-meta -> rnaseq_log2fc / rnaseq_de_neglog10p = 0 (expression-summary only).")

    agg = summarise_matrix(gene_mat, case_cols, control_cols, args.min_detect)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False)
    n_de = int((agg["rnaseq_de_neglog10p"] > 0).sum())
    print(
        f"Wrote {args.out}  ({len(agg)} genes; "
        f"mean_log_tpm max={float(agg['rnaseq_mean_log_tpm'].max()) if len(agg) else 0:.3f}; "
        f"{n_de} genes with DE signal)."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
