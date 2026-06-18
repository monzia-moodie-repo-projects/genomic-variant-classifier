from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


MATRIX = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.gene_tpm.tsv")
DESIGN = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\designs\gtex_v11_Brain_Cortex_vs_Whole_Blood.tsv")
OUT = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.tsv")
MANIFEST = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.manifest.json")

POS = "Brain_Cortex"
NEG = "Whole_Blood"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        out[order[i]] = min(prev, 1.0)
    return out


def main() -> int:
    if not MATRIX.exists():
        fail(f"missing matrix: {MATRIX}")
    if not DESIGN.exists():
        fail(f"missing design: {DESIGN}")

    design = pd.read_csv(DESIGN, sep="\t", dtype=str)
    matrix = pd.read_csv(MATRIX, sep="\t")

    sample_order = design["sample_id"].tolist()
    if matrix.columns[2:].tolist() != sample_order:
        fail("matrix sample order does not exactly match design order")

    pos_samples = design.loc[design["condition"] == POS, "sample_id"].tolist()
    neg_samples = design.loc[design["condition"] == NEG, "sample_id"].tolist()

    expr = matrix[sample_order].apply(pd.to_numeric, errors="coerce")
    if expr.isna().any().any():
        fail("matrix contains NaN/non-numeric expression values")
    if (expr < 0).any().any():
        fail("matrix contains negative TPM values")

    pos_tpm = expr[pos_samples].to_numpy(dtype=float)
    neg_tpm = expr[neg_samples].to_numpy(dtype=float)

    mean_pos = pos_tpm.mean(axis=1)
    mean_neg = neg_tpm.mean(axis=1)
    log2fc = np.log2(mean_pos + 1.0) - np.log2(mean_neg + 1.0)

    pos_log = np.log2(pos_tpm + 1.0)
    neg_log = np.log2(neg_tpm + 1.0)

    _, pvalues = ttest_ind(
        pos_log,
        neg_log,
        axis=1,
        equal_var=False,
        nan_policy="propagate",
    )

    pvalues = np.asarray(pvalues, dtype=float)
    status = np.full(len(pvalues), "welch_ttest", dtype=object)

    nan_mask = ~np.isfinite(pvalues)
    if nan_mask.any():
        pvalues[nan_mask] = 1.0
        status[nan_mask] = "degenerate_variance_pvalue_set_to_1"

    pvalues = np.clip(pvalues, np.nextafter(0, 1), 1.0)
    qvalues = bh_fdr(pvalues)
    neglog10p = -np.log10(pvalues)

    out = pd.DataFrame({
        "gene_id": matrix["Name"].astype(str),
        "gene_symbol": matrix["Description"].astype(str),
        "rnaseq_log2fc": log2fc,
        "rnaseq_de_pvalue": pvalues,
        "rnaseq_de_qvalue": qvalues,
        "rnaseq_de_neglog10p": neglog10p,
        "rnaseq_de_test_status": status,
        "mean_tpm_Brain_Cortex": mean_pos,
        "mean_tpm_Whole_Blood": mean_neg,
        "n_Brain_Cortex": len(pos_samples),
        "n_Whole_Blood": len(neg_samples),
        "contrast_id": "Brain_Cortex_vs_Whole_Blood",
        "leakage_guard": "normal_tissue_reference_not_variant_label",
    })

    if out.isna().any().any():
        fail("DE output contains NaN values after degeneracy handling")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)

    manifest = {
        "artifact": str(OUT),
        "matrix": str(MATRIX),
        "design": str(DESIGN),
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "sha256": sha256(OUT),
        "method": "Welch t-test on log2(TPM+1); degenerate variance p-values set conservatively to 1.0",
        "degenerate_rows": int(nan_mask.sum()),
        "leakage_guard": "normal_tissue_reference_not_variant_label",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"OK wrote DE features: {OUT}")
    print(f"rows={len(out)}")
    print(f"degenerate_rows={int(nan_mask.sum())}")
    print(f"sha256={manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
