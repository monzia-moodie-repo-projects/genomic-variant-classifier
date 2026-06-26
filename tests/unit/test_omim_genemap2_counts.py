"""OMIM genemap2 phenotype-counting tests.

Covers the rewrite that made genemap2.txt the source for BOTH disease counts and
the AD flag (fixing the long-standing omim_n_diseases ~88 bug that read mim2gene.txt).
Exercises: [non-disease] exclusion, {susceptibility}/?provisional inclusion,
(3) molecular-basis counting, AD detection, multi-row max aggregation, genemap2-only path.
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import pandas as pd
import pytest

from genomic_variant_classifier.data.omim import OMIMConnector


GENEMAP2_HEADER = (
    "# Chromosome\tGenomic Position Start\tGenomic Position End\tCyto Location\t"
    "Computed Cyto Location\tMIM Number\tGene/Locus And Other Related Symbols\t"
    "Gene Name\tApproved Gene Symbol\tEntrez Gene ID\tEnsembl Gene ID\tComments\t"
    "Phenotypes\tMouse Gene Symbol/ID"
)
_NCOL = len(GENEMAP2_HEADER.lstrip("# ").split("\t"))


def _row(gene: str, phenos: str) -> str:
    parts = [""] * _NCOL
    parts[0] = "chr1"
    parts[5] = "600000"
    parts[8] = gene          # Approved Gene Symbol
    parts[12] = phenos       # Phenotypes
    return "\t".join(parts)


@pytest.fixture
def genemap2_file(tmp_path: Path) -> Path:
    lines = [
        "# Copyright (c) 1966-2026 Johns Hopkins University.",
        "# Generated: 2026-06-20",
        GENEMAP2_HEADER,
        _row("GENE_A", "Disease one, 100001 (3), Autosomal dominant; Disease two, 100002 (3), Autosomal recessive"),
        _row("GENE_B", "{Susceptibility X}, 200001 (2); [Biomarker Y], 200002 (3); Real disease Z, 200003 (3)"),
        _row("GENE_C", "?Provisional disease, 300001 (3), Autosomal recessive"),
        _row("GENE_D", ""),
        _row("GENE_E", "E disease 1, 500001 (3)"),
        _row("GENE_E", "E disease 1, 500001 (3); E disease 2, 500002 (3), Autosomal dominant"),
        _row("GENE_F", "[Blood group thing], 600001 (3); [Another marker], 600002 (1)"),
    ]
    p = tmp_path / "genemap2.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _annotate(genemap2_file: Path, genes: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"gene_symbol": genes})
    # genemap2-only: mim2gene_path intentionally None (proves relaxed guard)
    conn = OMIMConnector(genemap2_path=str(genemap2_file))
    return conn.annotate_dataframe(df)


@pytest.mark.parametrize("gene,n_all,n_mol,is_ad", [
    ("GENE_A", 2, 2, 1),   # 2 diseases, both (3), one AD
    ("GENE_B", 2, 1, 0),   # {susc}(2) disease-not-mol + [non-disease] EXCLUDED + plain(3) disease+mol
    ("GENE_C", 1, 1, 0),   # ?provisional (3), AR
    ("GENE_D", 0, 0, 0),   # empty phenotypes
    ("GENE_E", 2, 2, 1),   # max across two rows
    ("GENE_F", 0, 0, 0),   # only [non-disease]
    ("GENE_UNKNOWN", 0, 0, 0),  # not in genemap2 -> defaults
])
def test_genemap2_counts(genemap2_file, gene, n_all, n_mol, is_ad):
    out = _annotate(genemap2_file, [gene])
    r = out.iloc[0]
    assert int(r["omim_n_diseases"]) == n_all
    assert int(r["omim_n_diseases_molecular"]) == n_mol
    assert int(r["omim_is_autosomal_dominant"]) == is_ad


def test_molecular_is_subset_of_all(genemap2_file):
    out = _annotate(genemap2_file, ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"])
    assert (out["omim_n_diseases_molecular"] <= out["omim_n_diseases"]).all()


def test_all_three_columns_present_and_int(genemap2_file):
    out = _annotate(genemap2_file, ["GENE_A"])
    for col in ("omim_n_diseases", "omim_n_diseases_molecular", "omim_is_autosomal_dominant"):
        assert col in out.columns
        assert out[col].dtype.kind == "i"


def test_genemap2_missing_returns_defaults(tmp_path):
    # No genemap2 path -> all defaults, no crash
    conn = OMIMConnector()
    out = conn.annotate_dataframe(pd.DataFrame({"gene_symbol": ["GENE_A"]}))
    assert int(out.iloc[0]["omim_n_diseases"]) == 0
    assert int(out.iloc[0]["omim_n_diseases_molecular"]) == 0
    assert int(out.iloc[0]["omim_is_autosomal_dominant"]) == 0
