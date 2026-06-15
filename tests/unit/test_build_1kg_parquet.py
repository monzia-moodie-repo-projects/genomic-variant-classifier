"""test_build_1kg_parquet.py -- Monzia Moodie

Reworked 1000G builder: GRCh38 AF_<POP> -> AFR_AF mapping (and GRCh37 AFR_AF fallback), genotype columns
skipped via maxsplit, cohort filter, chunked streaming, and the all-zero coverage gate.
"""
from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_1kg_parquet", Path(__file__).resolve().parents[2] / "scripts" / "build_1kg_parquet.py"
)
B = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(B)

_HDR = "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"


def _vcf(p, body):
    with gzip.open(p, "wt") as f:
        f.write(_HDR + body)


def test_grch38_fields_and_genotypes_ignored(tmp_path):
    d = tmp_path / "v"; d.mkdir()
    _vcf(d / "chr22.vcf.gz",
         "chr22\t100\t.\tA\tG\t.\tPASS\tAF=0.2;AF_AFR=0.30;AF_EUR=0.10;AF_EAS=0.05;AF_SAS=0.20;AF_AMR=0.15\tGT\t0|1\t1|1\n"
         "chr22\t200\t.\tC\tT,A\t.\tPASS\tAF=0.1,0.05;AF_AFR=0.12,0.03;AF_EUR=0.08,0.02;AF_EAS=0.01,0;AF_SAS=0.05,0.01;AF_AMR=0.07,0.02\tGT\t0|2\t1|0\n")
    out = tmp_path / "o.parquet"
    B.build([str(d / "chr22.vcf.gz")], str(out))
    g = pd.read_parquet(out).set_index("variant_id")
    assert list(g.columns) == ["allele_freq", "AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF"]
    assert g.loc["22:100:A:G", "AFR_AF"] == 0.30
    assert g.loc["22:200:C:A", "AFR_AF"] == 0.03


def test_grch37_field_names_also_work(tmp_path):
    d = tmp_path / "v"; d.mkdir()
    _vcf(d / "c.vcf.gz",
         "1\t15211\t.\tT\tG\t.\tPASS\tAF=0.6;AFR_AF=0.53;EUR_AF=0.73;EAS_AF=0.50;SAS_AF=0.64;AMR_AF=0.67\tGT\t0|0\t0|1\n")
    out = tmp_path / "o.parquet"
    B.build([str(d / "c.vcf.gz")], str(out))
    g = pd.read_parquet(out).set_index("variant_id")
    assert g.loc["1:15211:T:G", "EUR_AF"] == 0.73


def test_cohort_filter_keeps_only_cohort(tmp_path):
    d = tmp_path / "v"; d.mkdir()
    _vcf(d / "c.vcf.gz",
         "22\t100\t.\tA\tG\t.\tPASS\tAF_AFR=0.1\tGT\t0|0\t0|0\n"
         "22\t999\t.\tA\tG\t.\tPASS\tAF_AFR=0.2\tGT\t0|0\t0|0\n")
    pd.DataFrame({"variant_id": ["clinvar:22:100:A:G"]}).to_parquet(tmp_path / "cohort.parquet")  # real prefix
    keys = B._load_cohort_keys(str(tmp_path / "cohort.parquet"))
    out = tmp_path / "o.parquet"
    B.build([str(d / "c.vcf.gz")], str(out), cohort_keys=keys)
    assert pd.read_parquet(out)["variant_id"].tolist() == ["22:100:A:G"]


def test_coverage_gate_fires_on_unmatched_fields(tmp_path):
    d = tmp_path / "v"; d.mkdir()
    _vcf(d / "c.vcf.gz", "22\t100\t.\tA\tG\t.\tPASS\tAF=0.2;AC=5;AN=100\tGT\t0|0\t0|0\n")
    with pytest.raises(SystemExit):
        B.build([str(d / "c.vcf.gz")], str(tmp_path / "o.parquet"))


def test_chunking_equivalent_to_single(tmp_path):
    d = tmp_path / "v"; d.mkdir()
    body = "".join(f"22\t{100+i}\t.\tA\tG\t.\tPASS\tAF_AFR=0.1\tGT\t0|0\t0|0\n" for i in range(20))
    _vcf(d / "c.vcf.gz", body)
    o1 = tmp_path / "o1.parquet"; o2 = tmp_path / "o2.parquet"
    B.build([str(d / "c.vcf.gz")], str(o1))
    B.build([str(d / "c.vcf.gz")], str(o2), chunk_size=3)
    a = pd.read_parquet(o1).sort_values("variant_id").reset_index(drop=True)
    b = pd.read_parquet(o2).sort_values("variant_id").reset_index(drop=True)
    assert a.equals(b)
