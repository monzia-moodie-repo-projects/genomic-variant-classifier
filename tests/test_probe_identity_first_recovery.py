"""
tests/test_probe_identity_first_recovery.py  (2026-07-09)
Run: python -m pytest tests/test_probe_identity_first_recovery.py -v
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load("probe_identity_first_recovery", "scripts/probe_identity_first_recovery.py")


def _setup(tmp_path, cohort_type="deletion", vs_type="Deletion"):
    pd.DataFrame({
        "variant_id": ["clinvar:10:124404664:na:na"], "chrom": ["10"], "pos": [124404664],
        "type": [cohort_type], "bucket": ["RECOVER"], "ref": ["GAACTCCTGAAC"], "alt": ["G"],
        "variation_id": [1213865.0], "vcf_pos": [124404645.0], "source": ["raw"],
        "genome_verified": [True],
    }).to_csv(tmp_path / "disp.tsv", sep="\t", index=False)
    pd.DataFrame({"variant_id": ["clinvar:10:124404664:na:na"], "gene_symbol": ["OAT"],
                  "pathogenicity": ["pathogenic"]}).to_parquet(tmp_path / "coh.parquet", index=False)
    with gzip.open(tmp_path / "vs.txt.gz", "wt") as fh:
        fh.write("VariationID\tType\tGeneSymbol\tAssembly\tChromosome\tStart\n")
        fh.write(f"174\t{vs_type}\tOAT\tGRCh38\t10\t124404664\n")
        fh.write("1213865\tDeletion\tOAT\tGRCh38\t10\t124404646\n")
    with gzip.open(tmp_path / "raw.vcf.gz", "wt") as fh:
        fh.write("##fileDate=2026-03-15\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("10\t124404646\t1213865\tGAACTCCTGAAC\tG\t.\t.\t.\n")
    with gzip.open(tmp_path / "fresh.vcf.gz", "wt") as fh:
        fh.write("##fileDate=2026-07-09\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("10\t124404664\t174\tAT\tA\t.\t.\t.\n")
    (tmp_path / "g.fa").write_text(">10\n" + "N" * 124404663 + "AT" + "N" * 100 + "\n")


def _run(tmp_path):
    return pf.main([
        "--disposition", str(tmp_path / "disp.tsv"), "--cohort", str(tmp_path / "coh.parquet"),
        "--raw-vcf", str(tmp_path / "raw.vcf.gz"), "--fresh-vcf", str(tmp_path / "fresh.vcf.gz"),
        "--variant-summary", str(tmp_path / "vs.txt.gz"), "--fasta", str(tmp_path / "g.fa"),
        "--assembly", "GRCh38", "--per-band", "5", "--out", str(tmp_path / "s.tsv"),
    ])


def test_neighbor_misattach_detected(tmp_path):
    _setup(tmp_path)
    assert _run(tmp_path) == 0
    df = pd.read_csv(tmp_path / "s.tsv", sep="\t")
    row = df.iloc[0]
    assert str(row["cohort_varid"]) == "174"           # true variant at Start==pos
    assert str(row["probe_varid"]) == "1213865"        # neighbor the probe grabbed
    assert bool(row["probe_was_wrong"]) is True
    assert row["identity_first_verdict"] == "RECOVER_BY_ID"
    assert row["id_first_ref"] == "AT" and row["id_first_alt"] == "A"   # CORRECT allele by ID


def test_repeat_type_gets_no_seq_allele(tmp_path):
    _setup(tmp_path, vs_type="Microsatellite")
    assert _run(tmp_path) == 0
    df = pd.read_csv(tmp_path / "s.tsv", sep="\t")
    assert df.iloc[0]["identity_first_verdict"] == "REPEAT_NO_SEQ_ALLELE"
    assert pd.isna(df.iloc[0]["id_first_ref"])
