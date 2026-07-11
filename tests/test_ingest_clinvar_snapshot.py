"""Tests for ingest_clinvar_snapshot.py -- fidelity to the connector + drift handling."""
import sys, gzip, json
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_clinvar_snapshot as ING


def _write_vs(path: Path, header: list[str], rows: list[list[str]]):
    """Write a tab-separated .gz variant_summary with given header + rows."""
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


# A minimal 2026-07-style header: has all connector columns EXCEPT ProteinChange, plus extras.
HEADER_2026_07 = [
    "AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance", "ClinSigSimple",
    "RS# (dbSNP)", "Assembly", "Chromosome", "Start", "Stop",
    "ReferenceAllele", "AlternateAllele", "ReviewStatus", "VariationID",
    "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF", "SomaticClinicalImpact",
]

def _row(**kw):
    d = {c: "" for c in HEADER_2026_07}
    d.update(kw)
    return [d[c] for c in HEADER_2026_07]


def test_basic_ingest_shape_and_schema(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="BRCA1", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
             Chromosome="17", Start="43000", ReferenceAllele="A", AlternateAllele="G",
             ReviewStatus="criteria provided", VariationID="100", **{"RS# (dbSNP)": "123"}),
        _row(GeneSymbol="TP53", ClinicalSignificance="Benign", Assembly="GRCh37",
             Chromosome="17", Start="7500", ReferenceAllele="C", AlternateAllele="T",
             ReviewStatus="no assertion", VariationID="101"),
    ])
    df, man = ING.ingest(vs)
    # GRCh37 row filtered out -> 1 row
    assert len(df) == 1
    assert list(df.columns) == ING.CANONICAL_COLUMNS
    r = df.iloc[0]
    assert r["variant_id"] == "clinvar:17:43000:A:G"
    assert r["pos"] == 43000  # note: read as int by pandas
    assert r["pathogenicity"] == "pathogenic"
    assert r["source_db"] == "clinvar"
    assert r["source_id"] == 100
    assert r["metadata"]["review_status"] == "criteria provided"


def test_protein_change_known_drift_is_all_none_not_crash(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="BRCA1", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
             Chromosome="17", Start="43000", ReferenceAllele="A", AlternateAllele="G",
             ReviewStatus="x", VariationID="100"),
    ])
    df, man = ING.ingest(vs)
    # ProteinChange absent -> protein_change all-None, recorded as known drift, no crash
    assert df["protein_change"].isna().all()
    assert man["protein_change_all_null"] is True
    assert "ProteinChange" in man["missing_rename_sources"]


def test_na_alleles_become_none_like_connector(tmp_path):
    # A na:na row alongside populated rows: source resolves 'legacy' (ReferenceAllele populated
    # in the majority), and the na:na row normalizes to None:None (matching the stale parquet).
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="G", ClinicalSignificance="Uncertain", Assembly="GRCh38",
             Chromosome="7", Start="4781213", ReferenceAllele="na", AlternateAllele="na",
             ReviewStatus="x", VariationID="2"),
        _row(GeneSymbol="G", ClinicalSignificance="Benign", Assembly="GRCh38",
             Chromosome="7", Start="500", ReferenceAllele="A", AlternateAllele="G",
             ReviewStatus="x", VariationID="3"),
        _row(GeneSymbol="G", ClinicalSignificance="Benign", Assembly="GRCh38",
             Chromosome="7", Start="600", ReferenceAllele="C", AlternateAllele="T",
             ReviewStatus="x", VariationID="4"),
    ])
    df, man = ING.ingest(vs)
    assert man["allele_source"] == "legacy"
    r = df[df["pos"] == 4781213].iloc[0]
    # empty-allele tokens normalize to None (matching stale 'clinvar:chrom:pos:None:None')
    assert r["ref"] is None and r["alt"] is None
    assert r["variant_id"] == "clinvar:7:4781213:None:None"


def test_hard_missing_column_fails_loud(tmp_path):
    # Remove a NON-known-removed required column (Start) -> must raise, not silently proceed.
    header = [c for c in HEADER_2026_07 if c != "Start"]
    vs = tmp_path / "vs.txt.gz"
    with gzip.open(vs, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(["x"] * len(header)) + "\n")
    with pytest.raises(ValueError, match="SCHEMA DRIFT"):
        ING.ingest(vs)


def test_determinism_same_md5(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="B", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
             Chromosome="1", Start="500", ReferenceAllele="A", AlternateAllele="T",
             ReviewStatus="x", VariationID="5"),
    ])
    o1 = tmp_path / "a.parquet"; o2 = tmp_path / "b.parquet"
    ING.main(["--input", str(vs), "--output", str(o1)])
    ING.main(["--input", str(vs), "--output", str(o2)])
    assert ING._md5(o1) == ING._md5(o2)


def test_refuses_canonical_path_without_force(tmp_path, monkeypatch):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="B", ClinicalSignificance="Benign", Assembly="GRCh38",
             Chromosome="1", Start="1", ReferenceAllele="A", AlternateAllele="T",
             ReviewStatus="x", VariationID="9"),
    ])
    # simulate the canonical path under tmp
    canon = tmp_path / "data" / "processed" / "clinvar_grch38.parquet"
    canon.parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    rc = ING.main(["--input", str(vs), "--output", "data/processed/clinvar_grch38.parquet"])
    assert rc == 4  # refused


def test_refuses_overwrite_without_force(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="B", ClinicalSignificance="Benign", Assembly="GRCh38",
             Chromosome="1", Start="1", ReferenceAllele="A", AlternateAllele="T",
             ReviewStatus="x", VariationID="9"),
    ])
    out = tmp_path / "o.parquet"
    assert ING.main(["--input", str(vs), "--output", str(out)]) == 0
    assert ING.main(["--input", str(vs), "--output", str(out)]) == 3  # refuse overwrite
    assert ING.main(["--input", str(vs), "--output", str(out), "--force"]) == 0  # ok with force


def test_dash_and_mixed_empty_tokens_normalized(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="G", ClinicalSignificance="Uncertain", Assembly="GRCh38",
             Chromosome="1", Start="100", ReferenceAllele="A", AlternateAllele="-",
             ReviewStatus="x", VariationID="50"),
        _row(GeneSymbol="G", ClinicalSignificance="Uncertain", Assembly="GRCh38",
             Chromosome="1", Start="200", ReferenceAllele="NA", AlternateAllele=".",
             ReviewStatus="x", VariationID="51"),
    ])
    df, man = ING.ingest(vs)
    # alt '-' -> None; ref 'NA' -> None; alt '.' -> None
    row100 = df[df["pos"] == 100].iloc[0]
    row200 = df[df["pos"] == 200].iloc[0]
    assert row100["alt"] is None, "dash '-' must normalize to None"
    assert row100["ref"] == "A"
    assert row200["ref"] is None and row200["alt"] is None
    assert man["alt_empty_normalized"] >= 2


# ---- 2026-07 schema shape: ReferenceAllele all 'na', alleles in *VCF columns ----------------
HEADER_VCF = [
    "AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance", "ClinSigSimple",
    "RS# (dbSNP)", "Assembly", "Chromosome", "Start", "Stop",
    "ReferenceAllele", "AlternateAllele", "ReviewStatus", "VariationID",
    "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF",
]

def _rowv(**kw):
    d = {c: "" for c in HEADER_VCF}
    d.update(kw)
    return [d[c] for c in HEADER_VCF]


def test_vcf_schema_sources_alleles_from_vcf_columns(tmp_path):
    vs = tmp_path / "vs.txt.gz"
    # ReferenceAllele/AlternateAllele all 'na'; real alleles in *VCF; pos must come from Start
    rows = [
        _rowv(GeneSymbol="B", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
              Chromosome="17", Start="43000", ReferenceAllele="na", AlternateAllele="na",
              ReferenceAlleleVCF="A", AlternateAlleleVCF="G", PositionVCF="43000",
              ReviewStatus="x", VariationID="100"),
        _rowv(GeneSymbol="B", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
              Chromosome="17", Start="43100", ReferenceAllele="na", AlternateAllele="na",
              ReferenceAlleleVCF="GCTG", AlternateAlleleVCF="G", PositionVCF="43099",
              ReviewStatus="x", VariationID="101"),
    ]
    with gzip.open(vs, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(HEADER_VCF) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    df, man = ING.ingest(vs)
    assert man["allele_source"] == "vcf"
    assert len(df) == 2
    snv = df[df["pos"] == 43000].iloc[0]
    assert snv["ref"] == "A" and snv["alt"] == "G"
    assert snv["variant_id"] == "clinvar:17:43000:A:G"
    # padded deletion: pos MUST be Start (43100), NOT PositionVCF (43099)
    dele = df[df["pos"] == 43100].iloc[0]
    assert dele["ref"] == "GCTG" and dele["alt"] == "G"
    assert dele["variant_id"] == "clinvar:17:43100:GCTG:G"
    assert man["nana_rate"] < 0.5


def test_all_null_tripwire_fires(tmp_path):
    # Both legacy AND vcf allele columns empty -> _resolve_allele_source raises (no usable source)
    vs = tmp_path / "vs.txt.gz"
    rows = [
        _rowv(GeneSymbol="B", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
              Chromosome="1", Start="100", ReferenceAllele="na", AlternateAllele="na",
              ReferenceAlleleVCF="na", AlternateAlleleVCF="na", PositionVCF="100",
              ReviewStatus="x", VariationID="1"),
    ]
    with gzip.open(vs, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(HEADER_VCF) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    with pytest.raises(ValueError, match="NO USABLE ALLELE COLUMN"):
        ING.ingest(vs)


def test_legacy_schema_still_uses_referenceallele(tmp_path):
    # 2026-03 shape: ReferenceAllele populated -> allele_source 'legacy'
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, HEADER_2026_07, [
        _row(GeneSymbol="B", ClinicalSignificance="Pathogenic", Assembly="GRCh38",
             Chromosome="1", Start="500", ReferenceAllele="A", AlternateAllele="G",
             ReviewStatus="x", VariationID="7"),
    ])
    df, man = ING.ingest(vs)
    assert man["allele_source"] == "legacy"
    assert df.iloc[0]["ref"] == "A" and df.iloc[0]["alt"] == "G"


def test_resolver_rejects_majority_empty_legacy(tmp_path):
    # 62.5% na:na in the sample -> the source resolver itself rejects (empty-rate > 0.5),
    # raising before any parquet write. First line of defense.
    vs = tmp_path / "vs.txt.gz"
    rows = []
    for i in range(3):
        rows.append(_row(GeneSymbol="G", ClinicalSignificance="Benign", Assembly="GRCh38",
                         Chromosome="1", Start=str(100+i), ReferenceAllele="A",
                         AlternateAllele="G", ReviewStatus="x", VariationID=str(10+i)))
    for i in range(5):
        rows.append(_row(GeneSymbol="G", ClinicalSignificance="Uncertain", Assembly="GRCh38",
                         Chromosome="1", Start=str(200+i), ReferenceAllele="na",
                         AlternateAllele="na", ReviewStatus="x", VariationID=str(20+i)))
    _write_vs(vs, HEADER_2026_07, rows)
    with pytest.raises(ValueError, match="NO USABLE ALLELE COLUMN"):
        ING.ingest(vs)


def test_tripwire_fires_past_resolver(tmp_path):
    # Isolate the TRIPWIRE (second line of defense): make the first 20000 rows populated so the
    # resolver's sample passes ('legacy'), then append >20000 na:na rows so the FULL-file na:na
    # rate exceeds 0.5. Only the tripwire can catch this.
    vs = tmp_path / "vs.txt.gz"
    rows = []
    for i in range(20000):
        rows.append(_row(GeneSymbol="G", ClinicalSignificance="Benign", Assembly="GRCh38",
                         Chromosome="1", Start=str(1000000+i), ReferenceAllele="A",
                         AlternateAllele="G", ReviewStatus="x", VariationID=str(100000+i)))
    for i in range(30000):
        rows.append(_row(GeneSymbol="G", ClinicalSignificance="Uncertain", Assembly="GRCh38",
                         Chromosome="2", Start=str(2000000+i), ReferenceAllele="na",
                         AlternateAllele="na", ReviewStatus="x", VariationID=str(200000+i)))
    _write_vs(vs, HEADER_2026_07, rows)
    with pytest.raises(ValueError, match="ALL-NULL TRIPWIRE"):
        ING.ingest(vs)


def test_conflicting_classifications_map_to_uncertain():
    """The 2026-07-10 label fix: 'Conflicting classifications of pathogenicity' (and compounds)
    must map to 'uncertain', NOT 'pathogenic'. Confident+modifier strings must stay confident."""
    m = ING._map_pathogenicity
    # conflicting -> uncertain (the fix)
    for s in ("Conflicting classifications of pathogenicity",
              "Conflicting classifications of pathogenicity; other",
              "Conflicting classifications of pathogenicity; risk factor",
              "Conflicting classifications of pathogenicity; association",
              "Conflicting classifications of pathogenicity; drug response",
              "Conflicting classifications of pathogenicity; other; risk factor",
              "conflicting data from submitters"):
        assert m(s) == "uncertain", f"{s!r} should be uncertain, got {m(s)!r}"
    # confident + secondary modifier -> UNCHANGED (leading classification is authoritative)
    for s in ("Pathogenic", "Pathogenic; drug response", "Pathogenic; risk factor",
              "Pathogenic; other", "Pathogenic, low penetrance", "Pathogenic/Likely pathogenic",
              "Pathogenic/Likely pathogenic; risk factor"):
        assert m(s) == "pathogenic", f"{s!r} should stay pathogenic, got {m(s)!r}"
    for s in ("Benign", "Benign; risk factor", "Benign/Likely benign"):
        assert m(s) == "benign"
    for s in ("Likely pathogenic", "Likely pathogenic; drug response", "Likely pathogenic, low penetrance"):
        assert m(s) == "likely_pathogenic"
    assert m("Likely benign") == "likely_benign"
    assert m("Uncertain significance") == "uncertain"
    assert m("not provided") == "uncertain"
