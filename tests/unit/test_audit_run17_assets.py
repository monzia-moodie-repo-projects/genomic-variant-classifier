"""test_audit_run17_assets.py -- Author: Monzia Moodie
Validates the pure audit core (filesystem classification). The registry-importing
main is exercised on-machine; here we test the logic that decides the verdict.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import audit_run17_assets as A  # noqa: E402


def _write(p: Path, content=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_present_when_primary_exists_nonempty(tmp_path):
    _write(tmp_path / "data/processed/clinvar_grch38.parquet", b"data")
    rows = A.audit(tmp_path, [{"source": "clinvar",
                               "primary": "data/processed/clinvar_grch38.parquet",
                               "alternates": ["data/raw/clinvar/clinvar_GRCh38.vcf.gz"]}])
    assert rows[0]["status"] == "PRESENT" and rows[0]["present"] is True


def test_found_at_alt_when_primary_missing_but_alt_present(tmp_path):
    _write(tmp_path / "data/processed/spliceai_index.parquet", b"data")  # alt only
    rows = A.audit(tmp_path, [{"source": "spliceai",
                               "primary": "data/external/spliceai/spliceai_index.parquet",
                               "alternates": ["data/processed/spliceai_index.parquet"]}])
    assert rows[0]["status"] == "FOUND_AT_ALT"
    assert rows[0]["alternates_found"][0]["path"] == "data/processed/spliceai_index.parquet"


def test_missing_when_neither_present(tmp_path):
    rows = A.audit(tmp_path, [{"source": "uniprot",
                               "primary": "data/external/uniprot/uniprot_human_reviewed.parquet",
                               "alternates": []}])
    assert rows[0]["status"] == "MISSING" and rows[0]["alternates_found"] == []


def test_empty_file_counts_as_missing(tmp_path):
    _write(tmp_path / "data/processed/gnomad_v4_exomes.parquet", b"")  # zero bytes
    rows = A.audit(tmp_path, [{"source": "gnomad",
                               "primary": "data/processed/gnomad_v4_exomes.parquet"}])
    assert rows[0]["status"] == "MISSING" and rows[0]["present"] is False


def test_finngen_typo_alt_resolves(tmp_path):
    # corrected (non-typo) name exists; registry primary has the typo
    _write(tmp_path / "data/external/finngen/finngen_R12_annotated_variants_v1.gz", b"gz")
    rows = A.audit(tmp_path, [{"source": "finngen",
                               "primary": "data/external/finngen/finnge_R12_annotated_variants_v1.gz",
                               "alternates": ["data/external/finngen/finngen_R12_annotated_variants_v1.gz"]}])
    assert rows[0]["status"] == "FOUND_AT_ALT"


def test_directory_present_iff_nonempty(tmp_path):
    d = tmp_path / "data/raw/cache/alphafold"
    d.mkdir(parents=True)
    rows = A.audit(tmp_path, [{"source": "alphafold", "primary": "data/raw/cache/alphafold"}])
    assert rows[0]["status"] == "MISSING"          # empty dir
    _write(d / "P12345.json", b"{}")
    rows = A.audit(tmp_path, [{"source": "alphafold", "primary": "data/raw/cache/alphafold"}])
    assert rows[0]["status"] == "PRESENT"          # now has a file
