"""test_parse_gmt_encoding.py -- Monzia Moodie

parse_gmt must reject binary/compressed payloads LOUDLY (Reactome ships ReactomePathways.gmt.zip, so a
file saved from the .gmt URL is a ZIP), transparently read .gz, parse genuine non-UTF-8 pathway names,
and never return 0 sets silently.
"""
import gzip

import pytest

from genomic_variant_classifier.data.kg_edges import parse_gmt, reactome_edges


def _w(p, b):
    p.write_bytes(b)
    return str(p)


def test_zip_payload_rejected(tmp_path):
    p = _w(tmp_path / "ReactomePathways.gmt", b"PK\x03\x04\x14\x00\x00\x00\x08\x00\xc6\x00junk\x00\x00")
    with pytest.raises(ValueError, match="ZIP"):
        parse_gmt(p)


def test_binary_nul_rejected(tmp_path):
    p = _w(tmp_path / "x.gmt", b"text\x00\x01\xc6\xff binary")
    with pytest.raises(ValueError, match="binary"):
        parse_gmt(p)


def test_gzip_transparently_read(tmp_path):
    gp = tmp_path / "g.gmt.gz"
    with gzip.open(gp, "wb") as f:
        f.write(b"PathwayA\thttps://x\tBRCA1\tBRCA2\tTP53\n")
    sets = parse_gmt(str(gp))
    assert "BRCA1" in next(iter(sets.values()))


def test_valid_text_with_non_utf8_name(tmp_path):
    p = _w(tmp_path / "v.gmt",
           b"Pathway-\xc6-name\thttps://x\tBRCA1\tBRCA2\tTP53\nOther\thttps://y\tTP53\tMDM2\n")
    sets = parse_gmt(p)
    allg = {g for v in sets.values() for g in v}
    assert {"BRCA1", "BRCA2", "TP53", "MDM2"} <= allg
    assert len(reactome_edges(p)) >= 1


def test_zero_sets_rejected(tmp_path):
    p = _w(tmp_path / "e.gmt", b"# comment only\nonecol\n")
    with pytest.raises(ValueError, match="0 gene sets"):
        parse_gmt(p)
