"""test_parse_gmt_encoding.py -- Monzia Moodie

Reactome GMT pathway names contain non-UTF-8 bytes; parse_gmt must decode robustly (errors='replace')
instead of raising UnicodeDecodeError, with ASCII gene symbols left intact.
"""
from genomic_variant_classifier.data.kg_edges import parse_gmt, reactome_edges


def test_parse_gmt_handles_non_utf8_pathway_name(tmp_path):
    p = tmp_path / "r.gmt"
    p.write_bytes(b"Pathway-\xc6-name\thttps://x\tBRCA1\tBRCA2\tTP53\n"
                  b"Other\thttps://y\tTP53\tMDM2\n")
    sets = parse_gmt(str(p))
    assert len(sets) == 2
    all_genes = {g for genes in sets.values() for g in genes}
    assert {"BRCA1", "BRCA2", "TP53", "MDM2"} <= all_genes
    assert len(reactome_edges(str(p))) >= 1
