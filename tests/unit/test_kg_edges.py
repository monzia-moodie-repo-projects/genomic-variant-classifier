"""test_kg_edges.py  --  Monzia Moodie

Knowledge-graph gene-gene edge connectors. Pure-python (no torch), so every test
runs anywhere; the final test exercises the handoff into the hetero-GNN builder.
Messy inputs throughout: malformed/duplicate GMT lines, CRLF, a CSV preamble, an
oversized set that must be suppressed, cohort restriction, and self-pairs.
"""
import numpy as np
import pytest

from genomic_variant_classifier.data.kg_edges import (
    clingen_edges,
    co_membership_edges,
    parse_gene_group_table,
    parse_gmt,
    reactome_edges,
)


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parse_gmt_messy(tmp_path):
    gmt = _write(tmp_path / "m.gmt",
                 "PathA\tdescA\tBRCA1\tTP53\tBRCA1\n"      # dup gene
                 "\r\n"                                     # blank
                 "PathB\tdescB\tTP53\tMLH1\n"
                 "ShortLine\tonlydesc\n"                    # <3 fields -> skip
                 "PathA\tdescA2\tEGFR\n"                    # dup set -> merge
                 "PathC\tdescC\t\t  \t\n"                   # empty genes -> skip
                 "PathD\tdescD\tKRAS\tNRAS\tHRAS\r\n")
    sets = parse_gmt(gmt)
    assert sets["PathA"] == ["BRCA1", "TP53", "EGFR"]       # dedup + merge
    assert "ShortLine" not in sets and "PathC" not in sets
    assert sets["PathD"] == ["KRAS", "NRAS", "HRAS"]


def test_co_membership_guard_dedup_selfpairs():
    sets = {
        "small": ["A", "B", "C"],
        "dup": ["A", "B"],                                  # A-B again
        "self": ["X", "X"],                                 # 1 gene -> no edge
        "big": [f"g{i}" for i in range(50)],                # exceeds guard
    }
    edges = co_membership_edges(sets, max_set_size=10)
    es = set(edges)
    assert {("A", "B"), ("A", "C"), ("B", "C")} <= es
    assert all(a < b for a, b in edges)                     # canonical
    assert len(es) == len(edges)                            # deduped
    assert not any(a == b for a, b in edges)                # no self-pairs
    assert not any(g.startswith("g") for e in edges for g in e)  # big suppressed


def test_co_membership_restrict_to():
    sets = {"small": ["A", "B", "C"]}
    assert set(co_membership_edges(sets, restrict_to={"A", "B"})) == {("A", "B")}
    assert co_membership_edges(sets, restrict_to={"A"}) == []   # <2 survive


def test_parse_gene_group_table_preamble(tmp_path):
    csvp = _write(tmp_path / "clingen.csv",
                  "CLINGEN GENE VALIDITY,,,\n"
                  "generated 2026,,,\n"
                  "GENE SYMBOL,DISEASE LABEL,MOI,CLASSIFICATION\n"
                  "BRCA1,Hereditary breast cancer,AD,Definitive\n"
                  "BRCA2,Hereditary breast cancer,AD,Definitive\n"
                  "MLH1,Lynch syndrome,AD,Definitive\n"
                  ",Lynch syndrome,AD,Definitive\n"          # blank gene
                  "MSH2,Lynch syndrome,AD,Definitive\n")
    groups = parse_gene_group_table(csvp, "GENE SYMBOL", "DISEASE LABEL")
    assert groups["Hereditary breast cancer"] == ["BRCA1", "BRCA2"]
    assert groups["Lynch syndrome"] == ["MLH1", "MSH2"]      # blank-gene row dropped
    with pytest.raises(ValueError):
        parse_gene_group_table(csvp, "NOPE", "DISEASE LABEL")


def test_clingen_edges_and_hetero_handoff(tmp_path):
    from genomic_variant_classifier.models.hetero_gnn import build_hetero_gene_graph

    csvp = _write(tmp_path / "cg.csv",
                  "GENE SYMBOL,DISEASE LABEL\n"
                  "BRCA1,HBOC\nBRCA2,HBOC\nMLH1,Lynch\nMSH2,Lynch\n")
    gmt = _write(tmp_path / "rx.gmt",
                 "PathA\td\tBRCA1\tTP53\nPathB\td\tTP53\tMLH1\n")
    cg = set(clingen_edges(csvp))
    assert ("BRCA1", "BRCA2") in cg and ("MLH1", "MSH2") in cg
    assert ("BRCA1", "MLH1") not in cg                       # different diseases

    genes = ["BRCA1", "BRCA2", "MLH1", "MSH2", "TP53"]
    g = build_hetero_gene_graph(
        genes, np.zeros((5, 3), dtype=np.float32),
        {"shares_disease": clingen_edges(csvp, restrict_to=set(genes)),
         "shares_pathway": reactome_edges(gmt, restrict_to=set(genes))},
    )
    assert g.edge_index_by_rel["shares_disease"].shape[1] == 4   # 2 undirected pairs
    assert g.edge_index_by_rel["shares_pathway"].shape[1] > 0    # TP53 bridges PathA/PathB
