"""test_hetero_kg_wiring.py  --  Monzia Moodie

KG-edge-spec loading + STRING edge extraction for the live hetero_gnn_score
overwrite (run_phase2_eval --hetero-gnn).
"""
import pytest

from genomic_variant_classifier.models.hetero_gnn_scorer import (
    load_kg_edge_specs,
    string_graph_to_edges,
)


def test_load_kg_edge_specs_reactome_cohort(tmp_path):
    gmt = tmp_path / "r.gmt"
    gmt.write_text("PATH1\thttp://x\tGENEA\tGENEB\tGENEC\nPATH2\thttp://y\tGENEB\tGENED\n")
    out = load_kg_edge_specs([f"reactome:{gmt}"], {"GENEA", "GENEB", "GENEC"})
    assert set(out) == {"shares_pathway"}
    # GENED is out of cohort -> dropped before pairing
    assert out["shares_pathway"] == [("GENEA", "GENEB"), ("GENEA", "GENEC"), ("GENEB", "GENEC")]


def test_load_kg_edge_specs_merge_and_relations(tmp_path):
    gmt = tmp_path / "r.gmt"; gmt.write_text("P\tx\tGENEA\tGENEB\n")
    csv = tmp_path / "c.csv"; csv.write_text("GENE SYMBOL,DISEASE LABEL\nGENEA,DX\nGENEB,DX\n")
    out = load_kg_edge_specs([f"reactome:{gmt}", f"kegg:{gmt}", f"clingen:{csv}"], {"GENEA", "GENEB"})
    assert set(out) == {"shares_pathway", "shares_disease"}
    assert out["shares_pathway"] == [("GENEA", "GENEB")]      # reactome + kegg merged + deduped
    assert out["shares_disease"] == [("GENEA", "GENEB")]


def test_load_kg_edge_specs_errors():
    with pytest.raises(ValueError):
        load_kg_edge_specs(["noColonSpec"], set())
    with pytest.raises(ValueError):
        load_kg_edge_specs(["bogus:/tmp/x"], set())


def test_string_graph_to_edges_cohort_restricted():
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_edge("A", "B"); G.add_edge("B", "C"); G.add_edge("C", "Z")
    e = string_graph_to_edges(G, restrict_to={"A", "B", "C"})
    assert ("C", "Z") not in e        # Z out of cohort
    assert ("A", "B") in e and ("B", "C") in e and len(e) == 2
    assert len(string_graph_to_edges(G)) == 3   # no restriction -> all edges
