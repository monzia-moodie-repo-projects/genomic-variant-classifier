"""Executable witnesses of registered defects (owner ruling 68ca322f, 2026-11-10; set "ar" revision B).

Each test PASSES while its defect still exists, so the registration in the census ledger is evidence, not prose. A stage that
repairs a defect will see its witness FAIL: the repair must then remove the witness and the registration together, in the
same change. The four defects were reproduced on the shadow modules, which nothing in production imports; registering them
does not change any behaviour. Author: Monzia Moodie."""
from __future__ import annotations

import dataclasses
import gzip
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import graph_identity as G
from genomic_variant_classifier.information_sources import labels as L

REPO = Path(__file__).resolve().parents[2]


def _registered(defect_id: str) -> None:
    from genomic_variant_classifier.information_sources import census as C

    ids = {d["id"] for d in C.strict_load(REPO / C.LEDGER_PATH)["registered_defects"]}
    assert defect_id in ids, f"{defect_id} is witnessed here but not registered in the census ledger"


def test_witness_label_type_identity_1():
    """LABEL-TYPE-IDENTITY-1: a label record accepts non-integers equal to 0/1 and any non-empty gene string."""
    _registered("LABEL-TYPE-IDENTITY-1")
    for value in (np.bool_(True), 1.0, np.float64(1.0), Fraction(1)):
        L.LabelRecord("r", "HGNC:1", value, "train")
    for gene in ("GAS8", " HGNC:1", "HGNC:01"):
        L.LabelRecord("r", gene, 1, "train")
    stored = L.LabelAuthority([L.LabelRecord("r", "HGNC:1", np.float64(1.0), "train"),
                               L.LabelRecord("h", "HGNC:2", 0, "heldout")]).store_for_fit().read("r")
    assert type(stored) is np.float64


def test_witness_label_denied_read_audit_1():
    """LABEL-DENIED-READ-AUDIT-1: held-out, refused and unknown rows raise KeyError and leave no trace in the access log."""
    _registered("LABEL-DENIED-READ-AUDIT-1")
    store = L.LabelAuthority([L.LabelRecord("t", "HGNC:1", 1, "train"), L.LabelRecord("s", "HGNC:9", 0, "train"),
                              L.LabelRecord("h", "HGNC:2", 1, "heldout"),
                              L.LabelRecord("hs", "HGNC:9", 1, "heldout")]).store_for_fit()
    assert store.read("t") == 1
    for row in ("s", "h", "unknown"):
        with pytest.raises(KeyError):
            store.read(row)
    assert store.accesses == (("t", "HGNC:1"),)


def _tiny(tmp_path):
    links = tmp_path / "l.gz"
    links.write_bytes(gzip.compress((" ".join(G.STRING_HEADER) + "\n9606.A 9606.B 0 0 0 0 900 0 0 950\n"
                                     "9606.B 9606.A 0 0 0 0 900 0 0 950\n").encode(), mtime=0))
    hg = tmp_path / "h.txt"
    hg.write_bytes(b"hgnc_id\tsymbol\tstatus\tprev_symbol\talias_symbol\nHGNC:1\tAAA\tApproved\t\t\n"
                   b"HGNC:2\tBBB\tApproved\t\t\n")
    t, h = G.read_string_pairs(links), G.read_hgnc(hg)
    pol = G.GraphPolicy(source="dataset:string", release="12.0", source_sha256=t.source_sha256, info_sha256="a" * 64,
                        mapping_source="dataset:hgnc", mapping_source_sha256=h.source_sha256, regime=G.Regime.G1,
                        selection_channels=("experimental",), attribute_channels=("experimental",), threshold_milli=700,
                        quantization=G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    return t, h, pol


def test_witness_graph_mapping_receipt_authority_1(tmp_path):
    """GRAPH-MAPPING-RECEIPT-AUTHORITY-1: the protein-info digest is a caller-supplied argument, so a mapping built from
    DIFFERENT info content claims the pinned source and is accepted; an arbitrary mapping digest is accepted too."""
    _registered("GRAPH-MAPPING-RECEIPT-AUTHORITY-1")
    t, h, pol = _tiny(tmp_path)
    true_map, _ = G.map_string_proteins({"9606.A": "AAA", "9606.B": "BBB"}, "a" * 64, h)
    swapped, _ = G.map_string_proteins({"9606.A": "BBB", "9606.B": "AAA"}, "a" * 64, h)
    assert swapped.info_sha256 == pol.info_sha256 and swapped.mapping_sha256 != true_map.mapping_sha256
    G.accept_graph(pol, G.materialize_string_graph(t, swapped, pol))
    g = G.materialize_string_graph(t, true_map, pol)
    for fake in ("0" * 64, "1" * 64):
        mat = G.materialization_of(pol, g.nodes, g.edges, g.attributes, g.source_sha256, fake)
        G.accept_graph(pol, dataclasses.replace(g, mapping_sha256=fake, materialization=mat))


def test_witness_graph_mutable_materialization_1(tmp_path):
    """GRAPH-MUTABLE-MATERIALIZATION-1: a frozen dataclass does not freeze the arrays and dicts it holds; the pair table's
    stored digest goes stale after an in-place write, and the HGNC index and gene-set source can be edited after reading."""
    _registered("GRAPH-MUTABLE-MATERIALIZATION-1")
    t, h, _ = _tiny(tmp_path)
    before = t.content_sha256
    t.channels[0, 4] = 1
    assert t.content_sha256 == before != G.pair_table_digest(t.proteins, t.a, t.b, t.channels, t.combined)
    h.approved["ZZZ"] = "HGNC:99"
    gmt = tmp_path / "s.gmt"
    gmt.write_bytes(b"P\tR-1\tAAA\tBBB\n")
    s = G.read_gene_sets(gmt, h, source_id="reactome", relation="shares_pathway")
    s.sets["R-2"] = frozenset({"HGNC:1"})
    assert "ZZZ" in h.approved and "R-2" in s.sets
