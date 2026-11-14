"""Stage 1D-1b.4 (set "as"): graph acceptance re-derives from the pinned files (GRAPH-MAPPING-RECEIPT-AUTHORITY-1, repaired)
and materialised content is immutable (GRAPH-MUTABLE-MATERIALIZATION-1, repaired). Owner ruling 68ca322f: "the reader
returns content and digest as one object, and acceptance at a trust boundary re-derives the graph from the verified
inputs"; "bytes-backed read-only buffers, read-only mappings, digests re-verified at trust boundaries".
Fixtures are written as bytes (the byte-strict readers refuse a carriage return). Author: Monzia Moodie."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import graph_identity as G


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(tmp_path, *, swap_info: bool = False):
    pairs = [("9606.A", "9606.B", "0 0 0 100 900 0 0 950"), ("9606.A", "9606.C", "0 0 0 0 950 0 0 960"),
             ("9606.B", "9606.C", "0 0 0 0 100 0 0 400")]
    body = " ".join(G.STRING_HEADER) + "\n" + "".join(f"{a} {b} {s}\n{b} {a} {s}\n" for a, b, s in pairs)
    links = tmp_path / "links.gz"
    links.write_bytes(gzip.compress(body.encode("ascii"), mtime=0))
    names = {"9606.A": "AAA", "9606.B": "BBB", "9606.C": "CCC"}
    if swap_info:
        names = {"9606.A": "BBB", "9606.B": "AAA", "9606.C": "CCC"}
    info = tmp_path / ("info_swapped.gz" if swap_info else "info.gz")
    info.write_bytes(gzip.compress(("\t".join(G.STRING_INFO_HEADER) + "\n" + "".join(
        f"{p}\t{n}\t10\tx\n" for p, n in names.items())).encode("ascii"), mtime=0))
    hgnc = tmp_path / "hgnc.txt"
    hgnc.write_bytes(b"hgnc_id\tsymbol\tstatus\tprev_symbol\talias_symbol\nHGNC:1\tAAA\tApproved\t\t\n"
                     b"HGNC:2\tBBB\tApproved\t\t\nHGNC:3\tCCC\tApproved\t\t\n")
    return links, info, hgnc


def _policy(links, info, hgnc, **kw):
    base = dict(source="dataset:string", release="12.0", source_sha256=_sha(links), info_sha256=_sha(info),
                mapping_source="dataset:hgnc", mapping_source_sha256=_sha(hgnc), regime=G.Regime.G1,
                selection_channels=("experimental",), attribute_channels=("experimental", "coexpression"),
                threshold_milli=700, quantization=G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    base.update(kw)
    return G.GraphPolicy(**base)


def _built(links, info, hgnc, pol):
    t, h = G.read_string_pairs(links), G.read_hgnc(hgnc)
    m, _ = G.map_string_proteins(G.read_string_info(info), h)
    return t, h, m, G.materialize_string_graph(t, m, pol)


def test_acceptance_re_derives_the_graph_from_the_pinned_files(tmp_path):
    links, info, hgnc = _files(tmp_path)
    pol = _policy(links, info, hgnc)
    _, _, _, g = _built(links, info, hgnc, pol)
    assert g.edges == (("HGNC:1", "HGNC:2"), ("HGNC:1", "HGNC:3"))
    accepted = G.accept_graph(pol, g, links=links, info=info, hgnc=hgnc, scratch_dir=tmp_path, partitions=4)
    assert accepted == g.materialization == G.verify_graph_consistency(pol, g)


def test_a_mapping_of_swapped_identities_claiming_the_pinned_digest_is_refused_at_the_boundary(tmp_path):
    """The witness of GRAPH-MAPPING-RECEIPT-AUTHORITY-1, inverted: the swapped mapping is self-consistent (only
    re-derivation can tell), and acceptance refuses it."""
    links, info, hgnc = _files(tmp_path)
    pol = _policy(links, info, hgnc)
    t, h, _, _ = _built(links, info, hgnc, pol)
    _, swapped_file, _ = _files(tmp_path, swap_info=True)
    forged_info = G.StringInfo(G.read_string_info(swapped_file).names, pol.info_sha256, 0)    # claims the pinned digest
    forged, _ = G.map_string_proteins(forged_info, h)
    g = G.materialize_string_graph(t, forged, pol)
    assert G.verify_graph_consistency(pol, g) == g.materialization                         # self-consistent
    with pytest.raises(ValueError, match=r"differs in \['edges', 'mapping', 'materialization'\]"):
        G.accept_graph(pol, g, links=links, info=info, hgnc=hgnc, scratch_dir=tmp_path, partitions=4)


def test_a_file_that_is_not_the_pinned_one_is_refused(tmp_path):
    links, info, hgnc = _files(tmp_path)
    pol = _policy(links, info, hgnc)
    _, _, _, g = _built(links, info, hgnc, pol)
    _, swapped_file, _ = _files(tmp_path, swap_info=True)
    with pytest.raises(ValueError, match="sha256|SHA-256|digest"):
        G.accept_graph(pol, g, links=links, info=swapped_file, hgnc=hgnc, scratch_dir=tmp_path, partitions=4)


def test_the_mapping_digest_is_the_mapping_s_own_and_cannot_be_set(tmp_path):
    links, info, hgnc = _files(tmp_path)
    pol = _policy(links, info, hgnc)
    _, h, m, g = _built(links, info, hgnc, pol)
    assert g.mapping is m and g.mapping_sha256 == m.mapping_sha256
    with pytest.raises(ValueError, match="init=False"):
        dataclasses.replace(g, mapping_sha256="0" * 64)
    other, _ = G.map_string_proteins(G.StringInfo({"9606.A": "AAA", "9606.B": "BBB"}, pol.info_sha256, 0), h)
    with pytest.raises(ValueError, match="mapping_sha256|node universe"):
        G.verify_graph_consistency(pol, dataclasses.replace(g, mapping=other))
    with pytest.raises(ValueError, match="pinned protein-info"):
        G.verify_graph_consistency(pol, dataclasses.replace(g, mapping=dataclasses.replace(m, info_sha256="7" * 64)))
    nodes = g.nodes + ("HGNC:9",)                                       # a universe the mapping does not define,
    mat = G.materialization_of(pol, nodes, g.edges, g.attributes, g.source_sha256, g.mapping_sha256)   # manifest re-made
    with pytest.raises(ValueError, match="node universe is not the one its mapping defines"):
        G.verify_graph_consistency(pol, dataclasses.replace(g, nodes=nodes, materialization=mat))


def test_the_mapping_takes_its_digest_from_the_reader_never_from_a_caller(tmp_path):
    links, info, hgnc = _files(tmp_path)
    h = G.read_hgnc(hgnc)
    with pytest.raises(TypeError, match="StringInfo"):
        G.map_string_proteins({"9606.A": "AAA"}, h)
    with pytest.raises(TypeError):
        G.map_string_proteins({"9606.A": "AAA"}, "a" * 64, h)          # the 1D-1b.3 signature is gone
    m, _ = G.map_string_proteins(G.read_string_info(info), h)
    assert m.info_sha256 == _sha(info)
    with pytest.raises(ValueError, match="SHA-256"):
        G.StringInfo({"9606.A": "AAA"}, "not-a-digest", 0)


def test_materialised_content_is_read_only(tmp_path):
    """The witness of GRAPH-MUTABLE-MATERIALIZATION-1, inverted: every array is bytes-backed (its WRITEABLE flag cannot be
    set again) and every mapping read-only, so a stored digest cannot go stale."""
    links, info, hgnc = _files(tmp_path)
    pol = _policy(links, info, hgnc)
    t, h, _, g = _built(links, info, hgnc, pol)
    for arr in (t.a, t.b, t.channels, t.combined, g.attributes):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    with pytest.raises(ValueError):
        t.channels[0, 4] = 1
    assert t.content_sha256 == G.pair_table_digest(t.proteins, t.a, t.b, t.channels, t.combined)
    for table in (h.approved, h.previous, h.alias, G.read_string_info(info).names, g.report,
                  g.report["unmapped_endpoint_status_counts"]):
        with pytest.raises(TypeError):
            table["ZZZ"] = "HGNC:99"
    cls = G.classify_threshold(t, ("experimental",), 700)
    with pytest.raises(ValueError):
        cls.status[0] = 0
    with pytest.raises(TypeError):
        cls.decided_exactly["x"] = 1
    gmt = tmp_path / "s.gmt"
    gmt.write_bytes(b"P\tR-1\tAAA\tBBB\n")
    s = G.read_gene_sets(gmt, h, source_id="reactome", relation="shares_pathway")
    for table in (s.sets, s.names, s.member_status_counts):
        with pytest.raises(TypeError):
            table["R-2"] = frozenset({"HGNC:1"})
    proj = G.project_pathways(s, ["HGNC:1", "HGNC:2"], max_set_size=None)
    with pytest.raises(TypeError):
        proj.edges[("HGNC:1", "HGNC:3")] = 1.0


def test_frozen_and_plain_values_round_trip():
    value = {"a": [1, {"b": 2}], "c": None}
    frozen = G.frozen_value(value)
    with pytest.raises(TypeError):
        frozen["a"][1]["b"] = 3
    assert G.plain_value(frozen) == value
    with pytest.raises(TypeError):
        G.frozen_value({"s": {1, 2}})
    with pytest.raises(TypeError):
        G.plain_value({"x": np.zeros(1)})
    assert not G.readonly_array(np.zeros(0, np.int32)).flags.writeable
