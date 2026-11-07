"""The committed UniProt sequence fixture is exactly what was measured (ESM2-REAL-MODE-FIXTURE-1, 2026-11-06). Author: Monzia Moodie

tests/unit/_uniprot_fixture.py says why the fixture exists and where its bytes came from. These tests need no model weights and no
network, so they run everywhere: the fixture is checked on every run, not only where ESM-2 can load.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.unit import _uniprot_fixture as F


def test_the_fixture_is_three_uniprot_entries_in_canonical_bytes():
    """Line feeds only, ASCII, every sequence line 60 residues except each entry's last -- UniProt's own FASTA layout."""
    raw = F.FIXTURE.read_bytes()
    assert b"\r" not in raw, "carriage returns in the fixture: git or an editor rewrote it"
    assert raw.isascii() and raw.endswith(b"\n")
    entries = F.read_entries()
    assert [e["accession"] for e in entries] == ["P38398", "P04637", "P60484"]
    for block in raw.decode("ascii").split(">")[1:]:
        lines = block.rstrip("\n").split("\n")[1:]
        assert all(len(line) == 60 for line in lines[:-1]) and 0 < len(lines[-1]) <= 60


@pytest.mark.parametrize("accession", sorted(F.PINNED))
def test_each_entry_is_the_measured_sequence(accession):
    """Accession, entry name, gene, sequence version, length, the sequence's SHA-256 and the residue the real-mode test changes --
    all as measured on 2026-11-06 in both of the owner's sources (his index and UniProt's current entry), and Homo sapiens."""
    gene, entry_name, version, length, sha256, position, residue = F.PINNED[accession]
    (e,) = [x for x in F.read_entries() if x["accession"] == accession]
    assert e["database"] == "sp", "a Swiss-Prot (reviewed) entry"
    assert (e["entry_name"], e["gene"], e["version"], e["organism"]) == (entry_name, gene, version, "9606")
    assert len(e["sequence"]) == length
    assert F.sequence_sha256(e["sequence"]) == sha256
    assert e["sequence"][position - 1] == residue
    assert set(e["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY"), "only the twenty standard residues"


def test_the_built_index_has_the_columns_the_index_builder_writes(tmp_path):
    index = F.build_index(tmp_path)
    frame = pd.read_parquet(index)
    assert list(frame.columns) == ["gene_symbol", "uniprot_id", "entry_name", "sequence"]
    assert list(frame["gene_symbol"]) == ["BRCA1", "TP53", "PTEN"]


def test_the_connector_serves_the_fixture_sequences_with_no_network(tmp_path):
    """The path the real-mode test depends on, without the model: the connector, given the index built from the fixture, returns
    each pinned sequence and makes no live lookup (an index configured turns the network off)."""
    from genomic_variant_classifier.data.esm2 import ESM2Connector

    connector = ESM2Connector(cache_path=tmp_path / "esm2_cache.sqlite", uniprot_index_path=F.build_index(tmp_path))
    assert connector.allow_network is False
    for accession, (gene, _entry, _version, length, sha256, position, residue) in F.PINNED.items():
        sequence = connector._get_sequence(gene)
        assert sequence is not None and len(sequence) == length, accession
        assert F.sequence_sha256(sequence) == sha256 and sequence[position - 1] == residue, accession
