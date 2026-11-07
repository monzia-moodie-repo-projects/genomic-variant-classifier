"""The committed UniProt sequences of the ESM-2 real-mode test (ESM2-REAL-MODE-FIXTURE-1, 2026-11-06). Author: Monzia Moodie

WHY A FIXTURE. tests/unit/test_esm2_activation.py::test_esm2_not_in_stub_mode needs three protein sequences. Until 2026-11-06 it read
them from data/external/uniprot/uniprot_human_reviewed.parquet, which git ignores. MEASURED on the owner's machine (block DC, the
data census, 2026-11-06 08:07-09:28 UTC): the same commit and the same locked environment SKIPPED the test in a clean clone (V1) and
PASSED it in the checkout, which holds that file. A test whose verdict depends on an untracked file is not a test of the code.

WHAT IS COMMITTED. tests/fixtures/uniprot/esm2_real_mode_BRCA1_TP53_PTEN.fasta: UniProt's own FASTA entries for P38398 (BRCA1),
P04637 (TP53) and P60484 (PTEN), as rest.uniprot.org served them on 2026-11-06 07:05 UTC (block SQ, record
GVC_esm2_fixture_probe_20261106T070555Z.json, SHA-256 f68be0c4...). Each sequence was IDENTICAL to the one in the owner's index (the
sequences the passing run used; index SHA-256 3d8380bd..., 20,190 rows), and each carries the residue the test changes. The
sequence versions are pinned below with each sequence's SHA-256: an entry UniProt revises later does not change this fixture.

The index the connector reads is BUILT from the fixture, in the test's own temporary folder, with the columns
scripts/build_uniprot_index.py writes (gene_symbol, uniprot_id, entry_name, sequence).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "uniprot" / "esm2_real_mode_BRCA1_TP53_PTEN.fasta"

#: accession -> (gene, entry name, sequence version, length, SHA-256 of the sequence, position, residue there)
PINNED = {
    "P38398": ("BRCA1", "BRCA1_HUMAN", 2, 1863, "80d7199c8d1550a5ba8387043481c1acf0f070df410b47c2bc1375584b59bdf1", 1699, "R"),
    "P04637": ("TP53", "P53_HUMAN", 4, 393, "e3ca60f18e3b6e98986ad6749215ee3b7a3abc09c191675213d1f243c035c9a3", 175, "R"),
    "P60484": ("PTEN", "PTEN_HUMAN", 1, 403, "2675a6cc946076d70342977845afe793269c5987a998382100bede64d1e61b42", 129, "G"),
}


def read_entries(path: Path = FIXTURE) -> list:
    """The fixture's entries in file order: dicts with accession, entry_name, gene, version, header, sequence."""
    text = path.read_bytes().decode("ascii")
    entries, header, chunks = [], None, []
    for line in text.split("\n"):
        if line.startswith(">"):
            if header is not None:
                entries.append(_entry(header, "".join(chunks)))
            header, chunks = line, []
        elif line:
            chunks.append(line)
    if header is not None:
        entries.append(_entry(header, "".join(chunks)))
    return entries


def _entry(header: str, sequence: str) -> dict:
    # >sp|P38398|BRCA1_HUMAN <name> OS=Homo sapiens OX=9606 GN=BRCA1 PE=1 SV=2
    database, accession, rest = header[1:].split("|", 2)
    entry_name = rest.split(" ", 1)[0]
    fields = dict(t.split("=", 1) for t in rest.split(" ") if "=" in t and t.split("=", 1)[0] in ("GN", "SV", "OX"))
    return {"database": database, "accession": accession, "entry_name": entry_name, "gene": fields.get("GN"),
            "version": int(fields["SV"]) if "SV" in fields else None, "organism": fields.get("OX"), "header": header,
            "sequence": sequence}


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def build_index(directory: Path) -> Path:
    """A UniProt index parquet built from the fixture, in `directory`; the columns scripts/build_uniprot_index.py writes."""
    entries = read_entries()
    frame = pd.DataFrame({"gene_symbol": [e["gene"] for e in entries], "uniprot_id": [e["accession"] for e in entries],
                          "entry_name": [e["entry_name"] for e in entries], "sequence": [e["sequence"] for e in entries]})
    path = Path(directory) / "uniprot_index_from_fixture.parquet"
    frame.to_parquet(path, index=False)
    return path
