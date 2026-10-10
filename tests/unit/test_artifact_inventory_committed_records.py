"""The COMMITTED artifact-inventory records: permanent evidence, byte-pinned, and the index derived from them (owner rulings
2026-10-08c, 2026-10-08e, 2026-10-08f).

A record enters the repository only after its evidence was verified outside it (make_record_unit_v2.py 64f1387e, 2026-10-09): the
record's exact bytes, a strict round-trip through the typed owner, the collector it names, the summary and readiness decision that
bind it, the implementation the decision names (collector, record owner, admission and every loaded checkout module), the run record's
interval, and readiness RE-DERIVED from the record and the pinned plan documents equal to the decision the collector wrote; the
installer then binds those bytes to the owner's evidence ZIP by its SHA-256. Here the repository holds itself to that: every record
file is exactly the reviewed bytes, parses and renders identically, and the index is exactly render_index of the records (a
replaceable projection -- never edited by hand). A later measurement is a NEW record naming its predecessor, added to PINNED with its
own digest.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.repository_records.artifact_inventory import (
    ArtifactInventoryRecord, family_root, render_index, scan_records)

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT.joinpath(*family_root().parts)
#: Every committed record by file name -> SHA-256 of its exact bytes (the digest the evidence bundle and readiness decision bind).
PINNED = {"REC-ece23653bf8e45dbad3da26303870dc6.json": "bb716d48ad628ae593e104263efd7ca846db237d25dff2b3f35ba0018d192c22"}
#: The collector that measured them (verify_artifact_inventory.py, 34,373 bytes, delivered in artifact_inventory_v2_2026-10-08.zip
#: a9e6b69a1fcaafcee1da717c2deda066ad79bc60c71488f101204e7ca6e5e9a6): the records' verifier_sha256.
COLLECTOR_SHA256 = "cfc1cac85bd6e32c660a7ae055e54ae286c0c51fdcfb3ee582ae0272a7d39656"


def test_the_family_holds_exactly_the_pinned_records_and_the_index():
    assert FAMILY.is_dir(), FAMILY
    assert sorted(p.name for p in FAMILY.iterdir()) == sorted(list(PINNED) + ["index.json"])


@pytest.mark.parametrize("name", sorted(PINNED))
def test_every_record_is_exactly_the_reviewed_bytes_and_round_trips(name):
    raw = (FAMILY / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED[name]
    record = ArtifactInventoryRecord.parse(raw)
    assert record.render() == raw and record.record_id.value + ".json" == name
    assert record.verifier_sha256 == COLLECTOR_SHA256


def test_the_index_is_exactly_derived_from_the_records():
    """scan_records parses every record strictly (file name = record id); render_index requires ONE linear chain."""
    assert (FAMILY / "index.json").read_bytes() == render_index(scan_records(ROOT))


def test_the_chain_starts_at_one_first_record():
    records = scan_records(ROOT)
    assert len([r for r in records if r.previous_record_id is None]) == 1


def test_no_record_file_is_ignored():
    """`*.log`-style ignore rules have kept evidence out of commits before (2026-10-02). NUL-separated bytes and a sentinel that MUST be
    reported, so a non-discriminating instrument fails instead of passing vacuously on any platform."""
    paths = [(FAMILY / n).relative_to(ROOT).as_posix() for n in sorted(PINNED) + ["index.json"]]
    sentinel = "probe_not_evidence.log"
    out = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                         input=("\0".join(paths + [sentinel]) + "\0").encode("utf-8"), capture_output=True, cwd=ROOT)
    assert out.returncode in (0, 1), out.stderr.decode("utf-8", "replace")
    assert [x for x in out.stdout.decode("utf-8").split("\0") if x] == [sentinel]
