"""What the v4/schema3 source-evidence epoch emitted, immediately before retirement.

Phase 1C Unit 3A++.0. Created 2026-09-02. NO PRODUCTION CHANGE.

WHY A SECOND CORPUS
-------------------
`tests/fixtures/provenance_migration_v1/` already exists and MUST NOT be
touched. The two corpora answer different questions:

    provenance_migration_v1   Did relocating canonical ownership change
                              identity semantics?  (answered: no)

    source_evidence_epoch_v4  Exactly what did the final v4/schema3
                              implementation emit before it was retired?

THE HISTORICAL CONDITION THIS RECORDS
-------------------------------------
MEASURED 2026-09-02 in `provenance/source.py`:

    line 105   EVIDENCE_DOMAIN = "drift-source-evidence-manifest-v4"
    line 486   "schema_version": 3,

The domain says v4; the payload it digests says 3. `product` was added to
`SourceArtifactKey` on 2026-09-01, which changed key equality AND the record
shape -- the domain was bumped and the embedded literal was not.

This corpus does NOT clean that up. It records it, so a future reader can
determine why v4 emitted schema_version 3, why v5 exists, and why the old
digest differs -- without archaeology.

WHY CANONICAL JSON AND NOT PICKLE
---------------------------------
The migration corpus needed `PYTHONHASHSEED=0` because three fixtures pickle a
`frozenset` of a string-based enum, whose iteration order is randomized per
process.

This corpus needs no such pin. MEASURED: `SourceDependency.as_record` returns
`sorted(r.value for r in self.roles)` and `SourceEvidenceManifest.of` sorts
dependencies by `canonical_key`. No set iteration order reaches the canonical
record, so the JSON is deterministic by construction rather than by
environment.

Pickle remains the right oracle for old fully-qualified class locations. A
stable canonical representation is the right oracle for semantic identity.
Different oracles for different failure classes.

USAGE
=====
    python scripts/dev/freeze_source_evidence_epoch.py --out <dir>
    python scripts/dev/freeze_source_evidence_epoch.py --out <dir> --check

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from genomic_variant_classifier.provenance import (
    ArtifactKind,
    CoordinateContext,
    EVIDENCE_DOMAIN,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceEvidenceManifest,
    SourceRole,
)

#: The historical condition this corpus exists to preserve.
EPOCH_DOMAIN = "drift-source-evidence-manifest-v4"
EPOCH_EMBEDDED_SCHEMA_VERSION = 3
HISTORICAL_CONDITION = (
    "domain epoch and embedded payload version differ: the domain was bumped "
    "to v4 on 2026-09-01 when `product` was added to SourceArtifactKey, "
    "changing key equality and the canonical record shape, but the embedded "
    "literal `schema_version` was left at 3. Two writable declarations "
    "represented one semantic version, which permits divergence by "
    "construction. Recorded, not corrected: v4 means exactly what v4 "
    "historically meant."
)

GRCH38 = CoordinateContext.assembly("GRCh38")
GRCH37 = CoordinateContext.assembly("GRCh37")
NOBUILD = CoordinateContext.build_independent()


def _digest(label: str) -> str:
    """Deterministic and real-shaped. Never a repeated nucleotide literal:
    `test_no_live_module_fabricates_a_poly_window_literal` refuses those, and
    it is right to -- content cannot distinguish a real tract from a
    placeholder."""
    return hashlib.sha256(
        ("gvc-source-evidence-epoch-v4:" + label).encode("ascii")).hexdigest()


def _ident(source, kind, release, ctx, label, product=None):
    return SourceArtifactIdentity(
        key=SourceArtifactKey(source, kind, product),
        release_id=release, coordinate_context=ctx,
        artifact_sha256=_digest(label))


def _dep(identity, *roles):
    return SourceDependency(identity=identity, roles=frozenset(roles))


def _cases() -> dict:
    """The twelve representative cases the design authority names."""
    clinvar38 = _ident("clinvar", ArtifactKind.PRIMARY_RELEASE, "2026-08",
                       GRCH38, "clinvar-grch38")
    clinvar37 = _ident("clinvar", ArtifactKind.PRIMARY_RELEASE, "2026-08",
                       GRCH37, "clinvar-grch37")
    uniprot = _ident("uniprot", ArtifactKind.PRIMARY_RELEASE, "2026_03",
                     NOBUILD, "uniprot")
    tx = _ident("gencode", ArtifactKind.SEQUENCE_FASTA, "release_50",
                GRCH38, "gencode-transcripts", "transcripts")
    pc = _ident("gencode", ArtifactKind.SEQUENCE_FASTA, "release_50",
                GRCH38, "gencode-pc", "pc_transcripts")
    lnc = _ident("gencode", ArtifactKind.SEQUENCE_FASTA, "release_50",
                 GRCH38, "gencode-lncrna", "lncRNA_transcripts")
    dbnsfp = _ident("dbnsfp", ArtifactKind.PRIMARY_RELEASE, "4.5a",
                    GRCH38, "dbnsfp")
    later = _ident("clinvar", ArtifactKind.PRIMARY_RELEASE, "2026-09",
                   GRCH38, "clinvar-grch38")
    rebytes = _ident("clinvar", ArtifactKind.PRIMARY_RELEASE, "2026-08",
                     GRCH38, "clinvar-different-bytes")

    return {
        "clinvar_grch38": (_dep(clinvar38, SourceRole.OBSERVATION),),
        "clinvar_grch37": (_dep(clinvar37, SourceRole.OBSERVATION),),
        "build_independent_source": (_dep(uniprot,
                                          SourceRole.REFERENCE_SEQUENCE),),
        "gencode_transcripts": (_dep(tx, SourceRole.ANNOTATION),),
        "gencode_pc_transcripts": (_dep(pc, SourceRole.ANNOTATION),),
        "gencode_lncrna_transcripts": (_dep(lnc, SourceRole.ANNOTATION),),
        "multiple_source_roles": (_dep(clinvar38, SourceRole.OBSERVATION,
                                       SourceRole.LABEL),),
        "multiple_sources": (_dep(clinvar38, SourceRole.OBSERVATION),
                             _dep(dbnsfp, SourceRole.ANNOTATION)),
        "genomic_and_build_independent": (
            _dep(clinvar38, SourceRole.OBSERVATION),
            _dep(uniprot, SourceRole.REFERENCE_SEQUENCE)),
        "same_source_different_products": (
            _dep(tx, SourceRole.ANNOTATION), _dep(pc, SourceRole.ANNOTATION),
            _dep(lnc, SourceRole.ANNOTATION)),
        "same_key_different_release": (
            _dep(clinvar38, SourceRole.OBSERVATION),),
        "same_key_different_digest": (_dep(rebytes, SourceRole.OBSERVATION),),
        "_contrast_same_key_later_release": (
            _dep(later, SourceRole.OBSERVATION),),
    }


def build() -> bytes:
    """The whole corpus, as canonical JSON bytes. Writes nothing."""
    if EVIDENCE_DOMAIN != EPOCH_DOMAIN:
        raise RuntimeError(
            "the live EVIDENCE_DOMAIN is {!r}, not {!r}. This script freezes "
            "the v4 epoch and must run BEFORE the migration to v5."
            .format(EVIDENCE_DOMAIN, EPOCH_DOMAIN))

    cases = {}
    for name, deps in sorted(_cases().items()):
        manifest = SourceEvidenceManifest.of(deps)
        record = {"schema_version": EPOCH_EMBEDDED_SCHEMA_VERSION,
                  "dependencies": [d.as_record()
                                   for d in manifest.dependencies]}
        cases[name] = {
            "canonical_record": record,
            "digest": manifest.digest,
            "dependency_order": [list(d.canonical_key)
                                 for d in manifest.dependencies],
            # `manifest.keys` yields SourceArtifactKey OBJECTS, not
            # tuples. Their canonical_key is the three-field tuple.
            "keys_in_order": [list(k.canonical_key)
                              for k in manifest.keys],
            "identities": [d.identity.as_record()
                           for d in manifest.dependencies],
        }
    doc = {
        "identity_family": "source_evidence_manifest",
        "domain": EPOCH_DOMAIN,
        "embedded_schema_version": EPOCH_EMBEDDED_SCHEMA_VERSION,
        "known_historical_condition": HISTORICAL_CONDITION,
        "frozen_at": "phase-1c-unit-3a++.0",
        "case_count": len(cases),
        "cases": cases,
    }
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    payload = build()
    target = args.out / "epoch.json"
    if args.check:
        if not target.is_file():
            print("  [FAIL] {} is absent".format(target))
            return 1
        have = target.read_bytes()
        same = have == payload
        print("  epoch.json {} B on disk | {} B regenerated | IDENTICAL {}"
              .format(len(have), len(payload), same))
        print("  on disk    {}".format(hashlib.sha256(have).hexdigest()))
        print("  regenerated{}".format(hashlib.sha256(payload).hexdigest()))
        return 0 if same else 1

    args.out.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    doc = json.loads(payload.decode("utf-8"))
    print("  wrote {} ({:,} B, {} case(s))".format(
        target, len(payload), doc["case_count"]))
    print("  sha256 {}".format(hashlib.sha256(payload).hexdigest()))
    print("  domain {} | embedded schema_version {}".format(
        doc["domain"], doc["embedded_schema_version"]))
    for name in sorted(doc["cases"]):
        print("    {:<38} {}".format(name, doc["cases"][name]["digest"][:32]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
