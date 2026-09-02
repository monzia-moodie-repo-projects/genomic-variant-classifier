"""Freeze what a scientific identity MEANS, before its owner changes.

Phase 1C Unit 3A.0. Created 2026-09-02. NO PRODUCTION CHANGE.

WHY THIS RUNS BEFORE ANY CODE MOVES
-----------------------------------
Unit 3A moves canonical ownership of the identity substrate from
`monitoring.drift` into `provenance`. Its governing invariant is:

    UNIT 3A MAY CHANGE WHERE A SCIENTIFIC IDENTITY IS DEFINED.
    IT MUST NOT CHANGE WHAT THAT IDENTITY MEANS.

An invariant asserted only against the post-move code proves nothing: the
post-move code would agree with itself. So this script captures the answer
BEFORE the move, from the pre-move tree, and commits it.

TWO ORACLES, WHICH FAIL FOR DIFFERENT REASONS
---------------------------------------------
SEMANTIC. `semantic.json` records `canonical_key`, `as_record()`, `digest` and
`describe()` for a corpus of boundary cases. If relocation alters canonical
ordering, a domain string, a field name or a sort, these change and the
post-move test fails.

PERSISTENCE. `*.pickle` files record the actual serialized bytes. A pickle
names a class by MODULE AND QUALNAME, so if `monitoring.drift.source_release`
stops resolving `SourceArtifactKey`, these fail to load.

The persistence census on 2026-09-02 found no file containing BOTH a
serialization mechanism and one of these types. That establishes no DIRECT
STATIC coupling. It does NOT establish that no persisted object graph contains
one: a `TrainingState` holding a `SourceManifest` in one module, pickled by
`joblib.dump` in another, is invisible to same-file co-occurrence. This
repository has already paid for a class relocation once -- `migrate_pickles.py`
and the `_CNN1DModule.__module__` episode -- so the fixture is cheap insurance
against a failure mode it has seen before.

WHAT THE CORPUS COVERS
----------------------
Boundary cases, not happy paths:

    clinvar primary release          the ordinary case
    gencode transcripts              the three products that motivated the
    gencode pc_transcripts           product coordinate -- one authority, one
    gencode lncRNA_transcripts       release, one kind, three artifacts
    uniprot build-independent        CoordinateContext.build_independent(),
                                     which must not collapse into None
    grch37 / grch38 artifacts        both declared assemblies
    multi-role dependency            roles are digest-bearing
    multi-authority manifest         canonical ordering across sources
    all five component kinds         FEATURE_ENGINEERING, MISSINGNESS,
                                     NORMALIZATION, JOIN_POLICY,
                                     COORDINATE_POLICY

USAGE
=====
    python scripts/dev/build_provenance_migration_fixtures.py --out <dir>
    python scripts/dev/build_provenance_migration_fixtures.py --out <dir> --check

`--check` regenerates and compares WITHOUT writing, so the same script proves
the corpus still matches after the move.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation; UTC = Coordinated Universal Time.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

from genomic_variant_classifier.monitoring.drift import (
    ArtifactKind,
    CoordinateContext,
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
    TransformationComponent,
    TransformationComponentKind,
    TransformationIdentity,
)

def _fixture_digest(label: str) -> str:
    """A deterministic, REAL-SHAPED SHA-256 for a fixture.

    Never random or time-derived: the corpus must regenerate identically on any
    machine, or it cannot be an oracle.

    And never a single nucleotide letter multiplied to a fixed length.
    MEASURED 2026-09-02: the acceptance gate refused this payload because
    `tests/unit/test_no_content_based_poly_detection.py` forbids a live module
    from constructing a repeated-nucleotide literal. Two of the four original
    constants were flagged and two were not, the difference being only whether
    the letter names a base.

    That guard exists because content cannot distinguish a real poly-adenine
    tract from a fabricated placeholder window, and four earlier detectors
    were blind to the same 21,814 rows. It was right to refuse: a module that
    builds such a literal is indistinguishable from one fabricating a window,
    whatever it means by it.

    The example forms are deliberately NOT written out here, since the guard
    may inspect text rather than syntax, and a docstring is no place to
    reintroduce the shape it warns about.

    Deriving the value removes the shape entirely AND produces digests that
    look like digests, so a reader of a fixture is not invited to think
    `aaaa...` is meaningful.
    """
    return hashlib.sha256(
        ("gvc-provenance-migration-fixture:" + label).encode("ascii")
    ).hexdigest()


_A = _fixture_digest("alpha")
_B = _fixture_digest("beta")
_C = _fixture_digest("gamma")
_D = _fixture_digest("delta")


def _corpus() -> dict:
    """Every representative value, built once and reused."""
    grch38 = CoordinateContext.assembly("GRCh38")
    grch37 = CoordinateContext.assembly("GRCh37")
    nobuild = CoordinateContext.build_independent()

    clinvar = SourceArtifactIdentity(
        key=SourceArtifactKey("clinvar", ArtifactKind.PRIMARY_RELEASE),
        release_id="2026-08",
        coordinate_context=grch38,
        artifact_sha256=_A,
    )
    gencode_tx = SourceArtifactIdentity(
        key=SourceArtifactKey("gencode", ArtifactKind.SEQUENCE_FASTA,
                              "transcripts"),
        release_id="release_50",
        coordinate_context=grch38,
        artifact_sha256=_B,
    )
    gencode_pc = SourceArtifactIdentity(
        key=SourceArtifactKey("gencode", ArtifactKind.SEQUENCE_FASTA,
                              "pc_transcripts"),
        release_id="release_50",
        coordinate_context=grch38,
        artifact_sha256=_C,
    )
    gencode_lnc = SourceArtifactIdentity(
        key=SourceArtifactKey("gencode", ArtifactKind.SEQUENCE_FASTA,
                              "lncRNA_transcripts"),
        release_id="release_50",
        coordinate_context=grch38,
        artifact_sha256=_D,
    )
    uniprot = SourceArtifactIdentity(
        key=SourceArtifactKey("uniprot", ArtifactKind.PRIMARY_RELEASE),
        release_id="2026_03",
        coordinate_context=nobuild,
        artifact_sha256=_A,
    )
    legacy37 = SourceArtifactIdentity(
        key=SourceArtifactKey("dbnsfp", ArtifactKind.PRIMARY_RELEASE),
        release_id="4.5a",
        coordinate_context=grch37,
        artifact_sha256=_B,
    )

    multirole = SourceDependency(
        identity=clinvar,
        roles=frozenset({SourceRole.OBSERVATION, SourceRole.LABEL}),
    )
    single = SourceDependency(
        identity=uniprot,
        roles=frozenset({SourceRole.REFERENCE_SEQUENCE}),
    )
    three_products = SourceEvidenceManifest.of(tuple(
        SourceDependency(identity=i, roles=frozenset({SourceRole.ANNOTATION}))
        for i in (gencode_tx, gencode_pc, gencode_lnc)))
    multi_authority = SourceEvidenceManifest.of((multirole, single))

    acquisition = SourceAcquisition(
        identity=clinvar,
        provenance=SourceRetrievalProvenance(
            retrieved_at="2026-08-01T00:00:00Z",
            observed_row_count=4_400_000,
            origin_locator="fixture://clinvar",
            transport="fixture",
        ),
    )
    manifest = SourceManifest(
        evidence=SourceEvidenceManifest.of((multirole,)),
        acquisitions=(acquisition,),
    )

    #: All FIVE component kinds. Section 11 asks for this explicitly, and a
    #: corpus covering three would not notice a kind whose ordering moved.
    transformation = TransformationIdentity.of(tuple(
        TransformationComponent(kind=k, schema_version=1,
                                fingerprint=_fixture_digest(k.value))
        for k in TransformationComponentKind))

    return {
        "coordinate_grch38": grch38,
        "coordinate_grch37": grch37,
        "coordinate_build_independent": nobuild,
        "key_clinvar": clinvar.key,
        "key_gencode_transcripts": gencode_tx.key,
        "key_gencode_pc_transcripts": gencode_pc.key,
        "key_gencode_lncrna_transcripts": gencode_lnc.key,
        "identity_clinvar": clinvar,
        "identity_uniprot_build_independent": uniprot,
        "identity_dbnsfp_grch37": legacy37,
        "dependency_multi_role": multirole,
        "acquisition_clinvar": acquisition,
        "evidence_three_gencode_products": three_products,
        "evidence_multi_authority": multi_authority,
        "manifest_clinvar": manifest,
        "transformation_all_component_kinds": transformation,
    }


def _semantic(obj) -> dict:
    """Every scientifically meaningful representation the object exposes.

    `__module__` is DELIBERATELY ABSENT. It is the one thing Unit 3A is
    supposed to change, so including it here would make the semantic oracle
    fail by construction and prove nothing. Ownership is captured separately
    in `ownership.json`, which the post-move test asserts DID change.

    That split is the point: one file must be identical afterwards, the other
    must not.
    """
    out = {"type_name": type(obj).__name__}
    for attr in ("canonical_key", "digest", "evidence_digest"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            out[attr] = list(value) if isinstance(value, tuple) else value
    for meth in ("as_record", "describe"):
        if hasattr(obj, meth):
            out[meth] = getattr(obj, meth)()
    return out


class NondeterministicHashing(RuntimeError):
    """The interpreter would produce unstable pickle bytes."""


def _require_deterministic_hashing() -> None:
    """Refuse to emit bytes that cannot be reproduced.

    MEASURED 2026-09-02: three fixtures -- `dependency_multi_role`,
    `evidence_multi_authority` and `manifest_clinvar` -- carry
    `frozenset({SourceRole, ...})`. `SourceRole` is a str-based enum, so its
    hash is randomized per process, so the frozenset's ITERATION ORDER varies,
    so `pickle.dumps` emits different bytes on different runs. Three
    consecutive runs produced 68e9ee011ce2, efaa2a1ed0b5, 68e9ee011ce2 for one
    fixture of constant length.

    `--check` could not detect this: it compares `loaded == original`, and set
    equality is order-independent. It reported zero failures on genuinely
    different bytes.

    So the corpus REFUSES to generate under an unpinned seed rather than
    producing bytes whose later regeneration would differ for reasons having
    nothing to do with the migration it exists to judge.
    """
    seed = os.environ.get("PYTHONHASHSEED")
    if seed != "0":
        raise NondeterministicHashing(
            "PYTHONHASHSEED is {!r}; it must be '0'. Three fixtures contain a "
            "frozenset of a str-based enum, whose iteration order -- and "
            "therefore whose pickled bytes -- depends on per-process hash "
            "randomization. Run through main(), which re-executes with the "
            "seed pinned.".format(seed))


def build() -> dict:
    """Every corpus file, as {filename: bytes}. Writes nothing.

    Exposed so an INSTALLER can produce the corpus inside its own transaction
    rather than receiving sixteen binary pickles through a text channel. The
    same function backs `main()`, so the committed bytes and the bytes a
    developer generates by hand are produced by one code path -- not two that
    happen to agree.
    """
    _require_deterministic_hashing()
    corpus = _corpus()
    semantic = {name: _semantic(obj) for name, obj in sorted(corpus.items())}
    ownership = {name: "{}.{}".format(type(obj).__module__,
                                      type(obj).__qualname__)
                 for name, obj in sorted(corpus.items())}
    pickles = {name: pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
               for name, obj in sorted(corpus.items())}
    index = {
        "schema": 1,
        "generated_by": "build_provenance_migration_fixtures.py",
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "hash_seed": "0",
        "entries": {
            name: {"pickle": name + ".pickle",
                   "pickle_sha256": hashlib.sha256(blob).hexdigest(),
                   "pickle_bytes": len(blob)}
            for name, blob in sorted(pickles.items())
        },
    }
    out = {name + ".pickle": blob for name, blob in pickles.items()}
    out["semantic.json"] = _json_bytes(semantic)
    out["index.json"] = _json_bytes(index)
    out["ownership.json"] = _json_bytes(ownership)
    return out


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str]) -> int:
    # Re-execute ONCE with the hash seed pinned. Setting it inside a running
    # interpreter is too late: str hashing is already seeded.
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        import subprocess
        return subprocess.call([sys.executable, "-B", __file__] + list(argv),
                               env=env)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare; write nothing")
    args = ap.parse_args(argv)

    # ONE code path. An installer calling build() and a developer running
    # --out must produce identical bytes, or the corpus is not an oracle.
    payloads = build()
    corpus = _corpus()
    pickles = {n[:-len(".pickle")]: b for n, b in payloads.items()
               if n.endswith(".pickle")}
    sem_bytes = payloads["semantic.json"]
    idx_bytes = payloads["index.json"]
    own_bytes = payloads["ownership.json"]
    ownership = json.loads(own_bytes.decode("utf-8"))

    if args.check:
        bad = []
        for name, blob in pickles.items():
            p = args.out / (name + ".pickle")
            if not p.is_file():
                bad.append("{}: absent".format(name)); continue
            have = p.read_bytes()
            loaded = pickle.loads(have)
            if type(loaded) is not type(corpus[name]):
                bad.append("{}: loads as {}, expected {}".format(
                    name, type(loaded).__name__, type(corpus[name]).__name__))
            elif loaded != corpus[name]:
                bad.append("{}: loaded object is not equal".format(name))
        have_sem = (args.out / "semantic.json").read_bytes() \
            if (args.out / "semantic.json").is_file() else b""
        if have_sem != sem_bytes:
            bad.append("semantic.json DIFFERS -- a MEANING changed, which is "
                       "exactly what Unit 3A must not do")
        # ownership.json is expected to DIFFER after the move. Report the
        # comparison without failing on it, so the migration can SEE that
        # ownership moved rather than merely assuming it.
        own_path = args.out / "ownership.json"
        if own_path.is_file():
            frozen = json.loads(own_path.read_text(encoding="utf-8"))
            moved = [n for n in sorted(ownership)
                     if frozen.get(n) != ownership[n]]
            print("  ownership entries that MOVED: {} of {}".format(
                len(moved), len(ownership)))
            for n in moved:
                print("      {:<38} {}".format(n, frozen.get(n)))
                print("      {:<38} -> {}".format("", ownership[n]))
        for line in bad:
            print("  [FAIL] {}".format(line))
        print("  entries checked: {} | failures: {}".format(len(pickles), len(bad)))
        return 1 if bad else 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, blob in pickles.items():
        (args.out / (name + ".pickle")).write_bytes(blob)
    (args.out / "semantic.json").write_bytes(sem_bytes)
    (args.out / "index.json").write_bytes(idx_bytes)
    (args.out / "ownership.json").write_bytes(own_bytes)
    print("  wrote {} pickle(s), semantic.json ({} B), index.json ({} B), "
          "ownership.json ({} B)".format(
              len(pickles), len(sem_bytes), len(idx_bytes), len(own_bytes)))
    for name in sorted(pickles):
        print("    {:<38} {:>6} B  {}".format(
            name, len(pickles[name]),
            hashlib.sha256(pickles[name]).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
