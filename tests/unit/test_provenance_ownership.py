"""One class object, two import paths -- and a legacy surface that still holds.

Phase 1C Unit 3A. Created 2026-09-02.

    UNIT 3A MAY CHANGE WHERE A SCIENTIFIC IDENTITY IS DEFINED.
    IT MUST NOT CHANGE WHAT THAT IDENTITY MEANS.

The frozen corpus committed at `2d90c23` judges the second half: `semantic.json`
must stay byte-identical and every pre-move pickle must still load. This file
judges the FIRST half -- that ownership actually moved, that the legacy surface
is an EXACT alias rather than a copy, and that no concept acquired a second
authority.

WHY OBJECT IDENTITY, NOT EQUALITY
---------------------------------
A compatibility shim could be written three ways:

    from provenance.source import SourceArtifactKey        exact alias
    class SourceArtifactKey(provenance...SourceArtifactKey): ...   SUBCLASS
    @dataclass ... class SourceArtifactKey: ...            COPY

Only the first is safe. A subclass is a DIFFERENT runtime type, so
`except LegacySourceError` stops catching the canonical one and `isinstance`
silently narrows. A copy compares equal field-by-field while being a different
class entirely -- and `pickle` resolves by module and name, so a copy would
load old bytes into the wrong authority.

`assert legacy is canonical` kills all three bad strategies at once.

WHY `__module__` IS NOT FORGED
------------------------------
Reflection must report the REAL owner. This repository has already paid for a
class whose `__module__` was reassigned -- `_CNN1DModule` in
`variant_ensemble.py`, and the `scripts/migrate_pickles.py` machinery that
exists to repair such things. Old pickles still load because a pickle resolves
`module.Name` THROUGH the legacy module, and the legacy module hands back the
canonical object. No lie is required.

WHAT STAYS IN MONITORING
------------------------
Provenance defines states; monitoring compares them. `SourceDeltaKind`,
`SourceTransition`, `source_transitions`, `differing_releases`,
`differing_components` and the representation comparators are judgements about
TWO things, and they remain drift concerns.

Acronyms: AST = abstract syntax tree; SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import genomic_variant_classifier.monitoring.drift as drift
import genomic_variant_classifier.provenance as provenance

_SRC = Path(__file__).resolve().parents[2] / "src" / "genomic_variant_classifier"
_PROV = _SRC / "provenance"
_DRIFT = _SRC / "monitoring" / "drift"

#: Every name whose canonical owner moved in this unit, and where it now lives.
CANONICAL_OWNER = {
    "canonical_json": "genomic_variant_classifier.provenance.serialization",
    "domain_digest": "genomic_variant_classifier.provenance.serialization",
    "GENOME_ASSEMBLIES": "genomic_variant_classifier.provenance.coordinate",
    "CoordinateContext": "genomic_variant_classifier.provenance.coordinate",
    "CoordinateContextKind": "genomic_variant_classifier.provenance.coordinate",
    "CoordinateError": "genomic_variant_classifier.provenance.coordinate",
    "assemblies_in": "genomic_variant_classifier.provenance.coordinate",
    "ArtifactKind": "genomic_variant_classifier.provenance.artifact",
    "SourceRole": "genomic_variant_classifier.provenance.source",
    "SourceError": "genomic_variant_classifier.provenance.source",
    "SourceIdentityError": "genomic_variant_classifier.provenance.source",
    "SourceArtifactKey": "genomic_variant_classifier.provenance.source",
    "SourceArtifactIdentity": "genomic_variant_classifier.provenance.source",
    "SourceRetrievalProvenance": "genomic_variant_classifier.provenance.source",
    "SourceDependency": "genomic_variant_classifier.provenance.source",
    "SourceAcquisition": "genomic_variant_classifier.provenance.source",
    "SourceEvidenceManifest": "genomic_variant_classifier.provenance.source",
    "SourceManifest": "genomic_variant_classifier.provenance.source",
    "TransformationComponentKind":
        "genomic_variant_classifier.provenance.transformation",
    "TransformationComponent":
        "genomic_variant_classifier.provenance.transformation",
    "TransformationError":
        "genomic_variant_classifier.provenance.transformation",
    "TransformationIdentity":
        "genomic_variant_classifier.provenance.transformation",
}

#: Modules that became pure compatibility surfaces. Section 39: a shim that
#: acquires a definition has become a second authority.
LEGACY_SHIMS = (
    "_digest.py", "coordinate.py", "source_vocabulary.py",
    "source_release.py", "transformation.py",
)

#: Comparisons. These answer "what CHANGED between two things" and stay here.
MONITORING_ONLY = (
    "SourceDeltaKind", "SourceTransition", "source_transitions",
    "differing_releases", "differing_components", "RepresentationIdentity",
    "RepresentationDelta", "assert_same_representation",
)


# ---------------------------------------------------------------------------
# 1. ownership actually moved
# ---------------------------------------------------------------------------

#: `GENOME_ASSEMBLIES` is a tuple of strings, so it carries no `__module__`
#: to assert. It is EXCLUDED from this parametrization rather than skipped
#: inside it: a case that always skips is not a test, and it would have moved
#: the suite's skip count off the fifteen it has held all session. Its ALIAS
#: identity is still checked by test_legacy_and_canonical_are_the_SAME_OBJECT.
DEFINED_OWNERS = tuple(n for n in sorted(CANONICAL_OWNER)
                       if n != "GENOME_ASSEMBLIES")


@pytest.mark.parametrize("name", DEFINED_OWNERS)
def test_the_canonical_owner_is_provenance(name):
    owner = getattr(getattr(provenance, name), "__module__", None)
    assert owner is not None, "{} carries no __module__".format(name)
    assert owner == CANONICAL_OWNER[name], (name, owner)


def test_no_canonical_class_still_claims_monitoring():
    """The complement. A single missed relocation is invisible per-name."""
    wrong = []
    for name in sorted(CANONICAL_OWNER):
        owner = getattr(getattr(provenance, name), "__module__", "")
        if "monitoring" in owner:
            wrong.append((name, owner))
    assert not wrong, wrong


# ---------------------------------------------------------------------------
# 2. the legacy surface is an EXACT alias
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICAL_OWNER))
def test_legacy_and_canonical_are_the_SAME_OBJECT(name):
    """`is`, not `==`. A subclass or a copy would pass equality and fail this.

    MEASURED 2026-09-02: the drift package facade exports THIRTY-TWO names and
    does NOT include `canonical_json`, `domain_digest`, `EVIDENCE_DOMAIN` or
    `SourceIdentityError` -- the first two because `_digest` is private, the
    last because Unit 1 introduced it without publishing it
    (SOURCE-IDENTITY-ERROR-NOT-EXPORTED-1).

    So membership is DERIVED from `__all__` rather than assumed. Asserting the
    facade exports something it never exported would be this test inventing a
    requirement, not checking one.
    """
    if name in drift.__all__:
        assert getattr(drift, name) is getattr(provenance, name), name
    else:
        assert not hasattr(drift, name) or \
            getattr(drift, name) is getattr(provenance, name), name


@pytest.mark.parametrize(
    "module_name, name",
    [("source_release", "SourceError"),
     ("source_release", "SourceIdentityError"),
     ("coordinate", "CoordinateError"),
     ("transformation", "TransformationError")])
def test_EXCEPTION_identity_is_preserved(module_name, name):
    """Easy to miss, and the most damaging to get wrong.

    Two independently declared `class SourceError(ValueError)` compare unequal
    as types, so `except LegacySourceError` would stop catching the canonical
    one -- silently, at runtime, in an error path.
    """
    legacy_module = importlib.import_module(
        "genomic_variant_classifier.monitoring.drift." + module_name)
    legacy, canon = getattr(legacy_module, name), getattr(provenance, name)
    assert legacy is canon
    assert issubclass(canon, ValueError)
    try:
        raise canon("x")
    except legacy:
        pass
    else:                                       # pragma: no cover
        pytest.fail("{} raised canonical is not caught by legacy".format(name))


def test_the_drift_public_surface_did_NOT_shrink():
    """Section 46: public legacy imports broken == 0."""
    assert len(drift.__all__) == 32, sorted(drift.__all__)
    for name in drift.__all__:
        assert hasattr(drift, name), name


def test_module_LEVEL_legacy_paths_still_resolve():
    """A pickle resolves `module.Name`, not `package.Name`.

    So the per-MODULE legacy paths must work, not merely the package facade.
    """
    for mod, names in (
        ("_digest", ("canonical_json", "domain_digest")),
        ("coordinate", ("CoordinateContext", "CoordinateError")),
        ("source_vocabulary", ("ArtifactKind",)),
        ("source_release", ("SourceArtifactKey", "SourceEvidenceManifest",
                            "SourceIdentityError")),
        ("transformation", ("TransformationIdentity", "differing_components")),
    ):
        m = importlib.import_module(
            "genomic_variant_classifier.monitoring.drift." + mod)
        for n in names:
            assert hasattr(m, n), "{}.{}".format(mod, n)
            if n in CANONICAL_OWNER:
                assert getattr(m, n) is getattr(provenance, n), (mod, n)


# ---------------------------------------------------------------------------
# 3. one authority per concept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", LEGACY_SHIMS)
def test_a_legacy_module_defines_NOTHING(filename):
    """Section 39. This is what makes the strangler pattern enforceable.

    Without it, someone adds a class to the old namespace six months from now
    and the project has two authorities again -- which is exactly how
    `SourceName` came to invent eighteen members beside a manifest that
    already declared thirty-two sources.
    """
    tree = ast.parse((_DRIFT / filename).read_text(encoding="utf-8"))
    defined = [n.name for n in tree.body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert defined == [], (filename, defined)


@pytest.mark.parametrize("filename", LEGACY_SHIMS)
def test_a_shim_EXPORTS_everything_it_imports(filename):
    """A shim's `__all__` must equal the names it re-exports.

    MEASURED BY SABOTAGE 2026-09-02: dropping a name from a shim's `__all__`
    changed NO test. `test_the_drift_public_surface_did_NOT_shrink` checks the
    PACKAGE facade, and the name stays importable explicitly, so the loss is
    invisible -- until someone writes `from ...source_release import *` and
    silently gets less than the module claims to offer.

    A compatibility surface that under-declares itself is a surface that will
    be trimmed by accident.
    """
    tree = ast.parse((_DRIFT / filename).read_text(encoding="utf-8"))
    imported = {a.asname or a.name
                for n in tree.body if isinstance(n, ast.ImportFrom)
                for a in n.names if a.name != "annotations"}
    declared = {x.value for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "__all__"
                for x in n.value.elts}
    assert declared == imported, {
        "file": filename,
        "imported_but_not_exported": sorted(imported - declared),
        "exported_but_not_imported": sorted(declared - imported),
    }


def test_the_canonical_modules_are_not_SHIMS_themselves():
    """The complement of the shim test: provenance must hold real code.

    If a relocation went the wrong way -- provenance re-exporting from
    monitoring -- every alias test above would still pass, because
    `legacy is canonical` holds for either direction.
    """
    empty = []
    for path in sorted(_PROV.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not [n for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))]:
            empty.append(path.name)
    assert not empty, (
        "these canonical modules define nothing: {}".format(empty))


def test_every_relocated_name_has_exactly_ONE_definition():
    """Measured across the WHOLE source tree, not the two packages involved."""
    where = {n: [] for n in CANONICAL_OWNER}
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):      # pragma: no cover
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in where:
                    where[node.name].append(path.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in where:
                        where[t.id].append(path.name)
    duplicated = {n: f for n, f in where.items() if len(f) != 1}
    assert not duplicated, duplicated


# ---------------------------------------------------------------------------
# 4. the layering, both directions
# ---------------------------------------------------------------------------

def test_provenance_imports_NOTHING_from_a_higher_layer():
    forbidden = ("data", "models", "training", "evaluation", "monitoring",
                 "pipelines", "api", "agent_layer", "conformal", "reports")
    offenders = []
    for path in sorted(_PROV.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mod = ""
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            for bad in forbidden:
                if "genomic_variant_classifier.{}".format(bad) in mod:
                    offenders.append("{}:{} {}".format(path.name, node.lineno, mod))
    assert not offenders, offenders


def test_the_drift_kernel_now_imports_PROVENANCE():
    """The complementary direction. Absence of the forbidden import is not
    evidence that the intended one exists."""
    seen = set()
    for path in sorted(_DRIFT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and "genomic_variant_classifier.provenance" in node.module:
                seen.add(path.name)
    assert seen >= set(LEGACY_SHIMS), sorted(set(LEGACY_SHIMS) - seen)


def test_no_internal_provenance_module_imports_through_the_PACKAGE():
    """Section 26: `__init__` is outward-facing only.

    A module importing `from genomic_variant_classifier.provenance import X`
    would depend on the whole export surface and create hidden cycles as that
    surface grows.
    """
    offenders = []
    for path in sorted(_PROV.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) \
                    and node.module == "genomic_variant_classifier.provenance":
                offenders.append("{}:{}".format(path.name, node.lineno))
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# 5. what did NOT move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MONITORING_ONLY)
def test_a_COMPARISON_stays_in_monitoring(name):
    """Provenance defines states; monitoring compares them.

    MEASURED BY SABOTAGE 2026-09-02: checking only `hasattr(provenance, name)`
    was NOT enough. Adding `differing_components` to
    `provenance/transformation.py` left the package `__init__` untouched, so
    the package-level assertion still passed while a comparison had in fact
    leaked into the identity substrate.

    The MODULE is where a leak happens. The package surface is only where it
    eventually becomes visible.
    """
    assert hasattr(drift, name), name
    assert not hasattr(provenance, name), (
        "{} is a judgement about TWO things and does not belong to the "
        "identity substrate".format(name))
    leaked = []
    for path in sorted(_PROV.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) \
                    and node.name == name:
                leaked.append(path.name)
    assert not leaked, (
        "{} is DEFINED in {} -- provenance defines states, monitoring "
        "compares them".format(name, leaked))


def test_the_evidence_domain_did_NOT_move_with_the_module():
    """A namespace change is not an identity-semantic change.

    Bumping the domain because the Python path changed would give the same
    scientific evidence a different digest for a cosmetic reason.
    """
    # v5 since Phase 1C Unit 3A++.2 -- a DELIBERATE semantic migration, not a
    # namespace move. This test's claim is unchanged: relocating a module did
    # not move the domain. The later epoch migration did, for a stated reason,
    # and `test_source_evidence_epoch_v5.py` proves exactly what it changed.
    assert provenance.EVIDENCE_DOMAIN == "drift-source-evidence-manifest-v5"
    assert provenance.TRANSFORMATION_DOMAIN == "drift-transformation-identity-v1"
    # Through the MODULE path, which is what a legacy caller and a pickle use.
    # The package facade never exported these; see the note on
    # test_legacy_and_canonical_are_the_SAME_OBJECT.
    legacy_source = importlib.import_module(
        "genomic_variant_classifier.monitoring.drift.source_release")
    legacy_tx = importlib.import_module(
        "genomic_variant_classifier.monitoring.drift.transformation")
    assert legacy_source.EVIDENCE_DOMAIN == provenance.EVIDENCE_DOMAIN
    assert legacy_tx.TRANSFORMATION_DOMAIN == provenance.TRANSFORMATION_DOMAIN
