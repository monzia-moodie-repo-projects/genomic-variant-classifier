"""The source registry must read the manifest, not a remembered version of it.

Created 2026-08-29 after `AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`.

WHAT THESE TESTS GUARD
----------------------
`SourceName` in the drift package was written from a path census on the
assumption that nothing declared these sources. MEASURED against
`configs/data_manifest.yaml` on 2026-08-29:

    declared sources                32      SourceName members        18
    sources it cannot name          16      aliases it accepts         0
    members declared nowhere         2      aliases it invented       26

Four of the sixteen it cannot name are `irreplaceable` and constrained:
`tcga` and `topmed` are `controlled`, `rnaseq` and `validation_cohort` are
`review`.

TWO KINDS OF TEST HERE
----------------------
Most run against a SYNTHETIC manifest, so they test the reader's logic rather
than today's estate. Two run against the REAL file, because a reader that
parses a fixture perfectly and refuses the actual manifest is useless -- and
because the real file is the thing whose 32 declarations had no test at all.

Acronyms: YAML = YAML Ain't Markup Language; DUA = Data Use Agreement.

Author: Monzia Moodie
"""
from __future__ import annotations

from pathlib import Path

import pytest

from genomic_variant_classifier.data.source_registry import (
    DEFAULT_MANIFEST,
    SourceClass,
    SourceDeclaration,
    SourceLocation,
    SourceRegistry,
    SourceRegistryError,
    SourceTier,
)

_ROOT = Path(__file__).resolve().parents[2]
_REAL = _ROOT / DEFAULT_MANIFEST

_MINIMAL = """version: 1
sources:
  clinvar:
    location: external
    tier: public
    class: public_redownloadable
    aliases: [clinvar_fresh]
    version: "GRCh38 2024-07"
    acquire: "NCBI ClinVar VCF"
    regenerate: ""
    sync: false
    notes: ""
  tcga:
    location: external
    tier: controlled
    class: irreplaceable
    aliases: []
    version: ""
    acquire: "GDC; controlled-access via dbGaP."
    regenerate: ""
    sync: false
    notes: "CONTROLLED-ACCESS."
  reactome_gene_pathways:
    location: external
    tier: public
    class: regenerable_expensive
    aliases: []
    version: "built parquet"
    acquire: ""
    regenerate: "python scripts/build_reactome_parquet.py --gmt g --out o"
    sync: true
    notes: "Built artifact."
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "data_manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. It reads what the manifest declares
# ---------------------------------------------------------------------------

def test_every_declaration_is_typed(tmp_path):
    r = SourceRegistry.load(_write(tmp_path, _MINIMAL))
    d = r.declaration("tcga")
    assert d.tier is SourceTier.CONTROLLED
    assert d.cls is SourceClass.IRREPLACEABLE
    assert d.location is SourceLocation.EXTERNAL
    assert isinstance(d.tier, SourceTier), "a raw string here is the defect"


def test_the_registry_records_where_it_read_from(tmp_path):
    """Provenance in the type, as `StoragePolicy.source` does.

    A reader that cannot say where its values came from cannot be audited.
    """
    p = _write(tmp_path, _MINIMAL)
    assert SourceRegistry.load(p).manifest_source == str(p)


def test_published_and_derived_are_distinguished(tmp_path):
    """`acquire` and `regenerate` already separate them; the reader exposes it."""
    r = SourceRegistry.load(_write(tmp_path, _MINIMAL))
    assert r.declaration("clinvar").is_published
    assert not r.declaration("clinvar").is_derived
    assert r.declaration("reactome_gene_pathways").is_derived
    assert not r.declaration("reactome_gene_pathways").is_published


def test_declared_aliases_resolve_and_undeclared_ones_do_not(tmp_path):
    """The standard, section 3: a source has exactly ONE canonical name, and
    the manifest records aliases so the auditor can fold them away."""
    r = SourceRegistry.load(_write(tmp_path, _MINIMAL))
    assert r.canonical_for("clinvar_fresh") == "clinvar"
    assert r.canonical_for("clinvar") == "clinvar"
    for invented in ("ncbi-clinvar", "ncbi_clinvar", "ClinVarPlus"):
        with pytest.raises(SourceRegistryError):
            r.canonical_for(invented)


# ---------------------------------------------------------------------------
# 2. It refuses what raw dictionary access admits
# ---------------------------------------------------------------------------

def test_a_controlled_source_marked_for_sync_is_refused(tmp_path):
    """The standard, section 5: controlled data is backed up offline ONLY.

    `setup_data_tree.py` checks this with `m.get("tier") == "controlled"`, so a
    misspelled key would pass the gate. Here the tier is TYPED first.
    """
    text = _MINIMAL.replace(
        '    acquire: "GDC; controlled-access via dbGaP."\n    regenerate: ""\n    sync: false',
        '    acquire: "GDC"\n    regenerate: ""\n    sync: true')
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, text))
    assert "CONTROLLED" in str(exc.value)


@pytest.mark.parametrize(
    "old,new,fragment",
    [("tier: controlled", "tier: contrlled", "expected one of"),
     ("class: irreplaceable", "class: irreplacable", "expected one of"),
     ("location: external", "location: exernal", "expected one of")],
    ids=["tier", "class", "location"])
def test_a_misspelled_value_is_refused(tmp_path, old, new, fragment):
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, _MINIMAL.replace(old, new, 1)))
    assert fragment in str(exc.value)


def test_a_misspelled_KEY_is_refused(tmp_path):
    """THE DEFECT raw access permits. `meta.get("tier")` returns None for
    `teir`, and `None != "controlled"`, so the compliance gate passes."""
    text = _MINIMAL.replace("  tcga:\n    location: external",
                            "  tcga:\n    teir: controlled\n    location: external")
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, text))
    assert "teir" in str(exc.value)


def test_a_missing_required_field_is_refused(tmp_path):
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, _MINIMAL.replace(
            "    tier: controlled\n", "", 1)))
    assert "declares no tier" in str(exc.value)


def test_a_source_cannot_alias_itself(tmp_path):
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, _MINIMAL.replace(
            "aliases: [clinvar_fresh]", "aliases: [clinvar]", 1)))
    assert "its own alias" in str(exc.value)


def test_one_alias_cannot_belong_to_two_sources(tmp_path):
    """REPLACE tcga's alias list rather than prepending a second one.

    The first version of this test INSERTED `aliases: [clinvar_fresh]` above
    the existing `aliases: []`. YAML takes the LAST duplicate key, so tcga
    parsed with an empty list, no conflict existed, and the reader was right to
    accept it. The fixture was wrong, not the reader.
    """
    text = _MINIMAL.replace(
        "    class: irreplaceable\n    aliases: []",
        "    class: irreplaceable\n    aliases: [clinvar_fresh]", 1)
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, text))
    assert "claimed by both" in str(exc.value)


def test_the_alias_conflict_fixture_really_conflicts(tmp_path):
    """The fixture must exercise the defect, not merely differ.

    If the mutation did not actually give two sources one alias, the test
    above would pass against a reader that never checked.
    """
    import yaml

    text = _MINIMAL.replace(
        "    class: irreplaceable\n    aliases: []",
        "    class: irreplaceable\n    aliases: [clinvar_fresh]", 1)
    parsed = yaml.safe_load(text)["sources"]
    assert parsed["clinvar"]["aliases"] == ["clinvar_fresh"]
    assert parsed["tcga"]["aliases"] == ["clinvar_fresh"]


def test_an_alias_cannot_also_be_a_canonical_source(tmp_path):
    text = _MINIMAL.replace("aliases: [clinvar_fresh]", "aliases: [tcga]", 1)
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, text))
    assert "AND a canonical source" in str(exc.value)


def test_an_empty_sources_section_is_refused(tmp_path):
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(_write(tmp_path, "version: 1\nsources: {}\n"))
    assert "sources" in str(exc.value)


def test_a_missing_manifest_RAISES_rather_than_defaulting(tmp_path):
    """`StoragePolicy.load` falls back to documented defaults; this must not.

    One cannot invent 32 source declarations, and a fallback registry would
    silently answer questions about evidence this project does not have.
    """
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry.load(tmp_path / "absent.yaml")
    assert "no defensible default" in str(exc.value)


def test_declarations_are_ordered_so_two_reads_compare_equal(tmp_path):
    p = _write(tmp_path, _MINIMAL)
    assert SourceRegistry.load(p) == SourceRegistry.load(p)
    with pytest.raises(SourceRegistryError) as exc:
        SourceRegistry(declarations=tuple(reversed(
            SourceRegistry.load(p).declarations)), manifest_source=str(p))
    assert "canonical order" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. Against the REAL manifest
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL.is_file(), reason="data_manifest.yaml absent")
def test_the_real_manifest_loads():
    """A reader that parses a fixture and refuses the actual file is useless.

    This is also the first test of any kind over the 32 declarations: before
    today only the `storage:` block and one tier claim were bound.
    """
    r = SourceRegistry.load(_REAL)
    assert len(r.declarations) >= 30, r.describe()
    assert r.manifest_source.endswith("data_manifest.yaml")


@pytest.mark.skipif(not _REAL.is_file(), reason="data_manifest.yaml absent")
def test_no_controlled_source_is_marked_for_sync_in_the_real_manifest():
    """The compliance rule, checked against the real declarations.

    `audit_data_tree.py` exits 2 on this and `setup_data_tree.py` aborts, but
    both compare a RAW STRING. This asserts it over typed values.
    """
    r = SourceRegistry.load(_REAL)
    assert r.controlled, "the manifest declares no controlled source at all"
    offenders = [d.name for d in r.controlled if d.sync]
    assert offenders == [], (
        "controlled sources marked sync=true: {}. The standard, section 5: "
        "never to a personal cloud, which would breach the DUA."
        .format(offenders))
