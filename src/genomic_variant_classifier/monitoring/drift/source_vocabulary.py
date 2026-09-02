"""Compatibility import surface. Canonical ownership moved to genomic_variant_classifier.provenance.artifact.

Phase 1C Unit 3A, 2026-09-02.

THIS MODULE DEFINES NOTHING. Every name below is an EXACT re-export -- the
same class object, not a subclass and not a copy:

    from genomic_variant_classifier.provenance import ArtifactKind
    from genomic_variant_classifier.monitoring.drift import ArtifactKind as legacy
    assert legacy is ArtifactKind          # True

That identity is what makes the move safe. A subclass would be a DIFFERENT
runtime type, so `except LegacyError` would stop catching the canonical one
and an `isinstance` check would silently narrow.

`__module__` is NOT forged back to this path. Reflection reports the real
owner, which is the point: the previous class-relocation episode in this
repository -- `_CNN1DModule.__module__` and `scripts/migrate_pickles.py` --
is what happens when Python ownership is lied about. Old pickles still load
because a pickle resolves `module.Name` through THIS module, and this module
hands back the canonical object.

DO NOT ADD PROVENANCE SEMANTICS HERE. ArtifactKind is a LOCAL vocabulary for scientific artifacts, not a drift
concept.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.provenance.artifact import (
    ArtifactKind,)

__all__ = [
    "ArtifactKind",]
