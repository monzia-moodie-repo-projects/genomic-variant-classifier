"""Compatibility import surface. Canonical ownership moved to genomic_variant_classifier.provenance.coordinate.

Phase 1C Unit 3A, 2026-09-02.

THIS MODULE DEFINES NOTHING. Every name below is an EXACT re-export -- the
same class object, not a subclass and not a copy:

    from genomic_variant_classifier.provenance import CoordinateContext
    from genomic_variant_classifier.monitoring.drift import CoordinateContext as legacy
    assert legacy is CoordinateContext          # True

That identity is what makes the move safe. A subclass would be a DIFFERENT
runtime type, so `except LegacyError` would stop catching the canonical one
and an `isinstance` check would silently narrow.

`__module__` is NOT forged back to this path. Reflection reports the real
owner, which is the point: the previous class-relocation episode in this
repository -- `_CNN1DModule.__module__` and `scripts/migrate_pickles.py` --
is what happens when Python ownership is lied about. Old pickles still load
because a pickle resolves `module.Name` through THIS module, and this module
hands back the canonical object.

DO NOT ADD PROVENANCE SEMANTICS HERE. Coordinate semantics belong to provenance.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.provenance.coordinate import (
    CoordinateContext,
    CoordinateContextKind,
    CoordinateError,
    GENOME_ASSEMBLIES,
    assemblies_in,)

__all__ = [
    "CoordinateContext",
    "CoordinateContextKind",
    "CoordinateError",
    "GENOME_ASSEMBLIES",
    "assemblies_in",]
