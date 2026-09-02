"""Compatibility import surface. Canonical ownership moved to genomic_variant_classifier.provenance.serialization.

Phase 1C Unit 3A, 2026-09-02.

THIS MODULE DEFINES NOTHING. Every name below is an EXACT re-export -- the
same class object, not a subclass and not a copy:

    from genomic_variant_classifier.provenance import domain_digest
    from genomic_variant_classifier.monitoring.drift import domain_digest as legacy
    assert legacy is domain_digest          # True

That identity is what makes the move safe. A subclass would be a DIFFERENT
runtime type, so `except LegacyError` would stop catching the canonical one
and an `isinstance` check would silently narrow.

`__module__` is NOT forged back to this path. Reflection reports the real
owner, which is the point: the previous class-relocation episode in this
repository -- `_CNN1DModule.__module__` and `scripts/migrate_pickles.py` --
is what happens when Python ownership is lied about. Old pickles still load
because a pickle resolves `module.Name` through THIS module, and this module
hands back the canonical object.

DO NOT ADD PROVENANCE SEMANTICS HERE. This module is private (_digest); it exists only so existing intra-drift
imports keep resolving.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.provenance.serialization import (
    canonical_json,
    domain_digest,)

__all__ = [
    "canonical_json",
    "domain_digest",]
