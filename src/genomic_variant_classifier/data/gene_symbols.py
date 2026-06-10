"""Shared gene-symbol normalization for UniProt/sequence resolution.

A single source of truth for turning a raw cohort ``gene_symbol`` value into the
symbol(s) to look up. Used by every consumer that resolves a gene to a UniProt
sequence or entry (esm2, eve, protein_pipeline, database_connectors), so the
normalization rules -- especially the multi-gene delimiter handling -- never
drift between call sites.

Pure and dependency-free by design (no logging in a library module).

Multi-gene annotations
----------------------
Some cohort rows carry a ``gene_symbol`` that joins several overlapping genes
with a semicolon, e.g. ``"MYH11;NDE1"`` (adjacent on 16p13.11) or
``"CRIPAK;LOC126806945;UVSSA"``. A single-symbol index has no key for the joined
string, so those rows silently resolve to nothing. ``gene_symbol_candidates``
splits on ``;`` and yields each component so the first that exists in the lookup
wins.

It NEVER splits on ``-``. Hyphens are legitimate inside single HGNC symbols
(``HLA-A``, ``HLA-DRB1``, ``NKX2-1``) and inside readthrough-fusion names
(``JMJD7-PLA2G4B``, ``ATP5MF-PTCD1``); splitting those would corrupt valid
symbols and invent spurious lookups.
"""

from __future__ import annotations

_MULTIGENE_DELIM = ";"
_NULLISH = {"", "NAN", "NONE", "NA", "<NA>"}


def normalize_gene_symbol(raw: object) -> str:
    """Canonical single-symbol form: ``str`` -> stripped, upper-cased.

    Returns ``""`` for ``None`` and null-ish strings (``"nan"``, ``"none"``,
    ``"na"``, ``"<NA>"``), so callers can treat empty as "no usable symbol".
    Does NOT split multi-gene strings -- use :func:`gene_symbol_candidates`
    for that.
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    return "" if s in _NULLISH else s


def gene_symbol_candidates(raw: object) -> list[str]:
    """Ordered, de-duplicated lookup candidates for a raw gene_symbol.

    The full normalized symbol is always first (so an exact hit on a joined
    symbol, should the index ever contain one, wins). If it contains the ``;``
    multi-gene delimiter, each non-empty component follows, in order. Never
    splits on ``-``. Returns ``[]`` for a null-ish symbol.

    Examples
    --------
    ``"BRCA1"``              -> ``["BRCA1"]``
    ``"  myh11;nde1 "``      -> ``["MYH11;NDE1", "MYH11", "NDE1"]``
    ``"HLA-A"``              -> ``["HLA-A"]``           (never split on '-')
    ``"ECE2;EEF1AKMT4-ECE2"``-> ``["ECE2;EEF1AKMT4-ECE2", "ECE2", "EEF1AKMT4-ECE2"]``
    ``None`` / ``"nan"``     -> ``[]``
    """
    full = normalize_gene_symbol(raw)
    if not full:
        return []
    out = [full]
    if _MULTIGENE_DELIM in full:
        for part in full.split(_MULTIGENE_DELIM):
            p = part.strip()
            if p and p not in out:
                out.append(p)
    return out
