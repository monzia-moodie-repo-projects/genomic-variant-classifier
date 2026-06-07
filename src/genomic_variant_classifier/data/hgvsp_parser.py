"""HGVSp protein-change parser.

Bridges the gap between a protein-change string (ClinVar `Name`-derived, e.g.
``p.Asp1692Asn``) and the ``(protein_pos, wt_aa, mut_aa)`` triple that the ESM-2
and EVE connectors consume. Missense substitutions only: synonymous, nonsense,
frameshift, indel, extension, and start-loss forms return ``(None, None, None)``,
because ``esm2_delta_norm`` is defined as the ref-vs-alt residue embedding distance
and is only meaningful for a single-residue amino-acid substitution.

Pure functions, no I/O, no logging (library-module rule). Handles 3-letter and
1-letter codes, optional ``p.(...)`` parentheses, and optional accession prefixes
(``NP_009225.1:p....``).

Examples
--------
>>> parse_hgvsp("p.Asp1692Asn")
(1692, 'D', 'N')
>>> parse_hgvsp("p.(Arg1699Gln)")
(1699, 'R', 'Q')
>>> parse_hgvsp("p.Arg1699Ter")      # nonsense
(None, None, None)
>>> parse_hgvsp("p.Asp1692Asp")      # synonymous
(None, None, None)
"""

from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

__all__ = [
    "parse_hgvsp",
    "parse_am_protein_variant",
    "add_protein_columns",
    "THREE_TO_ONE",
]

# Standard 20 amino acids only. Deliberately EXCLUDES Ter/Stop/Sec/Pyl so that
# nonsense and selenocysteine forms fail to normalise and are rejected as
# non-missense.
THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
_ONE_LETTER: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# wt-residue, position, mut-residue. Each residue is a 3-letter code (Upper+2 lower)
# or a single uppercase letter. Optional p.( ) wrapper. Anchored: anything trailing
# (fs, del, ins, dup, ext, Ter, *, =, ?) fails the match and is rejected.
_HGVSP_RE = re.compile(
    r"^p\.\(?"
    r"(?P<wt>[A-Z][a-z]{2}|[A-Z])"
    r"(?P<pos>\d+)"
    r"(?P<mut>[A-Z][a-z]{2}|[A-Z])"
    r"\)?$"
)

_NONE3: tuple[None, None, None] = (None, None, None)

# AlphaMissense protein_variant: single-letter wt + 1-based pos + single-letter mut,
# e.g. "V123M". Tolerates an optional "p." prefix or "UNIPROT:" prefix.
_AM_RE = re.compile(r"^(?:p\.)?([A-Z])(\d+)([A-Z])$")


def _norm_aa(tok: str) -> Optional[str]:
    """Normalise a 1- or 3-letter residue token to a single uppercase letter,
    or None if it is not one of the standard 20 amino acids."""
    if len(tok) == 3:
        return THREE_TO_ONE.get(tok.capitalize())
    if len(tok) == 1 and tok.upper() in _ONE_LETTER:
        return tok.upper()
    return None


def parse_hgvsp(s: object) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Parse a protein-change string into ``(protein_pos, wt_aa, mut_aa)``.

    Returns ``(None, None, None)`` for missing, malformed, synonymous, or
    non-substitution variants. ``wt_aa``/``mut_aa`` are single uppercase letters;
    ``protein_pos`` is 1-based.
    """
    if s is None:
        return _NONE3
    if isinstance(s, float) and math.isnan(s):
        return _NONE3
    t = str(s).strip()
    if not t or t.lower() in {"nan", "none", "<na>", "."}:
        return _NONE3
    # Drop an accession prefix, e.g. "NP_009225.1:p.Asp1692Asn".
    if ":" in t:
        t = t.rsplit(":", 1)[-1].strip()
    m = _HGVSP_RE.match(t)
    if not m:
        return _NONE3
    wt = _norm_aa(m.group("wt"))
    mut = _norm_aa(m.group("mut"))
    if wt is None or mut is None:
        return _NONE3
    if wt == mut:                       # synonymous (e.g. p.Asp1692Asp)
        return _NONE3
    pos = int(m.group("pos"))
    if pos < 1:
        return _NONE3
    return (pos, wt, mut)


def parse_am_protein_variant(s: object) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Parse an AlphaMissense ``protein_variant`` (e.g. ``"V123M"``, tolerating an
    optional ``p.`` or ``UNIPROT:`` prefix) into ``(protein_pos, wt_aa, mut_aa)``.

    Single-letter codes only — AlphaMissense's native format. AlphaMissense lists
    only missense substitutions, but we still reject synonymous (wt == mut) and
    malformed values, returning ``(None, None, None)``.
    """
    if s is None:
        return _NONE3
    if isinstance(s, float) and math.isnan(s):
        return _NONE3
    t = str(s).strip()
    if not t or t.lower() in {"nan", "none", "<na>", "."}:
        return _NONE3
    if ":" in t:
        t = t.rsplit(":", 1)[-1].strip()
    m = _AM_RE.match(t)
    if not m:
        return _NONE3
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    if wt not in _ONE_LETTER or mut not in _ONE_LETTER:
        return _NONE3
    if wt == mut or pos < 1:
        return _NONE3
    return (pos, wt, mut)


def add_protein_columns(df: pd.DataFrame, source_col: str = "protein_change") -> pd.DataFrame:
    """Populate ``protein_pos`` (nullable Int64), ``wt_aa``, ``mut_aa`` on ``df``
    by parsing ``source_col``. Mutates and returns ``df``. If ``source_col`` is
    absent, the three columns are created empty (so downstream connectors still
    find them and fall back cleanly)."""
    n = len(df)
    if source_col not in df.columns:
        df["protein_pos"] = pd.array([pd.NA] * n, dtype="Int64")
        df["wt_aa"] = pd.array([None] * n, dtype="object")
        df["mut_aa"] = pd.array([None] * n, dtype="object")
        return df
    parsed = [parse_hgvsp(v) for v in df[source_col].to_numpy()]
    df["protein_pos"] = pd.array([p[0] if p[0] is not None else pd.NA for p in parsed], dtype="Int64")
    df["wt_aa"] = pd.array([p[1] for p in parsed], dtype="object")
    df["mut_aa"] = pd.array([p[2] for p in parsed], dtype="object")
    return df
