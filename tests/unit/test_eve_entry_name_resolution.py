"""Tests for the EVE entry-name -> HGNC resolution fix (Phase 0, Run 17).

EVE per-protein files are named by UniProt ENTRY NAME (e.g. "1433G_HUMAN.csv"),
not by HGNC symbol. Before this fix the connector keyed its lookup on the
entry-name prefix ("1433G"), so a join against an HGNC-keyed cohort ("YWHAG")
silently produced eve_score=0.5 everywhere (empirically 0/2 matched; 2/2 after).

This module guards:
  * the pure resolver helpers (_eve_stem_to_entry_name, load_eve_entry_map,
    resolve_eve_gene);
  * the end-to-end directory parse + annotate (HGNC cohort now covered 2/2);
  * the fail-loud behaviour (empty map / low resolution -> WARNING, never silent);
  * non-regression of the mutations_protein_name branch;
  * a data-gated corpus check that >=80% of the real 3,211 EVE filenames resolve
    via the rebuilt UniProt index;
  * a data-gated check that the rebuilt index actually carries entry_name.

Run:  pytest tests/unit/test_eve_entry_name_resolution.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.eve import (
    DEFAULT_SCORE,
    EVEConnector,
    _eve_stem_to_entry_name,
    load_eve_entry_map,
    resolve_eve_gene,
    _EVE_MIN_RESOLVED_FRACTION,
)

# Canonical real-data locations (tests that need them are skipped if absent).
_EVE_VARIANT_DIR = Path("data/external/eve/EVE_all_data/variant_files")
_UNIPROT_INDEX = Path("data/external/uniprot/uniprot_human_reviewed.parquet")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_eve_variant_csv(path: Path, wt: str, pos: int, mt: str, score: float) -> None:
    """Write a minimal EVE variant_files-style CSV (NO mutations_protein_name,
    matching the real 43-column files which carry none -- so the filename branch
    is exercised)."""
    pd.DataFrame(
        {"wt_aa": [wt], "position": [pos], "mt_aa": [mt], "EVE_scores_ASM": [score]}
    ).to_csv(path, index=False)


def _write_index(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows = [(gene_symbol, entry_name, uniprot_id), ...] -> index parquet."""
    pd.DataFrame(
        {
            "gene_symbol": [r[0] for r in rows],
            "uniprot_id": [r[2] for r in rows],
            "entry_name": [r[1] for r in rows],
            "sequence": ["M" for _ in rows],
        }
    ).to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# 1. Pure helper: _eve_stem_to_entry_name
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stem,expected",
    [
        ("1433G_HUMAN", "1433G_HUMAN"),
        ("TP53_HUMAN_singles_scores", "TP53_HUMAN"),  # EVE suffix tolerated
        ("p53_human", "P53_HUMAN"),                    # case-insensitive
        ("  BRCA1_HUMAN  ", "BRCA1_HUMAN"),            # whitespace stripped
        ("WEIRDNAME", "WEIRDNAME"),                    # no _HUMAN -> as-is (upper)
        ("", ""),
        (None, ""),
    ],
)
def test_eve_stem_to_entry_name(stem, expected):
    assert _eve_stem_to_entry_name(stem) == expected


# --------------------------------------------------------------------------- #
# 2. Pure helper: resolve_eve_gene
# --------------------------------------------------------------------------- #
def test_resolve_eve_gene_via_map():
    m = {"1433G_HUMAN": "YWHAG", "1433Z_HUMAN": "YWHAZ", "TP53_HUMAN": "TP53"}
    assert resolve_eve_gene("1433G_HUMAN", m) == ("YWHAG", True)
    assert resolve_eve_gene("1433Z_HUMAN", m) == ("YWHAZ", True)
    # EVE-style suffix still resolves the embedded entry name
    assert resolve_eve_gene("TP53_HUMAN_singles_scores", m) == ("TP53", True)


def test_resolve_eve_gene_fallback():
    m = {"1433G_HUMAN": "YWHAG"}
    # Not in map -> legacy prefix fallback, flagged unresolved
    assert resolve_eve_gene("BRCA1_HUMAN", m) == ("BRCA1", False)
    assert resolve_eve_gene("UNKNOWNZZ_HUMAN", m) == ("UNKNOWNZZ", False)
    # Already-HGNC stem with no _HUMAN -> prefix == itself
    assert resolve_eve_gene("BRCA1", m) == ("BRCA1", False)


# --------------------------------------------------------------------------- #
# 3. Pure helper: load_eve_entry_map
# --------------------------------------------------------------------------- #
def test_load_eve_entry_map_roundtrip(tmp_path):
    idx = tmp_path / "uni.parquet"
    _write_index(idx, [("YWHAG", "1433G_HUMAN", "P61981"),
                       ("YWHAZ", "1433Z_HUMAN", "P63104")])
    m = load_eve_entry_map(idx)
    assert m["1433G_HUMAN"] == "YWHAG"
    assert m["1433Z_HUMAN"] == "YWHAZ"


def test_load_eve_entry_map_degraded(tmp_path):
    # Missing entry_name column -> {} (caller logs loudly)
    no_en = tmp_path / "no_entry.parquet"
    pd.DataFrame({"gene_symbol": ["X"], "uniprot_id": ["Y"], "sequence": ["M"]}).to_parquet(no_en)
    assert load_eve_entry_map(no_en) == {}
    assert load_eve_entry_map(None) == {}
    assert load_eve_entry_map(tmp_path / "does_not_exist.parquet") == {}


# --------------------------------------------------------------------------- #
# 4. Integration: HGNC cohort is covered 2/2 via the entry-name map
# --------------------------------------------------------------------------- #
def test_end_to_end_hgnc_join_covered(tmp_path):
    eve_dir = tmp_path / "variant_files"
    eve_dir.mkdir()
    _write_eve_variant_csv(eve_dir / "1433G_HUMAN.csv", "R", 4, "A", 0.7727)
    _write_eve_variant_csv(eve_dir / "1433Z_HUMAN.csv", "K", 3, "A", 0.7122)
    idx = tmp_path / "uni.parquet"
    _write_index(idx, [("YWHAG", "1433G_HUMAN", "P61981"),
                       ("YWHAZ", "1433Z_HUMAN", "P63104")])

    conn = EVEConnector(eve_path=eve_dir, entry_map_path=idx)
    cohort = pd.DataFrame(
        {"gene_symbol": ["YWHAG", "YWHAZ"], "protein_change": ["p.R4A", "p.K3A"]}
    )
    out = conn.annotate_dataframe(cohort)
    covered = (out["eve_score"] != DEFAULT_SCORE).sum()
    assert covered == 2, f"HGNC cohort should be covered 2/2, got {covered}"
    assert abs(out["eve_score"].iloc[0] - 0.7727) < 1e-6
    assert abs(out["eve_score"].iloc[1] - 0.7122) < 1e-6


def test_without_map_is_zero_and_loud(tmp_path, caplog):
    eve_dir = tmp_path / "variant_files"
    eve_dir.mkdir()
    _write_eve_variant_csv(eve_dir / "1433G_HUMAN.csv", "R", 4, "A", 0.7727)
    conn = EVEConnector(eve_path=eve_dir, entry_map_path=None)  # no map
    cohort = pd.DataFrame({"gene_symbol": ["YWHAG"], "protein_change": ["p.R4A"]})
    with caplog.at_level(logging.WARNING):
        out = conn.annotate_dataframe(cohort)
    # 0 coverage on HGNC cohort...
    assert (out["eve_score"] != DEFAULT_SCORE).sum() == 0
    # ...but LOUD, not silent
    assert any("entry-name map empty" in r.message for r in caplog.records), \
        "missing map must emit a WARNING (not a silent zero)"


def test_low_resolution_warns(tmp_path, caplog):
    eve_dir = tmp_path / "variant_files"
    eve_dir.mkdir()
    for stem in ["1433G_HUMAN", "AAA_HUMAN", "BBB_HUMAN", "CCC_HUMAN", "DDD_HUMAN"]:
        _write_eve_variant_csv(eve_dir / f"{stem}.csv", "R", 4, "A", 0.9)
    idx = tmp_path / "uni.parquet"
    _write_index(idx, [("YWHAG", "1433G_HUMAN", "P61981")])  # resolves only 1/5
    conn = EVEConnector(eve_path=eve_dir, entry_map_path=idx)
    with caplog.at_level(logging.WARNING):
        conn.annotate_dataframe(pd.DataFrame({"gene_symbol": ["YWHAG"],
                                              "protein_change": ["p.R4A"]}))
    assert any("resolved" in r.message and "%" in r.message for r in caplog.records), \
        "low resolution fraction must emit a WARNING"


def test_mutations_protein_name_branch_unchanged(tmp_path):
    """The protein-name branch (used by existing test_eve.py fixtures) is untouched."""
    eve_dir = tmp_path / "variant_files"
    eve_dir.mkdir()
    pd.DataFrame(
        {
            "mutations_protein_name": ["TP53_HUMAN", "TP53_HUMAN"],
            "wt_aa": ["R", "R"], "position": [175, 248], "mt_aa": ["H", "Q"],
            "EVE_scores_ASM": [0.95, 0.88],
        }
    ).to_csv(eve_dir / "TP53_HUMAN_singles_scores.csv", index=False)
    idx = tmp_path / "uni.parquet"
    _write_index(idx, [("YWHAG", "1433G_HUMAN", "P61981")])
    conn = EVEConnector(eve_path=eve_dir, entry_map_path=idx)
    out = conn.annotate_dataframe(pd.DataFrame({"gene_symbol": ["TP53"],
                                                "protein_change": ["p.R175H"]}))
    assert abs(out["eve_score"].iloc[0] - 0.95) < 1e-6


# --------------------------------------------------------------------------- #
# 5. Data-gated: rebuilt UniProt index carries entry_name + real corpus coverage
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _UNIPROT_INDEX.exists(),
                    reason="UniProt index not present (run scripts/build_uniprot_index.py)")
def test_uniprot_index_has_entry_name_column():
    df = pd.read_parquet(_UNIPROT_INDEX)
    assert "entry_name" in df.columns, (
        "UniProt index lacks entry_name -- rebuild with the patched "
        "scripts/build_uniprot_index.py before staging."
    )
    # known anchors
    by_entry = dict(zip(df["entry_name"].astype(str).str.upper(),
                        df["gene_symbol"].astype(str).str.upper()))
    assert by_entry.get("1433G_HUMAN") == "YWHAG"
    assert by_entry.get("1433Z_HUMAN") == "YWHAZ"


@pytest.mark.skipif(
    not (_UNIPROT_INDEX.exists() and _EVE_VARIANT_DIR.exists()),
    reason="EVE variant_files and/or UniProt index not present",
)
def test_real_corpus_resolution_fraction():
    """At least _EVE_MIN_RESOLVED_FRACTION of the real 3,211 EVE filenames must
    resolve to an HGNC symbol via the rebuilt index. Expected ~0.99."""
    entry_map = load_eve_entry_map(_UNIPROT_INDEX)
    assert entry_map, "entry map empty -- index missing entry_name column?"
    stems = [p.stem for p in _EVE_VARIANT_DIR.glob("*.csv")]
    assert stems, "no EVE CSVs found"
    resolved = sum(1 for s in stems if resolve_eve_gene(s, entry_map)[1])
    frac = resolved / len(stems)
    assert frac >= _EVE_MIN_RESOLVED_FRACTION, (
        f"only {resolved}/{len(stems)} ({frac:.1%}) EVE filenames resolved entry-name "
        f"-> HGNC; expected >= {_EVE_MIN_RESOLVED_FRACTION:.0%}. Index may be stale."
    )
