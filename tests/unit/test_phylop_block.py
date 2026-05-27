"""tests/unit/test_phylop_block.py

Tests for `src/genomic_variant_classifier/data/phylop.py` (Connector 5 — Phase 2).

All tests inject a synthetic in-memory index so no BigWig file I/O occurs.
The `_make_connector` helper sets `conn._index` directly. The `_canonical_df`
helper produces a one-row canonical-schema DataFrame for variant lookups.

History
-------
Relocated from repository root to tests/unit/ on 2026-05-26 (commit follow-up
to 3a166f6). At its previous location, this file lived next to the project
root and used lazy method-local imports for `PhyloPConnector` and friends,
but referenced `pd.DataFrame` in a class-body type annotation at L32. With
typeguard 4.5.1 loaded as a pytest plugin and no module-level `import pandas
as pd`, pytest collection raised `NameError: name 'pd' is not defined`,
blocking every full-suite run since the A1 close-out work began.

This relocation moves the file under tests/unit/ where it belongs, adds the
three required module-level imports (numpy, pandas, pytest), and prepends
`from __future__ import annotations` for forward compatibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

class TestPhyloPConnector:
    """
    All tests inject a synthetic in-memory index to avoid the large BigWig
    file.  The _make_connector helper sets conn._index directly so no file
    I/O occurs.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_connector(rows: list[tuple]) -> "PhyloPConnector":
        """
        Return a PhyloPConnector pre-loaded with a synthetic index.

        rows: list of (chrom, pos, score) tuples — no file needed.
        """
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {
            (str(chrom), int(pos)): float(score)
            for chrom, pos, score in rows
        }
        return conn

    @staticmethod
    def _canonical_df(**overrides) -> pd.DataFrame:
        """Minimal canonical-schema DataFrame for one variant."""
        base = dict(
            chrom=["17"],
            pos=[43071077],
            ref=["G"],
            alt=["T"],
            gene_symbol=["BRCA1"],
        )
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)

    # ------------------------------------------------------------------
    # Basic lookup
    # ------------------------------------------------------------------
    def test_known_position_returns_real_score(self):
        """A position present in the index returns its exact score."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        score = conn.get_score("17", 43071077)
        assert abs(score - 4.532) < 1e-6

    def test_missing_position_returns_default(self):
        """A position absent from the index returns DEFAULT_SCORE."""
        from genomic_variant_classifier.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([])
        assert conn.get_score("1", 100) == DEFAULT_SCORE

    def test_custom_missing_value_honoured(self):
        """Caller-supplied missing_value overrides DEFAULT_SCORE."""
        conn = self._make_connector([])
        assert conn.get_score("1", 100, missing_value=-99.0) == -99.0

    def test_negative_score_returned_correctly(self):
        """Accelerated-evolution positions return negative scores."""
        conn = self._make_connector([("1", 925952, -3.7)])
        assert conn.get_score("1", 925952) == pytest.approx(-3.7)

    # ------------------------------------------------------------------
    # Chromosome normalisation
    # ------------------------------------------------------------------
    def test_chr_prefix_stripped_on_get_score(self):
        """'chr17' and '17' resolve to the same index entry."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        assert conn.get_score("chr17", 43071077) == pytest.approx(4.5)

    def test_lowercase_chrom_accepted(self):
        """'chr1', 'Chr1', 'CHR1' all normalise correctly."""
        conn = self._make_connector([("1", 925952, 2.1)])
        for prefix in ("chr1", "Chr1", "CHR1", "1"):
            assert conn.get_score(prefix, 925952) == pytest.approx(2.1)

    def test_chrM_maps_to_MT(self):
        """Mitochondrial chromosome normalises to 'MT'."""
        conn = self._make_connector([("MT", 1234, 1.0)])
        assert conn.get_score("chrM", 1234) == pytest.approx(1.0)
        assert conn.get_score("M", 1234) == pytest.approx(1.0)

    def test_sex_chromosome_X(self):
        conn = self._make_connector([("X", 50000, 3.2)])
        assert conn.get_score("chrX", 50000) == pytest.approx(3.2)

    # ------------------------------------------------------------------
    # annotate_dataframe
    # ------------------------------------------------------------------
    def test_annotate_adds_phylop_score_column(self):
        """annotate_dataframe must add a 'phylop_score' column."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert "phylop_score" in result.columns

    def test_annotate_returns_copy(self):
        """annotate_dataframe must not mutate the input DataFrame."""
        conn = self._make_connector([])
        df = self._canonical_df()
        original_cols = list(df.columns)
        _ = conn.annotate_dataframe(df)
        assert list(df.columns) == original_cols

    def test_annotate_correct_score_for_hit(self):
        """Matched position gets the real score, not the default."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.532)

    def test_annotate_default_for_miss(self):
        """Unmatched position gets DEFAULT_SCORE."""
        from genomic_variant_classifier.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == DEFAULT_SCORE

    def test_annotate_mixed_hits_and_misses(self):
        """Per-row resolution — one hit, one miss in the same DataFrame."""
        from genomic_variant_classifier.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = pd.DataFrame({
            "chrom": ["17", "1"],
            "pos":   [43071077, 999999],
        })
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.5)
        assert result.loc[1, "phylop_score"] == DEFAULT_SCORE

    def test_annotate_preserves_existing_columns(self):
        """All original columns must survive annotation."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        for col in df.columns:
            assert col in result.columns

    def test_annotate_no_nans(self):
        """Output 'phylop_score' column must have no NaNs."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert not result["phylop_score"].isnull().any()

    def test_annotate_replaces_existing_phylop_score(self):
        """If the DataFrame already has 'phylop_score' it is overwritten."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = self._canonical_df()
        df["phylop_score"] = 0.0
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.5)

    # ------------------------------------------------------------------
    # Stub mode (no file supplied)
    # ------------------------------------------------------------------
    def test_stub_mode_returns_default_without_crash(self):
        """PhyloPConnector(None) must not raise — returns DEFAULT_SCORE."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector, DEFAULT_SCORE
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {}
        assert conn.get_score("1", 100) == DEFAULT_SCORE

    def test_stub_mode_annotate_dataframe(self):
        """annotate_dataframe in stub mode fills every row with DEFAULT_SCORE."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector, DEFAULT_SCORE
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {}
        df = pd.DataFrame({"chrom": ["1", "17"], "pos": [100, 43071077]})
        result = conn.annotate_dataframe(df)
        assert (result["phylop_score"] == DEFAULT_SCORE).all()
