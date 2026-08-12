"""tests/unit/test_phylop_block.py

Tests for `src/genomic_variant_classifier/data/phylop.py` (Connector 5 - Phase 2).

All tests inject a synthetic in-memory index so no BigWig file input/output
occurs. The `_make_connector` helper sets `conn._index` directly. The
`_canonical_df` helper produces a one-row canonical-schema DataFrame.

History
-------
Relocated from repository root to tests/unit/ on 2026-05-26 (commit follow-up
to 3a166f6) after a typeguard/pytest collection failure.

REWRITTEN 2026-08-12 for PHYLOP-SOURCE-OWNERSHIP-1. The previous version of
this file defended a contract that is now scientifically inadmissible:

    test_annotate_replaces_existing_phylop_score
        "If the DataFrame already has 'phylop_score' it is overwritten."

That was not an oversight. It was a deliberate guarantee that this connector
would replace whatever canonical PhyloP value it found -- and in stub mode,
with no source at all, replace every one of them with 0.0. phyloP is a SIGNED
score: positive means conservation, negative means faster-than-neutral
evolution, and zero means NEUTRAL. So the guarantee was that an absent source
would assert neutral evolution across the entire cohort, destroying the 17,706
distinct values dbNSFP had already supplied.

    test_annotate_no_nans
        "Output 'phylop_score' column must have no NaNs."

is the same defect one layer down: forbidding NaN is what made 0.0 mandatory.
It is the identical shape as the five superseded no-NaN assertions repaired in
commit 48985d6.

The tests that were ORTHOGONAL to ownership -- chromosome normalisation, copy
semantics, column preservation -- are retained unchanged or strengthened. The
tests below deliberately assert the OWNERSHIP CONTRACT and not the lookup
substrate, so PHYLOPPERF-1 can replace the dictionary backend and the
assembly-registry platform commit can replace chromosome resolution without
reopening any scientific contract established here.

Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPhyloPConnector:
    """Synthetic in-memory index throughout; no file input/output."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_connector(rows: list[tuple]):
        """A PhyloPConnector pre-loaded with a synthetic index.

        Constructs with phylop_file=None and injects `_index`. This is why
        `available` is index-OR-path rather than path-only: were it path-only,
        every test in this file would silently exercise the stub no-op and pass
        while measuring nothing.
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
    # THE OWNERSHIP CONTRACT -- the reason this commit exists
    # ------------------------------------------------------------------
    def test_connector_never_modifies_canonical_phylop_score(self):
        """THE INVARIANT. A source connector may not redefine canonical evidence.

        The superseded test guaranteed the opposite. Here dbNSFP's values are
        already present and the connector has its own observation for the same
        locus -- and canonical evidence is still untouched, because resolving
        two sources is not a connector's authority.
        """
        from genomic_variant_classifier.data.phylop import CANONICAL_COLUMN
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = self._canonical_df()
        existing = pd.Series([-1.234], name=CANONICAL_COLUMN)
        df[CANONICAL_COLUMN] = existing

        out = conn.annotate_dataframe(df)

        pd.testing.assert_series_equal(
            out[CANONICAL_COLUMN], existing, check_names=False)
        assert out.loc[0, "phylop_bigwig"] == pytest.approx(4.5)

    def test_connector_publishes_only_its_owned_column(self):
        """Exactly one new column, and it is the declared one."""
        from genomic_variant_classifier.data.phylop import OUTPUT_COLUMN
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = self._canonical_df()
        out = conn.annotate_dataframe(df)
        added = set(out.columns) - set(df.columns)
        assert added == {OUTPUT_COLUMN}, "connector published {}".format(added)

    def test_owned_and_canonical_columns_are_distinct(self):
        """A structural guard: if these ever coincide the collision returns."""
        from genomic_variant_classifier.data.phylop import (
            CANONICAL_COLUMN, OUTPUT_COLUMN,
        )
        assert OUTPUT_COLUMN != CANONICAL_COLUMN
        assert OUTPUT_COLUMN == "phylop_bigwig"
        assert CANONICAL_COLUMN == "phylop_score"

    # ------------------------------------------------------------------
    # Stub mode -- a strict no-op
    # ------------------------------------------------------------------
    def test_stub_is_a_strict_noop(self):
        """No source configured: the frame is returned UNCHANGED.

        Not zero-filled, and not given an all-NaN column either. "The source
        did not participate" and "the source participated and observed nothing"
        are different facts, and an all-NaN column collapses them.
        """
        from genomic_variant_classifier.data.phylop import (
            OUTPUT_COLUMN, PhyloPConnector,
        )
        conn = PhyloPConnector(phylop_file=None)
        df = pd.DataFrame({
            "chrom": ["1", "17"],
            "pos": [100, 43071077],
            "phylop_score": [3.2, -0.8],
        })
        out = conn.annotate_dataframe(df)

        pd.testing.assert_frame_equal(out, df)
        assert OUTPUT_COLUMN not in out.columns

    def test_stub_cannot_destroy_existing_canonical_evidence(self):
        """The measured defect, as a test.

        Before 2026-08-12 this exact call replaced every phylop_score with 0.0.
        """
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        conn = PhyloPConnector(phylop_file=None)
        df = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "pos": [100, 101, 102],
            "phylop_score": [4.2, -1.3, 0.0],
        })
        out = conn.annotate_dataframe(df)
        assert out["phylop_score"].tolist() == [4.2, -1.3, 0.0]

    def test_stub_get_score_returns_none_not_zero(self):
        """None, not a numeric sentinel that can enter arithmetic as data."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        conn = PhyloPConnector(phylop_file=None)
        assert conn.get_score("1", 100) is None

    def test_availability_counts_an_injected_index(self):
        """Index-or-path. Path-only availability would make every injected-index
        test in this file a silent no-op."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        assert PhyloPConnector(phylop_file=None).available is False
        assert self._make_connector([("1", 100, 1.0)]).available is True

    # ------------------------------------------------------------------
    # unknown != zero, in both directions
    # ------------------------------------------------------------------
    def test_missing_observation_is_nan_not_zero(self):
        """Source queried, no observation at this locus."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = pd.DataFrame({"chrom": ["1"], "pos": [999999]})
        out = conn.annotate_dataframe(df)
        assert pd.isna(out.loc[0, "phylop_bigwig"])

    def test_a_genuine_zero_is_observed_not_missing(self):
        """The inverse, and equally load-bearing.

        0.0 is a real phyloP observation meaning neutral evolution. The
        migration away from 0.0-as-sentinel must not turn it into missingness
        -- the same trap already met with a pLI of exactly zero, where 259
        genes were being excluded from coverage counts.
        """
        conn = self._make_connector([("1", 100, 0.0)])
        df = pd.DataFrame({"chrom": ["1"], "pos": [100]})
        out = conn.annotate_dataframe(df)
        assert out.loc[0, "phylop_bigwig"] == 0.0
        assert not pd.isna(out.loc[0, "phylop_bigwig"])

    def test_get_score_returns_none_for_an_unobserved_locus(self):
        conn = self._make_connector([])
        assert conn.get_score("1", 100) is None

    def test_get_score_has_no_caller_supplied_sentinel(self):
        """The caller may not decide what an unobserved score means.

        `get_score(..., missing_value=0.0)` let a caller assert a specific
        biological value for an absent measurement. That is the semantic hole
        CONSTRAINTFILL-1 closed for gnomAD constraint, where a missing
        loss-of-function observed/expected ratio was recorded as 1.0 --
        "completely tolerant".
        """
        import inspect
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        params = list(inspect.signature(PhyloPConnector.get_score).parameters)
        assert "missing_value" not in params, (
            "get_score still accepts a caller-supplied sentinel: {}".format(params))
        annotate_params = list(
            inspect.signature(PhyloPConnector.annotate_dataframe).parameters)
        assert "missing_value" not in annotate_params

    # ------------------------------------------------------------------
    # Basic lookup -- retained
    # ------------------------------------------------------------------
    def test_known_position_returns_real_score(self):
        conn = self._make_connector([("17", 43071077, 4.532)])
        assert conn.get_score("17", 43071077) == pytest.approx(4.532)

    def test_negative_score_returned_correctly(self):
        """Accelerated-evolution positions carry negative scores. A repair that
        clipped or absolutised them would destroy the conservation/acceleration
        distinction phyloP exists to express."""
        conn = self._make_connector([("1", 925952, -3.7)])
        assert conn.get_score("1", 925952) == pytest.approx(-3.7)

    # ------------------------------------------------------------------
    # Chromosome normalisation -- retained, orthogonal to ownership
    # ------------------------------------------------------------------
    def test_chr_prefix_stripped_on_get_score(self):
        conn = self._make_connector([("17", 43071077, 4.5)])
        assert conn.get_score("chr17", 43071077) == pytest.approx(4.5)

    def test_lowercase_chrom_accepted(self):
        conn = self._make_connector([("1", 925952, 2.1)])
        for prefix in ("chr1", "Chr1", "CHR1", "1"):
            assert conn.get_score(prefix, 925952) == pytest.approx(2.1)

    def test_chrM_maps_to_MT(self):
        conn = self._make_connector([("MT", 1234, 1.0)])
        assert conn.get_score("chrM", 1234) == pytest.approx(1.0)
        assert conn.get_score("M", 1234) == pytest.approx(1.0)

    def test_sex_chromosome_X(self):
        conn = self._make_connector([("X", 50000, 3.2)])
        assert conn.get_score("chrX", 50000) == pytest.approx(3.2)

    # ------------------------------------------------------------------
    # annotate_dataframe -- retained and strengthened
    # ------------------------------------------------------------------
    def test_annotate_does_not_mutate_the_input(self):
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = self._canonical_df()
        before = df.copy(deep=True)
        _ = conn.annotate_dataframe(df)
        pd.testing.assert_frame_equal(df, before)

    def test_annotate_preserves_existing_columns(self):
        conn = self._make_connector([])
        df = self._canonical_df()
        out = conn.annotate_dataframe(df)
        for col in df.columns:
            assert col in out.columns
        pd.testing.assert_frame_equal(out[df.columns.tolist()], df)

    def test_annotate_correct_score_for_hit(self):
        conn = self._make_connector([("17", 43071077, 4.532)])
        out = conn.annotate_dataframe(self._canonical_df())
        assert out.loc[0, "phylop_bigwig"] == pytest.approx(4.532)

    def test_annotate_mixed_hits_and_misses(self):
        """Per-row resolution: one hit, one miss, in one frame."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = pd.DataFrame({"chrom": ["17", "1"], "pos": [43071077, 999999]})
        out = conn.annotate_dataframe(df)
        assert out.loc[0, "phylop_bigwig"] == pytest.approx(4.5)
        assert pd.isna(out.loc[1, "phylop_bigwig"])

    def test_annotate_preserves_row_order_and_index(self):
        """A vectorised backend that reorders rows would be fast and wrong.
        PHYLOPPERF-1 replaces the substrate; this contract must survive it."""
        conn = self._make_connector([("1", 10, 1.0), ("1", 30, 3.0)])
        df = pd.DataFrame({"chrom": ["1", "1", "1"], "pos": [30, 20, 10]},
                          index=[7, 8, 9])
        out = conn.annotate_dataframe(df)
        assert out.index.tolist() == [7, 8, 9]
        assert out.loc[7, "phylop_bigwig"] == pytest.approx(3.0)
        assert pd.isna(out.loc[8, "phylop_bigwig"])
        assert out.loc[9, "phylop_bigwig"] == pytest.approx(1.0)

    def test_annotate_requires_chrom_and_pos(self):
        from genomic_variant_classifier.data.phylop import PhyloPContractError
        conn = self._make_connector([("1", 100, 1.0)])
        try:
            conn.annotate_dataframe(pd.DataFrame({"chrom": ["1"]}))
        except PhyloPContractError as exc:
            assert "pos" in str(exc)
            return
        raise AssertionError("a frame without 'pos' was accepted")

    # ------------------------------------------------------------------
    # The transitional substrate is DECLARED, not assumed permanent
    # ------------------------------------------------------------------
    def test_canonical_ownership_is_recorded_as_TRANSITIONAL(self):
        """A1 does not make dbNSFP the permanent owner of phylop_score.

        It removes an illegal overwrite and establishes source-specific
        publication as the migration boundary. dbNSFP inherits the canonical
        name temporarily; the endpoint is that NEITHER connector owns it and a
        reconciler derives it from both observations.

        This is a constant rather than a comment because a surgical repair has
        a habit of becoming permanent architecture through inertia, and
        PHYLOP-RECONCILE-1 must change something a test can see.
        """
        from genomic_variant_classifier.data.phylop import (
            PHYLOP_CANONICALIZATION_STATE,
        )
        assert PHYLOP_CANONICALIZATION_STATE == "transitional_dbnsfp_inherited_v1"

    def test_the_transitional_substrate_is_recorded(self):
        """Transitional architecture otherwise becomes permanent architecture.

        PHYLOPPERF-1 changes the lookup marker; the assembly-registry platform
        commit changes the chromosome-resolution marker. Both are deliberately
        visible so nobody mistakes either for the approved endpoint.
        """
        from genomic_variant_classifier.data.phylop import (
            PHYLOP_CHROMOSOME_RESOLUTION, PHYLOP_LOOKUP_SUBSTRATE,
        )
        assert PHYLOP_LOOKUP_SUBSTRATE == "legacy_dict_v1"
        assert PHYLOP_CHROMOSOME_RESOLUTION == "legacy_normalise_chrom_v1"

    def test_the_backend_interface_is_what_the_connector_depends_on(self):
        """The connector must route through lookup_many, so PHYLOPPERF-1 can
        swap the engine without touching the ownership contract."""
        from genomic_variant_classifier.data.phylop import (
            DictPhyloPBackend, PhyloPConnector,
        )
        conn = self._make_connector([("1", 100, 2.5)])
        backend = conn._lookup_backend()
        assert isinstance(backend, DictPhyloPBackend)
        got = backend.lookup_many(pd.DataFrame({"chrom": ["1"], "pos": [100]}))
        assert isinstance(got, pd.Series)
        assert got.iloc[0] == pytest.approx(2.5)

    def test_a_backend_that_loses_row_identity_is_refused(self):
        from genomic_variant_classifier.data.phylop import PhyloPContractError

        class _BadBackend:
            def lookup_many(self, loci):
                return pd.Series([1.0], index=[999], dtype="float64")

        conn = self._make_connector([("1", 100, 1.0)])
        conn._lookup_backend = lambda: _BadBackend()
        try:
            conn.annotate_dataframe(pd.DataFrame({"chrom": ["1"], "pos": [100]}))
        except PhyloPContractError as exc:
            assert "row identity" in str(exc)
            return
        raise AssertionError("a backend that lost row identity was accepted")
