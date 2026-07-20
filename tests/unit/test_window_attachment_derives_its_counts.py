"""WindowAttachment derives its counts from two stored masks.

Until 2026-07-20 the class stored `usable` plus three integer counts, and each resolution tier
computed the breakdown by hand. Tier 1 got it wrong -- every unusable row was attributed to
`n_unmapped` while `n_placeholder` was hardcoded to zero -- and NO TEST COVERED THAT PATH,
because no test supplied an `ok` column alongside pre-attached windows. That is precisely the
shape the production cohort uses.

This file covers it, and pins the derivation so the counts cannot drift from the masks again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genomic_variant_classifier.data import seq_window_join as J


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        J.REF_WIN_COL: ["A" * 101] * n,
        J.ALT_WIN_COL: ["C" * 101] * n,
    })


def _att(key_found, builder_ok, provenance="parquet+ok") -> J.WindowAttachment:
    kf = np.asarray(key_found, dtype=bool)
    return J.WindowAttachment(_frame(len(kf)), kf, np.asarray(builder_ok, dtype=bool),
                              provenance)


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------

def test_every_count_is_derived_from_the_two_masks():
    """Hand-computed against a table small enough to check by eye.

        row  key_found  builder_ok   usable  unmapped  placeholder
        0      True       True        yes       .           .
        1      True       False        .        .          yes
        2      False      True         .       yes          .
        3      False      False        .       yes          .
    """
    att = _att([True, True, False, False], [True, False, True, False])
    assert att.n_rows == 4
    assert list(att.usable) == [True, False, False, False]
    assert att.n_usable == 1
    assert att.n_unmapped == 2, "rows 2 and 3 had no window located"
    assert att.n_placeholder == 1, "row 1 was located but the builder rejected it"
    assert att.usable_fraction == 0.25


def test_n_rows_cannot_disagree_with_the_frame():
    """Under the old layout n_rows was a stored int and nothing tied it to the frame."""
    for n in (0, 1, 7):
        att = _att([True] * n, [True] * n)
        assert att.n_rows == len(att.windows) == n


def test_usable_fraction_of_an_empty_attachment_is_zero_not_an_error():
    assert _att([], []).usable_fraction == 0.0


# ---------------------------------------------------------------------------
# The defect: tier 1 with an `ok` column
# ---------------------------------------------------------------------------

def test_tier1_with_ok_attributes_rejected_rows_to_placeholder_not_unmapped():
    """THE REGRESSION THIS FILE EXISTS FOR.

    A frame carrying windows AND an `ok` column resolves through tier 1. Rows the builder
    rejected are PRESENT -- they were located -- so they are builder-placeholders. Before
    2026-07-20 they were reported as `n_unmapped`, with `n_placeholder` hardcoded to zero.
    """
    meta = _frame(5)
    meta[J.OK_COL] = [True, True, False, False, False]

    att = J.attach_delta_windows(meta)

    assert att.provenance == "rows+ok"
    assert att.n_unmapped == 0, (
        "no row is unmapped -- every window is present on the frame. Reporting these as "
        "unmapped is the inverted attribution this test pins."
    )
    assert att.n_placeholder == 3, "three rows were located and rejected by the builder"
    assert att.n_usable == 2


def test_the_old_attribution_would_have_failed_this_test():
    """NEGATIVE CONTROL, computed rather than asserted.

    Reproduces the old formula -- n_unmapped = (~usable).sum(), n_placeholder = 0 -- and
    shows it disagrees with the new derivation on exactly this input. A regression test that
    would also have passed against the bug pins nothing.
    """
    meta = _frame(5)
    meta[J.OK_COL] = [True, True, False, False, False]
    att = J.attach_delta_windows(meta)

    old_n_unmapped = int((~att.usable).sum())
    old_n_placeholder = 0

    assert old_n_unmapped == 3 and att.n_unmapped == 0
    assert old_n_placeholder == 0 and att.n_placeholder == 3


def test_tier1_without_ok_still_reports_missing_windows_as_unmapped():
    """A null window means no window was located, at any tier."""
    meta = _frame(3)
    meta.loc[2, J.REF_WIN_COL] = None

    att = J.attach_delta_windows(meta)

    assert att.provenance == "rows"
    assert att.n_unmapped == 1
    assert att.n_placeholder == 0, "no verdict was available, so nothing was rejected"
    assert not att.usable[2]


# ---------------------------------------------------------------------------
# provenance_is_verified
# ---------------------------------------------------------------------------

def test_only_the_ok_tiers_report_a_verified_provenance():
    assert _att([True], [True], "rows+ok").provenance_is_verified
    assert _att([True], [True], "parquet+ok").provenance_is_verified
    assert not _att([True], [True], "rows").provenance_is_verified
    assert not _att([True], [True], "parquet").provenance_is_verified
    assert not _att([True], [True], "none").provenance_is_verified


def test_an_unverified_attachment_still_reports_usable_rows():
    """The trap this flag exists to expose.

    A "parquet" attachment fabricates builder_ok as all-True, so n_usable counts rows nobody
    checked. The number looks identical to a verified one; only provenance distinguishes them.
    """
    att = _att([True, True, True], [True, True, True], "parquet")
    assert att.n_usable == 3
    assert not att.provenance_is_verified


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------

def test_subset_recomputes_every_count_exactly():
    att = _att([True, True, False, False, True], [True, False, True, False, True])
    sub = att.subset([0, 1, 2])

    assert sub.n_rows == 3
    assert list(sub.usable) == [True, False, False]
    assert sub.n_usable == 1
    assert sub.n_unmapped == 1
    assert sub.n_placeholder == 1


def test_subset_carries_provenance_unchanged():
    """Selecting rows neither improves nor degrades the builder's verdict about them."""
    for prov in ("rows+ok", "parquet+ok", "rows", "parquet", "none"):
        sub = _att([True, True], [True, False], prov).subset([0])
        assert sub.provenance == prov
        assert sub.provenance_is_verified == (prov.endswith("+ok"))


def test_subset_accepts_a_boolean_mask():
    att = _att([True, False, True], [True, True, False])
    sub = att.subset(np.array([True, False, True]))
    assert sub.n_rows == 2
    assert list(sub.usable) == [True, False]


def test_subset_resets_the_frame_index():
    """predict_proba aligns positionally; a subset carrying its parent's index breaks that."""
    sub = _att([True] * 5, [True] * 5).subset([3, 4])
    assert list(sub.windows.index) == [0, 1]


def test_subset_of_a_subset_is_still_exact():
    att = _att([True, True, True, False], [True, False, True, True])
    sub = att.subset([0, 1, 2]).subset([1, 2])
    assert sub.n_rows == 2
    assert sub.n_placeholder == 1
    assert sub.n_usable == 1


def test_summary_names_the_tier_and_both_failure_modes():
    att = _att([True, True, False], [True, False, True], "parquet+ok")
    s = att.summary()
    assert "parquet+ok" in s
    assert "1/3 usable" in s
    assert "1 unmapped" in s
    assert "1 builder-placeholder" in s
