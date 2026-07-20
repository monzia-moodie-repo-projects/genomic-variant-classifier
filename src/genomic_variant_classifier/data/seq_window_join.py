"""Attach ref/alt delta sequence windows to a split's rows, alignment-safe.

The sequence CNN needs a 2-column [fasta_seq_ref, fasta_seq_alt] frame whose
row i corresponds to feature-matrix row i. Splits are produced by
DataPrepPipeline as positionally-aligned (X_*, meta_*) pairs, so we attach
windows to `meta` and rely on key identity -- never on row order between the
cohort parquet and the split.

Resolution order:
  1. `meta` already carries fasta_seq_ref + fasta_seq_alt  -> use directly.
  2. `seq_windows_path` given -> key-join on chrom:pos:ref:alt (order-preserving).
  3. otherwise -> placeholder for every row (NO SIGNAL), usable=False throughout.


WHAT CHANGED ON 2026-07-15, AND WHY (roadmap 6.28)
==================================================
This module used to return `(windows_df, n_unmapped)` and fabricate `"A" * window`
for any row it could not resolve. It now returns a `WindowAttachment` carrying an
explicit `usable` mask. The reason is a defect measured on the live artifact:

    seq_windows.manifest.json (2026-07-10 build, the artifact Run 17 will read):
        n_rows_built : 4420180
        n_ok         : 4398366
        n_poly       :   21814      <-- 0.494% of the cohort

THERE ARE TWO INDEPENDENT FABRICATION PATHS, AND THEY USE DIFFERENT FILL CHARACTERS:

    builder failure  -> `delta_window_builder.POLY = "N"` -> "N"*101 is WRITTEN INTO
                        the parquet, flagged `ok=False`, and counted as n_poly.
    join failure     -> this module's own fallback        -> "A"*101, never persisted.

Every consumer in the repository detected only the SECOND one. Three separate
poly-detectors existed, and all three compared against "A"*window:

    scripts/train.py:485          `X_seq_test[REF_WIN_COL] != _POLY_WIN`
    data/genomic_lm.py:201/250    `self._poly = "A" * self.window`; `_mapped_mask`
    this module (the fallback itself)

So all three were blind to the same 21,814 rows, which flowed into training as though
they were real sequence:

  * cnn_1d: `encode_sequence` gives "N" no branch, so those rows became an ALL-ZERO
    (101, 4) tensor; ref == alt, so the delta channels were identically zero too. It
    trained on them and said nothing.
  * Nucleotide Transformer: ref and alt are BOTH poly-N, so `||alt_emb - ref_emb||`
    is exactly 0.0 -- which genomic_lm's own docstring defines as "window unavailable
    / model unavailable (stub)". Three distinct conditions collapsed onto one value.
    That is the EXIT_NOT_CHECKED bug (scripts/run_drift_monitor.py): "I could not
    look" rendered identically to "I looked and found zero".

The ok-fraction is 4398366/4420180 = 99.507%, comfortably above
seq_window_manifest.MIN_OK_FRACTION (0.95), so no existing gate fired. The gate was
never wrong; nothing downstream ever asked it.

WHY CONTENT-MATCHING IS THE WRONG FIX -- this is the load-bearing point.
Extending the checks to also match "N"*window would be patchwork, and it would be
patchwork on top of an error of principle: **a window that reads "A"*101 may be REAL
DATA.** Poly-A tracts exist in the genome. Content cannot distinguish "the reference
genuinely says AAAA..." from "we gave up and typed A". Only PROVENANCE can.

`scripts/build_seq_windows.py:154` has been writing an `ok` column into the parquet
the whole time -- the ground truth was already in the artifact, and the join simply
never read it. This module now reads it. That is why the fix is a mask and not a
longer list of forbidden strings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REF_WIN_COL = "fasta_seq_ref"
ALT_WIN_COL = "fasta_seq_alt"
OK_COL = "ok"
_KEY_COLS = ("chrom", "pos", "ref", "alt")

#: Fill for rows with no real sequence. Its VALUE is deliberately uninteresting:
#: nothing may branch on it. `WindowAttachment.usable` is the only sanctioned way to
#: ask whether a row carries real sequence. Kept as "N" so that, should an unmasked
#: row ever reach `encode_sequence`, it yields an all-zero (visibly dead) tensor
#: rather than a plausible poly-A tract.
PLACEHOLDER_BASE = "N"


@dataclass(frozen=True)
class WindowAttachment:
    """Windows plus the provenance needed to know which of them are real.

    THE TWO MASKS ARE STORED. EVERYTHING ELSE IS DERIVED.

    Until 2026-07-20 this class stored `usable` as one combined mask alongside `n_rows`,
    `n_unmapped` and `n_placeholder` as separate integers, and asked each resolution tier to
    compute the breakdown by hand. Tier 1 computed it wrongly: it attributed every unusable
    row to `n_unmapped` and hardcoded `n_placeholder` to zero, so a cohort whose windows were
    already attached WITH an `ok` column reported its builder-placeholder rows as "unmapped"
    in the run log -- exactly inverted. No test covered that path, because no test supplied an
    `ok` column alongside pre-attached windows.

    Storing the components and deriving the aggregates makes that error unrepresentable: a
    tier must state BOTH masks, and every count follows from them. It also removes a latent
    inconsistency -- under the old layout nothing prevented `n_rows != len(windows)`.

    Attributes
    ----------
    windows:
        2-column [fasta_seq_ref, fasta_seq_alt] frame, 1:1 with `meta` (reset index).
    key_found:
        Boolean array, 1:1 with `meta`. True iff a window was LOCATED for this row -- found
        in the window source (tier 2), or already present and non-null on the frame (tier 1).
        Says nothing about whether that window is real; `builder_ok` says that.

        Named `key_found` rather than `mapped` deliberately: genomic_lm.py:386 and :432 bind
        `att.usable` to a local called `mapped`, and one word with two meanings in one
        codebase is how a mask gets read wrong.
    builder_ok:
        Boolean array, 1:1 with `meta`. The window BUILDER's own verdict -- True iff it could
        construct a real window from the reference. False means it could not: missing contig,
        out-of-range position, non-ACGT allele.

        WHEN NO `ok` COLUMN IS AVAILABLE THIS IS FABRICATED AS ALL-TRUE, and `provenance`
        records that. Ask `provenance_is_verified` before trusting it.
    provenance:
        Which resolution tier produced this attachment, and whether the builder's verdict
        travelled with it. Recorded so a run artifact can state HOW its windows were obtained,
        not merely that it had some.
    """

    windows: pd.DataFrame
    key_found: np.ndarray
    builder_ok: np.ndarray
    provenance: str

    @property
    def usable(self) -> np.ndarray:
        """True iff the row carries sequence that came from the reference genome.

        **Consumers must mask on this and nothing else.** Never compare window CONTENT to a
        placeholder string: a real window may legitimately equal any given string (poly-A
        tracts are real biology), and a filler character can change without warning.
        """
        return self.key_found & self.builder_ok

    @property
    def n_rows(self) -> int:
        return len(self.windows)

    @property
    def n_usable(self) -> int:
        return int(self.usable.sum())

    @property
    def n_unmapped(self) -> int:
        """Rows for which no window was located at all."""
        return int((~self.key_found).sum())

    @property
    def n_placeholder(self) -> int:
        """Rows LOCATED but flagged unusable by the builder.

        This is the count tier 1 used to report as zero while attributing these rows to
        `n_unmapped` instead.
        """
        return int((self.key_found & ~self.builder_ok).sum())

    @property
    def usable_fraction(self) -> float:
        return (self.n_usable / self.n_rows) if self.n_rows else 0.0

    @property
    def provenance_is_verified(self) -> bool:
        """True iff `builder_ok` came from the builder rather than being assumed.

        Tiers "rows" and "parquet" have no `ok` column to read, so they fabricate
        `builder_ok` as all-True. `n_usable` from such an attachment counts rows nobody
        checked. Only the "+ok" tiers carry a real verdict.
        """
        return self.provenance.endswith("+ok")

    def subset(self, idx) -> "WindowAttachment":
        """A row subset of this attachment -- still an attachment.

        `idx` is anything both DataFrame.iloc and numpy fancy-indexing accept: an integer
        array, a boolean mask of length n_rows, or a slice.

        `provenance` is carried through UNCHANGED. Selecting rows neither improves nor
        degrades the builder's verdict about the rows selected.

        This method could not have been written honestly before the counts became derived:
        given only a combined `usable` mask, the unmapped/placeholder breakdown of a slice is
        unrecoverable, so any subset would have had to report a stale or invented figure.
        """
        return WindowAttachment(
            windows=self.windows.iloc[idx].reset_index(drop=True),
            key_found=np.asarray(self.key_found)[idx],
            builder_ok=np.asarray(self.builder_ok)[idx],
            provenance=self.provenance,
        )

    def summary(self) -> str:
        return (
            f"windows[{self.provenance}]: {self.n_usable}/{self.n_rows} usable "
            f"({100.0 * self.usable_fraction:.3f}%), {self.n_unmapped} unmapped, "
            f"{self.n_placeholder} builder-placeholder"
        )


def _make_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["chrom"].astype(str) + ":" + df["pos"].astype(str)
        + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    )


def attach_delta_windows(
    meta: pd.DataFrame,
    seq_windows_path=None,
    window: int = 101,
) -> WindowAttachment:
    """Resolve [fasta_seq_ref, fasta_seq_alt] for every row of `meta`.

    Returns a WindowAttachment. Read `.usable` before trusting any row's sequence.
    """
    placeholder = PLACEHOLDER_BASE * window
    n = len(meta)

    # -- 1. windows already on the rows -> structurally aligned, no join. -------
    if REF_WIN_COL in meta.columns and ALT_WIN_COL in meta.columns:
        ref_s, alt_s = meta[REF_WIN_COL], meta[ALT_WIN_COL]
        # No `ok` column travels with a pre-attached frame, so provenance is limited
        # to presence. Stated plainly rather than papered over: if the caller hands us
        # a frame whose windows are already placeholders, we cannot know it. Prefer
        # tier 2, which carries the builder's own verdict.
        # A window is LOCATED for this row iff both columns are non-null. Nothing was
        # looked up at this tier, but the window is either here or it is not.
        key_found = ref_s.notna().to_numpy() & alt_s.notna().to_numpy()
        if OK_COL in meta.columns:
            builder_ok = meta[OK_COL].fillna(False).astype(bool).to_numpy()
            prov = "rows+ok"
        else:
            # No verdict travelled with the frame, so one is assumed. `provenance` records
            # that, and provenance_is_verified reports it.
            builder_ok = np.ones(n, dtype=bool)
            prov = "rows"
        out = pd.DataFrame(
            {
                REF_WIN_COL: ref_s.fillna(placeholder).astype(str).to_numpy(),
                ALT_WIN_COL: alt_s.fillna(placeholder).astype(str).to_numpy(),
            }
        )
        att = WindowAttachment(out, key_found, builder_ok, prov)
        logger.info("%s", att.summary())
        return att

    # -- 2. key-join from a windows parquet (order-preserving via .map). --------
    if seq_windows_path is not None:
        want = [*_KEY_COLS, REF_WIN_COL, ALT_WIN_COL]
        try:
            seq = pd.read_parquet(seq_windows_path, columns=[*want, OK_COL])
            has_ok = True
        except (ValueError, KeyError):
            # An artifact predating build_seq_windows' `ok` column. Do not silently
            # proceed as though every window were real -- that is the exact failure
            # this module now exists to prevent.
            seq = pd.read_parquet(seq_windows_path, columns=want)
            has_ok = False
            logger.warning(
                "seq windows at %s carry no '%s' column: builder-placeholder rows "
                "CANNOT be identified and will be treated as usable. Rebuild with "
                "scripts/build_seq_windows.py to restore provenance.",
                seq_windows_path, OK_COL,
            )

        seq = seq.assign(_key=_make_key(seq)).drop_duplicates("_key")  # window = f(key)
        mkey = _make_key(meta)
        r = mkey.map(seq.set_index("_key")[REF_WIN_COL])
        a = mkey.map(seq.set_index("_key")[ALT_WIN_COL])

        mapped = r.notna().to_numpy() & a.notna().to_numpy()
        if has_ok:
            ok = mkey.map(seq.set_index("_key")[OK_COL]).fillna(False).astype(bool).to_numpy()
        else:
            ok = np.ones(n, dtype=bool)


        out = pd.DataFrame(
            {
                REF_WIN_COL: r.fillna(placeholder).astype(str).to_numpy(),
                ALT_WIN_COL: a.fillna(placeholder).astype(str).to_numpy(),
            }
        )
        att = WindowAttachment(
            out, mapped, ok, "parquet+ok" if has_ok else "parquet",
        )
        logger.info("%s", att.summary())
        if att.n_placeholder:
            logger.warning(
                "%d/%d rows carry BUILDER-PLACEHOLDER windows (ok=False): no reference "
                "sequence exists for them. They are masked usable=False. Before "
                "2026-07-15 these were indistinguishable from real windows and were "
                "trained on.", att.n_placeholder, n,
            )
        return att

    # -- 3. no source -> no sequence signal at all. ----------------------------
    logger.warning(
        "no seq_windows_path: ALL %d rows get placeholder windows, usable=False. "
        "Any sequence model fitted on this attachment is fitting noise.", n,
    )
    out = pd.DataFrame({REF_WIN_COL: [placeholder] * n, ALT_WIN_COL: [placeholder] * n})
    _none = np.zeros(n, dtype=bool)
    return WindowAttachment(out, _none, _none, "none")
