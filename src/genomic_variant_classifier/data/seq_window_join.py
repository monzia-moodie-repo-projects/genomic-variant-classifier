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

    Attributes
    ----------
    windows:
        2-column [fasta_seq_ref, fasta_seq_alt] frame, 1:1 with `meta` (reset index).
    usable:
        Boolean array, 1:1 with `meta`. True iff the row carries sequence that came
        from the reference genome. **Consumers must mask on this and nothing else.**
        Never compare window CONTENT to a placeholder string: a real window may
        legitimately equal any given string (poly-A tracts are real biology).
    n_unmapped:
        Rows whose key was absent from the window source entirely.
    n_placeholder:
        Rows present in the source but flagged `ok=False` by the builder -- i.e. the
        builder could not construct a window (missing contig, out-of-range position,
        non-ACGT allele). These are the 21,814 the old code could not see.
    provenance:
        Which resolution tier produced this attachment; recorded so a run artifact can
        state HOW its windows were obtained, not merely that it had some.
    """

    windows: pd.DataFrame
    usable: np.ndarray
    n_rows: int
    n_unmapped: int
    n_placeholder: int
    provenance: str

    @property
    def n_usable(self) -> int:
        return int(self.usable.sum())

    @property
    def usable_fraction(self) -> float:
        return (self.n_usable / self.n_rows) if self.n_rows else 0.0

    def summary(self) -> str:
        return (
            f"windows[{self.provenance}]: {self.n_usable}/{self.n_rows} usable "
            f"({100.0 * self.usable_fraction:.3f}%), {self.n_unmapped} unmapped, "
            f"{self.n_placeholder} builder-placeholder"
        )

    def __iter__(self):
        """Back-compat shim: `wins, n_unmapped = attach_delta_windows(...)` unpacks.

        DEPRECATED AND DELIBERATELY LOSSY -- it drops `usable`, which is the entire
        point of this object. It exists only so that the 2026-07-15 change lands as
        one reviewable step instead of a seven-call-site rewrite in the same commit.

        MIGRATION STATUS, 2026-07-15 -- accurate as written, not aspirational:

            MIGRATED to `.usable`:
              src/genomic_variant_classifier/data/genomic_lm.py   (both call sites)

            STILL UNPACKING THE TUPLE -- these read `usable=False` rows as real:
              scripts/train.py:441,458,480          <-- feeds the CNN's X_seq. The
                                                        one that matters for Run 17.
              scripts/run_phase2_eval.py:425,426,427
              scripts/rekey_seq_windows_v2.py:145   <-- verification only, low risk

        `scripts/train.py:485` additionally computes its own poly-detector against
        `_POLY_WIN = "A" * 101`. PLACEHOLDER_BASE is now "N", so that check no longer
        matches anything and its `_n_real_test` count is now wrong in the OTHER
        direction -- it will report every row as real. **train.py must be migrated
        before Run 17.** Tracked as roadmap 6.28.

        This docstring is the todo list. When the list above is empty, delete this
        method and the `usable`-less path stops existing.
        """
        yield self.windows
        yield self.n_unmapped


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
        if OK_COL in meta.columns:
            usable = (
                ref_s.notna().to_numpy()
                & alt_s.notna().to_numpy()
                & meta[OK_COL].fillna(False).astype(bool).to_numpy()
            )
            prov = "rows+ok"
        else:
            usable = ref_s.notna().to_numpy() & alt_s.notna().to_numpy()
            prov = "rows"
        out = pd.DataFrame(
            {
                REF_WIN_COL: ref_s.fillna(placeholder).astype(str).to_numpy(),
                ALT_WIN_COL: alt_s.fillna(placeholder).astype(str).to_numpy(),
            }
        )
        att = WindowAttachment(out, usable, n, int((~usable).sum()), 0, prov)
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

        usable = mapped & ok
        n_unmapped = int((~mapped).sum())
        n_placeholder = int((mapped & ~ok).sum())

        out = pd.DataFrame(
            {
                REF_WIN_COL: r.fillna(placeholder).astype(str).to_numpy(),
                ALT_WIN_COL: a.fillna(placeholder).astype(str).to_numpy(),
            }
        )
        att = WindowAttachment(
            out, usable, n, n_unmapped, n_placeholder,
            "parquet+ok" if has_ok else "parquet",
        )
        logger.info("%s", att.summary())
        if n_placeholder:
            logger.warning(
                "%d/%d rows carry BUILDER-PLACEHOLDER windows (ok=False): no reference "
                "sequence exists for them. They are masked usable=False. Before "
                "2026-07-15 these were indistinguishable from real windows and were "
                "trained on.", n_placeholder, n,
            )
        return att

    # -- 3. no source -> no sequence signal at all. ----------------------------
    logger.warning(
        "no seq_windows_path: ALL %d rows get placeholder windows, usable=False. "
        "Any sequence model fitted on this attachment is fitting noise.", n,
    )
    out = pd.DataFrame({REF_WIN_COL: [placeholder] * n, ALT_WIN_COL: [placeholder] * n})
    return WindowAttachment(out, np.zeros(n, dtype=bool), n, n, 0, "none")
