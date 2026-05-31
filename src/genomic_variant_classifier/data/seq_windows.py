"""Reference/alt sequence-window extraction for the delta sequence CNN (Phase B).

Produces, for each variant, a pair of fixed-width nucleotide windows:

  * ``ref_window``  -- the reference sequence centered on the variant start
  * ``alt_window``  -- the same locus with the variant allele spliced in

The delta CNN consumes both and learns from ``E(alt) - E(ref)`` (the variant's
effect), not the absolute region -- this is the anti-memorization design chosen
for the honest baseline.

Centering convention (1-based ``pos`` per Ensembl/VCF): the *first* base of the
variant is placed at index ``HALF`` (50) of a ``WINDOW`` (101) string, so the
reference base and the first alt base occupy the same index in the two windows.
For SNVs the windows differ by a single base; for indels the downstream context
shifts/relengthens but the variant start stays centered.

Design notes:
  * No logging in this library module (project convention).
  * ``pyfaidx`` is imported lazily in ``open_reference`` only, so the core
    functions are testable with any ``{chrom: sequence}`` mapping (dict or
    ``pyfaidx.Fasta``) without the genome present.
  * Missing contig (e.g. the cohort's ``'Un'``) or a fully out-of-range
    position yields ``ref_window == alt_window`` (both poly-A), so the model's
    delta is exactly zero -- no spurious signal.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

WINDOW: int = 101
HALF: int = WINDOW // 2          # 50 -- index of the variant's first base
PAD_CHAR: str = "A"              # matches encode_sequence's pad in variant_ensemble


def _safe_slice(
    ref: Mapping,
    chrom: str,
    start0: int,
    end0: int,
) -> Optional[Tuple[int, str, int]]:
    """Return ``(left_pad, core, right_pad)`` for the 0-based half-open span
    ``[start0, end0)`` on ``chrom``, clamped to contig bounds with pad counts for
    the portions that fall outside. ``None`` if the contig is absent.

    ``left_pad + len(core) + right_pad == end0 - start0`` always holds.
    """
    try:
        rec = ref[chrom]
    except (KeyError, TypeError):
        return None
    n = len(rec)
    left_pad = max(0, -start0)
    right_pad = max(0, end0 - n)
    a = max(0, start0)
    b = min(n, end0)
    core = str(rec[a:b]).upper() if b > a else ""
    return left_pad, core, right_pad


def extract_ref_window(
    ref: Mapping,
    chrom: str,
    pos1: int,
    window: int = WINDOW,
) -> Optional[str]:
    """Reference window centered on ``pos1`` (1-based). ``None`` if contig absent."""
    half = window // 2
    start0 = (pos1 - 1) - half
    end0 = (pos1 - 1) + (window - half)
    res = _safe_slice(ref, chrom, start0, end0)
    if res is None:
        return None
    left_pad, core, right_pad = res
    win = (PAD_CHAR * left_pad) + core + (PAD_CHAR * right_pad)
    return win[:window].ljust(window, PAD_CHAR)


def extract_alt_window(
    ref: Mapping,
    chrom: str,
    pos1: int,
    ref_allele: str,
    alt_allele: str,
    window: int = WINDOW,
) -> Optional[str]:
    """Window with ``alt_allele`` spliced in at ``pos1``, variant start centered.

    Built as ``upstream(HALF) + alt + downstream-reference`` then trimmed/padded
    to ``window``. ``None`` if the contig is absent.
    """
    half = window // 2
    # Upstream: `half` bases immediately before pos (index 0..half-1 of the window)
    res_left = _safe_slice(ref, chrom, (pos1 - 1) - half, (pos1 - 1))
    if res_left is None:
        return None
    lp, left_core, _ = res_left
    left = (PAD_CHAR * lp) + left_core            # length == half

    # Downstream reference begins AFTER the (claimed) ref allele
    ds_start0 = (pos1 - 1) + max(1, len(ref_allele))
    res_ds = _safe_slice(ref, chrom, ds_start0, ds_start0 + window)
    downstream = "" if res_ds is None else res_ds[1]

    win = left + (alt_allele.upper() if alt_allele else "") + downstream
    return win[:window].ljust(window, PAD_CHAR)


def build_delta_windows(
    ref: Mapping,
    chrom: str,
    pos1: int,
    ref_allele: str,
    alt_allele: str,
    window: int = WINDOW,
) -> Tuple[str, str]:
    """Return ``(ref_window, alt_window)``, each exactly ``window`` long.

    If the contig is absent (e.g. ``'Un'``), both are poly-A so the model's
    delta is zero -- the correct no-signal fallback.
    """
    rw = extract_ref_window(ref, chrom, pos1, window)
    if rw is None:
        return PAD_CHAR * window, PAD_CHAR * window
    aw = extract_alt_window(ref, chrom, pos1, ref_allele, alt_allele, window)
    if aw is None:
        return PAD_CHAR * window, PAD_CHAR * window
    return rw, aw


def ref_matches(
    ref: Mapping,
    chrom: str,
    pos1: int,
    ref_allele: str,
) -> Optional[bool]:
    """Whether the reference at ``[pos1, pos1+len(ref_allele))`` equals
    ``ref_allele`` (build/contig sanity). ``None`` if contig absent.

    Aggregated across the cohort this gives a mismatch RATE; a high rate means
    the cohort and the reference disagree on build/contig and the windows are
    untrustworthy.
    """
    if not ref_allele:
        return None
    res = _safe_slice(ref, chrom, pos1 - 1, (pos1 - 1) + len(ref_allele))
    if res is None:
        return None
    left_pad, core, right_pad = res
    if left_pad or right_pad:
        return False
    return core == ref_allele.upper()


def open_reference(fasta_path):
    """Open a decompressed FASTA for random access via pyfaidx (lazy import)."""
    from pyfaidx import Fasta
    return Fasta(str(fasta_path), rebuild=False)
