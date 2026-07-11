"""delta_window_builder.py -- core sequence-window construction for the sequence CNN.

Builds the two-column [fasta_seq_ref, fasta_seq_alt] delta windows the one-dimensional
convolutional neural network (cnn_1d) needs, from a variant's (chrom, pos, ref, alt) plus the
GRCh38 reference genome. Replaces the poly-placeholder path that left cnn_1d degenerate
(Area Under the Receiver Operating Characteristic Curve 0.5419 on placeholder sequences).

Conventions PINNED by the feasibility probe (2026-07-10, reference-base match 200/200):
  - contig name = str(chrom) as-is (the reference and cohort both use bare '1'..'22','X','Y','MT')
  - position is 1-based; the reference base sits at 0-based index (pos - 1)
  - window = 101 base pairs, centered on the variant (50 bp flank each side for a single-nucleotide
    variant)

Single-nucleotide variants replace one center base. Insertions/deletions replace len(ref) bases
with alt and re-center to hold exactly `window` base pairs. Contig edges pad with 'N'. Any
reference-allele mismatch (fetched base != cohort ref) or fetch failure falls back to the poly
placeholder WITH an explicit reason recorded -- never silently.

This module is the correctness foundation (Step 1). Benchmark, full precompute, manifest, and the
retrain-side coherence gate build on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass

POLY = "N"  # placeholder base; a full poly window is POLY * window


@dataclass
class WindowResult:
    ref_window: str
    alt_window: str
    ok: bool           # True if built from the real reference and validated
    reason: str = ""   # why it fell back, if ok is False


def _norm_allele(a) -> str:
    return str(a).strip().upper() if a is not None else ""


def build_window(fetch, chrom, pos, ref, alt, window: int = 101) -> WindowResult:
    """Build one (ref_window, alt_window) pair.

    `fetch(contig, start0, length) -> str|None` returns `length` reference bases starting at the
    0-based coordinate `start0`, or None on failure. Injecting `fetch` keeps this function pure and
    unit-testable without a real genome.
    """
    poly = POLY * window
    ref = _norm_allele(ref); alt = _norm_allele(alt)
    try:
        pos = int(pos)
    except (TypeError, ValueError):
        return WindowResult(poly, poly, False, "bad_pos")
    if not ref or not alt:
        return WindowResult(poly, poly, False, "empty_allele")
    if any(c not in "ACGT" for c in ref) or any(c not in "ACGT" for c in alt):
        # non-ACGT alleles (e.g. '-', 'N', symbolic) are out of scope for a clean window
        return WindowResult(poly, poly, False, "non_acgt_allele")

    contig = str(chrom)
    half = window // 2                 # for window=101, half=50

    # Candidate offsets for where the ref allele STARTS, relative to the 0-based coordinate (pos-1).
    # Pinned by the indel-convention probe (2026-07-10, 5,000-indel sample, 0% unexplained):
    #   single-nucleotide, insertion, and equal-length multi-nucleotide variants: ref starts at
    #     pos-1 (offset 0) -- 100% of insertions and mnv, 100% of SNVs.
    #   deletions: ref starts at pos-2 (offset -1) -- 97%; the remaining 3% also match at offset 0
    #     (repetitive/homopolymer regions where both align), caught by trying 0 as well.
    # The builder accepts the FIRST offset whose genome slice equals the cohort ref, so correctness
    # is guaranteed by the ref-match self-check regardless of which offset is canonical.
    if len(alt) > len(ref) or len(alt) == len(ref):
        candidate_offsets = (0,)               # insertion / SNV / mnv: anchored at pos-1
    else:
        candidate_offsets = (-1, 0)            # deletion: pos-2 first, then pos-1 for the residual

    # Over-fetch generously so any candidate offset and the full window fit, then index into it.
    pad = max(len(ref), len(alt)) + 2
    center0 = pos - 1                          # variant locus (window centers here, like SNVs)
    fetch_start = center0 - half - pad
    fetch_len = window + 2 * pad + 2
    left_clip = 0
    if fetch_start < 0:
        left_clip = -fetch_start
        fetch_len += fetch_start
        fetch_start = 0
    raw = fetch(contig, fetch_start, fetch_len)
    if raw is None:
        return WindowResult(poly, poly, False, "fetch_failed")
    raw = raw.upper()
    # Index-aligned track: ref_track[i] corresponds to genome 0-based coord (track_origin0 + i).
    ref_track = ("N" * left_clip) + raw
    track_origin0 = fetch_start - left_clip

    # Find the offset whose genome slice equals the cohort ref.
    ref_start0 = None
    for off in candidate_offsets:
        cand_start0 = center0 + off
        idx = cand_start0 - track_origin0
        if idx < 0 or idx + len(ref) > len(ref_track):
            continue
        if ref_track[idx: idx + len(ref)] == ref:
            ref_start0 = cand_start0
            break
    if ref_start0 is None:
        # No candidate offset matches -- fall back transparently with the offset-0 observation.
        idx0 = center0 - track_origin0
        got = ref_track[idx0: idx0 + len(ref)] if 0 <= idx0 else ""
        return WindowResult(poly, poly, False, f"ref_mismatch(got={got!r},exp={ref!r})")

    ref_idx = ref_start0 - track_origin0        # index of the ref allele's first base in the track
    # Build the alt track by splicing alt in place of ref at the matched position.
    alt_track = ref_track[:ref_idx] + alt + ref_track[ref_idx + len(ref):]
    # The alt allele occupies the same start index; downstream bases shift by (len(alt)-len(ref)).

    # Center both windows on the VARIANT LOCUS (center0), so indel windows are positionally
    # comparable to SNV windows. The center index within each track:
    ref_center_idx = center0 - track_origin0
    # In the alt track, everything at or after ref_idx shifts; the variant locus maps to the same
    # left context, so we center on the same index (the splice begins at ref_idx <= center for
    # deletions at offset -1, and at center for offset 0).
    alt_center_idx = center0 - track_origin0

    def center_slice(track: str, cidx: int) -> str:
        lo = cidx - half
        hi = lo + window
        s = ""
        if lo < 0:
            s += "N" * (-lo)
            lo = 0
        s += track[lo:hi]
        if len(s) < window:
            s += "N" * (window - len(s))
        return s[:window]

    ref_window = center_slice(ref_track, ref_center_idx)
    alt_window = center_slice(alt_track, alt_center_idx)

    # Self-checks: exact length always; center base equals ref/alt for single-nucleotide variants.
    if len(ref_window) != window or len(alt_window) != window:
        return WindowResult(poly, poly, False, "bad_length")
    if len(ref) == 1 and len(alt) == 1:
        if ref_window[half] != ref:
            return WindowResult(poly, poly, False, "center_ref_mismatch")
        if alt_window[half] != alt:
            return WindowResult(poly, poly, False, "center_alt_mismatch")
    return WindowResult(ref_window, alt_window, True, "")
