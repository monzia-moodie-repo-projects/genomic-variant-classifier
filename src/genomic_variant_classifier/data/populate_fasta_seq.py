"""Materialize ref/alt sequence windows for the delta CNN (Phase B core).

Streams the clean cohort, extracts a ``(fasta_seq_ref, fasta_seq_alt)`` pair per
variant from the indexed GRCh38 reference, and writes a NEW parquet (every
original column preserved, in the original row order, plus the two window
columns) so the existing gene-disjoint splits are reproduced byte-for-byte.

Performance: variants are processed grouped by contig in sorted order, loading
each chromosome's sequence into memory ONCE (one sequential read) and slicing
in-RAM. This eliminates the per-row random disk seeks that dominate a cold
3.1 GB FASTA (~10x faster), while output stays in the original row order
(windows are placed back by original index before writing).

Lineage: the original pipeline used a single ``fasta_seq`` 101-bp window column
(never populated for real; connectors set it to ``None``/poly-A). This replaces
it with a two-window delta representation; the reference window corresponds to
the old ``fasta_seq`` and the poly-A fallback matches ``.fillna("A"*101)``.

Anchoring (diagnosed 2026-05-31): this cohort's ``pos`` comes from ClinVar
``variant_summary`` (Start = first changed base) while ref/alt were patched from
the ClinVar VCF (anchored one base earlier for deletions). SNV/MNV/insertion
align at delta 0; deletions align at delta -1. Each variant is anchored to the
position where its ref allele actually matches the genome
(``seq_windows.find_anchor``); the cohort's ``pos`` column is left untouched
(other joins depend on it). Single-base refs require an exact match.

Guards -- nothing fails silently:
  * EARLY ABORT: after the first contig (>= _EARLY_MIN resolvable rows seen), if
    the running unanchored rate exceeds ``abort_unanchored`` the pass aborts
    before writing anything.
  * AGGREGATE GUARDS (post-extraction, pre-write):
      - unanchored rate (resolvable contigs) > ``abort_unanchored`` -> abort
      - degenerate rate (anchored rows whose ref window is all-pad)
        > ``abort_degenerate``                                      -> abort
  * Output is written to ``<out>.tmp`` and promoted via atomic ``os.replace``
    only on success; on any write error the temp file is removed. Unmapped
    contigs (e.g. ``'Un'``) are expected zero-delta and counted separately.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from genomic_variant_classifier.data import seq_windows as sw

logger = logging.getLogger(__name__)

REQUIRED_COLS = ("chrom", "pos", "ref", "alt")
REF_COL = "fasta_seq_ref"
ALT_COL = "fasta_seq_alt"
_EARLY_MIN = 10_000
_POLY_A = sw.PAD_CHAR * sw.WINDOW


class GuardFailure(RuntimeError):
    """Raised when a data-quality guard trips; the temp output is removed."""


def reference_contigs(fasta) -> Set[str]:
    try:
        return set(fasta.keys())
    except AttributeError:
        return set(fasta)


def _missing_required(cohort_path: str) -> List[str]:
    names = set(pq.ParquetFile(cohort_path).schema_arrow.names)
    return [c for c in REQUIRED_COLS if c not in names]


def _load_contig(fasta, chrom: str) -> str:
    """Whole-contig sequence as an uppercase str (pyfaidx Fasta or mapping)."""
    rec = fasta[chrom]
    return str(rec[:]).upper()


def populate(
    cohort_path: str,
    fasta_path: str,
    out_path: str,
    *,
    batch_size: int = 100_000,
    abort_unanchored: float = 0.02,
    abort_degenerate: float = 0.01,
    max_shift: int = 3,
    progress_every: int = 200_000,
    window: int = sw.WINDOW,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"cohort not found: {cohort_path}")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"reference FASTA not found: {fasta_path}")
    missing = _missing_required(cohort_path)
    if missing:
        raise KeyError(f"cohort missing required columns: {missing}")
    existing = set(pq.ParquetFile(cohort_path).schema_arrow.names)
    if REF_COL in existing or ALT_COL in existing:
        raise KeyError(f"cohort already contains {REF_COL}/{ALT_COL}; refusing to overwrite")

    fasta = sw.open_reference(fasta_path)
    contigs = reference_contigs(fasta)

    pf = pq.ParquetFile(cohort_path)
    meta = pf.read(columns=list(REQUIRED_COLS)).to_pandas()
    if limit is not None:
        meta = meta.iloc[:limit]
    n = len(meta)
    meta = meta.reset_index(drop=True)            # RangeIndex 0..n-1 == original order
    meta["chrom"] = meta["chrom"].astype(str)

    ref_wins: List[str] = [_POLY_A] * n
    alt_wins: List[str] = [_POLY_A] * n

    n_unmapped = n_resolvable = n_shifted = n_unanchored = n_degenerate = 0
    unanchored_examples: List[str] = []
    t0 = time.time()

    def _rate(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    for chrom, grp in meta.groupby("chrom", sort=True):
        idxs = grp.index.to_numpy()
        if chrom not in contigs:
            n_unmapped += len(idxs)
            continue                              # ref_wins/alt_wins already poly-A
        seqstr = _load_contig(fasta, chrom)
        ref1 = {chrom: seqstr}
        poss = grp["pos"].to_numpy()
        refs = grp["ref"].tolist()
        alts = grp["alt"].tolist()
        for i, p, r, a in zip(idxs, poss, refs, alts):
            r = r or ""
            a = a or ""
            n_resolvable += 1
            apos = sw.find_anchor(ref1, chrom, int(p), r, max_shift=max_shift)
            if apos is None:
                n_unanchored += 1
                if len(unanchored_examples) < 10:
                    unanchored_examples.append(f"{chrom}:{p} ref={r!r}")
                continue
            if apos != int(p):
                n_shifted += 1
            rw, aw = sw.build_delta_windows(ref1, chrom, apos, r, a, window)
            ref_wins[i] = rw
            alt_wins[i] = aw
            if rw == _POLY_A:
                n_degenerate += 1
        del seqstr, ref1
        logger.info(
            "contig %s done: resolvable=%s shifted=%.4f unanchored=%.5f rate=%.0f/s",
            chrom, f"{n_resolvable:,}", _rate(n_shifted, n_resolvable),
            _rate(n_unanchored, n_resolvable), n_resolvable / max(time.time() - t0, 1e-6),
        )
        if n_resolvable >= _EARLY_MIN and _rate(n_unanchored, n_resolvable) > abort_unanchored:
            raise GuardFailure(
                f"EARLY ABORT: unanchored rate {_rate(n_unanchored, n_resolvable):.4f} "
                f"> {abort_unanchored} after {n_resolvable:,} resolvable rows; "
                f"examples: {unanchored_examples[:5]}"
            )

    unanchored_rate = _rate(n_unanchored, n_resolvable)
    degenerate_rate = _rate(n_degenerate, n_resolvable)
    if unanchored_rate > abort_unanchored:
        raise GuardFailure(
            f"unanchored rate {unanchored_rate:.4f} > {abort_unanchored}; "
            f"examples: {unanchored_examples[:5]}"
        )
    if degenerate_rate > abort_degenerate:
        raise GuardFailure(f"degenerate rate {degenerate_rate:.4f} > {abort_degenerate}")

    out_schema = pf.schema_arrow.append(pa.field(REF_COL, pa.string())) \
                                .append(pa.field(ALT_COL, pa.string()))
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    writer: Optional[pq.ParquetWriter] = None
    try:
        writer = pq.ParquetWriter(tmp_path, out_schema)
        off = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            if off >= n:
                break
            take = min(batch.num_rows, n - off)
            bt = batch.slice(0, take) if take < batch.num_rows else batch
            tbl = pa.Table.from_batches([bt])
            tbl = tbl.append_column(REF_COL, pa.array(ref_wins[off:off + take], pa.string()))
            tbl = tbl.append_column(ALT_COL, pa.array(alt_wins[off:off + take], pa.string()))
            writer.write_table(tbl)
            off += take
        writer.close()
        writer = None
        os.replace(tmp_path, out_path)
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    stats = {
        "total": n,
        "n_unmapped": n_unmapped,
        "n_resolvable": n_resolvable,
        "n_shifted": n_shifted,
        "n_unanchored": n_unanchored,
        "n_degenerate": n_degenerate,
        "shift_rate": _rate(n_shifted, n_resolvable),
        "unanchored_rate": _rate(n_unanchored, n_resolvable),
        "degenerate_rate": _rate(n_degenerate, n_resolvable),
        "elapsed_s": time.time() - t0,
    }
    logger.info("populate complete: %s", stats)
    return stats
