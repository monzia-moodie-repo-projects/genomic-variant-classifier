"""Materialize ref/alt sequence windows for the delta CNN (Phase B core).

Streams the clean cohort, extracts a ``(fasta_seq_ref, fasta_seq_alt)`` pair per
variant from the indexed GRCh38 reference via
``seq_windows.build_delta_windows``, and writes a NEW parquet (every original
column preserved, in the original row order, plus the two window columns) so the
existing gene-disjoint splits are reproduced byte-for-byte.

Lineage: the original pipeline used a single ``fasta_seq`` 101-bp window column
(never populated for real; connectors set it to ``None``/poly-A). This replaces
it with a two-window delta representation; the reference window corresponds to
the old ``fasta_seq`` and the poly-A fallback matches the old
``.fillna("A"*101)`` semantics.

Guards -- nothing fails silently:
  * EARLY ABORT: once enough resolvable-contig rows are seen, if the running
    ref-allele mismatch rate exceeds ``abort_mismatch`` the pass aborts within
    seconds (systemic build/coordinate problem) -- before wasting the full pass.
  * AGGREGATE GUARDS (end of pass):
      - mismatch rate (resolvable contigs) > ``abort_mismatch``   -> abort
      - degenerate rate (resolvable contigs whose ref window is all-pad)
        > ``abort_degenerate``                                     -> abort
  * Output is written to ``<out>.tmp`` and promoted via atomic ``os.replace``
    only on success; on any abort the temp file is removed and ``GuardFailure``
    is raised. Unmapped-contig rows (e.g. ``'Un'``) are expected zero-delta and
    counted separately -- they never count toward the degenerate guard.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq

from genomic_variant_classifier.data import seq_windows as sw

logger = logging.getLogger(__name__)

REQUIRED_COLS = ("chrom", "pos", "ref", "alt")
REF_COL = "fasta_seq_ref"
ALT_COL = "fasta_seq_alt"
_EARLY_MIN = 10_000          # resolvable rows seen before early-abort may trigger
_POLY_A = sw.PAD_CHAR * sw.WINDOW


class GuardFailure(RuntimeError):
    """Raised when a data-quality guard trips; the temp output is removed."""


def reference_contigs(fasta) -> Set[str]:
    """Set of contig names present in the reference (pyfaidx Fasta or mapping)."""
    try:
        return set(fasta.keys())
    except AttributeError:
        return set(fasta)  # plain mapping


def _missing_required(cohort_path: str) -> List[str]:
    names = set(pq.ParquetFile(cohort_path).schema_arrow.names)
    return [c for c in REQUIRED_COLS if c not in names]


def populate(
    cohort_path: str,
    fasta_path: str,
    out_path: str,
    *,
    batch_size: int = 100_000,
    abort_mismatch: float = 0.02,
    abort_degenerate: float = 0.01,
    progress_every: int = 200_000,
    window: int = sw.WINDOW,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Extract windows for every row of ``cohort_path`` into ``out_path``.

    Returns a stats dict. Raises ``GuardFailure`` (after removing the temp file)
    if a guard trips, ``FileNotFoundError`` if inputs are absent, or
    ``KeyError`` if required columns are missing.
    """
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"cohort not found: {cohort_path}")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"reference FASTA not found: {fasta_path}")
    missing = _missing_required(cohort_path)
    if missing:
        raise KeyError(f"cohort missing required columns: {missing}")
    if REF_COL in pq.ParquetFile(cohort_path).schema_arrow.names or \
       ALT_COL in pq.ParquetFile(cohort_path).schema_arrow.names:
        raise KeyError(f"cohort already contains {REF_COL}/{ALT_COL}; refusing to overwrite")

    fasta = sw.open_reference(fasta_path)
    contigs = reference_contigs(fasta)

    pf = pq.ParquetFile(cohort_path)
    out_schema = pf.schema_arrow.append(pa.field(REF_COL, pa.string())) \
                                .append(pa.field(ALT_COL, pa.string()))
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    total = 0
    n_unmapped = 0
    n_resolvable = 0
    n_mismatch = 0
    n_degenerate = 0
    mismatch_examples: List[str] = []
    t0 = time.time()
    writer: Optional[pq.ParquetWriter] = None

    def _rate(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    try:
        writer = pq.ParquetWriter(tmp_path, out_schema)
        for batch in pf.iter_batches(batch_size=batch_size):
            chroms = [str(c) for c in batch.column("chrom").to_pylist()]
            poss = batch.column("pos").to_pylist()
            refs = [(r or "") for r in batch.column("ref").to_pylist()]
            alts = [(a or "") for a in batch.column("alt").to_pylist()]

            ref_wins: List[str] = []
            alt_wins: List[str] = []
            for chrom, pos1, ref_a, alt_a in zip(chroms, poss, refs, alts):
                if chrom not in contigs:
                    n_unmapped += 1
                    ref_wins.append(_POLY_A)
                    alt_wins.append(_POLY_A)
                    continue
                n_resolvable += 1
                rw, aw = sw.build_delta_windows(fasta, chrom, int(pos1), ref_a, alt_a, window)
                ref_wins.append(rw)
                alt_wins.append(aw)
                if rw == _POLY_A:
                    n_degenerate += 1
                m = sw.ref_matches(fasta, chrom, int(pos1), ref_a)
                if m is False:
                    n_mismatch += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append(f"{chrom}:{pos1} ref={ref_a!r}")

            tbl = pa.Table.from_batches([batch])
            tbl = tbl.append_column(REF_COL, pa.array(ref_wins, pa.string()))
            tbl = tbl.append_column(ALT_COL, pa.array(alt_wins, pa.string()))
            writer.write_table(tbl)

            total += batch.num_rows
            if total % progress_every < batch_size:
                rate = total / max(time.time() - t0, 1e-6)
                logger.info(
                    "rows=%s mismatch=%.4f degenerate=%.4f unmapped=%s rate=%.0f/s",
                    f"{total:,}", _rate(n_mismatch, n_resolvable),
                    _rate(n_degenerate, n_resolvable), f"{n_unmapped:,}", rate,
                )

            # fail-fast: systemic mismatch detected early
            if n_resolvable >= _EARLY_MIN and _rate(n_mismatch, n_resolvable) > abort_mismatch:
                raise GuardFailure(
                    f"EARLY ABORT: mismatch rate {_rate(n_mismatch, n_resolvable):.4f} "
                    f"> {abort_mismatch} after {n_resolvable:,} resolvable rows; "
                    f"examples: {mismatch_examples[:5]}"
                )

            if limit is not None and total >= limit:
                logger.info("stopping after %s rows (--limit)", f"{total:,}")
                break

        writer.close()
        writer = None

        mismatch_rate = _rate(n_mismatch, n_resolvable)
        degenerate_rate = _rate(n_degenerate, n_resolvable)
        if mismatch_rate > abort_mismatch:
            raise GuardFailure(
                f"mismatch rate {mismatch_rate:.4f} > {abort_mismatch}; "
                f"examples: {mismatch_examples[:5]}"
            )
        if degenerate_rate > abort_degenerate:
            raise GuardFailure(
                f"degenerate rate {degenerate_rate:.4f} > {abort_degenerate}"
            )

        os.replace(tmp_path, out_path)
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    stats = {
        "total": total,
        "n_unmapped": n_unmapped,
        "n_resolvable": n_resolvable,
        "n_mismatch": n_mismatch,
        "n_degenerate": n_degenerate,
        "mismatch_rate": _rate(n_mismatch, n_resolvable),
        "degenerate_rate": _rate(n_degenerate, n_resolvable),
        "elapsed_s": time.time() - t0,
    }
    logger.info("populate complete: %s", stats)
    return stats
