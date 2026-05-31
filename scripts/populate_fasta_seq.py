#!/usr/bin/env python
"""CLI: materialize ref/alt sequence windows into a new cohort parquet.

Examples
--------
Smoke test on the first 50k rows (fast; verifies the real mismatch rate)::

    python scripts/populate_fasta_seq.py --limit 50000 \
        --out data/processed/clinvar_grch38_clean_seq.smoke.parquet

Full pass (~0.5 GB output; budget several minutes -- per-row pyfaidx seeks)::

    python scripts/populate_fasta_seq.py

Exits nonzero (and removes the temp file) if a data-quality guard trips.
"""

from __future__ import annotations

import argparse
import logging
import sys

from genomic_variant_classifier.data.populate_fasta_seq import populate, GuardFailure

DEF_COHORT = "data/processed/clinvar_grch38_clean.parquet"
DEF_FASTA = "data/external/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
DEF_OUT = "data/processed/clinvar_grch38_clean_seq.parquet"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default=DEF_COHORT)
    p.add_argument("--fasta", default=DEF_FASTA)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--abort-mismatch", type=float, default=0.02,
                   help="abort if ref-allele mismatch rate (resolvable contigs) exceeds this")
    p.add_argument("--abort-degenerate", type=float, default=0.01,
                   help="abort if poly-A degenerate rate (resolvable contigs) exceeds this")
    p.add_argument("--progress-every", type=int, default=200_000)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N rows (smoke test); rounds up to a batch")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        stats = populate(
            args.cohort, args.fasta, args.out,
            batch_size=args.batch_size,
            abort_mismatch=args.abort_mismatch,
            abort_degenerate=args.abort_degenerate,
            progress_every=args.progress_every,
            limit=args.limit,
        )
    except GuardFailure as e:
        logging.error("GUARD FAILED -- no output written: %s", e)
        return 2
    except (FileNotFoundError, KeyError) as e:
        logging.error("INPUT ERROR: %s", e)
        return 3

    logging.info(
        "OK  total=%s resolvable=%s unmapped=%s mismatch_rate=%.5f "
        "degenerate_rate=%.5f elapsed=%.1fs -> %s",
        f"{stats['total']:,}", f"{stats['n_resolvable']:,}", f"{stats['n_unmapped']:,}",
        stats["mismatch_rate"], stats["degenerate_rate"], stats["elapsed_s"], args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
