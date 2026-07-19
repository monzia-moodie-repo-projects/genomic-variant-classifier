#!/usr/bin/env python3
"""build_clean_seq_from_windows.py -- build the sequence-annotated cohort by JOIN.

PURPOSE
-------
Produces `data/processed/clinvar_grch38_clean_seq.parquet`: the clean cohort with
reference/alternate delta windows and their PROVENANCE attached.

    clinvar_grch38_clean.parquet          (cohort, no windows)
  + seq_windows/seq_windows.parquet       (windows + `ok` + `reason`)
  -> clinvar_grch38_clean_seq.parquet     (cohort + windows + ok + reason)

WHY A JOIN AND NOT A REBUILD
----------------------------
`seq_windows.parquet` is produced by `scripts/build_seq_windows.py` from the reference
genome, and MEASURED 2026-07-18 the clean cohort's keys are a STRICT SUBSET of it -- zero
keys missing. So this artifact can be assembled by a key join in minutes, with no access
to GRCh38, no pyfaidx, and no risk of a second builder disagreeing with the first.

Two window builders previously coexisted and disagreed. The superseded pair
(`data/seq_windows.py` + `data/populate_fasta_seq.py`, placeholder base "A") wrote this
same file with NO `ok` column, so `attach_delta_windows` took its has_ok=False branch and
declared every row usable behind a logger warning. Because "A" is a member of
encode_sequence's BASES, every given-up position one-hot-encoded to a CONFIDENT ADENINE.
This script replaces that producer: one builder, one convention, provenance carried.

WHAT THE OUTPUT GAINS
---------------------
`ok` and `reason` columns. `attach_delta_windows` then computes a real `usable` mask
instead of an unconditional array of True. Measured on the 2026-07-18 cohort: 723 rows of
4,399,089 are builder-placeholders (668 non-ACGT allele, 53 reference mismatch, 2 fetch
failed) and are now masked rather than trained on.

NO CONTENT CHECKS -- BY DESIGN
------------------------------
This script never constructs or compares against a placeholder literal. A window whose
content is a homopolymer MAY BE REAL; content cannot distinguish "the reference genuinely
says that" from "the builder gave up". Correctness is verified from the builder's own `ok`
column, and the expected counts are DERIVED FROM THE INPUTS at run time rather than
hardcoded, so they cannot go stale as the cohort changes.

EXIT CODES
----------
    0  output written and verified
    1  post-check failed; the existing output is untouched and the candidate is retained
    2  bad inputs (missing file, or the cohort is not a subset of the windows)
    3  environment problem (imports unavailable)

USAGE
-----
    python scripts/build_clean_seq_from_windows.py --dry-run
    python scripts/build_clean_seq_from_windows.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DEF_CLEAN = "data/processed/clinvar_grch38_clean.parquet"
DEF_WINDOWS = "data/processed/seq_windows/seq_windows.parquet"
DEF_OUT = "data/processed/clinvar_grch38_clean_seq.parquet"

KEY_COLS = ["chrom", "pos", "ref", "alt"]
REF_WIN_COL = "fasta_seq_ref"
ALT_WIN_COL = "fasta_seq_alt"
OK_COL = "ok"
REASON_COL = "reason"
ATTACHED = [REF_WIN_COL, ALT_WIN_COL, OK_COL, REASON_COL]


def _key(df):
    return (df["chrom"].astype(str) + ":" + df["pos"].astype(str)
            + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))


def _hash(pd, df):
    return pd.util.hash_pandas_object(_key(df), index=False).to_numpy()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clean", default=DEF_CLEAN)
    p.add_argument("--seq-windows", dest="seq_windows", default=DEF_WINDOWS)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--window", type=int, default=101)
    p.add_argument("--dry-run", action="store_true",
                   help="run every check, write nothing")
    p.add_argument("--no-backup", action="store_true",
                   help="do not copy an existing output aside before replacing it")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment guard
        print("ABORT: required import failed: {!r}".format(exc))
        return 3

    p_clean = Path(args.clean)
    p_win = Path(args.seq_windows)
    p_out = Path(args.out)
    p_tmp = p_out.with_suffix(".tmp.parquet")
    p_bak = p_out.with_name(p_out.name + ".bak")

    print("=" * 78)
    print("BUILD clean_seq BY JOIN")
    print("  cohort  : {}".format(p_clean))
    print("  windows : {}".format(p_win))
    print("  output  : {}".format(p_out))
    print("  mode    : {}".format("DRY RUN" if args.dry_run else "APPLY"))
    print("=" * 78)

    for label, p in (("cohort", p_clean), ("windows", p_win)):
        if not p.exists():
            print("ABORT: {} not found: {}".format(label, p))
            return 2

    # -- STEP 1: window lookup --------------------------------------------------------
    win = pd.read_parquet(p_win, columns=KEY_COLS + ATTACHED)
    n_win_rows = len(win)
    win["_h"] = _hash(pd, win)
    win = win.drop_duplicates("_h", keep="first").set_index("_h")
    print("  windows rows {:,} -> {:,} unique keys ({:,} duplicate row(s) collapsed)".format(
        n_win_rows, len(win), n_win_rows - len(win)))

    # -- STEP 2: coverage -------------------------------------------------------------
    clean_keys = _hash(pd, pd.read_parquet(p_clean, columns=KEY_COLS))
    n_clean = len(clean_keys)
    missing = int((~pd.Index(clean_keys).isin(win.index)).sum())
    print("  cohort rows  {:,};  keys absent from windows: {:,}".format(n_clean, missing))
    if missing:
        print("ABORT: the cohort is NOT a subset of the windows. The join would silently")
        print("       drop {:,} row(s). Rebuild the windows first with".format(missing))
        print("       scripts/build_seq_windows.py, then re-run.")
        return 2

    # EXPECTED provenance, DERIVED from the inputs -- never hardcoded, so it cannot
    # go stale when the cohort changes.
    sub = win.reindex(clean_keys)
    exp_bad = int((~sub[OK_COL].fillna(False).astype(bool)).sum())
    exp_reasons = Counter(
        sub.loc[~sub[OK_COL].fillna(False).astype(bool), REASON_COL]
        .astype(str).str.split("(").str[0])
    print("  expected unusable rows (from the builder's own ok column): {:,}".format(exp_bad))
    for k, v in sorted(exp_reasons.items()):
        print("      {:<20} {:>7,}".format(k, v))
    del sub

    if args.dry_run:
        print("\nDRY RUN complete. Every check above ran. Nothing was written.")
        return 0

    # -- STEP 3: streaming join -------------------------------------------------------
    if p_out.exists() and not args.no_backup and not p_bak.exists():
        shutil.copy2(p_out, p_bak)
        print("  backup: {}".format(p_bak.name))
    elif p_bak.exists():
        print("  backup already present, NOT overwritten: {}".format(p_bak.name))

    writer = None
    written = 0
    try:
        # Every reader is scoped. On Windows os.replace() fails with WinError 32 if any
        # handle on the source is still open; POSIX permits it, so this is a portability
        # requirement that testing on Linux alone cannot surface.
        with pq.ParquetFile(p_clean) as pf:
            for batch in pf.iter_batches(batch_size=args.chunk_size):
                d = batch.to_pandas()
                hit = win.reindex(_hash(pd, d))
                for c in ATTACHED:
                    d[c] = hit[c].to_numpy()
                tbl = pa.Table.from_pandas(d, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(p_tmp, tbl.schema)
                writer.write_table(tbl)
                written += len(d)
                del d, tbl
    finally:
        if writer is not None:
            writer.close()
    print("  rows written : {:,}  ({:.1f}s)".format(written, time.perf_counter() - t0))

    # -- STEP 4: post-check, PROVENANCE-BASED -----------------------------------------
    failures = []
    with pq.ParquetFile(p_tmp) as pf:
        n_out = pf.metadata.num_rows
        cols = [f.name for f in pf.schema_arrow]
    if n_out != n_clean:
        failures.append("row count {:,} != cohort {:,}".format(n_out, n_clean))
    for c in ATTACHED:
        if c not in cols:
            failures.append("missing column {}".format(c))

    if not failures:
        n_null = n_badlen = n_bad = 0
        reasons = Counter()
        with pq.ParquetFile(p_tmp) as pf:
            for b in pf.iter_batches(batch_size=args.chunk_size, columns=ATTACHED):
                d = b.to_pandas()
                n_null += int(d[REF_WIN_COL].isna().sum() + d[ALT_WIN_COL].isna().sum())
                for c in (REF_WIN_COL, ALT_WIN_COL):
                    s = d[c].dropna().astype(str)
                    n_badlen += int((s.str.len() != args.window).sum())
                m = ~d[OK_COL].fillna(False).astype(bool)
                n_bad += int(m.sum())
                for x in d.loc[m, REASON_COL].astype(str).str.split("(").str[0]:
                    reasons[x] += 1
                del d
        print("  null windows {} | wrong-length {} | unusable {:,}".format(
            n_null, n_badlen, n_bad))
        if n_null:
            failures.append("{} null window value(s)".format(n_null))
        if n_badlen:
            failures.append("{} window(s) not {} characters".format(n_badlen, args.window))
        if n_bad != exp_bad:
            failures.append("unusable rows {:,} != expected {:,}".format(n_bad, exp_bad))
        if dict(reasons) != dict(exp_reasons):
            failures.append("reason breakdown {} != expected {}".format(
                dict(reasons), dict(exp_reasons)))

    if failures:
        print("\nPOST-CHECK FAILED -- the existing output was NOT replaced:")
        for f in failures:
            print("  - {}".format(f))
        print("Candidate retained at {}".format(p_tmp))
        return 1

    os.replace(p_tmp, p_out)
    print("\nOK  {}  ({:.1f} MB, {:.1f}s total)".format(
        p_out.name, p_out.stat().st_size / (1024 * 1024), time.perf_counter() - t0))
    print("  Consumers now receive `ok`; attach_delta_windows computes a real usable mask.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
