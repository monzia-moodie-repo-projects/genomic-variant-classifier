"""
scripts/esm2_prefetch_uniprot.py
================================
Parallel pre-fetch of UniProt canonical sequences into the ESM-2 SQLite cache,
so the (forward-bound) annotate step never waits on the network.

WHY
---
The ESM-2 connector fetches one UniProt sequence per distinct gene via
esm2._fetch_uniprot_sequence (network; ~0.85 s/gene serial in the CPU probe).
For ~19k cohort genes that is ~4.5 h serial. UniProt lookups are I/O-bound
HTTP GETs, so a bounded thread pool collapses that to minutes.

DESIGN (correctness)
--------------------
- Parallelize ONLY the network fetch. sqlite3 connections are not safe to
  share across threads, and the bottleneck is the network, not the write, so
  ALL SQLite writes happen on the main thread as futures complete. Worker
  threads call _fetch_uniprot_sequence (requests.get -- thread-safe per call)
  and return results; the main thread persists them. The script asserts this
  invariant at write time.
- Idempotent / resumable: only genes NOT already in the cache are fetched;
  re-running after an interruption fetches just the remainder.
- Writes to the PERSISTENT ESM-2 cache by design (that is the point). Default
  target is the connector's default cache; override with --cache-path.

USAGE (repo root, .venv312 active)
----------------------------------
  # quick throughput trial on 100 genes first:
  python scripts/esm2_prefetch_uniprot.py --clinvar data/processed/clinvar_grch38.parquet --workers 8 --limit 100
  # then the full set:
  python scripts/esm2_prefetch_uniprot.py --clinvar data/processed/clinvar_grch38.parquet --workers 8

EXIT CODES
----------
  0  completed (per-gene failures are reported, not fatal)
  2  import problem or no missense genes in the cohort
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parallel UniProt pre-fetch into the ESM-2 cache")
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--cache-path", default=None,
                    help="SQLite cache to populate (default: ESM-2 connector default)")
    ap.add_argument("--workers", type=int, default=8,
                    help="bounded concurrent UniProt requests (default 8; be a good citizen)")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap on genes fetched this run (0 = all; for a quick trial)")
    args = ap.parse_args(argv)

    # Verified module-level API (esm2.py: _DEFAULT_CACHE 74, _open_cache 118,
    # _cache_get_sequence 137, _cache_put_sequence 144, _fetch_uniprot_sequence 183).
    from genomic_variant_classifier.data import esm2 as esm2_mod

    cache_path = Path(args.cache_path) if args.cache_path else esm2_mod._DEFAULT_CACHE
    print(f"cache target : {cache_path}")
    conn = esm2_mod._open_cache(cache_path)

    df = pd.read_parquet(args.clinvar, columns=["gene_symbol", "consequence"])
    miss = df[df["consequence"].fillna("").str.contains("missense", case=False)]
    genes_all = sorted({g for g in miss["gene_symbol"].dropna().astype(str) if g})
    if not genes_all:
        print("no missense genes in cohort -- nothing to fetch. STOP.")
        return 2

    todo = [g for g in genes_all if esm2_mod._cache_get_sequence(conn, g) is None]
    already = len(genes_all) - len(todo)
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    print(f"genes total   : {len(genes_all):,}")
    print(f"already cached: {already:,}")
    print(f"to fetch      : {len(todo):,}  (workers={args.workers}, timeout={args.timeout}s)")
    if not todo:
        print("cache already complete for this cohort. Done.")
        return 0

    main_tid = threading.get_ident()
    resolved = 0
    failed: list[str] = []
    done = 0
    t0 = time.time()

    def _fetch(gene: str):
        # worker thread: network only, NO sqlite here
        return gene, esm2_mod._fetch_uniprot_sequence(gene, args.timeout)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_fetch, g) for g in todo]
        for fut in as_completed(futs):
            gene, res = fut.result()
            if res is not None:
                uid, seq = res
                # invariant: writes happen only on the main thread
                assert threading.get_ident() == main_tid, "sqlite write off main thread"
                esm2_mod._cache_put_sequence(conn, gene, uid, seq)
                resolved += 1
            else:
                failed.append(gene)
            done += 1
            if done % 250 == 0 or done == len(todo):
                elapsed = max(time.time() - t0, 1e-9)
                rate = done / elapsed
                eta_min = (len(todo) - done) / max(rate, 1e-9) / 60.0
                print(f"  {done}/{len(todo)} ({resolved} ok, {len(failed)} failed) "
                      f"| {rate:.1f} genes/s | ETA {eta_min:.1f} min")

    dt_min = (time.time() - t0) / 60.0
    print(f"\nfetched {resolved}/{len(todo)} in {dt_min:.1f} min ({len(failed)} failed)")
    if failed:
        preview = ", ".join(failed[:15])
        print(f"failed (first 15): {preview}{' ...' if len(failed) > 15 else ''}")
        print("  (often transient, or genes with no reviewed human UniProt entry; "
              "re-run to retry only the missing ones.)")
    print(f"cache now holds sequences for {already + resolved:,}/{len(genes_all):,} cohort genes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
