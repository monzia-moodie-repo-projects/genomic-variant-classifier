#!/usr/bin/env python3
"""
prebuild_finngen_caches.py — one-time FinnGen full-index cache builder.
Phase-D Stage-1, 2026-07-04.

Builds the full-index parquet caches for BOTH FinnGen releases (R12, R13) by
invoking the patched FinnGenConnector._load_full_index() once per release.
Each build reads the ~30 GB .gz once (~20-25 min), writes
data/raw/cache/finngen_{prefix}full_index.parquet + .meta.json, then VERIFIES
the cache (row count, schema, fast reload).

Run from repo root with:
    $env:PYTHONPATH="src"; python -u scripts/prebuild_finngen_caches.py

Idempotent: if a valid cache already exists (matching source size+mtime), the
connector loads it in seconds instead of rebuilding — so re-running is cheap.
"""
from __future__ import annotations
import sys, time, json, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prebuild_finngen")

# --- ensure src on path ---
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from genomic_variant_classifier.data.finngen import FinnGenConnector  # noqa: E402

RELEASES = [
    # (label, gz_path, column_prefix)
    ("R12", "data/external/finngen/finnge_R12_annotated_variants_v1.gz", ""),
    ("R13", "data/external/finngen/finngen_R13_annotated_variants_v0.gz", "r13_"),
]

EXPECTED_SCHEMA = ["chrom", "pos", "ref", "alt", "af_fin", "af_nfsee"]


def build_and_verify(label: str, gz: str, prefix: str) -> bool:
    gzp = Path(gz)
    log.info("=" * 70)
    log.info("[%s] source: %s", label, gz)
    if not gzp.exists():
        log.error("[%s] SOURCE MISSING: %s — SKIP", label, gz)
        return False

    conn = FinnGenConnector(tsv_path=gzp, column_prefix=prefix)
    pq_path, meta_path = conn._cache_paths()
    log.info("[%s] target cache: %s", label, pq_path)

    already = pq_path.exists() and meta_path.exists()
    t0 = time.time()
    idx = conn._load_full_index()   # builds if absent, loads if valid cache
    dt = time.time() - t0
    log.info("[%s] _load_full_index returned %d rows in %.1fs (%s)",
             label, len(idx), dt, "cache-load" if (already and dt < 60) else "BUILT")

    # --- verify ---
    ok = True
    # schema
    missing = [c for c in EXPECTED_SCHEMA if c not in idx.columns]
    if missing:
        log.error("[%s] SCHEMA MISSING columns: %s", label, missing); ok = False
    else:
        log.info("[%s] schema OK: %s", label, list(idx.columns))
    # row count sanity (FinnGen releases have ~20M variants; expect >>1M)
    if len(idx) < 1_000_000:
        log.error("[%s] ROW COUNT SUSPICIOUS: %d (<1M) — cache may be wrong", label, len(idx)); ok = False
    else:
        log.info("[%s] row count OK: %d", label, len(idx))
    # cache files exist
    if not (pq_path.exists() and meta_path.exists()):
        log.error("[%s] cache files not written", label); ok = False
    else:
        sz = pq_path.stat().st_size / 1e6
        meta = json.loads(meta_path.read_text())
        log.info("[%s] cache parquet %.1f MB; sidecar source=%s size=%d",
                 label, sz, Path(meta.get("source","?")).name, meta.get("size", -1))
    # fast reload proof (second call must be seconds)
    t1 = time.time()
    conn2 = FinnGenConnector(tsv_path=gzp, column_prefix=prefix)
    idx2 = conn2._load_full_index()
    dt2 = time.time() - t1
    if dt2 > 60:
        log.error("[%s] RELOAD SLOW (%.1fs > 60s) — cache not being used!", label, dt2); ok = False
    else:
        log.info("[%s] fast reload OK: %.1fs, %d rows", label, dt2, len(idx2))
    if len(idx2) != len(idx):
        log.error("[%s] reload row mismatch %d vs %d", label, len(idx2), len(idx)); ok = False

    log.info("[%s] VERDICT: %s", label, "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    log.info("FinnGen cache pre-build starting (2 releases).")
    results = {}
    for label, gz, prefix in RELEASES:
        results[label] = build_and_verify(label, gz, prefix)
    log.info("=" * 70)
    log.info("SUMMARY: %s", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
    all_ok = all(results.values())
    log.info("ALL CACHES %s", "READY" if all_ok else "-- SOME FAILED (see above)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
