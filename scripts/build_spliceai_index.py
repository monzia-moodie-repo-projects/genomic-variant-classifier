#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_spliceai_index.py
"""
from __future__ import annotations
import argparse
import gzip
import logging
import sys
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def _parse_spliceai_info(info):
    best = (0.0, 0.0, 0.0, 0.0, "")
    best_max = -1.0
    for field in info.split(";"):
        if not field.startswith("SpliceAI="):
            continue
        for ann in field[len("SpliceAI="):].split(","):
            parts = ann.split("|")
            if len(parts) < 6:
                continue
            try:
                ds_ag, ds_al, ds_dg, ds_dl = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                symbol = parts[1]
                mx = max(ds_ag, ds_al, ds_dg, ds_dl)
                if mx > best_max:
                    best_max = mx
                    best = (ds_ag, ds_al, ds_dg, ds_dl, symbol)
            except (ValueError, IndexError):
                continue
    return best

def build_index(vcf_path, out_path, chunk_size=500_000, min_score=0.0):
    if not vcf_path.exists():
        logger.error("VCF not found: %s", vcf_path)
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("chrom",           pa.string()),
        pa.field("pos",             pa.int32()),
        pa.field("ref",             pa.string()),
        pa.field("alt",             pa.string()),
        pa.field("ds_ag",           pa.float32()),
        pa.field("ds_al",           pa.float32()),
        pa.field("ds_dg",           pa.float32()),
        pa.field("ds_dl",           pa.float32()),
        pa.field("splice_ai_score", pa.float32()),
        pa.field("symbol",          pa.string()),
    ])
    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
    chroms, positions, refs, alts = [], [], [], []
    ds_ags, ds_als, ds_dgs, ds_dls, scores, symbols = [], [], [], [], [], []
    total_lines = total_written = skipped_score = 0
    t0 = time.time()

    def _flush():
        nonlocal total_written
        if not positions:
            return
        writer.write_table(pa.table({
            "chrom": chroms, "pos": pa.array(positions, type=pa.int32()),
            "ref": refs, "alt": alts,
            "ds_ag": pa.array(ds_ags, type=pa.float32()),
            "ds_al": pa.array(ds_als, type=pa.float32()),
            "ds_dg": pa.array(ds_dgs, type=pa.float32()),
            "ds_dl": pa.array(ds_dls, type=pa.float32()),
            "splice_ai_score": pa.array(scores, type=pa.float32()),
            "symbol": symbols,
        }))
        total_written += len(positions)
        chroms.clear(); positions.clear(); refs.clear(); alts.clear()
        ds_ags.clear(); ds_als.clear(); ds_dgs.clear(); ds_dls.clear()
        scores.clear(); symbols.clear()

    logger.info("Streaming %s ...", vcf_path)
    with gzip.open(vcf_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            total_lines += 1
            if total_lines % 5_000_000 == 0:
                logger.info("  %9d lines | %7d written | %5d skipped | %.1f min",
                            total_lines, total_written, skipped_score, (time.time()-t0)/60)
            cols = line.rstrip("\n").split("\t", 8)
            if len(cols) < 8:
                continue
            chrom = cols[0].lstrip("chr")
            try:
                pos = int(cols[1])
            except ValueError:
                continue
            ref, alt = cols[3], cols[4]
            if "," in alt:
                continue
            ds_ag, ds_al, ds_dg, ds_dl, symbol = _parse_spliceai_info(cols[7])
            max_delta = max(ds_ag, ds_al, ds_dg, ds_dl)
            if max_delta < min_score:
                skipped_score += 1
                continue
            chroms.append(chrom); positions.append(pos); refs.append(ref); alts.append(alt)
            ds_ags.append(ds_ag); ds_als.append(ds_al); ds_dgs.append(ds_dg); ds_dls.append(ds_dl)
            scores.append(max_delta); symbols.append(symbol)
            if len(positions) >= chunk_size:
                _flush()
    _flush()
    writer.close()
    logger.info("Done. %d lines | %d written | %.1f min", total_lines, total_written, (time.time()-t0)/60)
    logger.info("Output: %s (%.1f MB)", out_path, out_path.stat().st_size/1024/1024)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vcf",        default="data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz")
    p.add_argument("--out",        default="data/processed/spliceai_index.parquet")
    p.add_argument("--chunk-size", type=int,   default=500_000)
    p.add_argument("--min-score",  type=float, default=0.0)
    args = p.parse_args()
    build_index(Path(args.vcf), Path(args.out), args.chunk_size, args.min_score)

if __name__ == "__main__":
    main()
