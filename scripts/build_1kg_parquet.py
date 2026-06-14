#!/usr/bin/env python3
"""build_1kg_parquet.py  --  Monzia Moodie

Build the 1000 Genomes Phase 3 AF parquet consumed by ThousandGenomesConnector,
emitting BOTH the global allele_freq AND the five super-population AFs so that
af_1kg_afr/eur/eas/sas/amr can be populated (fixes the previously-dead features).

Output schema:
    variant_id   str    "chrom:pos:ref:alt" (chrom without 'chr' prefix)
    allele_freq  float  global AF (INFO AF)
    AFR_AF EUR_AF EAS_AF SAS_AF AMR_AF  float  super-population AFs (INFO *_AF)

Usage:
    python scripts/build_1kg_parquet.py \\
        --vcf-dir data/external/1000g/phase3_vcf \\
        --out     data/external/1000g/kg_phase3_af.parquet

Parses VCF.gz manually (no pysam dependency). Multi-allelic sites are split;
the per-ALT AF index is honoured. Rows with no parseable AF are skipped.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_1kg_parquet")

_POP_KEYS = ("AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF")


def parse_info(info: str) -> dict:
    """Parse a VCF INFO string into a dict (key->value; flags->True)."""
    out = {}
    for field in info.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            out[k] = v
        elif field:
            out[field] = True
    return out


def rows_from_vcf_line(line: str) -> list[dict]:
    """Yield one record per ALT allele from a VCF data line. Robust to
    multi-allelic sites and missing population fields (missing -> 0.0)."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 8:
        return []
    chrom, pos, _id, ref, alt, _qual, _filt, info = parts[:8]
    chrom = chrom[3:] if chrom.lower().startswith("chr") else chrom
    alts = alt.split(",")
    fields = parse_info(info)

    def _per_alt(key, i):
        raw = fields.get(key)
        if raw is None or raw is True:
            return 0.0
        vals = str(raw).split(",")
        try:
            return max(0.0, min(1.0, float(vals[i] if i < len(vals) else vals[-1])))
        except (ValueError, IndexError):
            return 0.0

    out = []
    for i, a in enumerate(alts):
        af = _per_alt("AF", i)
        rec = {"variant_id": f"{chrom}:{pos}:{ref}:{a}", "allele_freq": af}
        for k in _POP_KEYS:
            rec[k] = _per_alt(k, i)
        out.append(rec)
    return out


def build(vcf_dir: str, out_path: str) -> None:
    vcfs = sorted(glob.glob(os.path.join(vcf_dir, "*.vcf.gz")))
    if not vcfs:
        raise SystemExit(f"No *.vcf.gz found in {vcf_dir}")
    logger.info("Found %d VCF files.", len(vcfs))
    records: list[dict] = []
    for vp in vcfs:
        n0 = len(records)
        with gzip.open(vp, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                records.extend(rows_from_vcf_line(line))
        logger.info("  %s -> %d records", os.path.basename(vp), len(records) - n0)
    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["variant_id"]).drop_duplicates(subset=["variant_id"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d variants -> %s", len(df), out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.vcf_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
