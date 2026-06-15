#!/usr/bin/env python3
"""build_1kg_parquet.py  --  Monzia Moodie

Build the 1000 Genomes super-population AF parquet consumed by ThousandGenomesConnector (fills
af_1kg_afr/eur/eas/sas/amr). Reworked for the GRCh38 30x high-coverage release (20220422_3202), whose
INFO uses AF_AFR/AF_EUR/AF_EAS/AF_SAS/AF_AMR (uppercase) -- NOT the GRCh37 Phase-3 AFR_AF naming and NOT
the lowercase AF_afr the old connector assumed. Key properties:

  * Multi-naming: each output column is filled from the first present INFO candidate, so the SAME builder
    works on GRCh38 high-coverage (AF_AFR), GRCh37 Phase-3 (AFR_AF), and lowercase (AF_afr) files.
  * INFO-only parse with split("\\t", 8): the genotype columns (3202 samples) are never materialised.
  * Streams from local paths OR https URLs, so the multi-GB genotype panels never need to be stored
    locally (honours the no->2GB-local rule); pair with --clinvar to cohort-filter to a small output.
  * Chunked Parquet writing (pyarrow): peak memory is bounded by --chunk-size, not the cohort.
  * Coverage gate: aborts if every super-pop AF column is all-zero (the silent-zero failure mode).

Output schema:
    variant_id  str    "chrom:pos:ref:alt" (chrom without 'chr')
    allele_freq float  global AF (INFO AF)
    AFR_AF EUR_AF EAS_AF SAS_AF AMR_AF  float  super-population AFs

Usage (stream the GRCh38 autosome panels, cohort-filter, small local output):
    python scripts/build_1kg_parquet.py \\
        --url-list data/external/1kgp/grch38_panel_urls.txt \\
        --clinvar  data/processed/clinvar_grch38_clean.parquet \\
        --out      data/external/1kgp/kg_grch38_af.parquet
Or from local files on G:\\ :
    python scripts/build_1kg_parquet.py --vcf-dir G:\\1kgp_vcf --clinvar ... --out ...
"""
from __future__ import annotations

import argparse
import glob
import gzip
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_1kg_parquet")

# output column -> ordered INFO key candidates (GRCh38 high-cov uppercase, GRCh37 Phase-3, lowercase)
_POP_OUT = ("AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF")
_POP_CANDIDATES = {
    "AFR_AF": ("AF_AFR", "AFR_AF", "AF_afr"),
    "EUR_AF": ("AF_EUR", "EUR_AF", "AF_eur"),
    "EAS_AF": ("AF_EAS", "EAS_AF", "AF_eas"),
    "SAS_AF": ("AF_SAS", "SAS_AF", "AF_sas"),
    "AMR_AF": ("AF_AMR", "AMR_AF", "AF_amr"),
}


def parse_info(info: str) -> dict:
    out = {}
    for field in info.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            out[k] = v
        elif field:
            out[field] = True
    return out


def _per_alt(keys: tuple, i: int, fields: dict) -> float:
    """First present candidate key wins; honour the per-ALT comma index; clamp to [0,1]."""
    for key in keys:
        raw = fields.get(key)
        if raw is None or raw is True:
            continue
        vals = str(raw).split(",")
        try:
            return max(0.0, min(1.0, float(vals[i] if i < len(vals) else vals[-1])))
        except (ValueError, IndexError):
            continue
    return 0.0


def rows_from_vcf_line(line: str, cohort_keys=None) -> list:
    parts = line.rstrip("\n").split("\t", 8)   # maxsplit: stop after INFO; never split 3202 genotypes
    if len(parts) < 8:
        return []
    chrom, pos, _id, ref, alt, _qual, _filt, info = parts[:8]
    chrom = chrom[3:] if chrom.lower().startswith("chr") else chrom
    alts = alt.split(",")
    fields = parse_info(info)
    out = []
    for i, a in enumerate(alts):
        vid = f"{chrom}:{pos}:{ref}:{a}"
        if cohort_keys is not None and vid not in cohort_keys:
            continue
        rec = {"variant_id": vid, "allele_freq": _per_alt(("AF",), i, fields)}
        for col in _POP_OUT:
            rec[col] = _per_alt(_POP_CANDIDATES[col], i, fields)
        out.append(rec)
    return out


def _iter_lines(src: str):
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            with gzip.GzipFile(fileobj=resp) as gz:
                for b in gz:
                    yield b.decode("utf-8", "replace")
    else:
        with gzip.open(src, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def _load_cohort_keys(clinvar_path: str) -> set:
    """Load cohort keys as bare 'chrom:pos:ref:alt', mirroring ThousandGenomesConnector's join key EXACTLY
    (built from chrom/pos/ref/alt with a leading 'chr' stripped) so the filter and the downstream join
    agree. Falls back to the variant_id column, stripping the 'clinvar:' prefix and any 'chr'."""
    df = pd.read_parquet(clinvar_path)
    if all(c in df.columns for c in ("chrom", "pos", "ref", "alt")):
        ch = df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
        keys = ch + ":" + df["pos"].astype(str) + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    elif "variant_id" in df.columns:
        keys = (df["variant_id"].astype(str)
                .str.replace(r"^clinvar:", "", regex=True)
                .str.replace(r"^chr", "", regex=True))
    else:
        raise SystemExit("--clinvar parquet needs (chrom,pos,ref,alt) or 'variant_id'")
    s = set(keys)
    logger.info("Cohort filter: %d unique variant_ids loaded from %s", len(s), clinvar_path)
    return s


def build(sources: list, out_path: str, cohort_keys=None, chunk_size: int = 2_000_000) -> None:
    cols = ["variant_id", "allele_freq", *_POP_OUT]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    nonzero = {c: 0 for c in _POP_OUT}
    buf: list = []

    def flush():
        nonlocal writer, total, buf
        if not buf:
            return
        df = pd.DataFrame.from_records(buf)
        df = df.dropna(subset=["variant_id"]).drop_duplicates(subset=["variant_id"])
        df = df[cols].astype({c: "float64" for c in cols[1:]})
        for c in _POP_OUT:
            nonzero[c] += int((df[c] > 0).sum())
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total += len(df)
        buf = []

    try:
        for src in sources:
            for line in _iter_lines(src):
                if line.startswith("#"):
                    continue
                buf.extend(rows_from_vcf_line(line, cohort_keys))
                if len(buf) >= chunk_size:
                    flush()
            flush()
            logger.info("  %s -> running total %d (kept)", os.path.basename(src.rstrip('/')), total)
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        raise SystemExit("No records written (cohort filter matched nothing, or no parseable lines).")
    if all(nonzero[c] == 0 for c in _POP_OUT):
        raise SystemExit(
            "COVERAGE GATE FAILED: every super-pop AF column is all-zero -- the INFO field names did not "
            "match any candidate. Inspect the VCF header (inspect_1kg_header.py) and extend _POP_CANDIDATES."
        )
    logger.info("Wrote %d variants -> %s", total, out_path)
    logger.info("Non-zero super-pop AF counts: %s", nonzero)


def _sources(args) -> list:
    if args.url_list:
        return [u.strip() for u in Path(args.url_list).read_text().splitlines() if u.strip() and not u.startswith("#")]
    if args.vcf_urls:
        return list(args.vcf_urls)
    if args.vcf_dir:
        v = sorted(glob.glob(os.path.join(args.vcf_dir, "*.vcf.gz")))
        if not v:
            raise SystemExit(f"No *.vcf.gz found in {args.vcf_dir}")
        return v
    raise SystemExit("provide one of --vcf-dir / --vcf-urls / --url-list")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf-dir", help="local dir of *.vcf.gz (e.g. on G:\\)")
    ap.add_argument("--vcf-urls", nargs="*", help="https URLs to stream (no local storage)")
    ap.add_argument("--url-list", help="text file of https URLs, one per line")
    ap.add_argument("--clinvar", help="cohort parquet; if given, keep only cohort variant_ids (small output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    args = ap.parse_args()
    keys = _load_cohort_keys(args.clinvar) if args.clinvar else None
    build(_sources(args), args.out, cohort_keys=keys, chunk_size=args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
