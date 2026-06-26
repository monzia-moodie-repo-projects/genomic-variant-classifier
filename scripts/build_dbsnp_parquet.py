#!/usr/bin/env python3
r"""build_dbsnp_parquet.py -- cohort-restricted dbSNP allele-frequency parquet.

Streams the bgzipped dbSNP RefSeq VCF (e.g. GCF_000001405.40.gz, build 157) ONCE,
sequentially (no tabix/bcftools/pysam needed -- pure stdlib gzip), keeping ONLY the
variants whose chrom:pos:ref:alt key is in the cohort, and writes a parquet with the
two columns DbSNPConnector reads:

    variant_id   str    "<bare-chrom>:<pos>:<ref>:<alt>"   (e.g. "7:4781213:G:A")
    allele_freq  float  population allele frequency (0.0 if not reported)

Why this exact format: DbSNPConnector._annotate builds its join key as
    chrom (chr-stripped) + ":" + pos + ":" + ref + ":" + alt
and left-joins against the parquet's `variant_id`. The parquet MUST use the SAME
key or every join silently misses (dbsnp_af=0 everywhere). Three traps handled:
  1. dbSNP CHROM is a RefSeq accession (NC_000001.11), NOT "1" -> mapped to bare chrom.
  2. ALT is multi-allelic (A,C,G) -> split into one variant_id per ALT.
  3. FREQ is a nested multi-study string (STUDY:ref,alt1,alt2|STUDY2:...) -> parsed,
     '.' = not reported, per-ALT freq taken from the most-preferred population that
     reports it (broad pops first; never an ancestry-specific pop unless it's the
     only one present).

Modes:
  --audit-studies N   : stream the first N data lines, report the distinct FREQ study
                        names + counts, then STOP. Run this FIRST to confirm the real
                        population labels before the full ~1-3h pass.
  (default)           : full cohort-restricted build.

Usage:
  # 1. Audit study names (fast -- first 2,000,000 lines):
  python scripts/build_dbsnp_parquet.py --vcf data/external/dbsnp/GCF_000001405.40.gz \
      --clinvar data/processed/clinvar_grch38_clean.parquet --audit-studies 2000000

  # 2. Full build (cohort-restricted; ~1-3h sequential stream):
  python scripts/build_dbsnp_parquet.py --vcf data/external/dbsnp/GCF_000001405.40.gz \
      --clinvar data/processed/clinvar_grch38_clean.parquet \
      --out data/external/dbsnp/dbsnp157_cohort.parquet
"""
from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("build_dbsnp_parquet")

# RefSeq accession core -> bare chrom (GRCh38 primary assembly).
_REFSEQ = {f"NC_{i:06d}": str(i) for i in range(1, 23)}
_REFSEQ["NC_000023"] = "X"
_REFSEQ["NC_000024"] = "Y"
_REFSEQ["NC_012920"] = "MT"

# Preferred FREQ study order, COVERAGE-FIRST, using the EXACT study labels confirmed
# present in this build-157 VCF via --audit-studies (Monzia's choice 2026-06-26):
#   dbGaP_PopFreq (1.48M) > TOPMED (900K) > GnomAD_genomes (845K) >
#   GnomAD_exomes (197K) > 1000Genomes_30X (190K) > 1000Genomes (145K)
# All are broad, non-ancestry-specific populations. Any other study (KOREAN, TOMMO,
# SGDP_PRJ, Korea4K, ...) is used only as a LAST resort (when none of the preferred
# pops report a given allele), never preferentially. Labels are case- and
# suffix-exact: "GnomAD_genomes" matches, "gnomAD"/"GnomAD" would NOT.
_POP_PREFERENCE = [
    "dbGaP_PopFreq",
    "TOPMED",
    "GnomAD_genomes",
    "GnomAD_exomes",
    "1000Genomes_30X",
    "1000Genomes",
]

_FREQ_RE = re.compile(r"(?:^|;)FREQ=([^;]+)")


def accession_to_chrom(acc: str) -> str | None:
    return _REFSEQ.get(acc.split(".")[0])


def get_freq_field(info: str) -> str:
    m = _FREQ_RE.search(info)
    return m.group(1) if m else ""


def parse_freq(freq_str: str) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    if not freq_str:
        return out
    for block in freq_str.split("|"):
        if ":" not in block:
            continue
        study, freqs = block.split(":", 1)
        vals: list[float | None] = []
        for v in freqs.split(","):
            v = v.strip()
            if v in (".", ""):
                vals.append(None)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(None)
        out[study] = vals
    return out


def alt_freq_for(studies: dict[str, list[float | None]], alt_idx_1based: int,
                 preference=_POP_PREFERENCE) -> float:
    ordered = [p for p in preference if p in studies] + [s for s in studies if s not in preference]
    for study in ordered:
        vals = studies[study]
        if alt_idx_1based < len(vals):
            f = vals[alt_idx_1based]
            if f is not None:
                return f
    return 0.0


def load_cohort_keys(clinvar_path: Path) -> set[str]:
    import pandas as pd
    df = pd.read_parquet(clinvar_path, columns=["chrom", "pos", "ref", "alt"])
    chrom = df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
    keys = (chrom + ":" + df["pos"].astype(str) + ":" +
            df["ref"].astype(str) + ":" + df["alt"].astype(str))
    return set(keys.tolist())


def main(argv=None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="bgzipped dbSNP RefSeq VCF (.gz)")
    ap.add_argument("--clinvar", required=True, help="cohort parquet (chrom/pos/ref/alt)")
    ap.add_argument("--out", help="output parquet path (required unless --audit-studies)")
    ap.add_argument("--audit-studies", type=int, default=0, metavar="N",
                    help="stream first N data lines, report FREQ study names, then stop")
    ap.add_argument("--progress-every", type=int, default=5_000_000,
                    help="log progress every N lines")
    ns = ap.parse_args(argv)

    vcf = Path(ns.vcf)
    if not vcf.exists():
        print(f"VCF not found: {vcf}", file=sys.stderr)
        return 2
    clinvar = Path(ns.clinvar)
    if not clinvar.exists():
        print(f"cohort parquet not found: {clinvar}", file=sys.stderr)
        return 2

    if ns.audit_studies:
        return _audit(vcf, ns.audit_studies, ns.progress_every)

    if not ns.out:
        print("--out is required for a full build (omit only with --audit-studies).", file=sys.stderr)
        return 2

    import pandas as pd

    logger.info("Loading cohort keys from %s ...", clinvar)
    cohort = load_cohort_keys(clinvar)
    logger.info("Cohort keys: %d", len(cohort))

    t0 = time.perf_counter()
    n_lines = 0
    n_emitted = 0
    n_unmapped_chrom = 0
    seen_keys: set[str] = set()
    out_ids: list[str] = []
    out_afs: list[float] = []

    with gzip.open(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            if n_lines % ns.progress_every == 0:
                logger.info("  ... %d data lines, %d emitted (%.1f min)",
                            n_lines, n_emitted, (time.perf_counter() - t0) / 60)
            # CHROM POS ID REF ALT QUAL FILTER INFO
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom = accession_to_chrom(parts[0])
            if chrom is None:
                n_unmapped_chrom += 1
                continue
            pos, ref, alt_field, info = parts[1], parts[3], parts[4], parts[7]
            alts = alt_field.split(",")
            studies = None  # parse lazily, only if a key matches
            for i, alt in enumerate(alts, start=1):
                key = f"{chrom}:{pos}:{ref}:{alt}"
                if key in cohort and key not in seen_keys:
                    if studies is None:
                        studies = parse_freq(get_freq_field(info))
                    af = alt_freq_for(studies, i)
                    out_ids.append(key)
                    out_afs.append(af)
                    seen_keys.add(key)
                    n_emitted += 1

    dt = (time.perf_counter() - t0) / 60
    logger.info("Done streaming: %d data lines, %d emitted, %d unmapped-chrom rows, %.1f min",
                n_lines, n_emitted, n_unmapped_chrom, dt)

    out_df = pd.DataFrame({"variant_id": out_ids, "allele_freq": out_afs})
    outp = Path(ns.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(outp, index=False)
    logger.info("Wrote %s (%d rows, %.1f MB)", outp, len(out_df),
                outp.stat().st_size / 1e6)

    # Coverage report against the cohort
    cov = n_emitted / max(len(cohort), 1) * 100
    nonzero = int((out_df["allele_freq"] > 0).sum())
    logger.info("Cohort coverage: %d / %d (%.2f%%) variants matched dbSNP; %d with AF>0.",
                n_emitted, len(cohort), cov, nonzero)
    if n_emitted == 0:
        logger.error("ZERO matches -- likely a key-format mismatch (chrom map / multiallelic). "
                     "Investigate before wiring.")
        return 4
    return 0


def _audit(vcf: Path, n_max: int, progress_every: int) -> int:
    """Stream up to n_max data lines, tally FREQ study names, then stop."""
    logger.info("AUDIT: scanning up to %d data lines for FREQ study names ...", n_max)
    studies = Counter()
    chrom_acc = Counter()
    n = 0
    t0 = time.perf_counter()
    with gzip.open(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 8:
                chrom_acc[parts[0].split(".")[0]] += 1
                for block in get_freq_field(parts[7]).split("|"):
                    if ":" in block:
                        studies[block.split(":", 1)[0]] += 1
            if n % progress_every == 0:
                logger.info("  ... %d lines (%.1f min)", n, (time.perf_counter() - t0) / 60)
            if n >= n_max:
                break
    print("\n=== FREQ study names (by frequency) ===")
    for name, cnt in studies.most_common():
        print(f"  {name:20s} {cnt:>10d}")
    print("\n=== CHROM accession cores (first few) ===")
    for acc, cnt in chrom_acc.most_common(10):
        mapped = accession_to_chrom(acc + ".0")
        print(f"  {acc:16s} -> {mapped}   ({cnt} lines)")
    print(f"\nScanned {n} data lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
