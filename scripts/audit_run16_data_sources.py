#!/usr/bin/env python3
"""audit_run16_data_sources.py -- discover + verify the data files behind every
Run-16 train.py data flag, so the production flag set is grounded in fact.

STRICTLY READ-ONLY. Never full-loads a parquet (footer schema + row count only --
the 16 GiB AlphaMissense cache lesson); reads only the first few lines of TSV/VCF.
Reports, per flag: every candidate found (path, size, sanity), or NOT FOUND with
the patterns searched. Ends with a production-flag summary.

Usage:  python scripts/audit_run16_data_sources.py
Author: Monzia Moodie."""
from __future__ import annotations

import glob
import gzip
import sys
from pathlib import Path

# Each entry: the train.py flag, its role, glob patterns (relative to repo root),
# the file kind for the sanity check, optional column expectations, size floor,
# and whether the --fast smoke already exercised it.
CANDIDATES = [
    dict(flag="--clinvar (cohort)", role="REQUIRED cohort (ref/alt + ReviewStatus)",
         globs=["data/processed/clinvar_grch38_clean_seq.parquet",
                "data/**/clinvar*clean_seq*.parquet"],
         kind="parquet", need_cols=["ref", "alt", "ReviewStatus"], smoke=True),
    dict(flag="--alphamissense (TSV)", role="REQUIRED: use the TSV, NOT the scores parquet (16 GiB OOM)",
         globs=["data/external/alphamissense/AlphaMissense_hg38.tsv.gz",
                "data/**/AlphaMissense*hg38*.tsv.gz"],
         kind="tsv_gz", smoke=True),
    dict(flag="  protein-coord index", role="REQUIRED for ESM-2: full index ~18 MB (0.29 MB = corrupted)",
         globs=["data/external/alphamissense/alphamissense_protein_index.parquet",
                "data/**/*protein*index*.parquet"],
         kind="parquet", min_mb=10.0, smoke=True),
    dict(flag="--gnomad-constraint", role="REQUIRED: gene_constraint_oe (loeuf)",
         globs=["data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv",
                "data/**/*constraint_metrics*.tsv"],
         kind="tsv", smoke=True),
    dict(flag="--esm2-uniprot-index", role="REQUIRED for ESM-2",
         globs=["data/external/uniprot/uniprot_human_reviewed.parquet",
                "data/**/uniprot*reviewed*.parquet"],
         kind="parquet", smoke=True),
    dict(flag="(SpliceAI cache)", role="active: splice_ai_score (auto-loaded)",
         globs=["data/raw/cache/spliceai_scores_snv.parquet",
                "data/**/spliceai*scores*.parquet", "data/**/spliceai*index*.parquet"],
         kind="parquet", smoke=True),
    # --- the 5 NOT exercised by the smoke: the production decision ---
    dict(flag="--gnomad (allele-freq)", role="OPTIONAL: allele_freq feature (distinct from constraint)",
         globs=["data/**/gnomad*.parquet", "data/**/*allele*freq*.parquet",
                "data/**/*gnomad*af*.parquet", "data/processed/gnomad*.parquet"],
         kind="parquet", need_cols=["variant_id", "allele_freq"],
         exclude=["constraint"], smoke=False),
    dict(flag="--uniprot", role="OPTIONAL: UniProt connector (--esm2-uniprot-index is separate)",
         globs=["data/external/uniprot/*.parquet", "data/**/uniprot*.parquet"],
         kind="parquet", smoke=False),
    dict(flag="--dbnsfp-path", role="OPTIONAL: SIFT/PolyPhen/etc.",
         globs=["data/**/dbNSFP*", "data/**/dbnsfp*"], kind="auto", smoke=False),
    dict(flag="--lovd-path", role="OPTIONAL: lovd_variant_class",
         globs=["data/**/lovd*.parquet", "data/**/LOVD*", "data/**/*lovd*.parquet"],
         kind="parquet", smoke=False),
    dict(flag="--finngen-path", role="OPTIONAL: FinnGen",
         globs=["data/**/finngen*", "data/**/FinnGen*", "data/**/*finngen*"],
         kind="auto", smoke=False),
]

BACKUP_SUFFIXES = (".bak", ".OOMbak", ".pre_reviewstatus.bak", ".orig")


def _mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / (1024 * 1024), 2)
    except OSError:
        return -1.0


def _is_backup(p: Path) -> bool:
    name = p.name
    return any(name.endswith(s) for s in BACKUP_SUFFIXES) or ".bak" in name


def _sanity_parquet(p: Path, need_cols):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return "ENV: pyarrow not importable"
    try:
        schema = pq.read_schema(p)           # footer only
        cols = list(schema.names)
        try:
            nrows = pq.ParquetFile(p).metadata.num_rows  # footer only
        except Exception:
            nrows = "?"
        head = ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else "")
        msg = f"parquet: {len(cols)} cols, {nrows} rows [{head}]"
        if need_cols:
            missing = [c for c in need_cols if c not in cols]
            msg += "  cols-needed: " + ("ALL PRESENT" if not missing else f"MISSING {missing}")
        return msg
    except Exception as e:
        return f"parquet: UNREADABLE ({type(e).__name__}: {e})"


def _sanity_text(p: Path, gzipped: bool):
    try:
        opener = gzip.open if gzipped else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            lines = []
            for line in f:
                if line.startswith("##"):       # VCF meta
                    continue
                lines.append(line.rstrip("\n"))
                if len(lines) >= 3:
                    break
        if not lines:
            return "text: empty / only comments"
        first = lines[0]
        ncol = len(first.split("\t"))
        preview = first[:90] + (" ..." if len(first) > 90 else "")
        kind = "vcf" if first.startswith("#CHROM") else "tsv"
        return f"{kind}: ~{ncol} tab-cols; first row: {preview!r}"
    except Exception as e:
        return f"text: UNREADABLE ({type(e).__name__}: {e})"


def _sanity(p: Path, kind: str, need_cols):
    name = p.name.lower()
    if kind == "auto":
        if name.endswith(".parquet"):
            kind = "parquet"
        elif name.endswith(".gz") or name.endswith(".bgz"):
            kind = "tsv_gz"
        else:
            kind = "tsv"
    if kind == "parquet":
        return _sanity_parquet(p, need_cols)
    if kind == "tsv_gz":
        return _sanity_text(p, gzipped=True)
    if kind == "tsv":
        return _sanity_text(p, gzipped=False)
    return f"size-only ({_mb(p)} MB)"


def _find(globs, exclude):
    hits = []
    for pat in globs:
        for m in glob.glob(pat, recursive=True):
            mp = Path(m)
            if mp.is_file() and mp not in hits:
                if exclude and any(x.lower() in str(mp).lower() for x in exclude):
                    continue
                hits.append(mp)
    return sorted(hits, key=lambda x: str(x).lower())


def main() -> int:
    if not Path("data").is_dir():
        print("ERROR: no ./data directory -- run from the repo root.")
        return 2
    print("=" * 78)
    print(" Run-16 data-source audit (read-only)")
    print("=" * 78)
    available_optional, missing_optional, degraded = [], [], []
    for c in CANDIDATES:
        need_cols = c.get("need_cols")
        hits = _find(c["globs"], c.get("exclude"))
        live = [h for h in hits if not _is_backup(h)]
        tag = "[smoke]" if c.get("smoke") else "[NOT in smoke]"
        print(f"\n{c['flag']:<26} {tag}")
        print(f"  role: {c['role']}")
        if not hits:
            print(f"  NOT FOUND  (searched: {c['globs']})")
            if not c.get("smoke"):
                missing_optional.append(c["flag"])
            continue
        for h in hits:
            mb = _mb(h)
            bk = "  <-- BACKUP/moved-aside" if _is_backup(h) else ""
            print(f"  - {h}  ({mb} MB){bk}")
            print(f"      {_sanity(h, c['kind'], need_cols)}")
        # verdicts
        if c.get("min_mb") and live:
            small = [h for h in live if 0 <= _mb(h) < c["min_mb"]]
            if small:
                print(f"  WARN: below {c['min_mb']} MB floor -> possibly corrupted/sample: {small}")
                degraded.append(c["flag"])
        if not c.get("smoke"):
            if live:
                available_optional.append((c["flag"], live[0]))
            else:
                missing_optional.append(c["flag"])
    print("\n" + "=" * 78)
    print(" PRODUCTION-FLAG SUMMARY (the 5 sources the smoke did NOT exercise)")
    print("=" * 78)
    if available_optional:
        print(" AVAILABLE to add to Run 16:")
        for flag, p in available_optional:
            print(f"   {flag:<24} {p}  ({_mb(p)} MB)")
    if missing_optional:
        print(" NOT FOUND (stay graceful stubs unless staged):")
        for flag in missing_optional:
            print(f"   {flag}")
    if degraded:
        print(f" DEGRADED (size floor) -- investigate: {degraded}")
    print("\nNote: 'cols-needed' MISSING means the file exists but lacks the columns the")
    print("connector reads -- a wrong-file match, not a usable source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
