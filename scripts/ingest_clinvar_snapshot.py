#!/usr/bin/env python
"""ingest_clinvar_snapshot.py (2026-07-09)

Single-purpose, drift-aware, fail-loud replica of ClinVarConnector.fetch() (the local-file
path in src/genomic_variant_classifier/data/database_connectors.py) that turns a raw ClinVar
variant_summary.txt.gz into the canonical 16-column processed parquet -- WITHOUT the caching,
the network download, or the side effect of writing to the canonical data/processed path.

WHY THIS EXISTS (rather than re-running the connector / run_phase2_eval.py):
  * The connector writes only via ClinVarConnector().fetch(...).to_parquet('data/processed/
    clinvar_grch38.parquet') -- an implicit, canonical-path write. Re-running it would risk
    clobbering the protected stale snapshot. This tool writes ONLY to an explicit --output.
  * The connector's df.rename() SILENTLY ignores columns absent from the input. On the 2026-07
    variant_summary, the 'ProteinChange' column was REMOVED by NCBI, so the connector would
    silently emit an all-None protein_change with no warning. This tool instead ASSERTS every
    rename-source column and LOUDLY reports any that are missing, recording them in a manifest.

FIDELITY: for every column that IS present, this reproduces the connector byte-for-byte:
    - pd.read_csv(sep='\t', low_memory=False), DEFAULT NaN handling (so ClinVar 'na' alleles
      become NaN/None exactly as the connector produces). NO dtype=str.
    - df[df['Assembly'] == assembly].copy()   (exact-case 'GRCh38')
    - the identical rename map, derived columns, metadata dict, variant_id concat, and the
      _to_canonical 16-column reindex.
NO allele/date/significance filtering (the connector is the permissive raw ingest; all
filtering happens downstream). Result is deterministic: same input -> identical md5.

EMPTY-ALLELE NORMALIZATION (Option A, chosen 2026-07-09):
  ClinVar's empty-allele token changed between the 2026-03 snapshot (uppercase 'NA', which
  pandas reads as null -> stored as None) and the 2026-07 snapshot (lowercase 'na', which
  pandas does NOT treat as null -> read as the literal string 'na'). To keep the fresh cohort
  comparable to the stale one AND to close a real gap (the canonical is_empty_allele omits the
  '-' token), this ingestion normalizes the FULL empty vocabulary {'', na, nan, none, '.', '-'}
  (case-insensitive, whitespace-stripped) to Python None before building variant_id. This means
  the fresh cohort quarantines allele-less rows identically to the stale cohort, and additionally
  catches '-' rows the classifier would otherwise miss. NOTE (pandas version): under pandas 2.x
  (the deployment target) an object column of None survives parquet round-trip as None, matching
  the stale parquet; under pandas 3.x the default string dtype may re-read it as NaN. Both are
  treated identically by is_empty_allele (both -> empty), so quarantine is correct regardless;
  any stale-vs-fresh diff MUST compare alleles via a normalized (is_empty_allele) view, never raw
  equality, so a None-vs-NaN-vs-'na' representation difference never registers as a real change.

Every acronym is expanded on first use: National Center for Biotechnology Information (NCBI),
Message-Digest-5 (MD5), Genome Reference Consortium Human build 38 (GRCh38), ClinVar Variation
Identifier (VariationID), Human Genome Variation Society protein notation (HGVSp).
"""
from __future__ import annotations
import sys, os, gzip, json, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

print("=== ingest_clinvar_snapshot START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

# The canonical 16 columns, in order -- identical to database_connectors.CANONICAL_COLUMNS.
CANONICAL_COLUMNS = [
    "variant_id", "source_db", "chrom", "pos", "ref", "alt", "gene_symbol",
    "transcript_id", "consequence", "pathogenicity", "allele_freq", "clinical_sig",
    "protein_change", "fasta_seq", "source_id", "metadata",
]

# Empty-allele token vocabulary. The stale 2026-03 parquet stored empties as Python None
# (uppercase 'NA' was a pandas default-NaN token then). The 2026-07 variant_summary uses
# lowercase 'na' (NOT a pandas default token) plus occasional '-'. To keep the fresh cohort's
# encoding IDENTICAL to the stale cohort's (so a stale-vs-fresh diff reflects real ClinVar
# changes, not a token-case artifact), we normalize the full empty vocabulary to None here,
# case-insensitively and whitespace-stripped. This mirrors how the canonical allele classifier
# treats empties, and additionally covers '-' (which the classifier's _NULL_TOKENS omits).
_EMPTY_ALLELE_TOKENS = frozenset({"", "na", "nan", "none", ".", "-"})


def _normalize_allele(v: object) -> object:
    """Return None for any empty-allele token (case-insensitive, stripped); else the value."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if str(v).strip().lower() in _EMPTY_ALLELE_TOKENS:
        return None
    return v


# The connector's rename map (variant_summary column -> canonical name).
RENAME_MAP = {
    "GeneSymbol":           "gene_symbol",
    "ClinicalSignificance": "clinical_sig",
    "Chromosome":           "chrom",
    "Start":                "pos",
    "ReferenceAllele":      "ref",
    "AlternateAllele":      "alt",
    "ProteinChange":        "protein_change",
    "VariationID":          "source_id",
    "RS# (dbSNP)":          "rs_id",
}
# Columns the connector also reads without renaming.
ALSO_READ = ["Assembly", "ReviewStatus"]


def _map_pathogenicity(sig: object) -> str:
    """Map a raw ClinVar clinical significance string to a 5-class pathogenicity label.

    This mirrors ClinVarConnector._map_pathogenicity EXCEPT for one deliberate, documented
    correction (2026-07-10): ClinVar's aggregate status 'Conflicting classifications of
    pathogenicity' (and its '; modifier' compound forms, and 'conflicting data from
    submitters') means submitters DISAGREE -- it is NOT a confident pathogenic call. The
    original connector fell through to the 'if pathogenic in s' substring branch (because the
    phrase contains the substring 'pathogenicity') and mislabeled ~161K rows as 'pathogenic'.
    We add an EARLY guard, before any substring match, that maps anything whose normalized text
    starts with 'conflicting' to 'uncertain'. This is the single correction; every other string
    is handled exactly as the original connector did. The canonical connector must receive the
    identical guard so the two stay in lockstep.
    """
    if not isinstance(sig, str) or not sig.strip():
        return "uncertain"
    s = sig.lower().strip()
    # Conflicting aggregate status = submitters disagree = not a confident call -> uncertain.
    # Must precede the 'pathogenic' substring checks below, which would otherwise misfire on the
    # 'pathogenicity' substring inside 'Conflicting classifications of pathogenicity'.
    if s.startswith("conflicting"):
        return "uncertain"
    if s.startswith("pathogenic"):
        return "pathogenic"
    if s.startswith("benign"):
        return "benign"
    if "likely pathogenic" in s:
        return "likely_pathogenic"
    if "likely benign" in s:
        return "likely_benign"
    if "pathogenic" in s:
        return "pathogenic"
    if "benign" in s:
        return "benign"
    return "uncertain"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _header_columns(path: Path) -> list[str]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n")
    return [c.lstrip("#") for c in first.split("\t")]


def _empty_rate(series) -> float:
    """Fraction of values that are null or an empty-allele token."""
    if len(series) == 0:
        return 1.0
    n = 0
    for v in series:
        if _normalize_allele(v) is None:
            n += 1
    return n / len(series)


def _resolve_allele_source(df: pd.DataFrame) -> str:
    """Decide, from ACTUAL column population (not header presence), where the alleles live.
    Returns 'legacy' (ReferenceAllele/AlternateAllele) or 'vcf' (ReferenceAlleleVCF/
    AlternateAlleleVCF). Raises if neither column pair carries alleles.

    Why population, not header: the 2026-07 variant_summary HAS a ReferenceAllele column but it
    is ~100% 'na' (NCBI deprecated it and moved alleles to the *VCF columns). Deciding on header
    presence alone silently produced an all-null cohort. We sample the actual data instead.
    """
    have_legacy = "ReferenceAllele" in df.columns and "AlternateAllele" in df.columns
    have_vcf = "ReferenceAlleleVCF" in df.columns and "AlternateAlleleVCF" in df.columns
    sample = df.head(20000)
    legacy_empty = _empty_rate(sample["ReferenceAllele"]) if have_legacy else 1.0
    vcf_empty = _empty_rate(sample["ReferenceAlleleVCF"]) if have_vcf else 1.0
    if have_legacy and legacy_empty <= 0.5:
        return "legacy"
    if have_vcf and vcf_empty <= 0.5:
        return "vcf"
    raise ValueError(
        f"NO USABLE ALLELE COLUMN: ReferenceAllele empty-rate={legacy_empty:.3f}, "
        f"ReferenceAlleleVCF empty-rate={vcf_empty:.3f} (sample of {len(sample):,} rows). "
        f"Neither carries alleles. Refusing to emit an all-null cohort. Inspect the "
        f"variant_summary schema.")


def ingest(input_path: Path, assembly: str = "GRCh38") -> tuple[pd.DataFrame, dict]:
    """Return (canonical_df, manifest). Fail loud on missing rename-source columns
    EXCEPT ones explicitly known-removed upstream (ProteinChange), which are recorded
    as known drift and emitted as an all-None canonical column."""
    manifest: dict = {
        "tool": "ingest_clinvar_snapshot.py",
        "input": str(input_path),
        "input_md5": _md5(input_path),
        "utc": datetime.now(timezone.utc).isoformat(),
        "assembly": assembly,
    }
    header = _header_columns(input_path)
    manifest["input_columns"] = header
    manifest["input_column_count"] = len(header)

    # Drift check: which rename-source columns are present / missing?
    KNOWN_REMOVED = {"ProteinChange"}  # NCBI removed the dedicated column from variant_summary
    missing = [c for c in RENAME_MAP if c not in header]
    missing_also = [c for c in ALSO_READ if c not in header]
    manifest["missing_rename_sources"] = missing
    manifest["missing_also_read"] = missing_also

    hard_missing = [c for c in missing if c not in KNOWN_REMOVED] + missing_also
    if hard_missing:
        # FAIL LOUD -- never silently proceed to an all-None canonical column.
        raise ValueError(
            f"SCHEMA DRIFT (unexpected): required source columns absent from "
            f"{input_path.name}: {hard_missing}. The connector's rename would silently "
            f"produce all-None outputs for these. Refusing to proceed. Investigate the "
            f"variant_summary schema before ingesting.")
    for c in missing:
        if c in KNOWN_REMOVED:
            print(f"  KNOWN DRIFT: source column '{c}' absent (removed upstream by NCBI). "
                  f"Its canonical target '{RENAME_MAP[c]}' will be all-None for this snapshot "
                  f"and must be EXCLUDED from any stale-vs-fresh diff.", flush=True)

    # Read exactly as the connector does: default NaN handling, no dtype coercion.
    op = gzip.open if str(input_path).endswith(".gz") else open
    print(f"  reading {input_path} (pd.read_csv sep=tab low_memory=False, default NaN) ...", flush=True)
    with op(input_path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)
    manifest["rows_raw"] = int(len(df))

    df = df[df["Assembly"] == assembly].copy()
    manifest["rows_after_assembly_filter"] = int(len(df))

    # Resolve which columns actually carry the alleles (population-based, not header-based).
    allele_source = _resolve_allele_source(df)
    manifest["allele_source"] = allele_source
    print(f"  allele source: {allele_source} "
          f"({'ReferenceAllele/AlternateAllele' if allele_source=='legacy' else 'ReferenceAlleleVCF/AlternateAlleleVCF'})",
          flush=True)

    # Build the effective rename map. pos ALWAYS comes from Start (matches the stale parquet's
    # coordinate convention, so the builder's padded-deletion pos-=1 correction applies to both
    # snapshots identically). Alleles come from whichever source is populated.
    effective_rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if allele_source == "vcf":
        # override ref/alt to come from the *VCF columns; drop the (empty) legacy mappings
        effective_rename.pop("ReferenceAllele", None)
        effective_rename.pop("AlternateAllele", None)
        effective_rename["ReferenceAlleleVCF"] = "ref"
        effective_rename["AlternateAlleleVCF"] = "alt"
    df = df.rename(columns=effective_rename)

    # Normalize empty-allele tokens to None so the fresh encoding matches the stale parquet.
    # Cast to object first: under pandas 3.0 StrDtype, .map returning None is re-coerced to <NA>;
    # object dtype preserves Python None exactly as the stale parquet stores it.
    # Build a genuine object-dtype Series via list comprehension: under pandas 3.0, both .map
    # and .astype(object) on a StrDtype column re-coerce Python None back to the string-NA, so
    # we materialize a plain Python list first, normalize, then rebuild as object dtype.
    if "ref" in df.columns:
        df["ref"] = pd.Series([_normalize_allele(v) for v in df["ref"].tolist()],
                              index=df.index, dtype=object)
    if "alt" in df.columns:
        df["alt"] = pd.Series([_normalize_allele(v) for v in df["alt"].tolist()],
                              index=df.index, dtype=object)

    df["source_db"]     = "clinvar"
    df["pathogenicity"] = df["clinical_sig"].apply(_map_pathogenicity)
    df["allele_freq"]   = None
    df["fasta_seq"]     = None
    df["transcript_id"] = None
    df["consequence"]   = None
    # ReviewStatus present (checked); rs_id present only if RS# (dbSNP) existed.
    rs_series = df["rs_id"] if "rs_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    rev_series = df["ReviewStatus"] if "ReviewStatus" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["metadata"] = [
        {"rs_id": rs, "review_status": rev}
        for rs, rev in zip(rs_series, rev_series)
    ]
    def _s(v):
        return "None" if v is None else str(v)
    df["variant_id"] = (
        "clinvar:" + df["chrom"].map(_s) + ":" +
        df["pos"].map(_s) + ":" +
        df["ref"].map(_s) + ":" +
        df["alt"].map(_s)
    )

    # ALL-NULL TRIPWIRE: if >50% of rows have BOTH ref and alt null, the allele source was
    # wrong (the exact corruption that produced the all-na:na fresh parquet). Fail loud; never
    # write a corrupt cohort. Real snapshots have <1% allele-less rows.
    _ref_null = df["ref"].map(lambda v: v is None)
    _alt_null = df["alt"].map(lambda v: v is None)
    _nana_rate = float((_ref_null & _alt_null).mean()) if len(df) else 1.0
    manifest["nana_rate"] = _nana_rate
    if _nana_rate > 0.5:
        raise ValueError(
            f"ALL-NULL TRIPWIRE: {100*_nana_rate:.1f}% of rows have both alleles null "
            f"(allele_source={allele_source}). This is the corruption signature of reading a "
            f"deprecated/empty allele column. Refusing to write. Expected allele-less rate <1%.")

    # _to_canonical: ensure every canonical column exists (fills missing with None), reindex.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    result = df[CANONICAL_COLUMNS].copy()
    manifest["ref_empty_normalized"] = int(result["ref"].isna().sum())
    manifest["alt_empty_normalized"] = int(result["alt"].isna().sum())
    manifest["rows_out"] = int(len(result))
    manifest["out_columns"] = list(result.columns)
    manifest["protein_change_all_null"] = bool(result["protein_change"].isna().all())
    return result, manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw variant_summary.txt.gz")
    ap.add_argument("--output", required=True, help="explicit output parquet path (never the canonical path unless you name it)")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--manifest", default=None, help="optional manifest json path (default: <output>.manifest.json)")
    ap.add_argument("--force", action="store_true", help="overwrite --output if it exists")
    a = ap.parse_args(argv)

    inp = Path(a.input)
    out = Path(a.output)
    if not inp.exists():
        print(f"FATAL: input not found: {inp}", flush=True); return 2
    if out.exists() and not a.force:
        print(f"REFUSING to overwrite existing {out} without --force.", flush=True); return 3
    # Guard: refuse to write the protected canonical stale parquet unless explicitly forced.
    canonical = Path("data/processed/clinvar_grch38.parquet")
    try:
        same = out.resolve() == canonical.resolve()
    except Exception:
        same = str(out).replace("\\", "/").endswith("data/processed/clinvar_grch38.parquet")
    if same and not a.force:
        print("REFUSING to write the canonical data/processed/clinvar_grch38.parquet "
              "(protected stale snapshot). Choose a different --output or pass --force "
              "if you truly intend to replace it.", flush=True)
        return 4

    result, manifest = ingest(inp, a.assembly)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    manifest["output"] = str(out)
    manifest["output_md5"] = _md5(out)

    mpath = Path(a.manifest) if a.manifest else out.with_suffix(out.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\n  rows raw               : {manifest['rows_raw']:,}", flush=True)
    print(f"  rows after GRCh38 filter: {manifest['rows_after_assembly_filter']:,}", flush=True)
    print(f"  rows out               : {manifest['rows_out']:,}", flush=True)
    print(f"  out columns            : {len(manifest['out_columns'])}", flush=True)
    print(f"  protein_change all-null: {manifest['protein_change_all_null']}", flush=True)
    print(f"  output                 : {out}", flush=True)
    print(f"  output md5             : {manifest['output_md5']}", flush=True)
    print(f"  manifest               : {mpath}", flush=True)
    print("=== ingest_clinvar_snapshot DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
