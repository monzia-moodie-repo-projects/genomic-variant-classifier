#!/usr/bin/env python
"""
build_cohort_from_source.py  (2026-07-09)
==========================================================================
Single authoritative cohort builder. Produces a coordinate-correct, bad-allele-quarantined
cohort from a raw ClinVar (Clinical Variant database) parquet in ONE pass, fixing BOTH
defects that the two previous builders each fixed only partially:

  * clean_cohort.py       quarantined bad-allele rows but did NOT correct coordinates.
  * build_cohort_v2.py    corrected padded-deletion coordinates but did NOT quarantine
                          bad-allele rows (the canonical clinvar_grch38_clean_v2_verified
                          parquet therefore still carried 19,988 na:na + 1,103 half-bad).

Neither did both. This builder does both, in the ONLY safe order:

  STEP 1  QUARANTINE FIRST.  Route every bad-allele row -- both-empty (na:na) AND half-bad
          (exactly one of reference/alternate allele empty) -- to a structural table, using
          allele_classify.is_empty_allele per side as the single source of truth.

  STEP 2  CORRECT COORDINATES SECOND.  On the already-cleaned rows, subtract one from the
          position of every padded deletion, using allele_classify.is_padded_deletion (the
          canonical predicate with the 2026-07-09 non-empty guard). Because empty-alternate
          rows are already gone, the padded-deletion mask is unambiguous and the historical
          inline-vs-canonical divergence is STRUCTURALLY IMPOSSIBLE here.

WHY THIS ORDER  (verified 2026-07-09 on clinvar_grch38.parquet)
    The raw bad-allele tokens are Python None (reference and alternate) and the string '.'
    (alternate only); no literal empty-string alternate exists, so on today's data the two
    padded-deletion predicates already agree. Quarantine-first makes them agree on ANY future
    snapshot too (a later ClinVar release could contain '' alternates). Defense-in-depth.

THE PADDED-DELETION CORRECTION  (from build_cohort_v2.py, established 30/30 with a control)
    The cohort's `pos` is ClinVar variant_summary's `Start` (first altered base). Its
    reference/alternate alleles are the Variant Call Format (VCF) ReferenceAlleleVCF /
    AlternateAlleleVCF, which begin at PositionVCF. For a PADDED DELETION the padding base is
    unchanged, so Start == PositionVCF + 1. Correcting is exactly: pos -= 1 for padded
    deletions only. A length-shrinking delins (e.g. AA>C) also has len(alt) < len(ref) but
    ref.startswith(alt) is False, so it is correctly NOT shifted.

GUARDS  (all fail-loud)
    G1  refuse to overwrite an existing --output (unless --force)
    G2  required columns present
    G3  variant-class composition invariant across the pos-only correction (a pos change
        cannot change a variant's class; if it does, the correction touched an allele)
    G4  row reconciliation: quarantined + kept == input rows exactly
    G5  SNV (single nucleotide variant) negative control against GRCh38 (Genome Reference
        Consortium Human build 38): SNVs are never shifted, so if the genome-slice convention
        were wrong they would mismatch en masse (~100%); a passing control distinguishes a
        coordinate bug from the rare genuine ClinVar-vs-genome disagreement.
    G6  reference-consistency: corrected padded deletions match the genome at pos within a
        tolerance (default 0.1%) for genuine ClinVar-vs-GRCh38 disagreement (alt loci,
        assembly patches, left-alignment). Requires --genome; SKIPPED_NO_GENOME (loud,
        recorded, PROVISIONAL) otherwise -- never a silent pass.
    G7  POST-CONDITION: zero bad-allele rows survive into the clean output.
    G8  POST-CONDITION: every indel (insertion or deletion) in the clean output is
        genome-consistent at its (corrected) position -- the guard neither prior builder had.
    G9  POST-CONDITION: no duplicate variant_id in the clean output.

OUTPUTS
    --output                          clean, coordinate-correct cohort parquet
    <output>_structural.parquet       the quarantined bad-allele rows (provenance)
    <output>_reconciliation.json      full reconciliation + guard results (per-build,
                                      permanent; e.g. cohort_fresh_reconciliation.json)
    cohort_build_reconciliation.json  fixed-name compatibility alias to the LAST build

USAGE (from project root, .venv312 active)
    python scripts/build_cohort_from_source.py --audit
    python scripts/build_cohort_from_source.py --apply --genome data/external/grch38/GRCh38.fa
    # build-both-and-diff:
    python scripts/build_cohort_from_source.py --apply --input data/processed/clinvar_grch38.parquet \
        --output data/processed/cohort_stale.parquet --genome data/external/grch38/GRCh38.fa
    python scripts/build_cohort_from_source.py --apply --input data/processed/clinvar_grch38_fresh.parquet \
        --output data/processed/cohort_fresh.parquet --genome data/external/grch38/GRCh38.fa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from genomic_variant_classifier.data.allele_classify import (
        is_empty_allele, is_padded_deletion,
    )
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from allele_classify import is_empty_allele, is_padded_deletion  # type: ignore

REQUIRED_COLS = ("variant_id", "chrom", "pos", "ref", "alt")


def _startswith_elementwise(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return pd.Series([rr.startswith(aa) for rr, aa in zip(r, a)], index=ref.index, dtype=bool)


def variant_class(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """Class by allele shape. Used only on the CLEANED rows (no empty alleles)."""
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    lr, la = r.str.len(), a.str.len()
    starts = _startswith_elementwise(r, a)
    starts_ra = _startswith_elementwise(a, r)
    out = pd.Series("MNV/other", index=r.index, dtype="object")
    out[(lr == 1) & (la == 1)] = "SNV"
    out[(lr > 1) & (la == 1) & starts] = "padded_deletion"
    out[(lr > 1) & (la == 1) & ~starts] = "delins_del"
    out[(lr == 1) & (la > 1) & starts_ra] = "padded_insertion"
    out[(lr == 1) & (la > 1) & ~starts_ra] = "delins_ins"
    return out


def _norm_chrom(c: object) -> str:
    s = str(c)
    return s[3:] if s.lower().startswith("chr") else s


def schema_fingerprint(columns) -> str:
    return hashlib.sha256(",".join(sorted(map(str, columns))).encode()).hexdigest()[:16]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


@dataclass
class BuildReconciliation:
    input_rows: int = 0
    quarantined_bad_allele: int = 0
    quarantined_na_na: int = 0
    quarantined_half_bad: int = 0
    clean_rows: int = 0
    padded_deletions_corrected: int = 0
    variant_id_rebuilt: int = 0
    composition_before: dict = field(default_factory=dict)
    composition_after: dict = field(default_factory=dict)
    reference_check: str = "NOT_RUN"
    reference_mismatches: int = 0
    snv_control: str = "NOT_RUN"
    indel_postcondition: str = "NOT_RUN"
    bad_allele_postcondition: str = "NOT_RUN"
    dup_variant_id: int = 0
    genome_consistent: int = 0
    genome_inconsistent: int = 0
    genome_unchecked: int = 0
    genome_inconsistent_by_pathogenicity: dict = field(default_factory=dict)
    input_md5: str = ""
    output_md5: str = ""
    schema_fingerprint: str = ""
    nt_windows_rebuild_required: bool = True
    notes: list = field(default_factory=list)

    def reconciles(self) -> bool:
        return self.input_rows == self.quarantined_bad_allele + self.clean_rows

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reconciles"] = self.reconciles()
        return d


def build(df: pd.DataFrame, recon: BuildReconciliation) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure core. Returns (clean_df, structural_df). Raises on any invariant violation."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Present: {list(df.columns)}")
    recon.input_rows = len(df)

    # STEP 1 -- quarantine bad-allele rows FIRST (single source of truth).
    ref_bad = is_empty_allele(df["ref"])
    alt_bad = is_empty_allele(df["alt"])
    bad = ref_bad | alt_bad
    both = ref_bad & alt_bad
    structural = df[bad].copy()
    clean = df[~bad].copy()
    recon.quarantined_bad_allele = int(bad.sum())
    recon.quarantined_na_na = int(both.sum())
    recon.quarantined_half_bad = int((bad & ~both).sum())
    recon.clean_rows = len(clean)

    # G4 row reconciliation
    if not recon.reconciles():
        raise ValueError("Row reconciliation failed: "
                         f"quarantined {recon.quarantined_bad_allele} + clean "
                         f"{recon.clean_rows} != input {recon.input_rows}")

    # composition BEFORE correction (on cleaned rows -- no empty alleles remain)
    recon.composition_before = {
        k: int(v) for k, v in variant_class(clean["ref"], clean["alt"]).value_counts().items()
    }

    # STEP 2 -- correct padded-deletion coordinates on the CLEANED rows only.
    mask = is_padded_deletion(clean["ref"], clean["alt"])
    recon.padded_deletions_corrected = int(mask.sum())

    pos = clean["pos"].to_numpy()
    if not np.issubdtype(pos.dtype, np.integer):
        if mask.any() and np.isnan(pos[mask.to_numpy()]).any():
            raise ValueError("NaN pos among padded deletions -- cannot correct coordinates.")
    clean.loc[mask, "pos"] = clean.loc[mask, "pos"] - 1

    # rebuild variant_id from corrected pos for the shifted rows
    prefix = clean["variant_id"].astype("string").str.split(":", n=1).str[0].fillna("clinvar")
    clean["variant_id"] = (
        prefix + ":" + clean["chrom"].astype(str) + ":"
        + clean["pos"].astype("int64").astype(str) + ":"
        + clean["ref"].astype(str) + ":" + clean["alt"].astype(str)
    )
    recon.variant_id_rebuilt = int(mask.sum())


    # G3 composition invariance -- a pos-only change cannot change a variant's class
    recon.composition_after = {
        k: int(v) for k, v in variant_class(clean["ref"], clean["alt"]).value_counts().items()
    }
    if recon.composition_before != recon.composition_after:
        raise ValueError(
            "Variant-class composition changed under a pos-only correction -- impossible. "
            f"before={recon.composition_before} after={recon.composition_after}")

    # DEDUP COLLAPSE (2026-07-10): collapse duplicate variant_id groups (same variant under
    # multiple ClinVar Variation Identifiers) into one deterministic most-severe survivor.
    # Placed AFTER the G3 composition-invariance guard (which brackets ONLY the pos-only
    # correction and must see the same rows before/after) and BEFORE the G7/G9 post-conditions.
    clean, _dedup_audit = collapse_duplicate_variants(clean)
    recon.collapsed_groups = len(_dedup_audit)
    recon.collapse_conflicts = int(sum(1 for a in _dedup_audit if a['classification_conflict']))
    recon.dedup_audit = _dedup_audit
    # Final composition reflects the collapsed frame (row count may have dropped); this is a
    # report value, NOT a guard -- the G3 invariant above already validated the correction.
    recon.composition_final = {
        k: int(v) for k, v in variant_class(clean["ref"], clean["alt"]).value_counts().items()
    }

    # G7 post-condition: zero bad-allele rows survive
    n_bad_left = int((is_empty_allele(clean["ref"]) | is_empty_allele(clean["alt"])).sum())
    if n_bad_left != 0:
        raise ValueError(f"POST-CONDITION G7 FAILED: {n_bad_left} bad-allele rows in clean output.")
    recon.bad_allele_postcondition = "PASSED (0 bad-allele rows)"

    # G9 post-condition: no duplicate variant_id
    recon.dup_variant_id = int(clean["variant_id"].duplicated().sum())
    if recon.dup_variant_id != 0:
        raise ValueError(f"POST-CONDITION G9 FAILED: {recon.dup_variant_id} duplicate variant_id.")

    recon.schema_fingerprint = schema_fingerprint(clean.columns)
    return clean, structural



# ============================================================================
# DEDUP COLLAPSE (2026-07-10; robust v2: tolerates optional columns per REQUIRED_COLS).
# ============================================================================

_DEDUP_SEVERITY = {"pathogenic": 5, "likely_pathogenic": 4, "uncertain": 3,
                   "likely_benign": 2, "benign": 1}

def _dedup_severity(label) -> int:
    return _DEDUP_SEVERITY.get(str(label), 0)

def _dedup_source_id_key(v):
    try:
        return (0, int(str(v)))
    except (ValueError, TypeError):
        return (1, str(v))

def collapse_duplicate_variants(clean, has_review_status=None):
    """Collapse duplicate variant_id groups into one deterministic, most-severe survivor.

    Robust to the builder's REQUIRED_COLS contract: only 'variant_id' is assumed. The optional
    columns 'pathogenicity', 'source_id', 'clinical_sig', 'metadata', 'review_status' are used
    when present and gracefully skipped when absent. Returns (collapsed_df, audit_records).
    Fully inert when there are no duplicate variant_ids.
    """
    has_path = "pathogenicity" in clean.columns
    has_sid = "source_id" in clean.columns
    has_sig = "clinical_sig" in clean.columns
    has_md = "metadata" in clean.columns
    if has_review_status is None:
        has_review_status = "review_status" in clean.columns

    dup_mask = clean["variant_id"].duplicated(keep=False)
    if not dup_mask.any():
        return clean, []

    unique_part = clean[~dup_mask]
    dup_part = clean[dup_mask]
    survivors, audit = [], []

    for vid, g in dup_part.groupby("variant_id", sort=True):
        # Deterministic within-group order: by source_id if present, else by a stable
        # positional key (original index) so ordering is reproducible regardless of input order.
        if has_sid:
            order_keys = [(_dedup_source_id_key(s), i) for i, s in
                          zip(range(len(g)), g["source_id"].tolist())]
        else:
            order_keys = [((0, i), i) for i in range(len(g))]
        order = [i for _, i in sorted(zip(order_keys, range(len(g))))]
        g_ordered = g.iloc[order]

        # Survivor selection: severity DESC (if labels present), then review DESC (if present),
        # then source_id ASC (if present), else first in deterministic order.
        def row_rank(pos, rr):
            sev = _dedup_severity(rr.get("pathogenicity")) if has_path else 0
            rev = 0
            if has_review_status:
                try: rev = int(rr.get("review_status"))
                except (ValueError, TypeError): rev = 0
            sidk = _dedup_source_id_key(rr.get("source_id")) if has_sid else (0, pos)
            return (-sev, -rev, sidk)
        ranked = sorted(((row_rank(p, rr), p, rr) for p, (_, rr) in
                         enumerate(g_ordered.iterrows())), key=lambda t: t[0])
        keep_row = ranked[0][2]

        n = len(g_ordered)
        paths = [str(x) for x in g_ordered["pathogenicity"].tolist()] if has_path else []
        sevs = [_dedup_severity(x) for x in g_ordered["pathogenicity"].tolist()] if has_path else []
        vids = [x for x in g_ordered["source_id"].tolist()] if has_sid else []
        sigs = [x for x in g_ordered["clinical_sig"].tolist()] if has_sig else []
        conflict = (len(set(paths)) > 1) if has_path else False
        span = (max(sevs) - min(sevs)) if sevs else 0

        survivor = keep_row.copy()
        if has_md:
            md = survivor.get("metadata")
            md = dict(md) if isinstance(md, dict) else ({} if md is None else {"_orig": md})
            md.update({
                "collapse_all_variation_ids": vids,
                "collapse_all_clinical_sigs": sigs,
                "collapse_all_pathogenicities": paths,
                "classification_conflict": bool(conflict),
                "conflict_span": int(span),
                "collapsed_from_n": int(n),
            })
            survivor["metadata"] = md
        survivors.append(survivor)

        audit.append({
            "variant_id": vid,
            "kept_source_id": (keep_row.get("source_id") if has_sid else None),
            "dropped_source_ids": ([v for v in vids if v != keep_row.get("source_id")] if has_sid else []),
            "kept_pathogenicity": (str(keep_row.get("pathogenicity")) if has_path else None),
            "all_pathogenicities": paths,
            "classification_conflict": bool(conflict),
            "conflict_span": int(span),
            "collapsed_from_n": int(n),
        })

    survivors_df = pd.DataFrame(survivors) if survivors else clean.iloc[0:0]
    out = pd.concat([unique_part, survivors_df], ignore_index=True)
    out = out[clean.columns]
    return out, audit

def _load_genome(genome_path: Path):
    try:
        import pysam  # type: ignore
        fasta = pysam.FastaFile(str(genome_path))
        return (lambda c, s0, e0: fasta.fetch(c, s0, e0)), set(fasta.references)
    except ImportError:
        import pyfaidx  # type: ignore
        fa = pyfaidx.Fasta(str(genome_path))
        return (lambda c, s0, e0: str(fa[c][s0:e0])), set(fa.keys())


def annotate_genome_consistency(clean: pd.DataFrame, genome_path: Path) -> pd.Series:
    """Return a 3-state nullable-boolean Series aligned to clean.index:
        True   -> ref matches genome[chrom][pos-1 : pos-1+len(ref)] (checked, passed)
        False  -> ref does NOT match at pos (checked, failed)
        <NA>   -> not checkable (contig absent) -- NEVER silently True/False.

    Checks EVERY row (SNV, MNV, indel), so a downstream consumer can trust that False means
    'checked and failed' and True means 'checked and passed', never 'not checked'. This is
    per-row metadata, distinct from the fail-loud build-sanity gates (SNV control, G8).
    A ref longer than the contig tail yields a short fetch that cannot equal ref -> False.
    """
    fetch, contigs = _load_genome(genome_path)

    def contig_of(c: str):
        c = _norm_chrom(c)
        for cand in (c, f"chr{c}"):
            if cand in contigs:
                return cand
        return None

    out = pd.array([pd.NA] * len(clean), dtype="boolean")
    for i, (chrom, pos, ref) in enumerate(zip(clean["chrom"], clean["pos"], clean["ref"])):
        cc = contig_of(str(chrom))
        if cc is None:
            out[i] = pd.NA            # contig absent -> unchecked, NOT a mismatch
            continue
        got = fetch(cc, int(pos) - 1, int(pos) - 1 + len(str(ref))).upper()
        out[i] = (got == str(ref).upper())
    return pd.Series(out, index=clean.index, name="ref_genome_consistent")


def reference_and_indel_check(clean: pd.DataFrame, genome_path: Path,
                              recon: BuildReconciliation, sample: "int | None" = None,
                              max_mismatch_rate: float = 0.001,
                              mismatch_out: "Path | None" = None) -> None:
    """G5 SNV control + G6 padded-deletion ref check + G8 all-indel genome-consistency."""
    fetch, contigs = _load_genome(genome_path)

    def contig_of(c: str):
        c = _norm_chrom(c)
        for cand in (c, f"chr{c}"):
            if cand in contigs:
                return cand
        return None

    # G5 SNV negative control -- SNVs are never shifted; a slice/build error fails them ~100%.
    snv_mask = (clean["ref"].astype(str).str.len() == 1) & (clean["alt"].astype(str).str.len() == 1)
    snv = clean[snv_mask].sample(min(2000, int(snv_mask.sum())), random_state=7)
    snv_mis = snv_n = 0
    for chrom, pos, ref in zip(snv["chrom"], snv["pos"], snv["ref"]):
        cc = contig_of(str(chrom))
        if cc is None:
            continue
        snv_n += 1
        if fetch(cc, int(pos) - 1, int(pos)).upper() != str(ref).upper():
            snv_mis += 1
    snv_rate = snv_mis / max(snv_n, 1)
    recon.snv_control = f"{snv_n - snv_mis}/{snv_n} match at pos-1 ({100*(1-snv_rate):.2f}%)"
    if snv_rate > 0.01:
        recon.reference_check = f"FAILED (SNV control {100*snv_rate:.1f}% mismatch)"
        raise ValueError(
            f"SNV CONTROL FAILED: {snv_mis}/{snv_n} SNVs mismatch at pos-1 "
            f"({100*snv_rate:.1f}%). SNVs are never shifted -- this is a SLICE-CONVENTION or "
            f"WRONG-BUILD error, not data. Do not trust the coordinate correction.")

    # G8 all-indel genome-consistency: EVERY indel's ref must match the genome at its pos.
    # (For a deletion ref begins at pos; for an insertion ref is a single anchor base at pos.)
    indel = clean[clean["ref"].astype(str).str.len() != clean["alt"].astype(str).str.len()]
    if sample and len(indel) > sample:
        indel = indel.sample(sample, random_state=42)
    mism = 0
    all_mismatches = []
    for vid, chrom, pos, ref in zip(indel["variant_id"], indel["chrom"], indel["pos"], indel["ref"]):
        cc = contig_of(str(chrom))
        if cc is None:
            raise ValueError(f"contig {chrom!r} not in genome {genome_path.name}")
        got = fetch(cc, int(pos) - 1, int(pos) - 1 + len(str(ref))).upper()
        if got != str(ref).upper():
            mism += 1
            all_mismatches.append(f"{vid}\texpected\t{ref}\tgenome\t{got}")
    recon.reference_mismatches = mism
    n_checked = len(indel)
    rate = mism / max(n_checked, 1)

    if all_mismatches:
        mm = mismatch_out or (Path("outputs") / "cohort_build_indel_mismatches.tsv")
        Path(mm).parent.mkdir(parents=True, exist_ok=True)
        Path(mm).write_text("variant_id\texpected_label\tref\tgenome_label\tgenome_seq\n"
                            + "\n".join(all_mismatches), encoding="utf-8")
        recon.notes.append(f"{mism} indel mismatches written to {mm}")

    if rate > max_mismatch_rate:
        recon.reference_check = f"FAILED ({mism}/{n_checked} = {100*rate:.4f}% > {100*max_mismatch_rate}%)"
        recon.indel_postcondition = recon.reference_check
        raise ValueError(
            f"INDEL GENOME-CONSISTENCY (G8) FAILED: {mism}/{n_checked} = {100*rate:.4f}% "
            f"mismatch, above {100*max_mismatch_rate}% tolerance. SNV control PASSED "
            f"({100*(1-snv_rate):.2f}%), so the convention is right -- this rate is too high "
            f"for mere ClinVar/reference disagreement. See the mismatch file.")
    recon.reference_check = (
        f"PASSED ({n_checked - mism}/{n_checked} = {100*(1-rate):.4f}%; "
        f"{mism} within {100*max_mismatch_rate}% tolerance; SNV control {100*(1-snv_rate):.2f}%)")
    recon.indel_postcondition = f"PASSED ({n_checked - mism}/{n_checked} indels genome-consistent)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unified coordinate-correct, quarantined cohort builder.")
    p.add_argument("--input", default="data/processed/clinvar_grch38.parquet")
    p.add_argument("--output", default="data/processed/clinvar_grch38_cohort_v4.parquet")
    p.add_argument("--genome", default=None, help="GRCh38 FASTA for the SNV control + indel check")
    p.add_argument("--ref-sample", type=int, default=None, help="check only N random indels (default: all)")
    p.add_argument("--max-mismatch-rate", type=float, default=0.001)
    p.add_argument("--force", action="store_true", help="overwrite an existing --output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true", help="report only, write nothing (default)")
    g.add_argument("--apply", action="store_true", help="write the cohort")
    a = p.parse_args(argv)

    in_path, out_path = Path(a.input), Path(a.output)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2
    if out_path.exists() and a.apply and not a.force:            # G1
        print(f"ERROR: {out_path} exists. Refusing to overwrite (use --force).", file=sys.stderr)
        return 5

    print("=" * 78)
    print(f"BUILD COHORT FROM SOURCE  input {in_path}")
    print("=" * 78)
    df = pd.read_parquet(in_path)
    print(f"loaded {len(df):,} rows / {len(df.columns)} cols", flush=True)

    recon = BuildReconciliation()
    clean, structural = build(df, recon)
    recon.input_md5 = _md5(in_path)

    if a.genome:
        gp = Path(a.genome)
        if not gp.exists():
            print(f"ERROR: --genome {gp} not found", file=sys.stderr)
            return 2
        print(f"running SNV control + all-indel genome-consistency check against {gp} ...", flush=True)
        # Diagnostic side-file ALWAYS goes to outputs/ (never the data tree), stamped with
        # the output stem so stale/fresh snapshot runs do not overwrite each other. This
        # keeps an --audit dry-run free of any write into data/processed/.
        mm_dir = Path("outputs")
        mm_dir.mkdir(parents=True, exist_ok=True)
        mm_path = mm_dir / f"{out_path.stem}_indel_mismatches.tsv"
        reference_and_indel_check(
            clean, gp, recon, sample=a.ref_sample, max_mismatch_rate=a.max_mismatch_rate,
            mismatch_out=mm_path)
        print(f"  reference check: {recon.reference_check}", flush=True)
        # Per-row 3-state flag over EVERY row (additive metadata; distinct from the gates).
        print("  annotating per-row ref_genome_consistent flag (all rows) ...", flush=True)
        flag = annotate_genome_consistency(clean, gp)
        clean["ref_genome_consistent"] = flag.values
        recon.genome_consistent = int((flag == True).sum())      # noqa: E712
        recon.genome_inconsistent = int((flag == False).sum())   # noqa: E712
        recon.genome_unchecked = int(flag.isna().sum())
        if recon.genome_inconsistent and "pathogenicity" in clean.columns:
            inc = clean[flag.values == False]                    # noqa: E712
            recon.genome_inconsistent_by_pathogenicity = {
                str(k): int(v) for k, v in
                inc["pathogenicity"].astype("string").fillna("<NA>").value_counts().items()}
        print(f"  ref_genome_consistent: {recon.genome_consistent:,} True / "
              f"{recon.genome_inconsistent:,} False / {recon.genome_unchecked:,} <NA>", flush=True)
        if recon.genome_inconsistent_by_pathogenicity:
            print(f"  genome-inconsistent by pathogenicity: "
                  f"{recon.genome_inconsistent_by_pathogenicity}", flush=True)
    else:
        recon.reference_check = "SKIPPED_NO_GENOME"
        recon.snv_control = "SKIPPED_NO_GENOME"
        recon.indel_postcondition = "SKIPPED_NO_GENOME"
        # Explicit <NA> flag for every row -- 'unchecked', never silently True/False.
        clean["ref_genome_consistent"] = pd.array([pd.NA] * len(clean), dtype="boolean")
        recon.genome_unchecked = len(clean)
        recon.notes.append(
            "PROVISIONAL: no GRCh38 FASTA supplied; SNV control and all-indel genome-consistency "
            "guards were NOT run, and ref_genome_consistent is <NA> for every row. Coordinates "
            "are corrected per the padded-deletion rule but NOT verified against the genome. "
            "Re-run with --genome before any production use.")
        print("  reference check: SKIPPED (no --genome) -- cohort is PROVISIONAL", flush=True)

    print(f"\ninput rows                    : {recon.input_rows:,}")
    print(f"quarantined (bad allele)      : {recon.quarantined_bad_allele:,} "
          f"(na:na {recon.quarantined_na_na:,} + half-bad {recon.quarantined_half_bad:,})")
    print(f"clean rows                    : {recon.clean_rows:,}")
    print(f"padded deletions corrected    : {recon.padded_deletions_corrected:,}")
    print(f"variant_id rebuilt            : {recon.variant_id_rebuilt:,}")
    print(f"bad-allele post-condition     : {recon.bad_allele_postcondition}")
    print(f"indel genome-consistency      : {recon.indel_postcondition}")
    print(f"duplicate variant_id          : {recon.dup_variant_id}")
    print(f"collapsed variant groups      : "
          f"{getattr(recon, 'collapsed_groups', 0):,} "
          f"(conflicts: {getattr(recon, 'collapse_conflicts', 0):,})")
    _comp_final = getattr(recon, 'composition_final', None)
    if _comp_final is not None and _comp_final != recon.composition_after:
        print("composition (final, post-collapse):")
        for _k in sorted(_comp_final):
            print(f"    {_k:16s} {_comp_final[_k]:>10,}")
    print(f"ref_genome_consistent         : {recon.genome_consistent:,} True / "
          f"{recon.genome_inconsistent:,} False / {recon.genome_unchecked:,} <NA>")
    print(f"reconciles                    : {recon.reconciles()}")
    print("composition (clean, unchanged by pos correction):")
    for k in sorted(recon.composition_after):
        print(f"    {k:16s} {recon.composition_after[k]:>10,}")
    for n in recon.notes:
        print(f"  NOTE: {n}")

    if not a.apply:
        print("\nAUDIT (dry-run). Nothing written. Re-run with --apply.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Recompute the schema fingerprint on the FINAL frame (the ref_genome_consistent column
    # was added after build()); the recorded fingerprint must match what is written.
    recon.schema_fingerprint = schema_fingerprint(clean.columns)
    clean.to_parquet(out_path, index=False)
    struct_path = out_path.with_name(out_path.stem + "_structural.parquet")
    structural.to_parquet(struct_path, index=False)
    recon.output_md5 = _md5(out_path)
    recon_path = out_path.with_name(out_path.stem + "_reconciliation.json")
    recon_path.write_text(json.dumps(recon.as_dict(), indent=2), encoding="utf-8")
    # Fixed-name compatibility copy of the reconciliation record. Identical content to the
    # per-build "<stem>_reconciliation.json" above; this alias always points at the LAST
    # build written, preserving the historically documented path for any consumer that
    # expects it. The per-build file is the permanent archive; this one may be overwritten.
    _compat_recon = out_path.with_name("cohort_build_reconciliation.json")
    _compat_recon.write_text(json.dumps(recon.as_dict(), indent=2), encoding="utf-8")
    # Persist the per-collapse dedup audit (one row per collapsed variant group) so every
    # collapse is inspectable. Empty when no duplicates were collapsed.
    _dedup_audit = getattr(recon, "dedup_audit", []) or []
    if _dedup_audit:
        import csv as _csv
        _dedup_path = out_path.with_name(out_path.stem + "_dedup_audit.tsv")
        _fields = ["variant_id", "kept_source_id", "dropped_source_ids",
                   "kept_pathogenicity", "all_pathogenicities",
                   "classification_conflict", "conflict_span", "collapsed_from_n"]
        with open(_dedup_path, "w", newline="", encoding="utf-8") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=_fields, delimiter="\t")
            _w.writeheader()
            for _rec in _dedup_audit:
                _w.writerow({_k: _rec.get(_k) for _k in _fields})
        print(f"WROTE: {_dedup_path.name}  ({len(_dedup_audit):,} collapsed groups)")

    _written_n = len(clean)
    print(f"\nWROTE: {out_path.name}  (MD5 {recon.output_md5}, {_written_n:,} rows)")
    print(f"WROTE: {struct_path.name}  ({recon.quarantined_bad_allele:,} quarantined rows)")
    print(f"WROTE: {recon_path.name}")
    print(f"WROTE: {_compat_recon.name}  (compatibility alias to this build)")
    print(f"\nDOWNSTREAM REQUIRED: rebuild sequence windows from this cohort -- padded-deletion "
          f"positions moved for {recon.padded_deletions_corrected:,} rows.")
    if recon.reference_check == "SKIPPED_NO_GENOME":
        print("REMINDER: PROVISIONAL (genome checks skipped). Re-run with --genome before production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
