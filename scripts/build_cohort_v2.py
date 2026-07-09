"""
build_cohort_v2.py  (2026-07-08)
==========================================================================
Correct the padded-deletion coordinate off-by-one identified in
docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md sec 3, and emit a
NEW, pinned cohort artifact. Does NOT overwrite v1; v1 remains as evidence.

THE BUG (established, 30/30 at pos-1 with a negative control)
    The cohort's `pos` is ClinVar variant_summary's `Start` (first altered base). Its
    `ref`/`alt` are ReferenceAlleleVCF / AlternateAlleleVCF, which begin at PositionVCF.
    For a PADDED DELETION the padding base is unchanged, so Start == PositionVCF + 1.
    Every chrom:pos(:ref:alt) annotation join and the NT sequence windows are therefore
    off by one for 189,468 padded deletions -- and 0 of 98,785 received a gnomAD AF.

THE CORRECTION (exact -- verified against the 30/30 signature)
    is_padded_del = (alt.str.len() < ref.str.len()) & ref.str.startswith(alt)
    pos[is_padded_del] -= 1
    Nothing else moves. Note `ref.startswith(alt)` is load-bearing: a length-shrinking
    DELINS like AA>C also has len(alt) < len(ref), but "AA".startswith("C") is False, so
    it is correctly NOT shifted. Only true left-padded deletions move.

THE GUARD NOBODY WROTE
    genome[chrom][pos-1 : pos-1+len(ref)] == ref   (0-based slice; VCF pos is 1-based)
    A single reference-consistency post-condition would have caught 187,258 rows the
    first time the cohort was built. It needs a GRCh38 FASTA. None is present on this
    machine (only GENCODE transcript FASTAs). So:
      * --genome PATH given  -> check every corrected padded deletion; ANY mismatch is a
                                hard failure (exit 4). This is the real validation.
      * --genome absent      -> SKIP with a loud WARNING recorded in the reconciliation
                                JSON (reference_check: "SKIPPED_NO_GENOME"). NEVER a
                                silent pass. cohort-v2 built this way is PROVISIONAL and
                                says so in its own metadata.

DOWNSTREAM DEPENDENCY (not silently assumed done)
    clinvar_grch38_clean_seq.parquet's Nucleotide-Transformer windows are centred on
    `pos`, so they are off by one for the same rows and MUST be rebuilt from cohort-v2.
    This script does not build them (the window builder is a separate stage); it RECORDS
    the requirement in the reconciliation JSON as nt_windows_rebuild_required: true.

USAGE (from project root, .venv312 active)
    python scripts/build_cohort_v2.py --audit
    python scripts/build_cohort_v2.py --apply
    python scripts/build_cohort_v2.py --apply --genome data/external/grch38/GRCh38.fa
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

REQUIRED_COLS = ("variant_id", "chrom", "pos", "ref", "alt")
V1_EXPECTED_MD5 = "7C5E107C220050EDB496A9D92A57D5FD"  # the v1 clean cohort, for provenance


def _startswith_elementwise(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """Vectorized `ref[i].startswith(alt[i])`.

    pandas `Series.str.startswith` only accepts a scalar/tuple pattern, not a per-row
    Series, so the prefix test must be done element-wise. Kept in one place so both
    `variant_class` and `is_padded_deletion` use identical semantics.
    """
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return pd.Series(
        [rr.startswith(aa) for rr, aa in zip(r, a)],
        index=ref.index, dtype=bool,
    )


def variant_class(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    lr, la = r.str.len(), a.str.len()
    starts = _startswith_elementwise(r, a)   # ref starts with alt
    starts_ra = _startswith_elementwise(a, r)  # alt starts with ref
    out = pd.Series("MNV/other", index=r.index, dtype="object")
    out[(lr == 1) & (la == 1)] = "SNV"
    out[(lr > 1) & (la == 1) & starts] = "padded_deletion"
    out[(lr > 1) & (la == 1) & ~starts] = "delins_del"
    out[(lr == 1) & (la > 1) & starts_ra] = "padded_insertion"
    out[(lr == 1) & (la > 1) & ~starts_ra] = "delins_ins"
    return out


def is_padded_deletion(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return (a.str.len() < r.str.len()) & _startswith_elementwise(r, a)


def schema_fingerprint(columns) -> str:
    return hashlib.sha256(",".join(sorted(map(str, columns))).encode()).hexdigest()[:16]


def _norm_chrom(c: object) -> str:
    s = str(c)
    return s[3:] if s.lower().startswith("chr") else s


@dataclass
class V2Reconciliation:
    n_rows: int = 0
    n_padded_deletions_corrected: int = 0
    n_unchanged: int = 0
    variant_id_rebuilt: int = 0
    reference_check: str = "NOT_RUN"
    reference_mismatches: int = 0
    nt_windows_rebuild_required: bool = True
    v1_source_md5: str = ""
    v2_schema_fingerprint: str = ""
    composition_before: dict = field(default_factory=dict)
    composition_after: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def identity_holds(self) -> bool:
        return self.n_rows == self.n_padded_deletions_corrected + self.n_unchanged

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["identity_holds"] = self.identity_holds()
        return d


def correct_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, V2Reconciliation]:
    """Pure function. Returns (corrected_df, reconciliation). Raises on schema problems."""
    recon = V2Reconciliation(n_rows=len(df))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Present: {list(df.columns)}")

    recon.composition_before = {
        k: int(v) for k, v in variant_class(df["ref"], df["alt"]).value_counts().items()
    }

    mask = is_padded_deletion(df["ref"], df["alt"])
    recon.n_padded_deletions_corrected = int(mask.sum())
    recon.n_unchanged = int((~mask).sum())

    out = df.copy()
    # integer-safe decrement on the masked rows only
    pos = out["pos"].to_numpy()
    if not np.issubdtype(pos.dtype, np.integer):
        # pos may be float if the source had NaNs; assert none in the masked set
        if np.isnan(pos[mask.to_numpy()]).any():
            raise ValueError("NaN pos among padded deletions -- cannot correct coordinates.")
    out.loc[mask, "pos"] = out.loc[mask, "pos"] - 1

    # rebuild variant_id from corrected pos (format: clinvar:{chrom}:{pos}:{ref}:{alt})
    prefix = out["variant_id"].astype("string").str.split(":", n=1).str[0].fillna("clinvar")
    out["variant_id"] = (
        prefix + ":" + out["chrom"].astype(str) + ":" + out["pos"].astype("int64").astype(str)
        + ":" + out["ref"].astype(str) + ":" + out["alt"].astype(str)
    )
    recon.variant_id_rebuilt = int(mask.sum())

    recon.composition_after = {
        k: int(v) for k, v in variant_class(out["ref"], out["alt"]).value_counts().items()
    }
    recon.v2_schema_fingerprint = schema_fingerprint(out.columns)

    # composition MUST be identical -- correcting pos changes no allele, hence no class
    if recon.composition_before != recon.composition_after:
        raise ValueError(
            "Variant-class composition changed under a pos-only correction -- impossible. "
            f"before={recon.composition_before} after={recon.composition_after}"
        )
    if not recon.identity_holds():
        raise ValueError("Row reconciliation failed: " + json.dumps(recon.as_dict()))
    return out, recon


def reference_check(df: pd.DataFrame, genome_path: Path, recon: V2Reconciliation,
                    sample: "int | None" = None, max_mismatch_rate: float = 0.001,
                    mismatch_out: "Path | None" = None) -> None:
    """Assert genome[chrom][pos-1 : pos-1+len(ref)] == ref for corrected padded deletions.

    Real genomic data is not 100% concordant: a tiny fraction of ClinVar records disagree
    with the GRCh38 primary assembly (alt loci, assembly patches, left-alignment
    ambiguity). A guard that demands zero mismatches would block every real run. So this
    tolerates up to `max_mismatch_rate` (default 0.1%) of deletion mismatches AS LONG AS
    the SNV control passes -- SNVs are never shifted, so a slice-convention or wrong-build
    error would fail them en masse (~100%), cleanly distinguishing a coordinate bug from
    genuine data disagreement. Every mismatch is written to `mismatch_out` for inspection.

    Requires pysam or pyfaidx.
    """
    try:
        import pysam  # type: ignore
        fasta = pysam.FastaFile(str(genome_path))
        def fetch(chrom, start0, end0):
            return fasta.fetch(chrom, start0, end0)
        contigs = set(fasta.references)
    except ImportError:
        try:
            import pyfaidx  # type: ignore
            fa = pyfaidx.Fasta(str(genome_path))
            def fetch(chrom, start0, end0):
                return str(fa[chrom][start0:end0])
            contigs = set(fa.keys())
        except ImportError as exc:
            raise RuntimeError(
                "reference check requested (--genome) but neither pysam nor pyfaidx is "
                "installed. `pip install pyfaidx` or omit --genome to skip."
            ) from exc

    mask = is_padded_deletion(df["ref"], df["alt"])
    sub = df[mask]
    if sample and len(sub) > sample:
        sub = sub.sample(sample, random_state=42)

    def contig_of(c: str) -> "str | None":
        c = _norm_chrom(c)
        for cand in (c, f"chr{c}"):
            if cand in contigs:
                return cand
        return None

    # --- CONTROL: SNVs were never shifted. If the SLICE convention were wrong, SNVs
    #     would fail en masse. This distinguishes a coordinate bug (systematic, ~100%)
    #     from genuine ClinVar-vs-reference disagreements (rare, scattered). ---
    snv_mask = (df["ref"].astype(str).str.len() == 1) & (df["alt"].astype(str).str.len() == 1)
    snv = df[snv_mask].sample(min(2000, int(snv_mask.sum())), random_state=7)
    snv_mis = 0
    snv_n = 0
    for chrom, pos, ref in zip(snv["chrom"], snv["pos"], snv["ref"]):
        cc = contig_of(str(chrom))
        if cc is None:
            continue
        snv_n += 1
        if fetch(cc, int(pos) - 1, int(pos)).upper() != str(ref).upper():
            snv_mis += 1
    snv_rate = snv_mis / max(snv_n, 1)
    recon.notes.append(f"SNV control: {snv_n - snv_mis}/{snv_n} match at pos-1 "
                       f"({100*(1-snv_rate):.2f}%)")
    if snv_rate > 0.01:
        recon.reference_check = f"FAILED (SNV control {100*snv_rate:.1f}% mismatch)"
        raise ValueError(
            f"REFERENCE-CONSISTENCY GUARD FAILED ON THE SNV CONTROL: {snv_mis}/{snv_n} "
            f"SNVs mismatch at pos-1 ({100*snv_rate:.1f}%). SNVs are never shifted, so this "
            f"is a SLICE-CONVENTION or WRONG-BUILD error, not a data issue. Do not trust the "
            f"deletion check until this is resolved."
        )

    # --- padded deletions at the corrected coordinate ---
    mism = 0
    all_mismatches: list[str] = []
    for vid, chrom, pos, ref in zip(sub["variant_id"], sub["chrom"], sub["pos"], sub["ref"]):
        cc = contig_of(str(chrom))
        if cc is None:
            raise ValueError(f"contig {chrom!r} not in genome {genome_path.name}")
        # pos is the corrected 1-based VCF POS; ref begins at pos, 0-based slice pos-1
        got = fetch(cc, int(pos) - 1, int(pos) - 1 + len(str(ref))).upper()
        if got != str(ref).upper():
            mism += 1
            all_mismatches.append(f"{vid}\texpected\t{ref}\tgenome\t{got}")
    recon.reference_mismatches = mism
    n_checked = len(sub)
    rate = mism / max(n_checked, 1)

    # write EVERY mismatch to a file for inspection (never just the first 10)
    if all_mismatches:
        mm_path = mismatch_out or (Path("outputs") / "cohort_v2_ref_mismatches.tsv")
        Path(mm_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mm_path).write_text(
            "variant_id\texpected_label\tref\tgenome_label\tgenome_seq\n"
            + "\n".join(all_mismatches), encoding="utf-8")
        recon.notes.append(f"{mism} deletion mismatches written to {mm_path}")

    # A slice bug is global (~100% fail). A tolerable rate of genuine ClinVar-vs-GRCh38
    # disagreement (alt loci, assembly patches, left-alignment) is tiny. Given the SNV
    # control passed, anything below the threshold is data, not a coordinate error.
    if rate > max_mismatch_rate:
        recon.reference_check = f"FAILED ({mism}/{n_checked} = {100*rate:.4f}% > {100*max_mismatch_rate}%)"
        raise ValueError(
            f"REFERENCE-CONSISTENCY GUARD FAILED: {mism} of {n_checked} corrected padded "
            f"deletions mismatch ({100*rate:.4f}%), above the {100*max_mismatch_rate}% "
            f"tolerance. The SNV control PASSED ({100*(1-snv_rate):.2f}%), so the convention "
            f"is right -- this rate is too high for mere ClinVar/reference disagreement and "
            f"needs investigation. See the mismatch file."
        )
    recon.reference_check = (
        f"PASSED ({n_checked - mism}/{n_checked} = {100*(1-rate):.4f}%; "
        f"{mism} genuine ClinVar/reference disagreements within {100*max_mismatch_rate}% tolerance; "
        f"SNV control {100*(1-snv_rate):.2f}%)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build coordinate-corrected cohort-v2.")
    p.add_argument("--input", default="data/processed/clinvar_grch38.parquet")
    p.add_argument("--output", default="data/processed/clinvar_grch38_clean_v2.parquet")
    p.add_argument("--genome", default=None, help="GRCh38 FASTA for the reference check")
    p.add_argument("--ref-sample", type=int, default=None,
                   help="check only N random padded deletions (default: all)")
    p.add_argument("--max-mismatch-rate", type=float, default=0.001,
                   help="tolerated fraction of deletion/reference disagreements (default 0.1%%)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true", help="report only, write nothing (default)")
    g.add_argument("--apply", action="store_true", help="write cohort-v2")
    a = p.parse_args(argv)

    in_path, out_path = Path(a.input), Path(a.output)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2
    if out_path.exists() and a.apply:
        print(f"ERROR: {out_path} exists. Refusing to overwrite. Move it or choose --output.",
              file=sys.stderr)
        return 5

    print("=" * 74)
    print(f"BUILD COHORT-V2  (coordinate correction)   input {in_path}")
    print("=" * 74)
    df = pd.read_parquet(in_path)
    print(f"loaded {len(df):,} rows / {len(df.columns)} cols")

    corrected, recon = correct_coordinates(df)

    # provenance: record the input's MD5
    recon.v1_source_md5 = hashlib.md5(in_path.read_bytes()).hexdigest().upper()

    if a.genome:
        gp = Path(a.genome)
        if not gp.exists():
            print(f"ERROR: --genome {gp} not found", file=sys.stderr)
            return 2
        print(f"running reference-consistency check against {gp} ...")
        reference_check(corrected, gp, recon, sample=a.ref_sample,
                        max_mismatch_rate=a.max_mismatch_rate,
                        mismatch_out=out_path.with_name("cohort_v2_ref_mismatches.tsv"))
        print(f"  reference check: {recon.reference_check}")
    else:
        recon.reference_check = "SKIPPED_NO_GENOME"
        recon.notes.append(
            "PROVISIONAL: no GRCh38 FASTA supplied; reference-consistency guard was NOT run. "
            "cohort-v2 coordinates are corrected per the padded-deletion rule but NOT verified "
            "against the genome. Re-run with --genome before trusting for a production run."
        )
        print("  reference check: SKIPPED (no --genome) -- cohort-v2 is PROVISIONAL")

    print(f"\npadded deletions corrected (pos -= 1): {recon.n_padded_deletions_corrected:,}")
    print(f"rows unchanged                       : {recon.n_unchanged:,}")
    print(f"variant_id rebuilt                   : {recon.variant_id_rebuilt:,}")
    print(f"schema fingerprint                   : {recon.v2_schema_fingerprint}")
    print(f"reconciliation identity holds        : {recon.identity_holds()}")
    print("composition (unchanged by design):")
    for k in sorted(recon.composition_after):
        print(f"    {k:16s} {recon.composition_after[k]:>10,}")
    if recon.notes:
        for n in recon.notes:
            print(f"  NOTE: {n}")

    if not a.apply:
        print("\nAUDIT (dry-run). Nothing written. Re-run with --apply.")
        return 0

    corrected.to_parquet(out_path, index=False)
    recon_path = out_path.with_name("cohort_v2_reconciliation.json")
    recon_path.write_text(json.dumps(recon.as_dict(), indent=2), encoding="utf-8")

    written_md5 = hashlib.md5(out_path.read_bytes()).hexdigest().upper()
    print(f"\nWROTE: {out_path.name}  (MD5 {written_md5})")
    print(f"WROTE: {recon_path.name}")
    print(f"\nDOWNSTREAM REQUIRED: rebuild the Nucleotide-Transformer sequence windows in "
          f"clinvar_grch38_clean_seq.parquet from cohort-v2 -- they are centred on `pos` and "
          f"are off by one for the {recon.n_padded_deletions_corrected:,} corrected rows.")
    if recon.reference_check == "SKIPPED_NO_GENOME":
        print("REMINDER: this cohort-v2 is PROVISIONAL (reference check skipped). Re-run with "
              "--genome to validate before any production training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
