# COHORT-V2 COORDINATE CORRECTION — STATUS & RESOLUTION

**Document date:** 2026-07-09
**HEAD at writing:** `d4d4615`
**Author:** Monzia Moodie
**Scope:** the padded-deletion coordinate bug, its correction, its genome-level
verification, and the state of everything downstream. Written so any researcher can read
this alone and know exactly where the project stands.

---

## 1. Executive summary

A coordinate off-by-one affecting **187,245 padded deletions** (4.24% of the 4,420,180-row
ClinVar cohort) was identified 2026-07-08, corrected by a purpose-built ground-up tool, and
**verified against the GRCh38 primary assembly on 2026-07-09**: 187,235 of 187,245 corrected
deletions (99.9947%) match the reference at the corrected position, with a 2000/2000 SNV
control confirming the coordinate convention. The 10 residual mismatches are genuine
ClinVar-vs-GRCh38 disagreements, not artifacts. The corrected cohort
(`clinvar_grch38_clean_v2_verified.parquet`, MD5 `F3152F671E920A0A0C19A696563002E0`) is the
new canonical cohort. Cohort-v1 (`clinvar_grch38_clean.parquet`, MD5
`7C5E107C220050EDB496A9D92A57D5FD`) is retained untouched as evidence.

The correction is **not yet propagated** to the Nucleotide-Transformer sequence windows or to
the positional annotations — those are the immediate next steps and are unblocked now that a
GRCh38 FASTA is on disk.

---

## 2. The bug (established 2026-07-08)

**Root cause.** The cohort's `pos` column was populated from ClinVar `variant_summary.Start`
(the first *altered* base). Its `ref`/`alt` came from `ReferenceAlleleVCF` /
`AlternateAlleleVCF`, which begin at `PositionVCF`. For a **left-padded deletion** — where the
first reference base is unchanged and carried as an anchor — `Start = PositionVCF + 1`. Every
such row therefore had a `pos` one greater than the VCF position at which its `ref` allele
begins.

**Scope of the escalation.** Because the join keys and sequence-window centres all used `pos`:
every `chrom:pos` annotation merge (gnomAD, SpliceAI, phyloP, CADD, dbNSFP, 1000G, dbSNP,
COSMIC, AlphaMissense) missed all padded deletions, and every Nucleotide-Transformer window
was centred one base off. Empirically, 0 of 98,785 padded deletions in an earlier regime
received a gnomAD allele frequency, versus 33.7% of matched insertions.

**Why only padded deletions.** The offset is exactly and only present when the ALT allele is a
strict prefix of the REF allele and shorter (`alt.len < ref.len AND ref.startswith(alt)`).
SNVs, insertions, delins, and MNVs are all correctly positioned. A length-shrinking delins
such as `AA>C` is NOT a padded deletion (`"AA".startswith("C")` is false) and is correctly
left unshifted.

---

## 3. The correction (2026-07-08, commit `11474f5`)

`scripts/build_cohort_v2.py` is a standalone builder, not a patch to the v1 cohort pipeline.
Its pure core, `correct_coordinates(df)`:

1. computes `is_padded_deletion = (alt.len < ref.len) & ref.startswith(alt)` element-wise
   (pandas `Series.str.startswith` rejects a per-row Series argument, so an explicit
   element-wise prefix check is used — a bug the tests caught before any data was touched);
2. applies `pos -= 1` on exactly those rows and nowhere else;
3. rebuilds `variant_id` from the corrected position;
4. asserts the variant-class composition is byte-identical before and after (a pos-only
   change cannot alter any allele, hence cannot alter any class), and that
   `n_corrected + n_unchanged == n_rows`;
5. records a schema fingerprint, the full composition, the v1 source MD5, and a
   reconciliation JSON;
6. writes to a NEW path and refuses to overwrite an existing file (exit 5).

**Applied to the real cohort:** 187,245 padded deletions corrected, 4,232,935 rows unchanged,
identity holds. Composition (invariant): SNV 4,102,868; padded_deletion 187,245;
padded_insertion 90,177; MNV/other 36,566; delins_del 2,282; delins_ins 1,042.

The incident report's original estimate was 187,258; the exhaustive count is 187,245 (Δ13 =
0.007%). Two independent methods agreeing to four significant figures.

---

## 4. Genome verification (2026-07-09, commits `70eb254`, `da379ac`)

A GRCh38 primary assembly FASTA was acquired to `data/external/grch38/GRCh38.fa` (plain
contig names `1`/`2`/`X`, matching ClinVar's `chrom`). Before this session only GENCODE
*transcript* FASTAs were present — spliced mRNA, useless for genomic coordinate checks.

**Two self-inflicted validation bugs occurred and were both caught by data-derived controls,
not by reasoning:**

- The preflight `scripts/check_grch38_fasta.py` initially asserted five known reference bases
  typed *from memory*; three were wrong, producing a FALSE failure on a correct genome. Fixed
  by removing all hardcoded literals — the base check now reads truth from the cohort's own
  SNVs.
- The reference guard initially hard-failed on ANY mismatch. Real genomic data is not 100%
  concordant, so this would block every legitimate run. Fixed to tolerate a small,
  configurable rate WHILE adding an SNV control that hard-fails on any systematic
  slice/build error regardless of tolerance — a coordinate bug cannot be tolerated away.

**The decisive evidence** (`scripts/diagnose_coordinate_convention.py`, which asserts no
hardcoded base and uses SNVs as the never-shifted control):

- **SNV control: 2000/2000 (100.00%)** match at `pos-1` — the standard 1-based-VCF-to-0-based
  convention is confirmed; wrong-build and slicing errors are ruled out.
- **Padded deletions: 187,235/187,245 (99.9947%)** match GRCh38 at the CORRECTED position;
  0/500 sampled matched at the ORIGINAL position.

A slice-convention error is global — it would fail ~100% of rows. A 99.9947% pass rate is
logically incompatible with a coordinate bug. The correction is right.

**The 10 residual mismatches** (`data/processed/cohort_v2_ref_mismatches.tsv`, frozen at
`docs/audits/evidence/2026-07-09/`): 5 are large multi-kb deletions in repeat/low-complexity
regions where ClinVar's left-alignment differs from a naive VCF slice; 4 are single-base
reference discrepancies where ClinVar's ref allele disagrees with the GRCh38 primary assembly
(a known ClinVar data issue); 1 is a rearrangement. All 10 are at the chromosome:position the
variant claims. Disposition: **FLAG, do not correct** — ClinVar's intended representation
cannot be reconstructed without submitter data, and 0.0053% is within published ClinVar/
reference indel discordance. A `ref_genome_mismatch` flag should be added when cohort-v2 is
annotated.

---

## 5. Artifacts and provenance (2026-07-09, commit `d4d4615`)

**Canonical cohort:** `data/processed/clinvar_grch38_clean_v2_verified.parquet`
(MD5 `F3152F671E920A0A0C19A696563002E0`, 135.45 MB). The provisional
`clinvar_grch38_clean_v2.parquet` — byte-identical, produced before the genome check — was
removed to eliminate a stale-artifact hazard (two identical files of record beside one
reconciliation JSON). `data/processed/` is gitignored (multi-GB parquet).

**Frozen evidence** under `docs/audits/evidence/2026-07-09/` (version-controlled):
`cohort_v2_reconciliation.json` (reference_check PASSED, 10 mismatches, identity holds),
`cohort_v2_ref_mismatches.tsv` (the 10 itemized), `cohort_v2_verified.txt` (full build log),
`coord_diagnostic.txt` (the SNV-control proof), `grch38_preflight.txt` (the genome
validation).

**Test coverage:** `tests/test_build_cohort_v2.py`, 11 tests, all passing on Python 3.12.10 /
pandas 2.3.3. Includes: padded-deletion-only shift, delins-not-shifted, variant_id rebuild,
composition invariance, the reference guard PASS path, the guard hard-fail path, the
tolerance path, and — critically — the SNV-control-catches-a-slice-bug path.

**Commit lineage (this thread):**
`11474f5` cohort-v2 builder + 11 tests →
`70eb254` genome-verified diagnostic + de-hardcoded preflight →
`da379ac` tolerant guard + SNV control →
`d4d4615` stale-artifact cleanup + frozen evidence (**HEAD**).

---

## 6. What remains (dependency-ordered)

1. **[IMMEDIATE, needs the genome — now on disk] Rebuild the Nucleotide-Transformer
   sequence windows from verified cohort-v2.** The windows are centred on `pos` and are off
   by one for the 187,245 corrected rows. `genomiclm_delta_norm` (a top smoke-test feature)
   currently encodes the wrong sequence for every padded deletion. NOTE: the exact
   window-builder stage in the repo must be located and read before rebuilding — do not
   assume its interface.
2. **Re-annotate cohort-v2.** Positional joins (gnomAD, SpliceAI, phyloP, CADD, dbNSFP) will
   now hit all 187,245 deletions. Add a `ref_genome_mismatch` flag for the 10.
3. **Evaluator/provenance fixes (no genome needed):** `evaluator.py::_calibration_error`
   `p==1.0` bin bug (ECE under-reported); `_bootstrap_ci` unstratified resampling; manifest
   `scikit-learn` version capture (records `not_installed` falsely); add `min_review_tier`,
   cohort MD5, and schema fingerprint to `save_manifest(config=...)`.
4. **Wire `run_phase2_eval.py` → `RunArtifactWriter`** to close the provenance gap (run14/
   run17-class runs currently produce un-joinable OOF).
5. **Re-derive splits from cohort-v2**; record cohort MD5, schema fingerprint, and per-class
   counts.
6. **Re-baseline** stratified by representation, with Type-1 circularity disclosed (see the
   leakage analysis, `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`), and a ClinVar-
   independent validation stratum if reachable.
7. **Local smoke → VM smoke → launch.**
8. **Purge the 1.95 GB AlphaFold blob** from git history to unblock the GitHub push (the repo
   is ~26 commits ahead of origin, push-blocked).

---

## 7. Recurring lesson, recorded for the next researcher

Every correct resolution this session came from a control that could not lie — a hash, a
commit date, an sklearn cross-check to 1e-16, a `git ls-files`, and finally an SNV control
(2000/2000; 187,235/187,245) — never from arguing toward an answer. Twice the *correction to a
correction* was itself wrong on the first pass (hardcoded reference bases; a zero-tolerance
guard), and both times the way out was a data-derived control, not more reasoning.

The reference guard refused to write an unverified cohort every single time it fired. Broken
validators cost a diagnostic cycle; they never produced a corrupt production artifact. That is
the entire purpose of "fail loud, never silent," and it earned its keep here.
