# Storage Action Ledger — GenAssoc — 2026-07-03

Purpose: a complete, auditable record of every large file's disposition during the
post-AlphaFold-build disk-management pass. Nothing is deleted until its copy is
verified (rclone check = 0 differences, or external-drive checksum match).

Context: C: drive was 40.2 GB free / 936 GB total (~96% full) after the AlphaFold
cohort build. rclone remote for Google Drive: `genvarcla:`.

## Decisions table

| item | path | size | action | destination | reason | status |
|---|---|---|---|---|---|---|
| dbNSFP full index | data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak | 0.88 GB | BACK UP (keep local) | genvarcla:GenAssoc-backup/2026-07-03/dbnsfp | ONLY copy of the 85,299,339-row dbNSFP index; the expected live file dbnsfp_full_index.parquet is MISSING; code references it in 5 places. Loads OK. NOT disposable. | **VERIFIED on Drive (0 diff)** |
| dbNSFP clinvar index | data/external/dbnsfp/dbnsfp_clinvar_index.parquet | 0.03 GB | BACK UP (keep local) | (same dbnsfp folder) | small derived index; backed up with the dir | **VERIFIED on Drive (0 diff)** |
| EVE all_data | data/external/eve/EVE_all_data/ | 63.36 GB | BACK UP (keep local) | genvarcla:GenAssoc-backup/2026-07-03/eve | User-requested Drive backup of MSAs + all_data. Extracted tree intact (14,932 files). | **VERIFIED on Drive: 0 differences, 14933 matching files, 108.68 GiB, 4h45m** |
| EVE MSAs (subset of above) | data/external/eve/EVE_all_data/MSAs/ | 24.15 GB | BACK UP | (within eve backup) | User-requested. eve.py has 0 MSA/alignment references -> NOT consumed by pipeline; preserved to Drive per instruction. | **VERIFIED (within eve backup)** |
| EVE zip | data/external/eve/EVE_all_data.zip | 8.91 GB | BACK UP, then candidate for local delete | genvarcla:GenAssoc-backup/2026-07-03/eve | Redundant with the extracted tree. Now backed up. Candidate for LOCAL delete to reclaim 8.91 GB. | **VERIFIED on Drive (within eve backup)** |
| AlphaFold cohort parquet | data/external/alphafold/alphafold_cohort.parquet | 0.11 GB | BACK UP (keep local) | genvarcla:GenAssoc-backup/2026-07-03/alphafold_cohort | Irreplaceable derived artifact: 9,960,360 residue rows, 18,034 structures, verified. Keep local (pipeline reads it). | **VERIFIED on Drive (0 diff)** |
| AlphaFold coverage json | data/external/alphafold/alphafold_coverage.json | <0.01 GB | BACK UP (keep local) | (same) | Coverage accounting (18,034 usable + 268 unusable = 18,302). | **VERIFIED on Drive (0 diff)** |
| AlphaFold CIF cache | data/raw/cache/alphafold/ | 8.77 GB | ARCHIVE (tar.gz) -> Drive, then candidate for local delete | genvarcla:GenAssoc-backup/2026-07-03/alphafold_cif_cache.tar.gz | 18,079 CIFs. Regenerable only WHILE AlphaFold DB serves v6 (not guaranteed forever). Archived for reproducibility. tar.gz = 1.904 GiB. | **VERIFIED on Drive (0 diff)** |
| FinnGen R12 | data/external/finngen/finnge_R12_annotated_variants_v1.gz | 29.92 GB | ARCHIVE to EXTERNAL drive | (external) | Raw source; referenced by eval pipeline. Disaster-safe on Drive + fast-restore on external (both requested). | **VERIFIED on Drive (0 diff, 3 files); external D: pending** |
| FinnGen R13 | data/external/finngen/finngen_R13_annotated_variants_v0.gz | 27.72 GB | ARCHIVE to EXTERNAL drive | (external) | as above | **VERIFIED on Drive (within finngen backup); external D: pending** |
| phyloP bigwig | data/external/phylop/hg38.phyloP100way.bw | 9.19 GB | ARCHIVE to EXTERNAL drive | (external) | Raw conservation track; referenced by eval pipeline. Drive + external (both requested). | **VERIFIED on Drive (0 diff, 9.192 GiB); external D: pending** |

## Verification protocol

- Google Drive uploads: `rclone copy --checksum` then `rclone check --one-way`
  must report `0 differences` before the item is considered safe.
- External-drive archives: copy, then compare SHA-256 of source vs destination;
  delete original ONLY on match.
- No local deletion of any item until its verified-copy condition is met AND
  (for pipeline-referenced items) its downstream use is confirmed not to need the
  local path.

## Open issues flagged during this pass

1. dbNSFP live index MISSING: `dbnsfp_full_index.parquet` does not exist; only the
   `.OOMbak` does. Code expects the live name. Likely a mid-rebuild OOM casualty.
   ACTION NEEDED (separate from storage): decide whether to rename/restore the
   .OOMbak to the expected filename so run17/eval does not fail. Do NOT delete.

## Event log (appended as actions complete)

- 2026-07-03: Ledger opened. Decisions recorded above.
- 2026-07-03 13:48: dbNSFP backup VERIFIED -> genvarcla:GenAssoc-backup/2026-07-03/dbnsfp
  (rclone check: 0 differences, 2 matching files). OOMbak = only copy of 85.3M-row index, now safe on Drive.
- 2026-07-03 18:58: EVE backup VERIFIED -> genvarcla:GenAssoc-backup/2026-07-03/eve
  (rclone check: 0 differences, 14933 matching files, 108.68 GiB transferred, 4h45m). MSAs + all_data + zip all on Drive.
- STILL PENDING: cohort parquet+coverage upload; CIF cache tar.gz upload; FinnGen+phyloP external-drive archive.
- 2026-07-03 19:08: AlphaFold cohort backup VERIFIED -> genvarcla:GenAssoc-backup/2026-07-03/alphafold_cohort
  (rclone check: 0 differences, 2 matching files). Cohort parquet (110MB) + coverage json on Drive.
- STILL PENDING: CIF cache tar.gz upload; FinnGen+phyloP external-drive archive; dbNSFP live-index restoration.
- 2026-07-03 19:30: CIF cache tar.gz (1.904 GiB) VERIFIED on Drive (0 diff).
- 2026-07-03 21:22: phyloP (9.192 GiB) VERIFIED on Drive (0 diff, 1 file).
- 2026-07-03 21:59: FinnGen VERIFIED on Drive (0 diff, 3 matching files).
- Drive quota confirmed: 5 TiB total, 4.03 TiB free -- ample.
- External drive D: confirmed exFAT (no FAT32 4GB limit), 676 GB free, writable ("My Passport").
- ALL SIX groups now disaster-safe on Google Drive. External D: fast-restore copy of FinnGen+phyloP PENDING.
- STILL PENDING: FinnGen+phyloP -> D: external; dbNSFP live-index restoration; deliberate local reclaim.
- NO LOCAL DELETIONS performed yet.
