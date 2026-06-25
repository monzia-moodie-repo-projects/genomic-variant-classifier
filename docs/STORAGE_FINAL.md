# STORAGE_FINAL — Canonical Storage Architecture (2026-06-25)

**Owner:** Monzia Moodie · **Project:** GenAssoc v1 (genomic-variant-classifier — *whole-genome*, multi-modal variant pathogenicity classifier)
**Status:** Drive consolidation COMPLETE & certified. Single canonical root. No scatter on G:. Source root retired.

> This document supersedes prior `SESSION_2026-06-24_storage_architecture.md` / `STORAGE_RUNBOOK_2026-06-24.md` draft notes for the storage layer.

---

## 1. The two-root problem (resolved)

Before 2026-06-25 there were **two** Drive trees on `genvarcla:` (Google Drive / Google One, NOT paid GCS):

| Root | Role | Disposition |
|------|------|-------------|
| `genvarcla:genomic-variant-data/` | scattered consolidation tree (gnomAD 24/24 landed here) | **RETIRED 2026-06-25** — all contents migrated, dir removed, verified empty (`directory not found`) |
| `genvarcla:genomic-variant-classifier/` | the manifest-declared base (`configs/data_manifest.yaml` → `base: genomic-variant-classifier/data`) | **CANONICAL going forward** (Monzia's decision 2026-06-25) |

Everything from `genomic-variant-data/` was moved server-side (rclone, metadata-only, no bytes through the laptop) into `genomic-variant-classifier/`, then the old root was deleted.

---

## 2. Canonical Drive layout (`genvarcla:genomic-variant-classifier/`)

```
genomic-variant-classifier/
├── data/
│   ├── external/        # third-party reference inputs (read-only)
│   │   ├── gnomad/       # 24 exome VCFs (chr1-22,X,Y) + 2 constraint files
│   │   ├── finngen/      # R12 .gz + r13/annotations/R13 .gz + r13_docs/ (9)
│   │   ├── reference/    # GRCh38 primary assembly .fa (+ .fai)
│   │   ├── revel/        # revel_with_transcript_ids (6.5 GB) + v1.3 zip archive
│   │   ├── eve/          # 14,933 files (EVE_all_data variant_files + MSAs)
│   │   ├── gtex/ dbsnp/ clingen/ omim/ gencode/ alphamissense/
│   │   ├── clinvar/ clinvar_fresh/ dbnsfp/ spliceai/ string/ uniprot/
│   │   ├── alphafold/ esm2/ hgnc/ phylop/ rnaseq/ vep/ lovd/ 1kgp/ 1000genomes/
│   ├── cache/           # esm2/esm2_cache.sqlite (data-adjacent build cache)
│   ├── processed/  raw/  reference/  splits/  models/  synthetic/
├── models/             # trained pipelines: v1/, phase2/, phase4/
├── outputs/            # runs/ (run9-15 ensembles + run15_full.tar), experiments/, phase2_*/, ...
├── results/
│   └── manifests/      # 14 provenance CSVs + sha256 (migration audit artifacts)
├── config/  logs/  notebooks/  src/
```

**Placement principle (anti-scatter):** `data/` holds *data inputs* only. Trained models → repo-level `models/`. Run/experiment outputs → `outputs/`. Provenance/audit artifacts → `results/manifests/`. Build caches that are data-adjacent → `data/cache/`. This mirrors the repo-level structure so local (C:) and cloud (G:) **complement, not conflict**.

---

## 3. gnomAD raw — integrity table (all human chromosomes, NO gaps)

24/24 v4.1 **exome** site VCFs verified in `data/external/gnomad/`. Spot-check MD5s (match upstream source exactly):

| Contig | MD5 |
|--------|-----|
| chrY | `d500cf5a73c53f02d1b95f1e092f2e49` |
| chr7 | `c41cd52571b001cf7d3e388a668e4dff` |
| chrX | `5b7b17d3d4cff22c20480a908c861a28` |

Full set present: chr1–chr22, chrX, chrY (`*.vcf.bgz`, count = 24). Constraint metrics: `gnomad.v4.1.constraint_metrics.tsv` (md5 `14df4b2acb581fcbbb2a82a3a555fd35`), `...constraint_index.parquet` (md5 `dbfdf2b976dfc699c7a16e490b92f454`).

**Scope note:** these are *exome* VCFs. gnomAD *genome* VCFs are a separate future acquisition if/when whole-genome scope requires them — not a gap in the current exome set, a scope decision.

---

## 4. Dedup verifications (hash-confirmed before any deletion)

| Asset | Source hash | Canonical hash | Verdict |
|-------|-------------|----------------|---------|
| finngen R12 .gz | `ea0736347bb00bbff62757e59b8cba8f` | `ea0736347bb00bbff62757e59b8cba8f` | identical → source deleted on move |
| reference .fa | `a65212262de00761b43869d5c08c8e4d` | `a65212262de00761b43869d5c08c8e4d` | identical → source deleted on move |

---

## 5. Drive storage rules (STANDING)

- **`genvarcla:` = Google Drive (Google One ~5TB), NOT paid GCS.** gsutil broken at SDK level — use `gcloud storage` + `rclone` only.
- **G: (Drive-for-Desktop / DriveFS) is a streaming cache-view, NOT a disk.** Its free space is synthetic; large writes cache onto C: (`%LOCALAPPDATA%\Google\DriveFS`) → caused all past robocopy ERROR 112. **NEVER bulk-write large files through G:.**
- **Use rclone (Drive API direct) for ALL Drive reorg** — server-side moves are metadata-only; no bytes traverse the laptop.
- **All large files live in the Google cloud location** and are pulled as needed; local C: keeps working copies + code-referenced paths.
- **Redundancy only where absolutely necessary** for integrity (e.g., REVEL zip archive kept alongside uncompressed working file as pristine source).

### rclone gotchas learned this migration
- `rclone move <dir>` of a *whole directory* with `--dry-run` prints "Skipped server-side directory move" and **enumerates nothing** — dir-level dry-run reveals nothing about contents. Use file-level `lsf -R` to inspect first.
- `rclone lsf --files-only` **without `-R`** counts only the top level — a nested file (e.g. finngen R13 under `r13/annotations/`) reads as "missing". **Always `-R` for count checks.**
- A name/path mismatch (file directly under `external/` vs assumed `external/revel/`) produces persistent "object not found" on every delete/move verb while the file still *lists*. Diagnose with `rclone lsjson --recursive` (returns ID/Size/IsDir/MimeType) before forcing anything. The REVEL phantom was this — resolved by `moveto` with the corrected path.
- `rclone rmdirs` removes only EMPTY dirs (no-op if content present). "directory not found" on `lsf` of a removed path = success signal.

---

## 6. C: local layout reconciliation (in progress)

C: must complement Drive. Verdicts:

| C: dir | Verdict |
|--------|---------|
| `model/` (7) | pykan side-effect checkpoint dir — **gitignored** (line 99). Not the model store. |
| `models/` (240) | real model store (mirrors Drive `models/`). Keep. |
| `manifests/` (2) | regenerable layout-audit output (`build_data_layout_manifest.py`) — **gitignored + committed** (caf5ecd). Distinct from Drive `results/manifests/`. |
| `lovd/` (7 JSON) | **stale subset** of `data/external/lovd/raw/` (which has the same JSONs + .txt + _api.json). Pending hash-confirm → delete. |
| `agent_data/` (85) | one-off `arch_cleanup_stage*` scripts + logs. Untracked. Archive or leave. NOT the agent SharedState (that's `src/.../agent_layer/agent_state.json`). |

---

## 7. Re-pull commands (disaster recovery)

```powershell
# gnomAD raw exomes (public, free, gs bucket):
rclone copy genvarcla:genomic-variant-classifier/data/external/gnomad <local-dest>
# reference genome (Ensembl ~3GB, slow):
#   Homo_sapiens.GRCh38.dna.primary_assembly.fa + samtools faidx for .fai
# All canonical external data:
rclone copy genvarcla:genomic-variant-classifier/data/external <local-dest>
```

---

## 8. Provenance

Migration executed 2026-06-25, server-side via rclone on `genvarcla:`. Repo HEAD at close: **caf5ecd** (`origin/main`). No data lost; every dataset hash-verified in its new canonical home; source root `genomic-variant-data/` retired and verified empty.
