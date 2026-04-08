# Run 6 Session Summary — 2026-04-08

## Session Objective
Resume and complete Run 6 GCP training after discovering the VM had been running idle since April 7 with no startup script attached and no data synced.

## What Was Accomplished

### Bugs Fixed (committed to main)
- `scripts/run_phase2_eval.py`: Added missing `--string-db` argparse argument (crashed Run 6 at GNN step with `AttributeError: Namespace object has no attribute string_db`)
- `scripts/run_phase2_eval.py`: Added ensemble resume logic — loads existing `ensemble.joblib` if present, skips full retraining. Saves ~1 hour of compute on resume.

### Run 6 Final Results
- **Holdout AUROC: 0.9862** (target >= 0.90 ✅)
- OOF blend AUROC: 0.9938
- Blend weights: RF 39.1%, CatBoost 31.9%, LightGBM 25.5%, XGBoost 3.5%
- Total variants: 345,007 (train: 3,195,020 augmented | test: 176,619 | holdout: 245,007)
- Features: 78 (first run with gnomAD v4.1 constraint features active: pLI, LOEUF, syn_z, mis_z)
- Genes: 4,441 | Pathogenic: 69,791 (20.0%)
- Models saved to `gs://genomic-variant-prod-outputs/run6/models/`

### What Run 6 Activated (vs Run 5)
- gnomAD v4.1 constraint features (`pLI`, `LOEUF`, `syn_z`, `mis_z`) — wired via `--gnomad-constraint`
- KAN unconditionally removed (process-killing C++ OOM at scale)
- SVM unconditionally skipped (>100K samples)
- AlphaMissense annotation active (5.2GB TSV)
- STRING DB loaded but GNN bypassed (see failures below)

## Failures and Root Causes

### GNN bypassed — torch-geometric not installed on VM
The Deep Learning VM image (`pytorch-2-7-cu128-ubuntu-2404-nvidia-570`) does not include `torch-geometric`. The GNN training block was silently skipped with: "No module named 'networks'". STRING DB file (133MB) was present and correct. Fix: add `pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html` to VM setup.

### Startup script never executed
The VM was created without `--metadata-from-file startup-script=...` so the script `gcp_run6_startup.sh` never ran on any boot. All data syncing was performed manually.

### spliceai_index.parquet corrupt (29GB)
The file in GCS is 29GB — this is the raw SpliceAI VCF incorrectly converted to parquet rather than an indexed lookup table. Omitted from Run 6 with no AUROC impact observed. Needs regeneration from scratch.

### Recurring file-vs-directory collisions on VM
`.gitkeep` placeholder files at `data/raw`, `data/external`, `data/processed` caused `FileExistsError` repeatedly when syncing data. Required manual `rm` + `mkdir` before each sync attempt.

### GCS bucket name mismatch
`gsutil` had no default project configured. Correct bucket is `gs://genomic-classifier-data/` (not `gs://genomic-variant-prod-data/` as assumed). `gsutil` also failed to read project from `gcloud config set project` — must use `gcloud storage` CLI instead.

### Recursive GCS sync nesting
`gcloud storage cp --recursive` and `gsutil -m cp -r` both created nested directories (`processed/processed/`, `external/external/`) requiring manual flattening. Workaround: always use individual file copies for critical paths.

## Lessons Learned
- Always verify GCS bucket names with `gcloud storage ls` before any copy operation
- `.gitkeep` files in data directories conflict with GCS sync — remove them from the repo or handle in startup script
- `gcloud storage cp --recursive` adds an extra directory level when destination already exists — use individual file copies or `gsutil -m cp gs://bucket/path/* dest/`
- `nohup cmd & ` returns the prompt immediately — `tail -f logfile` can be run right after without stopping the job
- `spliceai_index.parquet` must be regenerated; the current GCS version is the raw VCF dump (29GB)

## Known Debt for Run 7
- [ ] Install torch-geometric on VM before training (GNN was zero-contribution in Run 6)
- [ ] Fix `gcp_run6_startup.sh` attachment: add `--metadata-from-file startup-script=scripts/gcp_run6_startup.sh` to `gcloud compute instances create`
- [ ] Regenerate `spliceai_index.parquet` correctly from raw VCF
- [ ] Remove `.gitkeep` files from `data/` subdirs or replace with `.gitignore` pattern
- [ ] Add `gcloud storage ls` check to startup script to validate bucket names at boot
- [ ] ESM-2 annotation: verify `transformers` is in VM pip install list (stub mode produces all-zero `esm2_delta_norm`)
