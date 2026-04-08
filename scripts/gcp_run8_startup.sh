#!/bin/bash
# =============================================================================
# GCP VM Startup Script — Run 8
# GPU-enabled (T4). Runs automatically on boot.
# Incorporates all lessons from Runs 6 and 7.
# =============================================================================
set -euo pipefail
exec > /var/log/genomic_run8.log 2>&1

echo "=== Startup: $(date) ==="

# --- Constants ---------------------------------------------------------------
REPO_DIR=/home/monzi/genomic-variant-classifier
GCS_DATA=gs://genomic-variant-prod-outputs/run6/data/data
GCS_OUT=gs://genomic-variant-prod-outputs/run8

# --- Trap: upload models and shut down on ANY exit ---------------------------
# Fires on success, failure, crash, OOM, or any other cause.
trap 'echo "=== EXIT TRAP: $(date) ===" >> /var/log/genomic_run8.log
      gsutil -m cp -r $REPO_DIR/models/v1/ $GCS_OUT/models/ 2>/dev/null || true
      gsutil cp /var/log/genomic_run8.log $GCS_OUT/logs/startup.log 2>/dev/null || true
      gsutil cp $REPO_DIR/logs/training_run8.log $GCS_OUT/logs/training_run8.log 2>/dev/null || true
      echo "=== Shutting down: $(date) ===" >> /var/log/genomic_run8.log
      sudo shutdown -h now' EXIT

# --- GPU verification ---------------------------------------------------------
echo "=== Verifying GPU: $(date) ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'FATAL: cuda not available — aborting Run 8'
print('GPU ok:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
"
echo "=== GPU verified: $(date) ==="

# --- Fix git safe directory (startup runs as root, repo owned by monzi) ------
git config --global --add safe.directory $REPO_DIR

# --- Pull latest code --------------------------------------------------------
cd $REPO_DIR
git pull || true

# --- Fix .pth bridge: venv sees system torch ---------------------------------
SITE_PKG=/home/monzi/venv/lib/python3.12/site-packages
grep -q "/usr/local/lib/python3.12/dist-packages" $SITE_PKG/system.pth 2>/dev/null || \
    echo "/usr/local/lib/python3.12/dist-packages" >> $SITE_PKG/system.pth
grep -q "/usr/lib/python3/dist-packages" $SITE_PKG/system.pth 2>/dev/null || \
    echo "/usr/lib/python3/dist-packages" >> $SITE_PKG/system.pth

# --- Activate venv -----------------------------------------------------------
source /home/monzi/venv/bin/activate

# Remove any broken pip-installed torch from venv
pip show torch 2>/dev/null && pip uninstall torch -y || true

# Verify venv sees system torch, pandas, and pyg
python -c "import torch; import pandas; print('torch', torch.__version__, '| pandas', pandas.__version__, '| cuda:', torch.cuda.is_available())"

# --- Install torch-geometric system-wide (depends on system torch) -----------
echo "=== Installing torch-geometric: $(date) ==="
sudo pip3 install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html \
    --break-system-packages --quiet
python -c "import torch_geometric; print('pyg ok:', torch_geometric.__version__)"

# --- Verify transformers for ESM-2 -------------------------------------------
python -c "import transformers; print('transformers ok:', transformers.__version__)" || \
    pip install transformers --quiet

# --- Fix .gitkeep collisions -------------------------------------------------
for p in data/raw data/processed data/external; do
    [ -f "$p" ] && rm "$p" && mkdir -p "$p" || true
done

# --- Sync data from GCS ------------------------------------------------------
echo "=== Syncing data: $(date) ==="
gcloud config set storage/parallel_composite_upload_enabled False

mkdir -p data/processed data/external/gnomad \
         data/external/string data/raw/cache \
         data/external/alphamissense/AlphaMissense_hg38.tsv \
         logs models/v1

gcloud storage cp $GCS_DATA/processed/clinvar_grch38.parquet          data/processed/
gcloud storage cp $GCS_DATA/processed/gnomad_v4_exomes.parquet        data/processed/
gcloud storage cp $GCS_DATA/processed/gene_pathogenic_counts.parquet  data/processed/
gcloud storage cp $GCS_DATA/processed/gene_summary.parquet            data/processed/
gcloud storage cp $GCS_DATA/processed/dbsnp_index.parquet             data/processed/
gcloud storage cp $GCS_DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
                 data/external/gnomad/
gcloud storage cp "$GCS_DATA/external/alphamissense/AlphaMissense_hg38.tsv/AlphaMissense_hg38.tsv" \
                 "data/external/alphamissense/AlphaMissense_hg38.tsv/"
gcloud storage cp $GCS_DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz \
                 data/external/string/
gcloud storage cp $GCS_DATA/external/string/9606.protein.info.v12.0.txt.gz \
                 data/external/string/
echo "=== Data sync complete: $(date) ==="

# --- Launch Run 8 ------------------------------------------------------------
echo "=== Starting Run 8: $(date) ==="
PYTHONPATH=$REPO_DIR python scripts/run_phase2_eval.py \
    --clinvar       data/processed/clinvar_grch38.parquet \
    --gnomad        data/processed/gnomad_v4_exomes.parquet \
    --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
    --alphamissense "data/external/alphamissense/AlphaMissense_hg38.tsv/AlphaMissense_hg38.tsv" \
    --string-db     data/external/string/9606.protein.links.detailed.v12.0.txt.gz \
    --min-review-tier 2 \
    --skip-svm --skip-nn --n-folds 5 \
    --output models/v1 \
    2>&1 | tee logs/training_run8.log

echo "=== Run 8 complete: $(date) ==="
# trap fires here automatically — uploads models and shuts down
