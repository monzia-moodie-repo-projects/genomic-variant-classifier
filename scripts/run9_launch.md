# Run 9 Launch Runbook

**Target**: AUROC > 0.990 with SpliceAI + GNN + 5 more base models active.
**Known state**: ESM-2 will be STUB MODE in Run 9. See
`docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md` for root
cause and the Run 10 remediation plan. Do not treat ESM-2 stub log
lines as failures during Run 9.
**Baseline**: Run 8 = AUROC 0.9863 (5 models active, GNN=0, SpliceAI=0,
ESM-2 stub). Run 9 should beat this with SpliceAI and GNN newly active.

## Pre-launch gate (local, before spending money)

```powershell
cd C:\Projects\genomic-variant-classifier
C:\Projects\genomic-variant-classifier\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# Clear pyc caches (standing practice)
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# Run the scripted preflight. Fails loudly on any gap.
python scripts\preflight_check.py
echo "preflight exit: $LASTEXITCODE"

# Fast iteration mode (skips the 5-10 min pytest):
#   python scripts\preflight_check.py --skip-pytest
# Offline mode (no gcloud auth):
#   python scripts\preflight_check.py --skip-gcs
```

**Do not proceed if exit code is non-zero.** Fix every FAIL line before
continuing. The preflight gates GPU spending; bypassing it is how we
got Run 8's silent-zero problems in the first place.

## Create the Vast.ai instance

Instance spec (matches Run 8's successful config):
- GPU: **RTX 4090** (24 GB VRAM; plenty for this workload)
- RAM: **>= 60 GB** (dbNSFP + SpliceAI lookups are memory-heavy)
- Disk: **>= 200 GB** (parquets + caches + model artifacts)
- CUDA: **12.x** (matches the Docker image torch was built against)
- Template: the PyTorch image you used for Run 8

Budget: Run 8 was 4270s on RTX 4090 @ $0.388/hr = ~$0.46. Run 9 will be
longer because GNN and 5 more base models train this time (ESM-2 is
still stub, so no ESM-2 cost). **Budget $5-10 and destroy the instance
the moment final metrics print**, per standing rule.

## On-VM setup (after SSH)

```bash
# 1. Clone and pull latest
cd /workspace
git clone https://${GITHUB_TOKEN}@github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
cd genomic-variant-classifier
git log -1 --oneline   # confirm HEAD matches what you committed

# 2. Pull data from GCS (gcloud should be pre-auth'd on the instance
#    image; if not, `gcloud auth login` then re-run)
mkdir -p data/external/spliceai data/external/string data/external/alphamissense data/raw/clinvar
gcloud storage cp gs://genomic-variant-prod-outputs/data/processed/spliceai_index.parquet data/external/spliceai/
gcloud storage cp gs://genomic-classifier-data/data/external/string/9606.protein.links.detailed.v12.0.txt.gz data/external/string/
gcloud storage cp gs://genomic-classifier-data/data/external/string/9606.protein.info.v12.0.txt.gz data/external/string/
# ... (ClinVar, AlphaMissense, dbNSFP, etc. per your Run 8 setup)

# 3. Install requirements. Pin transformers<5 based on local testing --
#    transformers 5.x is installed and importable locally but we have
#    not validated the ESM-2 connector's forward pass against it.
pip install -r requirements.txt
pip install "transformers>=4.40,<5.0"

# 4. Run the on-VM preflight
bash scripts/preflight_vm.sh
```

**If the VM preflight fails, DO NOT start training.** Paste the failure
output into a new Claude session if you need help.

## Launch training

```bash
# From the repo root, on the VM
mkdir -p logs
python scripts/run_phase2_eval.py \
  --output-dir /workspace/outputs/run9 \
  --string-db data/external/string/9606.protein.links.detailed.v12.0.txt.gz \
  2>&1 | tee logs/run9.log
```

## Success checklist -- grep these strings in logs/run9.log

### Required (non-ESM-2)
```bash
# GNN on GPU (not CPU fallback)
grep -E "GNN device: cuda|GNN.*cuda:0" logs/run9.log

# STRING DB loaded with non-zero edges (the whole point of Run 9)
grep -E "STRING DB.*loaded [1-9][0-9]* edges|STRING.*edges: [1-9]" logs/run9.log

# SpliceAI annotated non-zero variants
grep -E "SpliceAI.*[1-9][0-9]+ variants|splice.*annotated.*[1-9]" logs/run9.log

# AlphaMissense still working (regression check from Run 8 fix)
grep -E "AlphaMissense.*[1-9][0-9]+ variants" logs/run9.log

# Final holdout AUROC
grep -E "holdout.*AUROC|AUROC:.*0\.9[89]" logs/run9.log | tail -5
```

### ESM-2: stub is EXPECTED in Run 9, not a failure

```bash
# ESM-2 IS expected to log stub-mode WARNING in Run 9 (see INCIDENT doc).
# This grep SHOULD return 1+ line. Zero lines here would mean the
# connector was never called, which is a different bug.
grep -i "ESM-2 stub mode" logs/run9.log
# expected: 1+ lines

# ESM-2 should also log that it found the required columns missing.
# This is the informational log that pointed us to the parser gap.
grep -i "ESM-2: columns .* absent" logs/run9.log
# expected: 1+ lines mentioning {protein_pos, wt_aa, mut_aa}
```

### Post-training sanity check on the feature frame

```bash
# Confirm esm2_delta_norm column exists (even though all-zero)
# AND that it is indeed all-zero (not partially-populated garbage).
python -c "
import pandas as pd
df = pd.read_parquet('outputs/run9/features_holdout.parquet')
d = df['esm2_delta_norm']
print(f'esm2_delta_norm: n={len(d)} zero_frac={(d==0).mean():.4f} '
      f'mean={d.mean():.6f} max={d.max():.6f}')
"
# Expected in Run 9: zero_frac == 1.0000, mean == 0, max == 0.
# When Run 10 lands the HGVSp parser, this should flip to
# zero_frac << 1 and max > 0.
```

## Shutdown -- the moment you see final metrics

1. **Save outputs to GCS** (do this before destroying):
   ```bash
   gcloud storage cp -r /workspace/outputs/run9 gs://genomic-variant-prod-outputs/run9/
   gcloud storage cp logs/run9.log gs://genomic-variant-prod-outputs/run9/logs/
   ```

2. **Verify the GCS upload landed** (2026-04-17 INCIDENT-verification rule):
   ```bash
   gcloud storage ls -l gs://genomic-variant-prod-outputs/run9/
   # Paste this output into the session doc.
   ```

3. **Destroy the Vast.ai instance** -- web console -> Destroy (not Stop).
   Stop keeps storage billing alive.

4. **Confirm $0/hr billing** at https://cloud.vast.ai/billing/.

**This order matters.** Shutdown ordering is critical: upload must
complete BEFORE destroy.

## Post-run documentation (mandatory, per standing rule #2)

After shutdown:

1. `docs/sessions/SESSION_YYYY-MM-DD_run9.md` -- full session record:
   - metrics table (AUROC, AUPRC, MCC, F1, Brier)
   - OOF AUROC per base model (all 10, to confirm which contributed)
   - GNN edge count (proves STRING DB loaded)
   - `esm2_delta_norm` statistics: zero-fraction (should be 1.0), max,
     importance rank (should be at or near 0)
   - SpliceAI feature importance rank
   - total GPU cost and wall time
   - gcloud storage ls receipt showing Run 9 outputs in GCS

2. Append to `docs/CHANGELOG.md` under `## YYYY-MM-DD --- Run 9`.

3. Update `docs/ROADMAP.md`:
   - check off SpliceAI / GNN activation items
   - ESM-2 item still OPEN, blocked on HGVSp parser (Run 10)
   - formally schedule Spectral Path Regression (Coombs 2025) eval
     for Run 10 alongside the HGVSp parser

4. Commit docs as a separate commit from any code changes.

## What to do if it fails partway

If training crashes or a feature (other than ESM-2) is silently zero:

1. Destroy the instance anyway -- debugging doesn't require GPU billing.
2. Copy logs to GCS and to local disk.
3. Write `docs/incidents/INCIDENT_YYYY-MM-DD_run9-<slug>.md` with:
   - exact error
   - root cause (if determined) or "unknown, see logs"
   - the gcloud storage ls confirmation that relevant GCS files exist
   - the plan to fix before Run 10

The 2026-04-17 INCIDENT-verification standing rule applies: no
"RESOLVED" without a gcloud storage ls receipt.