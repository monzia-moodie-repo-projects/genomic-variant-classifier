# Run 16 Launch Runbook

Grounded in the Run-14 proven launch. Every command is PowerShell-safe: NO angle-bracket
placeholders (PowerShell treats `<` as a reserved operator). Where you must supply a value,
it is a quoted variable assignment you edit. Local preflight is already GREEN; this covers
instance creation through teardown. Author: Monzia Moodie.

Repo is PUBLIC (clone needs no token). Local SSH key: C:\Users\monzi\.ssh\id_lambda_run8.

---

## A. Create the instance (two separate paste blocks)

Offers are ephemeral -- search fresh; if `create` says the offer is gone, re-search.

```powershell
vastai search offers "reliability > 0.99 dlperf >= 80 pcie_bw >= 12 gpu_name = RTX_4090 cuda_max_good >= 12.0 disk_space >= 200 rentable = true" --order "dph_total" --limit 12
```

Pick the `ID` of a cheap offer with enough RAM for full-cohort data-prep (>= 32 GB RAM
preferred; the 1.49M-row pandas/feature step is memory-heavy). Then (edit the quoted value):

```powershell
$OfferId = "38381901"   # EXAMPLE ONLY -- offers expire; pick a FRESH verified offer with >= 64 GB RAM (prefer ~128) to avoid data-prep OOM
vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 200 --ssh --direct --label run16
```

Note the instance ID it returns (edit the quoted value), then wait for it to come up:

```powershell
$InstanceId = "12345678"   # the instance ID returned by 'create'
vastai show instance $InstanceId
```

Image note: Run 14 launched on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. If your
Run-15 prep settled on the Vast.ai PyTorch image instead, use that -- the runbook's clone
step (B) is conditional, so it is a no-op if the image already auto-cloned the repo.

---

## B. Get the endpoint and bootstrap the repo on the box

When `actual_status` is `running`, get the SSH URL and parse it (one source of truth):

```powershell
vastai ssh-url $InstanceId            # prints e.g. ssh://root@ssh5.vast.ai:23456

$Url = "ssh://root@ssh5.vast.ai:23456"   # paste the REAL line above, exactly
if ($Url -match 'ssh://(?:([^@]+)@)?([^:]+):(\d+)') { $SshUser=$Matches[1]; $SshHost=$Matches[2]; $SshPort=$Matches[3] }
"$SshUser @ ${SshHost}:${SshPort}"        # sanity print
```

Bootstrap: clone (idempotent) + check out the pushed HEAD. Uses the CRLF-stripped heredoc
pattern (standing PS hygiene rule):

```powershell
$Bootstrap = @'
set -e
cd /workspace
if [ ! -d genomic-variant-classifier ]; then
  git clone https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
fi
cd genomic-variant-classifier
git fetch origin
git checkout main
git pull --ff-only
echo "REMOTE_HEAD: $(git rev-parse --short HEAD)"
echo "BOOTSTRAP_DONE"
'@ -replace "`r`n", "`n"

$Bootstrap | ssh -i C:\Users\monzi\.ssh\id_lambda_run8 -p $SshPort -o StrictHostKeyChecking=accept-new root@$SshHost bash -s
```

Expected last lines: `REMOTE_HEAD: 5619f8e` (must match your local HEAD) then `BOOTSTRAP_DONE`.
If REMOTE_HEAD differs, the clone is stale -- the preflight in C will also flag it.

---

## C. Preflight gate (must be GREEN before staging)

```powershell
python scripts\preflight_run16.py --ssh-url $Url --ssh-key C:\Users\monzi\.ssh\id_lambda_run8
```

Expect: local checks OK, then `ssh connect` / `gpu` (4090) / `disk` (>= 25 GB) PASS.
If RED here, STOP and fix -- do not stage.

---

## D. Stage the data (preflight-gated; scp + verify + symlink bridge)

```powershell
python scripts\stage_run16.py --ssh-url $Url --ssh-key C:\Users\monzi\.ssh\id_lambda_run8 --scp-hf-cache
```

It re-runs the preflight gate, scp's the 10-file manifest into the cloned repo (do-not-ship
excluded), verifies each remote byte-size, scp's the ESM-2 HF cache, and creates the
/workspace/{data,outputs} symlink bridge (escalating any rm -rf for you to run manually).
It STOPS at the on-box gates. Add `--dry-run` first if you want to preview the exact plan.

---

## E. On the box: env, train, gates

SSH in:

```powershell
ssh -i C:\Users\monzi\.ssh\id_lambda_run8 -p $SshPort -o StrictHostKeyChecking=accept-new root@$SshHost
```

Then (bash, on the box):

> **CORRECTED 2026-07-13 -- DO NOT RUN THE OLD `sed` STEP.**
>
> This runbook used to instruct you to `sed -i` the **installed** `imodelsx` package file
> here, rewriting `test_size=test_size` to `test_size=self.test_size` (and likewise for
> `random_state` and `shuffle`) inside `site-packages`. **That step is gone. Do not perform
> it by hand, and do not restore it.**
>
> **What it was for.** `imodelsx` 1.0.13's `KANClassifier.__init__` accepts `test_size`,
> `random_state` and `shuffle` and then *discards* them, while its `fit()` reads them as
> **bare names** -- so `fit()` raises `NameError` on any unmodified install. The `sed`
> redirected the lookup onto `self`, and `models/kan.py` supplied `self.<name>`. Together
> they worked.
>
> **Why it was dangerous.** The `sed` ran only in the Run 11 / Run 16 launch scripts and on
> the developer's laptop. It **never** ran in Continuous Integration, **never** in Docker,
> and **never** in `scripts/vm_bootstrap_run.sh` -- the Run 17 path. So the Kolmogorov-Arnold
> Network raised `NameError` in every Continuous Integration run, the ensemble's bare
> `except Exception` swallowed it, and a **twelve-model** ensemble was trained and reported
> as healthy. It also left the developer's virtual environment holding a library **no clean
> machine had**, so "it passes on my machine" was load-bearing from 2026-05 until 2026-07-13.
>
> **What replaces it.** The repair is now applied **in-process**, at import, by
> `models/kan.py::_repair_imodelsx_kan_bare_names()`. It applies identically in every
> environment, handles both the pristine and the legacy sed-patched source form, and is
> covered by `tests/unit/test_kan_actually_fits.py`. Nothing needs to be patched by hand.
>
> `scripts/vm_bootstrap_run.sh` section E now **fits** `KANClassifier` (and every other base
> model) rather than merely importing it, and refuses the launch if any of them cannot train.
> An import check cannot see a bug in `fit()` -- which is precisely how this survived.

```bash
cd /workspace/genomic-variant-classifier
pip install -r requirements.txt --break-system-packages
# NO imodelsx sed. The KAN repair is in-process (models/kan.py). See the note above.
python -c "import catboost, lightgbm, xgboost, torch; print('env OK', torch.cuda.is_available())"
# Prove KAN actually TRAINS on this box before spending money -- import is not enough:
python -c "
import numpy as np
from genomic_variant_classifier.models.kan import KANClassifier, _IMODELSX_KAN_REPAIR
rng = np.random.default_rng(0); X = rng.standard_normal((40, 5)); y = (X[:, 0] > 0).astype(int)
KANClassifier(hidden_sizes=[8]).fit(X, y)
print('KAN FITS; imodelsx_repair =', _IMODELSX_KAN_REPAIR)
"
```

Train (the LAUNCH_CONTRACT_run16.md Sec.1 flag set; nohup so it survives SSH drops):

```bash
nohup python scripts/train.py \
  --clinvar           data/processed/clinvar_grch38_clean_seq.parquet \
  --alphamissense     data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
  --gnomad            data/processed/gnomad_v4_exomes.parquet \
  --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
  --dbnsfp-path       data/external/dbnsfp/dbnsfp_clinvar_index.parquet \
  --lovd-path         data/external/lovd/lovd_all_variants.parquet \
  --esm2-model        esm2_t33_650M_UR50D \
  --esm2-uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet \
  --esm2-device       cuda \
  --out-dir           outputs/run16 \
  > /workspace/run16_full.log 2>&1 &
tail -f /workspace/run16_full.log
```

After data-prep produces the splits, BEFORE trusting the run, the Sec.4 gates:

```bash
python scripts/run_schema_drift_check.py --matrix outputs/run16/splits/X_train.parquet   # green
python scripts/audit_smoke_feature_population.py outputs/run16/splits                     # LOVD must be > 0
```

Watch (LAUNCH_CONTRACT Sec.6): cnn_1d still ~0.5 at full scale = real defect; LOVD > 0;
blend weights must DIVERGE from uniform 0.0769; checkpoint each base model < 30 min else ABORT.

---

## F. SCP results back, then teardown (SEPARATE, after verifying the run)

```powershell
scp -i C:\Users\monzi\.ssh\id_lambda_run8 -P $SshPort -r root@${SshHost}:/workspace/genomic-variant-classifier/outputs/run16 C:\Projects\genomic-variant-classifier\outputs\
```

Only after outputs are safely back and verified (irreversible -- separate paste, manual verify):

```powershell
echo y | vastai destroy instance $InstanceId
```
