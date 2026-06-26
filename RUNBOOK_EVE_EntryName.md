# RUNBOOK — EVE entry-name → HGNC resolution (Run 17)

## The bug, in one line
EVE per-protein files are named by **UniProt entry name** (`1433G_HUMAN.csv`), so the
connector keyed its lookup on the prefix `1433G`; the cohort keys on the HGNC symbol
`YWHAG`, so the join silently missed and `eve_score` stayed `0.5` for the entire run.
Empirically **0/2** matched before the fix, **2/2** after.

## What the fix does
Adds an `entry_name` column to the UniProt index (`build_uniprot_index.py`, UniProt's
`id` field), then resolves each EVE filename `1433G_HUMAN → YWHAG` via that column
before keying the lookup (`eve.py`). The map path is threaded explicitly through
`AnnotationConfig` (`--eve-entry-map`) in **both** `run_phase2_eval` and
`regen_splits_local` (option A: regen now mirrors EVE). The flag is **independent**
in code; the launch script points both `--esm2-uniprot-index` and `--eve-entry-map`
at the same `$UNIPROT_INDEX` (one definition point → no drift). A **fail-loud** guard
warns if the map is empty or resolves <80% of files — so a stale/missing index is
never a silent zero.

---

## STEP 0 — Pre-flight (repo clean, on the right commit)
```powershell
cd C:\Projects\genomic-variant-classifier
git status                      # confirm clean / expected
git rev-parse --short HEAD
$env:PYTHONPATH = "src"
```

## STEP 1 — Dry-run the installer (no files changed)
```powershell
.\Install_EVE_EntryName_Resolution.ps1 -Check
```
Expect: every patcher reports `CHECK: ... anchors found`, no modifications.

## STEP 2 — Apply
```powershell
.\Install_EVE_EntryName_Resolution.ps1
```
Expect: 5 patchers `PASS`, py-compile OK, CRLF audit `0 of 6`, `bash -n` OK,
pytest green (the 2 corpus/index tests **SKIP** here — they need the rebuilt index).

## STEP 3 — Rebuild the UniProt index (REQUIRED — it now needs entry_name)
> ~30 MB streamed download from UniProt, 1–2 min. This is the one network step;
> it cannot be done offline.
```powershell
python scripts\build_uniprot_index.py
```
Expect: `Wrote NN,NNN genes -> data\external\uniprot\uniprot_human_reviewed.parquet`.
Then confirm the new column + corpus coverage (these tests now run, not skip):
```powershell
python -m pytest tests\unit\test_eve_entry_name_resolution.py -v -k "entry_name or corpus"
```
Expect: `test_uniprot_index_has_entry_name_column PASSED`,
`test_real_corpus_resolution_fraction PASSED` (≥80%; typically ~99%).

## STEP 4 — Re-run the empirical probe (the decisive 0/2 → 2/2)
```powershell
$env:PYTHONPATH = "src"
@'
import pandas as pd, shutil, tempfile
from pathlib import Path
from genomic_variant_classifier.data.eve import EVEConnector

src = Path("data/external/eve/EVE_all_data/variant_files")
idx = "data/external/uniprot/uniprot_human_reviewed.parquet"
tmp = Path(tempfile.mkdtemp(prefix="eve_probe_"))
pairs = [("1433G_HUMAN","YWHAG"), ("1433Z_HUMAN","YWHAZ")]
rows=[]
for stem,hgnc in pairs:
    f = src / (stem + ".csv"); shutil.copy(f, tmp / f.name)
    d = pd.read_csv(f, dtype=str); d.columns=[c.strip() for c in d.columns]
    d["EVE_scores_ASM"]=pd.to_numeric(d["EVE_scores_ASM"],errors="coerce")
    d=d.dropna(subset=["EVE_scores_ASM"]); d=d[d["wt_aa"].str.strip()!=d["mt_aa"].str.strip()]
    r=d.iloc[0]; pc="p.%s%d%s"%(r["wt_aa"].strip(), int(float(r["position"])), r["mt_aa"].strip())
    rows.append({"gene_symbol":hgnc,"protein_change":pc})
cohort=pd.DataFrame(rows)
conn=EVEConnector(eve_path=str(tmp), entry_map_path=idx)
out=conn.annotate_dataframe(cohort)
hits=int((out["eve_score"]!=0.5).sum())
print(out.to_string()); print("HGNC-key matches:", hits, "/ 2")
assert hits==2, "STILL BROKEN -- index missing entry_name or stale"
shutil.rmtree(tmp); print("PROBE PASS: 2/2")
'@ | python -
```
Expect: `HGNC-key matches: 2 / 2` and `PROBE PASS: 2/2`.

## STEP 5 — Re-upload the rebuilt index to Drive
> The parquet changed (new column), so the Drive copy is now stale.
```powershell
rclone copy data\external\uniprot\uniprot_human_reviewed.parquet `
  genvarcla:genomic-variant-classifier/data/external/uniprot/ -P
rclone lsf genvarcla:genomic-variant-classifier/data/external/uniprot --files-only
```

## STEP 6 — Unblock the launch EVE path + variant_files-only staging
> Until now the launch script pointed `--eve-path` at `data/external/eve` (the bundle
> root). EVE's non-recursive glob needs the **variant_files** subdir, and staging
> should tar only that (~10 GB) not the 63 GB bundle. This is a SEPARATE change —
> apply it only **after** STEP 4 proves 2/2 (wiring a path to a 0-coverage join would
> be the same silent-zero in disguise).
>
> These two edits are intentionally NOT in the installer (they touch the launch EVE
> path + a staging script and must follow the proven probe). Confirm the exact lines
> first, then patch:
```powershell
# Confirm current EVE path wiring + staging tar target before editing:
Select-String -Path scripts\launch_run17_baseline.sh -Pattern 'EVE_DIR=|--eve-path'
Select-String -Path scripts\Stage_Run17_EVE_ESM2.ps1 -Pattern 'eve|variant_files|tar' 2>$null
```
Then (after review) point `EVE_DIR` at `…/EVE_all_data/variant_files`, add a
`0-CSV abort` guard, and set the staging tar to `variant_files/` only.
*(Hold for a dedicated patch once you paste those two regions.)*

## STEP 7 — ALL-MODELS smoke WITH EVE/ESM-2 on
```powershell
# tiny train, no --skip flags, --string-db auto; watch the EVE coverage line go non-zero
python scripts\smoke_all_models.py --clinvar data\processed\clinvar_grch38.parquet `
  --eve-path data\external\eve\EVE_all_data\variant_files `
  --eve-entry-map data\external\uniprot\uniprot_human_reviewed.parquet `
  --esm2-uniprot-index data\external\uniprot\uniprot_human_reviewed.parquet `
  --string-db auto --smoke-n 3000
```
> NOTE: `smoke_all_models.py` does not yet accept `--eve-path/--eve-entry-map`. If it
> errors on those flags, that's a separate (small) wiring task — flag it and I'll add
> them the same way. Until then, exercise EVE via `regen_splits_local` (option A now
> wires it) on a tiny ClinVar slice and watch the `11/17 (EVE)` coverage log.

## STEP 8 — Provision + full run (only after smoke is clean)
State estimated time + $ cost, get explicit approval, then launch per the Run-17
preflight. After the run, postflight `verify_eve_esm2_coverage.py` confirms EVE
coverage is non-zero on the ELIGIBLE-MISSENSE denominator.

---

## Rollback
Every patcher wrote a `.bak` before editing. To revert a file:
```powershell
# example: revert eve.py
Copy-Item src\genomic_variant_classifier\data\eve.py.pre_entry_resolver.bak `
          src\genomic_variant_classifier\data\eve.py -Force
```
Backups: `*.pre_entry_name.bak` (builder), `*.pre_entry_resolver.bak` (eve.py),
`*.pre_eve_entry_map.bak` (real_data_prep / run_phase2 / regen / launch).
Prefer `git checkout -- <file>` if you want a guaranteed-clean revert.

## Cost note (option A consequence)
`regen_splits_local` now parses the EVE CSVs (option A mirror), so a prep-only split
check is no longer EVE-free — it will parse `variant_files` when `--eve-path` is
passed. Omit `--eve-path` for a fast split-only check; pass it when you want the
prep-check to also exercise EVE resolution before a full run.
