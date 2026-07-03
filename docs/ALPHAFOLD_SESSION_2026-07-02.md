# AlphaFold Phase-D Activation — Session Record, 2026-07-02

**Scope.** Bring the four AlphaFold structural features (`alphafold_plddt`,
`solvent_accessibility`, `secondary_structure_context`, `dist_to_active_site`)
from "wired, stubbed" to a validated, buildable cohort parquet. Repo:
`C:\Projects\genomic-variant-classifier`. Environment: Windows 11 Pro,
PowerShell 5.1, Python 3.12.10, `.venv312`.

This document records every defect found, its diagnosis, the fix, and the
verification, so any researcher can reconstruct exactly what happened and why the
final coverage decision was made.

---

## Entry state

The four AF features were wired parquet-first (`AnnotationConfig.alphafold_path`
-> `real_data_prep` step-14a -> `run_phase2_eval` / `launch_run17_baseline.sh`),
feature count 91 (unchanged; data-source change, not schema), 15/15 AF unit tests
green, but no cohort parquet existed. The verify script `verify_alphafold_build.ps1`
Stage 2 (`--max-genes 50`) aborted with exit 4 ("zero residue features extracted").

## Defect 1 — stale AlphaFold DB download URL (v4 -> v6)

**Symptom.** Every CIF fetch 404'd; Stage 2 fast-failed with zero structures.
**Root cause.** `build_alphafold_parquet.py` hard-coded
`.../AF-{acc}-F1-model_v4.cif`. AlphaFold DB released v6 (synced to UniProt
2025_03); coordinates carried over from v4 but entries were relabelled and old
per-file `v4` URLs 404 for UniProt-synced entries. Confirmed live: `P04637`
(TP53) `v4` -> 404, `v6` -> 200; prediction API reports `latestVersion=6`.
**Fix (Install_alphafold_url_fix.ps1).** Resolve the current cif URL from the
prediction API (`/api/prediction/{acc}` -> `cifUrl`) rather than templating a
version; save under the server's filename; reject any non-CIF payload (a non-200
or non-`data_`/`_atom_site` body is never written). Added 4 offline fetch-path
unit tests (previously the fetch path had zero coverage — which is how a stale
URL passed 15/15). Live end-to-end fetch of P04637 confirmed a real v6 CIF.

## Defect 2 — O(n^2) Shrake-Rupley RSA (large-protein bottleneck)

**Symptom.** After Defect 1, Stage 2 appeared to hang. Profiling showed
`per_residue_rsa` was 21.66 s of a 21.7 s extraction for A2M (1474 res, 11,496
atoms); the all-atoms neighbour scan is O(n^2). A2M-scale was tens of seconds;
TTN-scale (~34k res) would have been hours -> the apparent "hang".
**Fix (Install_alphafold_rsa_vectorize.ps1).** Vectorized neighbour search
(scipy cKDTree ball query) + numpy occlusion, PROVEN numerically identical to the
original (max RSA diff 0.0 at the real 192-point sphere, on synthetic structures
up to DMD scale; empty-input and fail-loud parity verified). A2M extraction:
~23 s -> 2.0 s on the target machine, 1474 rows (identical). Added a naive-
reference identity test (fast path must equal O(n^2) bit-for-bit) and a perf-
regression tripwire. Also added per-gene progress logging to the driver so a slow
run is never again mistaken for a hang.

**Stage 2 result after Defects 1-2:** exit 0, 50/50 structures, 50,335 rows,
sentinel fractions 0.000, real active-site distances (0.0 .. 181.32). PASS.

## Defect 3 — silent isoform mis-selection (data-correctness)

**Symptom (found via a large-protein timing probe).** Several Tier-A giants
returned "download None" (TTN, MUC16, OBSCN, MUC5AC), and the prediction API
returns MULTIPLE records for ~54% of genes.
**Root cause.** The API returns one record PER ISOFORM, entryId
`AF-{acc}-{N}-F1`, each with isoform-specific residue numbering. The fix from
Defect 1 took `data[0]`, which is not guaranteed to be the canonical record.
Observed: AARS1 `data[0]` = canonical (968) but `data[1]` = a LONGER isoform
(992); "pick longest" would choose wrong. DYST/SYNE1 have NO record matching the
canonical sequence (canonical 7570/8797; longest available 2649/1725). Attaching
an isoform structure to canonical `protein_pos` silently mis-numbers all four
features — worse than a sentinel because it is not flagged.
**Fix (Install_alphafold_canonical_selection.ps1).** Select the record whose
`uniprotSequence` EXACTLY equals our canonical UniProt index sequence; if none
matches, return None -> the gene is a documented coverage miss (sentinel), never a
mis-numbered substitute. Threads the canonical sequence through
`_download_cif`/`_resolve_cif_url` via an accession->sequence map. Verified on
real records: AARS1/ABCB1 -> base canonical record; DYST/SYNE1 -> None. Added
3 unit tests pinning this behaviour.

## Coverage decision (ACCEPTED, documented)

A canonical-sequence-match audit over 400 sampled cohort genes gave a
**98.2% usable (canonical structure) rate**. Unusable genes are AFDB length-ceiling
giants and isoform-only entries. These are ACCEPTED as a documented coverage gap:
their variants receive structural sentinels for the four AF features but retain all
other ~87 features AND the ESM-2 sequence branch (no length ceiling), so they are
structurally-blind but not feature-blind. ESMFold-folding the giants remains a
clean future enhancement, not a blocker.

Known unusable, high-variant genes (missense variant counts):
- TTN (14,397), NEB (4,263), OBSCN (3,901), PLEC (3,757), CDH23 (2,184),
  SZT2 (1,643), SRRM2 (968), MUC16 (297), ITPR2 (294), and others.

The build now writes `alphafold_coverage.json` (per gene: canonical-ok / unusable
/ selected entryId) next to the cohort parquet, and a dormant hard gate aborts
(exit 5) only if the usable-gene fraction falls below 0.90 — catching a wholesale
selection regression while passing the accepted ~2% gap.

## Verification standard used throughout

Every code change was delivered as a hash-verified `Install_*.ps1` with SHA-256
backups, guarded single-match `str.replace` (count==1 abort), byte-compile, full
unit-suite (0 skipped), and a live end-to-end post-check gate. Each installer's
patcher was additionally extracted verbatim from the finished script and re-run
against a faithful source reconstruction before delivery.

## Remaining before the full build

1. Run `Install_alphafold_canonical_selection.ps1`; re-run the 50-gene probe.
2. Parallelization decision: serial full build is ~15-24 h (dominated by ~2.6 s/gene
   network x 18,302). Concurrent downloads + a process pool for extraction is the
   next validated change if wall-clock matters.
3. Local checkpoint commit of the AF files + wiring edits (restore point). Hold the
   push until `origin` is reset from `monzia-moodie-repo-projects` to the reclaimed
   `monzia-moodie` root: `git remote set-url origin https://github.com/monzia-moodie/genomic-variant-classifier.git`.
