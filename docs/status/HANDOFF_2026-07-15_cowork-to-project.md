# HANDOFF — 2026-07-15 — Claude Cowork → Claude Project

**For Monzia. Written to be pasted into a Project session and read cold by a Claude with no
prior context.**

**Read this whole document before touching anything. Then read, in this order:**
1. `memory.md` (the Project's standing-instruction record) — **non-optional, and the single
   biggest failure of the Cowork session was not reading it**
2. `CLAUDE.md` (repository operating doctrine)
3. `docs/ROADMAP.md` §6A (current state), §6B (forward plan), §6 (open register), §7 (the
   four root patterns)
4. `docs/status/STATUS_2026-07-15_evidence-separated.md` (every claim sorted into
   MEASURED / INFERRED / UNKNOWN / WRONG)

---

## 0. WHAT THIS SESSION WAS, AND HOW MUCH TO TRUST IT

A single Claude Cowork session on **2026-07-15**, working directly against
`C:\Projects\genomic-variant-classifier` with Windows-side file tools and PowerShell run by
Monzia.

**It produced five wrong conclusions.** They are listed in §7 of this document and in §4 of
the status file. The pattern is consistent and worth knowing before you inherit anything:

> **When it measured, it was right. When it extrapolated past the measurement, it was wrong.**

`n_poly: 21814` was real. `is_fast: False` was real. `1966 collected` was real. Every failure
was an inference shipped as a finding.

**Therefore:** treat §2 (measured) as reproducible-on-re-run. Treat §3 (inferred) as
unproven. Do not build on §3.

---

## 1. STATE OF THE TREE — READ THIS FIRST

| | |
|---|---|
| **HEAD** | `f078953` — plus commits made at the end of this session (see `git log`) |
| **PUSHED?** | **NO. Nothing has been pushed.** Continuous Integration has seen none of it. |
| **Suite** | **1966 collected**, **9 FAILING** (see §5) |
| **Working tree** | Uncommitted changes remain — `git status --short` |
| **Last sealed run** | **Run 15** (`032a2ab`, 2026-06-09). Run 16 was an expensive failure. Run 17 not launched. |

**`docs/sessions/_chat_transcripts/` may exist and must be deleted.** ~60 MB of chat
transcript was copied into the repository during this session while diagnosing a tooling
question. It is untracked. **Delete it; it must never be committed to a public repo.**

---

## 2. MEASURED — reproducible; the command is named

### Contract and roster

| thing | value | command |
|---|---|---|
| Tabular features | **95** | `EXPECTED_TABULAR_FEATURE_COUNT`, enforced against `TABULAR_FEATURES` at import |
| Base models | **13** | live `VariantEnsemble.base_estimators` |
| Agents | **22** | `scripts/check_agents_active.py` → *"22 agents (registered=22, scheduled=22) … 0 dormant"* |
| Tests | **1966** | `pytest tests/ --collect-only -q` → "1966 tests collected in 16.76s", exit 0 |
| Last agent run | **2026-06-20 (`age=25.06d`)** | reported by every agent — **while the checker calls them ACTIVE** |

### Cohort

```
clinvar_grch38_pathfix.parquet         4,420,180 rows | 28,350 genes
clinvar_grch38_clean.parquet           4,399,089 rows | NO fasta_seq_ref/alt
clinvar_grch38_clean_seq.parquet       4,399,089 rows | HAS fasta_seq_ref/alt | NO `ok`
seq_windows/seq_windows.parquet        4,420,180 rows | HAS fasta_seq_ref/alt + `ok` + `reason`

pathfix − clean = 21,091   <-- see §3.1, this number matters enormously

pathogenicity:  uncertain 2,718,963 (61.5%) | likely_benign 1,083,576 | benign 276,240
                pathogenic 229,602 | likely_pathogenic 111,799     [sums to 4,420,180]
Trainable (excl. uncertain): 1,701,217  — 341,401 pos / 1,359,816 neg (1:3.98)
fasta_seq column: null 4,420,180 / non-null 0   <-- 100% EMPTY
```

### Sequence windows

```
seq_windows.manifest.json:
  n_rows_built 4,420,180 | n_ok 4,398,366 | n_poly 21,814  (0.4935%)
  poly_reason_breakdown: empty_allele 19,988 | non_acgt_allele 1,771
                         ref_mismatch 53 | fetch_failed 2      [sums to 21,814]
  builder_version: delta_window_builder/2026-07-10-stepB
  cohort_path: clinvar_grch38_pathfix.parquet
  build_utc: 2026-07-11T01:24:22Z
```

### Hardware / storage

```
Get-Disk       -> Number 0 ONLY.  NVMe KIOXIA 1024GB, 953.87 GB, BusType RAID
Get-Partition  -> every partition DiskNumber 0.  C: = partition 3, 935.59 GB
Get-Volume     -> C: 935.59 GB, 6.63 GB free.  G: NOT LISTED (cloud mount)
Get-PSDrive C,G -> G: Description = "Google Drive"
```

**ONE physical disk. ~6.62 GB free.** `G:` is DriveFS streaming cache (memory.md: rclone
remote `genvarcla:`, ~5 TB; **never bulk-write through `G:`**). Its `Free` property reports
the host disk and is meaningless.

**Reclaimable locally (~78 GB), none of it a genomic source:**
- `AppData\Local\Docker\wsl\disk\docker_data.vhdx` — **72.46 GB**
- `genomic-variant-classifier_GITBACKUP_2026-07-11` — 1.8189 GB (CI green; handoff says safe)
- `genomic-variant-classifier_BACKUP_2026-07-11` — holds the **1.90 GB AlphaFold blob excised
  from git history**, still on disk
- project `outputs/` — 9.80 GB (gitignored, handoff-classified reclaimable)

**Archive candidates (~196 GB):** gnomAD VCFs in `C:\Users\monzi\data\` — **`chr17` appears
TWICE** (`data\raw\` and `data\external\`), so this starts with de-duplication, not upload.

### The four architectures — none is complete

| | state | evidence |
|---|---|---|
| **JEPA** | **NOT STARTED** | `jepa` appears in 2 files, both prose. `pretrain`/`self_supervised`/`ssl`/`embedding_dim` → zero hits in `src/`. |
| **Conformal** | **6 of 14 modules** | `conformal/` has `scores`, `split`, `calibrate`, `coverage`, `grouped`, `mondrian`. 28 tests. LAC matches MAPIE **element-wise exactly**. MISSING: RAPS, ordinal CRC, multilabel, gene_ranking, risk_control, artifacts, config, subgroup, evaluation, monitoring. |
| **Metric stack** | **PARTIAL** | `evaluation/metrics.py`, 30 tests: AUROC, AUPRC+lift, Brier, ECE, calibration slope/intercept, bootstrap CI, stratified. Required panel is several times larger. |
| **RNA** | **EXPRESSION ONLY** | `rnaseq.py` (5 feats), `gtex.py` (6), `rna_pipeline.py` (splice). **No RNA sequence foundation model. No transcriptomic foundation model.** Nothing from the 2026-07-15 RNA specification exists. |

### Foundation-model layer actually present

WIRED: Nucleotide Transformer (DNA), ESM-2 (protein), AlphaFold (structure), GAT + hetero-KG
(networks).
**ABSENT: RNA sequence, RNA structure, transcriptomics, DNA long-context.**

**UNDOCUMENTED FINDING:** `data/primateai3d.py` exists as a connector and **no PrimateAI-3D
feature is in `TABULAR_FEATURES`.** A dormant connector nobody has recorded — HGMD's exact
shape before 6.21a. Disposition it: wire it, or record why not.

---

## 3. INFERRED — unproven; do NOT build on these

### 3.1 THE 21,091 / CLEAN-COHORT HYPOTHESIS — the highest-stakes open question

`memory.md` records:
- *"Clean cohort is required: `clinvar_grch38.parquet` raw must go through
  `clean_cohort.py --apply` before use; null ref/alt rows cause gate failures and are
  scientifically incorrect"*
- Stage-1 capstone smoke FAILED; root cause = *"`clinvar_grch38.parquet` retains **21,091**
  structural/CNV rows with null/empty ref/alt"*

**4,420,180 − 4,399,089 = 21,091.** Exactly.

**HYPOTHESIS:** `clean`/`clean_seq` are the **cleaned** cohort (alleleless removed, per the
standing rule); `seq_windows.parquet` was built from **`pathfix`, which still contains them**
— and its `empty_allele: 19,988` are those very rows. If so, **the artifact carrying
provenance is the one built on the wrong cohort**, which is the inverse of what roadmap 6.29
originally claimed.

**NOT VERIFIED. Verify before acting:**
```powershell
# Does clean_cohort.py --apply remove exactly 21,091? Are the removed rows the alleleless ones?
# Does pathfix minus alleleless == clean's 4,399,089?
```

### 3.2 `maxentscan_delta` may be the next `genomiclm_llr` — HIGHEST-VALUE OPEN LEAD

```python
pipelines/rna_pipeline.py:340
    ref_col = alt_col = df["fasta_seq"].fillna("")
```

`ref == alt`, reading a column **measured at 100% NULL across all 4,420,180 rows**. This is
the *fallback* beneath a preferred `fasta_seq_ref`/`fasta_seq_alt` branch.
`maxentscan_delta` is in `TABULAR_FEATURES` **and `build_reference_slice` feeds it**
(`rng.uniform(-10, 10, n)`).

**That is 6.27's exact shape:** feature dead on real data, fixture supplying what the
connector never produces, stage 5 grading the fixture.

**Falsifier:** read `maxentscan_delta`'s distribution in a real engineered matrix. If
non-constant, this is wrong. **NOT DONE. One command settles it.**

### 3.3 The 0.5% razor may not transfer

`run_phase2_eval.py` aborts (`return 2`) when unusable > 0.5%. 21,814/4,420,180 = **0.4935%**
— a margin of ~287 rows. **But the 21,814 are keyed to `pathfix`, and the launchers load
`clean_seq` — a different cohort.** How many of the 21,814 even exist there is **unknown**.
The arithmetic may not apply at all.

---

## 4. WHAT WAS CHANGED — commit by commit

| commit | what |
|---|---|
| `a889684` | `ci(lockfile)` — **`check_lock_satisfies.py` was UNTRACKED while `ci.yml:85` invoked it**; first push would have failed on a missing file. Gate rebuilt to assert **satisfaction**, not byte-identity. `requirements-api.lock` regenerated on Linux/3.11. Cascade broken (a red lockfile job had been skipping **1,936 tests**). |
| `8d7df86` | `fix(genomiclm)` — **`genomiclm_llr` was identically 0.0 for all 4,420,180 rows since the connector was written.** `tok(win, return_offsets_mapping=True)` raises `NotImplementedError` on the slow `EsmTokenizer`; a bare `except Exception` swallowed it into a below-threshold `logger.debug`, marked `# pragma: no cover`. Replaced with `centre_token_index()`. 12 tests + `ast` tripwires. |
| `9157133` | `fix(seq-windows)` — `WindowAttachment` + `usable` mask from the builder's `ok` column. `_mapped_mask`, `self._poly` deleted. Tier 1's hardcoded `return out, 0` fixed. |
| `184f2c6` | `fix(agent-layer)` — `run_agents.py` bare orchestrator import + `sys.path` hack removed (two module objects for one package). |
| `f078953` | `docs(readme,claude)` — 22 agents, CLAUDE.md §4.1 |
| *(end of session)* | README restore+rewrite; ROADMAP 6.25–6.29 + §6A/§6B; status doc; memory.md corrections |

### Uncommitted at handoff

- `variant_ensemble.py` — `single_sequence_mode`, `_build_single_channels`,
  `_assert_no_null_windows`, strict `_encode_batch`, `X_seq: pd.Series` → `pd.DataFrame`
- `train.py` — migrated to `.usable`; `has_sequences` now reads **train** (it read **test**,
  while `cnn_1d` is fitted on train)
- `run_phase2_eval.py` — migrated; 0.5% gate now counts placeholders
- `rekey_seq_windows_v2.py` — migrated (its write gate had been silently disarmed)
- `tests/unit/test_no_content_based_poly_detection.py` — NEW
- rewrites of `test_seq_window_join.py`, `test_train_cnn_activation.py`

---

## 5. THE 9 FAILING TESTS

**7 × `_encode_batch` contract change.** `test_encode_batch_dispatch`,
`test_poly_a_series_fallback_no_crash`, `test_single_fasta_seq_mode`,
`test_cnn1d_pickles_after_fit`, `test_ensemble_save_load_with_cnn1d`, and 2 catboost
integration tests. **Every one passes a `pd.Series`** — as the (false) `X_seq: pd.Series`
annotation instructed. `scripts/train.py` has always passed a **DataFrame**
(`_att_train.windows`). The tolerant adapter let the two disagree, so **the suite was green
for years on a code path the run never executes**, for the sequence model's only input.

**Fix:** give them 2-column `[fasta_seq_ref, fasta_seq_alt]` fixtures, or
`single_sequence_mode=True` where a single sequence is genuinely the point.
`test_poly_a_series_fallback_no_crash` tests a fallback that no longer exists — rewrite to
assert the raise.

**2 × poly ban**, reporting real findings:
- **Six UTF-8 byte-order-mark files** (`build_spliceai_index`, `compute_within_gene_auroc`,
  `diagnose_phase2_prediction_reconstruction`, `download_finngen_R10_DEPRECATED`,
  `parse_dbsnp_freq_query_tsv`, `validate_gencode_assets`). Python strips the mark;
  `ast.parse(read_text("utf-8"))` dies on it. **memory.md already recorded this hazard.**
- Remaining poly-literal construction sites: `preflight_run16_inputs.py:23`,
  `probe_cohort_seq_density.py:19`, `run9_ablations.py:640`, `correctness_harness.py:39`,
  `seq_windows.py` ×4, `populate_fasta_seq.py:59`. Monzia's instruction: **fix every site,
  allowlist nothing.** Note `run9_ablations.py` fabricates poly-A for its entire ablation by
  design and `preflight_run16_inputs.py` gates a finished run — for those two, "fix" likely
  means **archive to `scripts/forensics/`** (roadmap 6.8 precedent), not rewrite. History of
  record should not be rewritten to satisfy a present-day gate.

---

## 6. FORWARD PLAN — Monzia's stated requirement, 2026-07-15

> **JEPA, conformal prediction, the full expanded metric stack, and the RNA architecture must
> be fully incorporated and rigorously checked, evaluated, tested, validated and verified,
> with evidence, before a smoke test is even discussed.**

**Run 17 is not next. No smoke test is due.** Full plan in `docs/ROADMAP.md` §6B. Summary:

**P0 — integrity:** 6.29 (verify the clean-cohort hypothesis FIRST), the 9 red tests,
`maxentscan_delta`, `primateai3d`, disk.

**P1 — RNA.** Per the 2026-07-15 specification, **both** categories are needed:
- RNA **sequence**: **RiNALMo** (state of the art; strong generalisation to unseen RNA
  families — the property this project needs, since splits are gene-disjoint), **RNA-FM**
  (mature generalist), **ERNIE-RNA** (structural priors; directly relevant to variant effects
  on RNA structure/splicing)
- **Transcriptomic**: **Geneformer** or **scGPT** — these learn from expression matrices, so
  they are additive to `rnaseq_*`/`gtex_*`, not a replacement
- Optional: UTR-LM (5′ UTR, translation efficiency)

**THE CONSTRAINT, LEARNED THE HARD WAY:** every foundation model wired so far has been
**collapsed to two scalars** before touching the feature contract — and one of those scalars
was **silently zero for its entire life** (6.27). Wiring four more the same way will fail the
same way. Each needs: a fail-loud connector (no bare `except`), a real test of the
**connector** (not of a fixture), and provenance in the run artifacts.

**P2 — conformal, completed.** Phase 1 done and MAPIE-cross-checked. Remaining: RAPS; ordinal
contiguous sets over the five ACMG tiers with distance-weighted risk control (Pathogenic→
Likely Pathogenic is not Pathogenic→Benign); **VUS deferral — 61.5% of the cohort is
`uncertain`, so this is the majority case, not an edge case**; modality-signature calibration;
gene-candidate sets; subgroup/temporal validation; `artifacts.py` fail-closed provenance.

**P3 — metric panel.** Validity (per-class, worst-group, coverage gap, bootstrap CI),
efficiency (set-size distribution, singleton/empty rates), clinical behaviour
(pathogenic-exclusion, severe-error, deferral burden, PPV among singletons), selective
prediction (risk-at-coverage, AURC), robustness under missing modality and shift.

**P4 — JEPA.** **Both design documents agree and it is not the obvious order:**
> *"Do not start with JEPA until this supervised fusion branch is reproducible, calibrated,
> attribution-validated, and benchmarked."*

Order: **expose embeddings → masked fusion v1 → validate attribution → benchmark against the
stacker → THEN JEPA pretraining.**

**Step 1 is a build, not a wiring job.** Every embedding is **destroyed on creation** —
`genomic_lm.py:284` computes 512 dims and reduces it to `np.linalg.norm(...)` on the next
line; ESM-2, the GAT and the tabular network do the same. Fusion v1 needs those vectors.

**Arithmetic that decides the design:** at 4,420,180 rows, float16, ref+alt only, NT + ESM-2
only, pooled only → **≈14.7 GB against 6.62 GB free.** Token-level (which the design
explicitly requires) → ~154 GB. **Scoped to the 1,701,217 trainable rows → ≈5.7 GB.** This is
a real constraint on a real disk.

**Run 17's shape if all four land:** three arms in ONE run — 13-model stacker, Fusion v1
(Stage 0), JEPA-pretrained Fusion v1 (Stage 1) — identical folds, identical gene-disjoint
test. Paired and internally valid, satisfying both Monzia's "JEPA in Run 17" and the
documents' "Stage 0 before Stage 1".

---

## 7. THE FIVE WRONG CONCLUSIONS — so they are not inherited

| claimed | refutation |
|---|---|
| *"NT masks the poly-A fallback; the CNN does not"* | `genomic_lm.py:201` `self._poly = "A"*window`. **Both** miss the builder's poly-**N**. |
| *"G: is the same physical volume as C:; the storage plan is built on a false premise"* | `Get-PSDrive` prints `Description: Google Drive`. Observation right, **inference wrong**. **`memory.md` already recorded this.** Also violated the standing rule against calling Monzia's statements "premises". |
| *"This output is stale"* (2nd instance) | Read **571 of 3416 lines (16%)** and concluded. New content began at line 3355. |
| *"`populate_fasta_seq`/`seq_windows` are a dead island"* | Coupled by **filename**, not import: `scripts/populate_fasta_seq.py:28 DEF_OUT = ".../clinvar_grch38_clean_seq.parquet"`, which every launcher names. The import-graph probe was blind to it and its silence was read as evidence. |
| *"Run 17 as scripted drops cnn_1d"* | `launch_run17_baseline.sh:304` calls **`run_phase2_eval.py`**, for which the file semantics is correct. |

**Plus the two structural failures:**
- **`memory.md` went unread for the entire session** — 144 lines of standing instructions,
  named in the system prompt, while Monzia twice said "should be in your memory". At least two
  session "findings" were already in it.
- **`conversation_search` / `/recent_chats` do not exist in Cowork.** Monzia asked for them in
  nearly every prompt. **The session never said so.** That is precisely the silent-failure
  class this project exists to eliminate — a thing that doesn't happen, reports nothing, and
  the caller proceeds believing it did.

---

## 8. DOCUMENTATION DEBT — mandatory per memory.md, NOT DONE

`memory.md`, *"Documentation (mandatory, unrequested)"*:

- [ ] **`SESSION_2026-07-07` … `SESSION_2026-07-15`** — `docs/sessions/` **stops at
      2026-07-06**. Nine days missing, covering the git rewrite, both CI failure classes, the
      drift revival, the Run-15 silent-zero discovery, the README audit, and this session.
      Reconstructable from `git log`, `docs/status/REMEDIATION_2026-07-13*`, the 2026-07-11
      handoff, and roadmap §6.25–6.29.
- [ ] **LIVING METRICS glossary** — AUROC/AUPRC/F1/MCC/Brier/OOF/calibration/gnn_score/
      odds-ratio/Cramér's V/bootstrap-CI/feature-importance/gates: formula, range, why, where
      applied, how it varied per run.
- [ ] **Per-model algorithm comparison** — how each model works, performed, differs, plus
      interpretation.
- [ ] **`ROADMAP.docx`** — required alongside `.md`, never updated this session.
- [ ] **Push.** Nine-plus commits local. *"Commit and push all docs at session end; never
      needs to be requested."*
- [ ] **The 2026-07-11 handoff is not in the repo.** It exists only as a file Monzia pastes
      into chats. It is the best record of 07-11. Add it.

---

## 9. DOES THE PROJECT EXECUTE FLAWLESSLY END-TO-END?

**No, and there is no evidence that it does.**

- 9 tests failing
- 6.29 open, and its framing withdrawn pending §3.1
- **The end-to-end path was never executed** — no smoke run, no `run_phase2_eval` invocation,
  no artifact produced
- Nothing pushed; CI has seen none of it
- Last end-to-end evidence is **Run 15** — and 6.21 established its feature space was **46%
  constant zero**, so it is evidence of execution, not correctness

**An end-to-end verification is a deliverable to build, not a status to report.** Minimum:
green suite → 6.29 resolved → `maxentscan_delta` verified → local smoke producing artifacts →
feature census printed beside them proving every declared feature carried information.

---

## 10. IF YOU READ NOTHING ELSE

1. **Read `memory.md` first.** Every session. It is the standing-instruction record and it is
   authoritative on Monzia's instructions. It is **stale on project state** (says 91 features
   against a contract of 95) — and Monzia's own rule governs: *"Artifacts beat memory: when
   project files conflict with session memory, artifacts win."*
2. **Verify §3.1 before touching the launchers.** The clean-cohort rule may invert the entire
   6.29 framing.
3. **`maxentscan_delta` is one command from being the next `genomiclm_llr`.**
4. **Do not ship an inference as a finding.** Measurements held all session. Extrapolations
   failed five times out of five.
5. **When an instruction names a capability you don't have, say so in that turn.** Do not
   silently drop it.
