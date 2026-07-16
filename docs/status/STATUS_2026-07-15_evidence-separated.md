# STATUS — 2026-07-15 — evidence-separated

**Purpose.** Every claim made during the 2026-07-15 session, sorted by *what kind of thing it
is*: something measured, something inferred, or something unknown. Written because the session's
assistant produced **five wrong conclusions** and Monzia cannot currently tell, from the code
comments alone, which statements are evidence and which are talk.

**READ `memory.md` BEFORE THIS DOCUMENT.** The Claude Project's `memory.md` holds Monzia's
standing instructions. It went UNREAD for the whole 2026-07-15 session, and at least two of
the "findings" below (the Google Drive semantics, the UTF-8 byte-order-mark hazard) were
already recorded in it. It also invalidates this document's framing of 6.29 -- see the
roadmap entry.

**This document exists to be audited by someone other than its author.** Nothing in it should be
believed because it is asserted here. Section 1 items carry the command and the output; re-run
them. Section 2 items carry a falsifier; test them. Section 3 items are open questions and must
not be built upon.

**Status of the work: NOT PUSHED.** Nine commits sit local. Seven tests are red by design
(pending fixture rewrites). Two gates are red because they are reporting real findings.

---

## 0. HOW TO READ THIS

| bucket | means | how to treat it |
|---|---|---|
| **MEASURED** | A command was run; the output is quoted. | Reproducible. Trust after re-running. |
| **INFERRED** | A conclusion drawn from measurements, not itself measured. | **Do not build on it.** Each carries what would falsify it. |
| **UNKNOWN** | Named, not established. | Open question. |
| **WRONG** | Asserted during the session and since refuted. | Recorded so it is not re-inherited. |

---

## 1. MEASURED

### 1.1 `genomiclm_llr` has been identically 0.0 for every row, since the connector was written

```
> AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-100m-multi-species')
is_fast: False | class: EsmTokenizer
OFFSET RAISES -> NotImplementedError : return_offset_mapping is not available when using
Python tokenizers.
```

`genomic_lm._masked_centre_logratio` located the variant's centre token with
`tok(win, return_offsets_mapping=True)`. HuggingFace raises `NotImplementedError` for that
argument on every **slow** tokeniser. It raised on every window. A bare `except Exception`
swallowed it into `logger.debug` — below the default level, so **nothing printed** — and carried
`# pragma: no cover`.

Cohort scale: **4,420,180 rows**. Feature is in `TABULAR_FEATURES`.

**Corroborating evidence, independent of the above:** the 2026-07-11 handoff records the
Run-17 smoke audit as `genomiclm_llr` = **dead in all splits**, while `genomiclm_delta_norm`
(which never touches offset mapping) = **ALIVE**. Two independent observations agreeing.

**Status:** FIXED (`centre_token_index`), 12 tests, `ast` tripwires banning the call's return.

### 1.2 21,814 cohort rows carry fabricated sequence, with a full reason breakdown

`data/processed/seq_windows/seq_windows.manifest.json`, read verbatim:

```json
"n_rows_built": 4420180,
"n_ok":         4398366,
"n_poly":         21814,
"poly_reason_breakdown": {
    "empty_allele":     19988,
    "non_acgt_allele":   1771,
    "ref_mismatch":        53,
    "fetch_failed":         2
},
"builder_version": "delta_window_builder/2026-07-10-stepB",
"build_utc": "2026-07-11T01:24:22.385232+00:00",
"cohort_path": "data\\processed\\clinvar_grch38_pathfix.parquet"
```

19,988 + 1,771 + 53 + 2 = **21,814** ✓. 4,398,366 + 21,814 = **4,420,180** ✓.

**91.6% are `empty_allele`** — alleleless variants. This is a rebuild target with an existing
workstream (`recover_alleleless_*`, `classify_alleleless_by_type`), not random noise.

ok-fraction = 4,398,366 / 4,420,180 = **99.507%**, above `MIN_OK_FRACTION` (0.95), so no
existing gate fires.

### 1.3 The two window artifacts are different objects

```
data/processed/clinvar_grch38_clean_seq.parquet | rows = 4,399,089 | 549,050,035 B | 6/12/2026
   cols: ['variant_id','source_db','chrom','pos','gene_symbol','transcript_id','pathogenicity',
          'allele_freq','clinical_sig','protein_change','fasta_seq','source_id','metadata',
          'ref','alt','consequence','fasta_seq_ref','fasta_seq_alt','ReviewStatus']

data/processed/seq_windows/seq_windows.parquet  | rows = 4,420,180 | 445,419,552 B | 7/10/2026
   cols: ['chrom','pos','ref','alt','fasta_seq_ref','fasta_seq_alt','ok','reason']
```

- **21,091 rows apart.** One month apart.
- **Only the new one has `ok` / `reason`** — i.e. provenance.
- `clinvar_grch38_clean.parquet` (the `--clinvar` target) = 4,399,089 rows and carries
  **NO `fasta_seq_ref` / `fasta_seq_alt`**. Windows can only arrive via `--seq-windows`.

### 1.4 `--seq-windows` has two incompatible meanings

```
train.py:102-107          "--seq-windows", default="data/processed/seq_windows"
                          help="DIRECTORY holding ... seq_windows.parquet + seq_windows.manifest.json"
train.py:435              _seq_win_parquet = Path(args.seq_windows) / "seq_windows.parquet"

run_phase2_eval.py:49-52  "--seq-windows", default="data/processed/clinvar_grch38_clean_seq.parquet"
                          help="Parquet with fasta_seq_ref/fasta_seq_alt delta windows"
run_phase2_eval.py:414    _seq_win = Path(args.seq_windows)
run_phase2_eval.py:436    attach_delta_windows(_meta_train_seq, _seq_win)
```

**One flag. One repository. Directory vs file.**

### 1.5 The Run 17 launcher invokes `run_phase2_eval.py`

```
launch_run17_baseline.sh:304   python scripts/run_phase2_eval.py $ARGS
```

Confirmed by flag ownership: `--min-review-tier`, `--unseen-gene-holdout`, `--max-train` exist
**only** in `run_phase2_eval.py`. So the launcher's `--seq-windows <file>` is **correct for the
script it calls.**

### 1.6 Every launcher points at the artifact WITHOUT provenance

```
launch_run17_baseline.sh:181       --seq-windows $DATA/processed/clinvar_grch38_clean_seq.parquet
launch_run17_r12only.sh:183        (same)
launch_run17_r13only.sh:183        (same)
launch_run17_rnaseq_ablation.sh:42 (same)
launch_run15_baseline.sh:133       (same)
smoke_all_models.py:31             (same)
preflight_gate.py:27               "--seq-windows": "processed/clinvar_grch38_clean_seq.parquet"
```

That artifact has no `ok` column (§1.3). So `attach_delta_windows` takes its
`except (ValueError, KeyError)` branch and logs *"no 'ok' column: builder-placeholder rows CANNOT
be identified and will be treated as usable"*.

### 1.7 Six Python source files carry a UTF-8 byte-order mark

Cross-validated by two independent methods (a `pytest` gate reading bytes, and a PowerShell
`[IO.File]::ReadAllBytes` scan) returning the **identical** list:

```
scripts\build_spliceai_index.py
scripts\compute_within_gene_auroc.py
scripts\diagnose_phase2_prediction_reconstruction.py
scripts\download_finngen_R10_DEPRECATED.py
scripts\parse_dbsnp_freq_query_tsv.py
scripts\validate_gencode_assets.py
```

Python's tokeniser strips it (PEP 263), so they import and test green.
`ast.parse(read_text(encoding="utf-8"))` raises `SyntaxError: invalid non-printable character
U+FEFF` — so every source-analysis gate in the suite either dies or must silently skip them.

### 1.8 An invalid escape sequence, and a Google Drive path already in the codebase

```
scripts\download_finngen_R10_DEPRECATED.py:6: SyntaxWarning: invalid escape sequence '\M'
    G:\My Drive\genomic-variant-data
```

`\M` is deprecated and becomes a `SyntaxError` in a future Python. Separately: the project
**already has a `genomic-variant-data` folder on Google Drive**, named in source.

### 1.9 Hardware and storage

```
Get-Disk       -> Number 0 ONLY. NVMe KXG60ZNV1T02 KIOXIA 1024GB, 953.87 GB, BusType RAID
Get-Partition  -> EVERY partition is DiskNumber 0. C: is partition 3, 935.59 GB
Get-Volume     -> C: NTFS "== 1 T A", 935.59 GB, 6.63 GB free.  G: NOT LISTED (cloud mount)
Get-PSDrive C,G -> G: Description = "Google Drive"
```

**One physical disk. 6.62–6.64 GB free.** `G:` is Google Drive (owner-stated ~5 TB, ~4.5 TB
free); its `Free` property reports the **host disk's** geometry and is meaningless.
Corroboration: `Get-Volume` shows a `Box` mount as a **FAT32 volume of 935.59 GB** — FAT32 caps
files at 4 GB, so that number is fiction too. Two cloud mounts, both parroting `C:`.

**Top consumers** (`C:\Users` = 444.98 GB; `C:\Documents and Settings` is a junction to it, the
same bytes double-counted):

| GB | path |
|---|---|
| 72.46 | `AppData\Local\Docker\wsl\disk\docker_data.vhdx` |
| ~196 | gnomAD VCFs under `C:\Users\monzi\data\` (24 files; **chr17 appears TWICE** — `data\raw\` and `data\external\`) |
| 31.78 | `pagefile.sys` |
| 29.92 + 27.72 | FinnGen R12 + R13 |
| 18.11 | `esm2_cache.sqlite` |
| 16.68 | Ubuntu WSL `ext4.vhdx` |
| 1.90 | `..._BACKUP_2026-07-11\data\alphafold_cif_cache_2026-07-03.tar.gz` — **the blob excised from git history, still on disk** |
| 1.8189 | `..._GITBACKUP_2026-07-11` (measured) |

Project: `data` 160.86 GB, `outputs` 9.80 GB.

### 1.10 Cohort labels

```
pathogenicity      count
uncertain        2,718,963   (61.5%)
likely_benign    1,083,576
benign             276,240
pathogenic         229,602
likely_pathogenic  111,799
TOTAL            4,420,180   ✓
```

Trainable (excluding `uncertain`) = **1,701,217** — 341,401 positive / 1,359,816 negative
(1:3.98). `fasta_seq`: **null 4,420,180 / non-null 0 — the column is 100% empty.**

### 1.11 Suite state

```
pytest tests/ --collect-only -q   ->  1963 tests collected   (EXIT 0)
pytest tests/ --assert-suite-size ->  1963 collected, 1956 passed, 7 skipped  (EXIT 0)
```

Then +4 (`test_no_content_based_poly_detection.py`) → **1967 collected**, currently
**9 failed / 1951 passed / 7 skipped**. `EXPECTED_SUITE_SIZE` = 1963 and is now **stale by 4**.

### 1.12 `single_sequence_mode` behaves as specified

```
delta mode  _in_channels: 13
delta mode rejects Series -> OK
single mode _in_channels: 5
single mode encoded shape: (4, 5, 101) (expect (4, 5, 101))
get_params carries the flag: True
```

### 1.13 22 agents, not 13

`scripts/check_agents_active.py` → *"22 agents (registered=22, scheduled=22) … 0 dormant"*.
Corroborated by AST transitive-inheritance scan (22) and a live
`Orchestrator._register_agents()` (22). **All show `age=25.06d` — last orchestrator run
2026-06-20.**

---

## 2. INFERRED — do not build on these

### 2.1 The `ok`-column gap makes today's provenance work inert on Run 17's path

**Inference:** because the launchers name `clinvar_grch38_clean_seq.parquet` (§1.6) and that file
has no `ok` column (§1.3), `WindowAttachment.usable` degrades to `notna()` and the 21,814
placeholder rows (§1.2) are counted as real.

**Falsifier:** run `run_phase2_eval.py` with the launcher's arguments against a small cohort and
read the `seq windows [train]: ...` summary line. If `provenance` reports `parquet+ok`, this is
wrong. **NOT DONE.**

**Caveat:** the 21,814 are keyed to the **pathfix** cohort (4,420,180). `clean_seq.parquet` is a
**different, 21,091-row-smaller** cohort. How many of the 21,814 exist in `clean_seq` at all is
**unknown** (§3.2).

### 2.2 The 0.5% abort gate is now a razor

`run_phase2_eval.py:~465` aborts (`return 2`) when unusable > 0.5%. 21,814 / 4,420,180 =
**0.4935%** — a margin of **0.0065 percentage points (~287 rows)**.

**Inference:** if the gate ever reads a provenance-carrying artifact, it will pass by a hair.

**Falsifier:** the arithmetic assumes the placeholder count transfers to whatever cohort is
actually loaded, and §2.1's caveat says it may not. **UNVERIFIED.**

### 2.3 `maxentscan_delta` may be the next `genomiclm_llr`

```
pipelines/rna_pipeline.py:340   ref_col = alt_col = df["fasta_seq"].fillna("")
```

`ref_col = alt_col`, reading a column measured at **100% null** (§1.10). `patch_rna_maxentscan_
delta.py` shows this is a **fallback** beneath a preferred `fasta_seq_ref`/`fasta_seq_alt` branch.
`maxentscan_delta` is in `TABULAR_FEATURES` **and the harness fixture feeds it**
(`rng.uniform(-10, 10, n)`) — the exact shape of §1.1.

**Falsifier:** read `maxentscan_delta`'s distribution in a real engineered matrix. If non-constant,
this is wrong. **NOT DONE. This is the single highest-value unverified lead in this document.**

### 2.4 `preflight_gate.py` cannot detect the failure it is named for

```python
if flag == "--seq-windows" and str(v).strip() == "":
    rows.append(("FAIL", f"{flag} empty -> CNN forced to poly-A (silent CNN degradation)"))
```

**Inference:** it fails only on *empty*. A non-empty path to a wrong/stale artifact passes.

**Falsifier:** none needed — read the code. But whether it matters depends on §2.1.

### 2.5 `test_launch_run17.py` checks a proxy

`test_required_flags_present[--seq-windows ]` is parametrised on the **flag string** and asserts
the substring appears in the launcher. It cannot see what the value points at.

---

## 3. UNKNOWN

1. **Does `run_phase2_eval.py` work against `seq_windows/seq_windows.parquet`?** Its schema is
   8 columns vs `clean_seq`'s 19. Repointing the launchers depends on this. **Not tested.**
2. **How many of the 21,814 placeholder rows exist in `clean_seq.parquet`?** Different cohorts,
   21,091 rows apart.
3. **Is `maxentscan_delta` dead on real data?** (§2.3)
4. **Why have the agents not run since 2026-06-20** (`age=25.06d`) while
   `check_agents_active.py` calls them ACTIVE with 0 dormant? The checker's threshold has not
   been read.
5. **Is `populate_fasta_seq.py` intended to be retired?** It is not import-dead — it *produces*
   `clean_seq.parquet`, which every launcher names. Whether the project intends `build_seq_windows`
   to supersede it is an **owner decision**, not a code fact.
6. **Roadmap 6.5** — the correctness-harness sanity model still does not converge. The register
   calls it *"the most scientifically substantive item left."* Untouched today.
7. **`requirements.txt` is Windows/Python-3.14 resolved** (`colorama` present,
   `nvidia-nccl-cu12` absent) → unpinned NCCL on the graphics-processing-unit box.
   Unaddressed today.

---

## 4. WRONG — asserted this session, since refuted

| claimed | refutation |
|---|---|
| *"Nucleotide Transformer masks the poly-A fallback; the CNN does not."* | `genomic_lm.py:201` `self._poly = "A" * window`. **Both** miss the builder's poly-**N**. Half-true. |
| *"G: is the same physical volume as C:, so the storage plan is built on a false premise."* | `Get-PSDrive C,G` prints `Description: Google Drive`. The **inference** was wrong; the observation was right. **Monzia's plan was sound; mine wasn't.** And `memory.md` already recorded `G:` as a DriveFS streaming cache with a ~5TB rclone remote behind it -- so this was not even a discovery, it was a rediscovery of a fact already on file, at Monzia's expense. It also violates his standing instruction: *"Never label his statements as 'beliefs,' 'premises,' or 'assumptions.'"* |
| *"This output is stale"* (2nd instance) | Read **571 of 3416 lines (16%)** and concluded. The new content began at line 3355. |
| *"`populate_fasta_seq` / `seq_windows` are a dead island."* | Coupled by **filename**, not import: `scripts/populate_fasta_seq.py:28 DEF_OUT = ".../clinvar_grch38_clean_seq.parquet"`, which every launcher names. The import-graph probe was structurally blind to it, and its silence was read as evidence. |
| *"Run 17 as scripted drops cnn_1d."* | `launch_run17_baseline.sh:304` calls **`run_phase2_eval.py`**, for which the file semantics is correct. Reasoned from `train.py`'s contract without checking which script runs. |

**Pattern:** every item above is an *inference shipped as a finding*. The measurements in §1 have
held. The extrapolations have not. Five in one session.

---

## 5. CHANGES MADE — 2026-07-15

### Committed (9 commits, **NOT PUSHED**)

| commit | scope |
|---|---|
| `a889684` | `ci(lockfile)`: adds `scripts/check_lock_satisfies.py`, which `ci.yml:85` had been invoking **while untracked** — the first push would have failed on a missing file. Root pattern (d). Plus the Linux/3.11-regenerated `requirements-api.lock`. |
| `8d7df86` | `fix(genomiclm)`: §1.1. `centre_token_index`, handler + pragma deleted, 12 tests. |
| `9157133` | `fix(seq-windows)`: `WindowAttachment` + `usable` from the `ok` column; `_mapped_mask` and `self._poly` deleted; tier 1's hardcoded `return out, 0` fixed. |
| `184f2c6` | `fix(agent-layer)`: `run_agents.py` bare orchestrator import + `sys.path` hack removed. |
| `f078953` | `docs(readme,claude)`: 22 agents, 1963 tests, CLAUDE.md §4.1 storage vocabulary. |

(Four earlier commits from the prior session are also unpushed.)

### Uncommitted

- `variant_ensemble.py` — `single_sequence_mode`, `_build_single_channels`,
  `_assert_no_null_windows`, strict `_encode_batch`, `X_seq: pd.Series` → `pd.DataFrame` on
  `fit`/`predict_proba`/`predict`/`evaluate`.
- `train.py` — migrated to `.usable`; `has_sequences` now reads **train** (it read **test**, the
  wrong split, while `cnn_1d` is fitted on train).
- `run_phase2_eval.py` — migrated to `.usable`; the 0.5% gate now counts placeholders.
- `rekey_seq_windows_v2.py` — migrated (its write gate had been silently disarmed).
- `tests/unit/test_no_content_based_poly_detection.py` — NEW, 4 tests.
- Rewrites of `test_seq_window_join.py`, `test_train_cnn_activation.py`.

### Red, by design

- 7 tests from the `_encode_batch` contract change. **All pass a `pd.Series`** — as the false
  `X_seq: pd.Series` annotation instructed. Production has always passed a DataFrame
  (`train.py: X_seq_train = _att_train.windows`). The suite was green on a shape the run never
  sends.
- 2 poly-ban tests, reporting §1.7 and the remaining construction sites.

---

## 6. THE ROADMAP DEBT — must be closed before anything else

**The register ends at 6.24 (2026-07-14). Nothing from 2026-07-15 is in it.**

Meanwhile the following **dangling references** were written into code today, pointing at
entries that **do not exist**:

| reference | files |
|---|---|
| `roadmap 6.26` | `run_agents.py`, `EXPECTED_SUITE_SIZE` |
| `roadmap 6.27` | `genomic_lm.py`, `test_genomiclm_llr_is_computed.py` |
| `roadmap 6.28` | `genomic_lm.py`, `seq_window_join.py`, `train.py`, `run_phase2_eval.py`, `rekey_seq_windows_v2.py`, `variant_ensemble.py`, `CLAUDE.md`, `test_no_content_based_poly_detection.py`, `test_seq_window_join.py`, `test_train_cnn_activation.py` |

This is **6.23's own defect** — *"it pointed at four files that do not exist"* — committed
approximately fifteen times, by the author of that entry, on the same day.

**Required:** write 6.25–6.28 into `docs/ROADMAP.md` §6 with the evidence above, **or** strip
every reference. A pointer to nothing is worse than no pointer: it reads as provenance.

---

## 7. RUN 17 READINESS — honest

| item | state |
|---|---|
| `genomiclm_llr` | **FIXED.** Would have hard-failed Run 17 at `_assert_no_dead_features` after full data prep, on paid compute. |
| Window provenance | **BUILT, INERT ON THE CONFIGURED PATH** (§2.1). |
| `cnn_1d` input contract | **FIXED**, 7 tests pending rewrite. |
| `maxentscan_delta` | **UNVERIFIED LEAD** (§2.3). Highest value. |
| Roadmap | **NOT UPDATED.** ~15 dangling refs. |
| README | Owner reports it "looks a mess." Correction blockquotes stacked on corrections rather than a clean rewrite. |
| Disk | 6.62 GB free. ~78 GB reclaimable locally without touching a genomic source. ~196 GB gnomAD → Google Drive; **de-duplicate `chr17` first**. |
| Push | **Nothing pushed.** |
| Local smoke | **NOT RUN.** |
| VM smoke | **NOT RUN.** |
| Independent audit | **RECOMMENDED.** One source of truth has been wrong five times today. The project's own doctrine (root pattern (d)) says a green from a single source is evidence about that source. |

---

## 8. THE AUTHOR'S OWN FAILURE MODE, for the auditor's benefit

Read §1 and §4 together. The measurements held; the extrapolations did not, five times.

The mechanism: this session wrote its lessons into **code comments** — long, argued, quoting
CLAUDE.md's meta-rule that *"a finding in a document is a comment"* — while putting its own
findings in exactly that place, and leaving the **roadmap**, the living record, untouched.
Discovery is visible. Verification is not. It optimised for the visible one, turn after turn,
and shipped surface area faster than it verified it.

**Treat every claim in this document as unproven until re-run. That is not modesty; it is the
correct prior given §4.**
