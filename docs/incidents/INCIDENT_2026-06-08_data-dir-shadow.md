# INCIDENT 2026-06-08 — `data/` directory shadowed by a non-directory → 79 test failures

- **Status:** Root-caused; fix + guard test delivered; recovery runbook below.
- **Severity:** BLOCKER (full suite red: 79 failed / 696 passed / 6 skipped). Run 15 gated until green.
- **Component:** `src/genomic_variant_classifier/data/database_connectors.py` (`FetchConfig`), `src/genomic_variant_classifier/data/real_data_prep.py` (`DataPrepConfig`).
- **Class:** Recurring path-collision in the repo-root `data/` tree (same family as the prior `.gitkeep` → `.gitignore` collision).

---

## 1. Symptom

`python -m pytest -q` collapsed from the prior-session baseline of **775 passed / 6 skipped** to **79 failed / 696 passed / 6 skipped**. Every one of the 79 failures terminates in the identical final frame:

```
self = WindowsPath('data'), mode = 511, parents = True, exist_ok = True
    os.mkdir(self, mode)
E   FileExistsError: [WinError 183] Cannot create a file when that file already exists: 'data'
```

Two failure entry points, one cause:

- **Connector tests** (`AlphaMissense`, `ClinGen`, `CADD`, `GTEx`, `SpliceAI`, `dbSNP`, `EVE`, `HGMD`, `OMIM`, `VEP`, `BaseConnector`, ESM-2, LOVD-flow): die in `FetchConfig.__post_init__` → `database_connectors.py:72` → `self.cache_dir.mkdir(parents=True, exist_ok=True)` on the relative default `Path("data/raw/cache")`.
- **Pipeline tests** (`TestAnnotationPipeline`, LOVD matrix flow): die in `DataPrepConfig.__post_init__` → `real_data_prep.py:135` → `self.output_dir.mkdir(parents=True, exist_ok=True)` on `data/splits`.

(The ESM-2 and LOVD-flow cases reach the same `os.mkdir('data')` via `esm2.py:_open_cache` and `_annotate_scores → SpliceAIConnector(...)`, respectively — different call stacks, same terminal frame.)

## 2. Root cause

`pathlib.Path.mkdir(parents=True, exist_ok=True)` recurses bottom-up: `data/raw/cache` → `data/raw` → `data`. When it reaches the repo-root `data` it calls `os.mkdir('data')`, which raises `FileExistsError [WinError 183]`. The `exist_ok=True` branch only swallows that error **if the existing path is a directory**. Here it re-raises — therefore **`data` exists but is not a directory.**

On this repo that means one of:
- a stray **file** named `data` (e.g. an aborted redirect, an editor "Save As", a partial download), or
- a **dangling symlink / junction** named `data` (this project routinely uses `ln -s`/junctions on Vast.ai; a broken link reports `exists()==True`, `is_dir()==False`, and triggers the same WinError 183).

Corroborating evidence from `git diff --stat` at incident time — the entire **tracked** contents of `data/` read as deleted, because a non-directory shadows every path under `data/`:

```
 data/external/.gitignore                       |   3 ---
 data/processed/.gitignore                      |   3 ---
 data/processed/gene_pathogenic_counts.parquet  | Bin 125042 -> 0 bytes
 data/processed/gene_summary.parquet            | Bin 369657 -> 0 bytes
 data/raw/.gitignore                            |   3 ---
```

**Design fault that amplified a one-file mistake into 79 failures:** connector/pipeline construction was **not side-effect-free**. `FetchConfig.__post_init__` and `DataPrepConfig.__post_init__` performed eager, CWD-relative `mkdir` at construction time, so merely instantiating any connector — including stub-mode and unit tests that never touch disk — did filesystem I/O. One shadowing entry in the repo root detonated the whole suite with a cryptic error far from its cause.

**Not the cause:** the constraint-OE patch committed this session (`9b99a24`, `gene_constraint_oe` ← gnomAD `loeuf`) is orthogonal; it only changes feature *values* at data-prep regeneration and touches no path logic.

## 3. Fix

Two layers — recovery (restores green with **no code change**) then hardening (prevents recurrence and makes any future shadow self-explanatory).

### 3a. Recovery (operator, on the Windows machine)
Forensic-first: **identify** what `data` is before touching it (large *untracked* data — `clinvar_grch38.parquet`, `spliceai_index.parquet`, the STRING graph pickle, the ClinVar VCF — may be at stake and is not in git). Then preserve aside, restore the tracked tree with `git checkout HEAD -- data`, recreate untracked subdirs, and re-run pytest. See the runbook (STEP 1).

### 3b. Hardening (code)
- **`patch_fetchconfig_lazy_mkdir.py`** — removes the eager `mkdir` from `FetchConfig.__post_init__` (construction is now side-effect-free; the cache dir is created lazily in `_save_cache`, which already `mkdir`s its parent before `to_parquet`, so caching is unaffected). Adds a module-level `ensure_dir()` helper that converts WinError 183 into a clear `NotADirectoryError` naming the offending component, and routes `_save_cache` through it.
- **`patch_datapreprep_mkdir_guard.py`** (optional, defense-in-depth) — wraps the single `self.output_dir.mkdir(...)` in `DataPrepConfig.__post_init__` in a `try/except → NotADirectoryError` with the same clear message.
- **`tests/unit/test_data_dir_not_shadowed.py`** — three guards: (1) repo-root `data` is a dir or absent; (2) `FetchConfig()` creates no directory on construction (regression lock for the lazy-mkdir change); (3) `ensure_dir()` raises a clear `NotADirectoryError`, never WinError 183, on a file-shadowed path.

## 4. Verification

- `python -m pytest -q` returns to **green** after STEP 1 **before** any code change (proves the diagnosis: the bug was environmental state, not code).
- After applying the patchers + adding the guard test: full suite green with the three new guard tests passing (≈ 778 passed / 6 skipped).
- `python scripts/maintenance/patch_fetchconfig_lazy_mkdir.py` is idempotent (re-run prints SKIP, no-ops) and AST-verifies the patched module.

## 5. Prevention

- Construction of configs/connectors is now side-effect-free; directories are created lazily at first write only.
- `ensure_dir()` guarantees any future path-shadow surfaces as a clear, actionable `NotADirectoryError` — never a cryptic WinError 183 buried 79 times.
- `test_data_dir_not_shadowed.py` turns the entire failure class into **one** fast, unambiguous signal at the top of every suite run.
- Operational note: when symlinking `data/` / `outputs/` on Vast.ai, always `rm -rf` the target before `ln -s`, and never leave a dangling link in the repo root (a broken junction reproduces this exact incident).

## 6. Secondary findings (tracked, non-blocking)

- **MEDIUM (env):** `Windows fatal exception 0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) during collection of `tests/unit/test_ablate_gnn.py`, from a `torch_scatter`/`torch_sparse` ABI mismatch against the installed `torch`. The test is `importorskip`-guarded so the suite survives, but the local install is broken. GNN is GPU-only on Vast.ai, so local breakage is acceptable; optional cleanup: `pip uninstall torch-scatter torch-sparse` locally to silence the process fault.
- **LOW (debt):** pandas `FutureWarning` on `.fillna` downcasting at `variant_ensemble.py` (lines ~369, 390, 394, 433, 438, 446, 451, 456, 463, 468) and `real_data_prep.py:388`. Non-blocking; fix pattern `result[col] = result[col].fillna(default).infer_objects(copy=False)`. Deferred — locate with the diagnostic in the runbook before patching.
- **DRIFT:** `docs/ROADMAP.docx` was missing from `Downloads`, so the install/commit steps for it failed (`pathspec 'docs/ROADMAP.docx' did not match`). Resolved permanently by generating it in-place from `docs/ROADMAP.md` via `scripts/make_roadmap_docx.py`.
- **CODE REVIEW — `protein_pipeline.py`** (currently dead: `protein_cache_dir` is `None`): (a) `line.lstrip("_atom_site.")` strips a *character set*, not the prefix — corrupts mmCIF column names like `auth_seq_id`; use `removeprefix("_atom_site.")`. (b) the bottom `get_alphafold_features(uniprot_id, aa_position)` wrapper passes a UniProt id where the mapper expects an HGNC symbol → always returns defaults. (c) RSA / secondary-structure / `dist_to_active_site` are heuristic approximations (linear-sequence distance ×3.8 Å as a 3-D Cα proxy) presented as structural measurements — flag as approximate to avoid injecting pseudo-signal.
- **CODE REVIEW — `dbsnp.py`:** clean; only nit is a redundant cache of the full AF lookup.
