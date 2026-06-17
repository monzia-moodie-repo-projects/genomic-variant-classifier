# Data Layout Standard

**Status:** canonical convention for this repository and a reusable template for
every project going forward.
**Author:** Monzia Moodie

This standard makes a project's genetic/genomic data tree predictable, auditable,
and safe to back up. It is enforced by a small toolset that reads one manifest
(`configs/data_manifest.yaml`): `setup_data_tree.py`, `audit_data_tree.py`,
`sync_data_to_gdrive.py`, and `preflight_data_guard.py` (all under
`scripts/maintenance/`).

## 1. The decision: local-canonical `data/`, Google Drive as a synced mirror

The repository's `data/` MUST be a **real local directory**, not a junction or
symlink into a cloud-synced folder. Google Drive is a one-directional **backup
mirror**, written by `rclone` on demand. Rationale (the options weighed):

- **(a) `data/` is a junction to `G:\My Drive\...\data` (the historical setup).**
  Pros: single copy, auto-versioned by Drive. Cons that disqualify it: when G:
  is unmounted/unsynced the junction dangles and every run/test fails (the
  2026-06-14 incident); Google Drive for Desktop streams files on demand, so a
  "file" may be a placeholder until hydrated -> I/O stalls and partial reads
  during prep/training; the sync client can contend for file locks mid-write;
  and **any controlled-access data (TCGA/dbGaP, TOPMed individual-level, HGMD,
  OMIM) sitting in a personal Drive likely breaches its DUA/license.**
- **(b) local-canonical + rclone mirror (CHOSEN).** Runs/tests read and write a
  real local directory -> fast, no cloud-FS stalls, and the dangling-junction
  failure class is eliminated outright. Backup is explicit and selective, so it
  never sits in the training I/O path, and controlled data is simply never put
  in the sync set. Cost: duplicate storage for the synced subset, and the mirror
  must be refreshed (a single `sync_data_to_gdrive.py` call). Worth it.
- **(c) local-canonical + Drive as backup only.** This is (b) with the strictest
  sync policy; folded into (b) via the manifest's per-source `sync` flag.

Most robust (no dependence on G: at run time), cleanest (one canonical local
tree), and most secure (controlled data never auto-leaves the machine).

## 2. Canonical tree

```
data/
  external/    third-party reference inputs, READ-ONLY, one dir per source
  raw/         raw downloads not yet processed
  raw/cache/   connector caches (always regenerable)
  processed/   pipeline-built artifacts (regenerable; named by the pipeline)
  reference/   small tracked schemas/manifests (committed)
  interim/     scratch / intermediate (regenerable, ephemeral)
  splits/      train/val/test splits (regenerable)
```

Each subtree carries a `.gitignore` containing `*` and `!.gitignore`, so the
directory is tracked but its contents are not. The only data committed to git is
small reference material explicitly whitelisted in the repo-root `.gitignore`.

## 3. Source naming

One directory per logical source under `data/external/`, named in
`lower_snake_case`, with **no version suffix** (the version lives in the
manifest, not the directory name). Aliases are forbidden: a source has exactly
one canonical name. The manifest records known aliases so the auditor can flag
and guide migration. Examples from this repo: `1kgp` (not `1000g`), `dbsnp`
(not `dbsnp156`), `spliceai` (not `spliceai_scores`), `hgmd` (not `hgmd_pro`),
`clingen` (move the stray `ClinGen-Gene-Disease-Summary.csv` into `clingen/`).

Built artifacts whose paths are already referenced by code (e.g.
`data/external/reactome_gene_pathways.parquet`) are kept where the code expects
them and recorded in the manifest as `regenerable_expensive`; they are NOT moved
(moving them would break connectors). New built artifacts should live under
`data/external/<source>/<source>.parquet` or in `processed/`.

## 4. Keep vs. regenerate (the four classes)

Every source is classified, which drives backup policy:

- **irreplaceable** -- cannot be re-obtained cheaply (your own cohort RNA-seq, a
  controlled-access extract). Back up.
- **regenerable_expensive** -- rebuildable but costs real compute (built
  parquets, ESM-2 embeddings). Back up to save recompute.
- **regenerable_cheap** -- connector caches, splits, quick processed parquets.
  Never back up; regenerate on demand.
- **public_redownloadable** -- stable public URL (ClinVar, gnomAD, GTEx). Do not
  back up by default (re-download); the manifest's `acquire` field records how.

## 5. Backup / sync policy (security-aware)

Three destinations, decided per source by `tier` and `class`:

- **cloud-backup (rclone -> Google Drive):** `sync: true`, `tier != controlled`,
  and `class` in {irreplaceable, regenerable_expensive}.
- **offline-only:** any `tier: controlled` source (HGMD, OMIM, COSMIC, TCGA,
  TOPMed individual-level). Back these up **encrypted / offline only** -- NEVER
  to a personal cloud, which would breach the DUA/license. The tooling hard-
  **aborts** if a controlled source is ever marked `sync: true`.
- **regenerable:** everything else -- not backed up; rebuilt via `regenerate`.

`setup_data_tree.py` generates `configs/rclone_data_filter.txt` from these rules;
`sync_data_to_gdrive.py` is dry-run by default and re-checks the controlled gate
before any call.

## 6. The manifest (`configs/data_manifest.yaml`)

Tracked in git (it is metadata, not data, and must survive a dangling `data/`).
Per source: `location`, `tier`, `class`, `aliases`, `version`, `acquire`,
`regenerate`, `sync`, `notes`. It is the single source of truth; the four
scripts derive everything from it. `tier: review` marks sources whose access
tier or leakage-independence must be confirmed before they are synced or used.

## 7. Tooling

- `setup_data_tree.py` -- idempotent, non-destructive: creates the skeleton +
  `.gitignore`s + the rclone filter. Never deletes or moves.
- `audit_data_tree.py` -- read-only: `data/` link status (real dir / working
  junction / dangling / shadow / missing), per-source inventory, alias/orphan/
  naming hygiene, controlled-in-sync compliance, security-aware size rollup.
  Exit 0 clean / 1 warnings / 2 blocked.
- `sync_data_to_gdrive.py` -- rclone mirror, dry-run by default, controlled gate.
- `preflight_data_guard.py` -- fail-loud `assert_data_usable()` to call at the
  top of any run/test before `data/` is touched.

## 8. Adopting this in a new project

1. Copy `configs/data_manifest.yaml`, the four `scripts/maintenance/` scripts,
   and this standard.
2. Edit the manifest for the new project's sources.
3. Run `python scripts/maintenance/setup_data_tree.py` to build the skeleton.
4. Keep `data/` a real local directory; configure an rclone remote for backup.
5. Call `assert_data_usable()` early in run scripts; run `audit_data_tree.py`
   before each run and at session end. Never let `data/` become a cloud junction.
