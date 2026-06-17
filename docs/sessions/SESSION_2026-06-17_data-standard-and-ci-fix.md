# Session 2026-06-17 -- data-layout standard shipped + CI feature-count drift fixed

## Summary
Two milestones landed and were verified end-to-end:
1. A reusable, project-agnostic **data-layout standard** (manifest + maintenance tooling)
   was added and the live `data/` tree brought into compliance -- final audit **CLEAN**,
   32 manifest sources (commits `47bc887` then `40f16f0`).
2. **CI was restored to green** after a feature-count regression that had been red since the
   RNA-seq commit `1f3c2e0` (commit `11e14a3`, CI run #442 Success on both pytest 3.11 + 3.12).

## Part 1 -- data-layout standard (40f16f0)
Canonical `data/` layout (external / raw / raw/cache / processed / reference / interim / splits),
each data subtree carrying an ignore-all `.gitignore` (`reference/` excluded -- it holds TRACKED
schemas). Single source of truth `configs/data_manifest.yaml` (32 sources: location / tier / class /
aliases / version / acquire / regenerate / sync / notes). Five manifest-driven tools under
`scripts/maintenance/`: setup (skeleton + .gitignores + rclone filter), audit (read-only inventory,
link-status, alias/orphan/naming hygiene, controlled-in-sync compliance, security-aware backup
rollup), consolidate_aliases (copy-verify-remove, dry-run default), sync_data_to_gdrive (rclone
mirror, controlled-source gate), preflight_data_guard (fail-loud `assert_data_usable`). Backup policy
is security-aware: controlled sources (HGMD/OMIM/COSMIC/TCGA/TOPMed) are NEVER synced to personal cloud
(tooling hard-aborts if one is marked sync:true).

### Key findings during rollout
- `data/` is ALREADY a real local directory (not a junction) -- the 2026-06-14 dangling-junction
  remediation had replaced it. Migration runbook is therefore moot; kept for other machines.
- The two "extra" external dirs were EMPTY aliases (`1000genomes`, `clinvar_fresh`, 0 files each) --
  ClinVar data lives in `raw/clinvar` + `processed/clinvar_grch38_*`. Both removed via consolidate.
- `external/reference` is the **GRCh38 primary-assembly FASTA (~3.8GB)**, a real orphan now registered
  (code-referenced; distinct from the top-level `data/reference/` schema dir).

### Two self-inflicted bugs caught + fixed this session
- `47bc887` message ran ahead of its content (consolidate tool not yet installed, `reference` not yet
  registered, aliases not yet removed). Reconciled by follow-up `40f16f0` (no history rewrite).
- `setup_data_tree.py` wrongly wrote an ignore-all `.gitignore` into the TRACKED `reference/` subtree
  (would silently ignore future schema files). Fixed: `reference/` excluded from ignore-all.

## Part 2 -- CI feature-count fix (11e14a3)
See `docs/incidents/INCIDENT_2026-06-17_feature-count-drift.md`. Fork C (`1f3c2e0`) widened
`TABULAR_FEATURES` 82 -> 87 (5 `rnaseq_*` cols) without bumping the two guardrails that pin the count.
A local SUBSET pytest run masked it; the full CI suite failed 5 tests across 3 commits. Fix: bump
`EXPECTED_TABULAR_FEATURE_COUNT` 82 -> 87 and add the 5 `rnaseq_*` to `KNOWN_ZERO_DEFAULT`.

## Verification
- Data audit: **VERDICT CLEAN** -- no aliases/orphans/violations; 32 sources; reference 3.8GB recognized.
- Full unit suite (local .venv312): **1309 passed, 2 skipped, 41 warnings** (warnings all pre-existing:
  LR lbfgs non-convergence, LGBM feature-name notices, n_components>n_samples in a tiny fixture).
- CI run #442 (`11e14a3`): **Success** 7m12s -- lockfile drift, pytest 3.11, pytest 3.12, Docker smoke all green.

## Open / follow-ups
- `external/reference` holds BOTH `.fa` (3005MB, indexed by `.fai`) AND `.fa.gz` (841MB, unindexed dup).
  Nothing in `src/` references it (likely the not-yet-wired source for empty `fasta_seq` features).
  Before reclaiming the 841MB, grep `config.yaml` / `scripts/` / notebooks; if unused, the `.fa.gz` is the
  re-derivable one to drop. `reference` is `sync:false` (public) -- consider `sync:true` (slow re-download).
- GTEx bulk + RNA-seq parquets still to be built; reactome activation (`c61ede6`) not yet smoke-verified.
- Lesson recorded: never validate a feature-matrix change with a test SUBSET -- the full-suite tripwires exist to catch count drift.
