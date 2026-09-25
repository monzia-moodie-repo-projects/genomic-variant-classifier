# DECISION 2026-09-24 -- gnomAD 4.1.1 approval recorded; approval is not adoption

Author: Monzia Moodie. Resolves the comparison in issue #17 ("source-monitor needs review").

## What is recorded

The Genome Aggregation Database (gnomAD) release **4.1.1** is **approved** by the owner. The approval was first
recorded in the owner's rulings preserved on 2026-09-22 (decision.txt generations `6cb17af5...` and `57859012...`),
restated on 2026-09-23 (`366556f7...`, `2c01bd51...`), and restated again on 2026-09-24 ("Since you have already
approved 4.1.1 ..."). The date of the original grant is not recorded in the repository; the 2026-09-15 session record
documents the *discovery* of 4.1.1, not its approval.

## Scope -- quoted from the rulings

- "Record gnomAD 4.1.1 approval separately from verified acquisition, qualification, and actual use. Product
  identities matter: a release change for constraint does not automatically establish a matching change for every
  frequency product."
- "Approval of gnomAD 4.1.1 remains distinct from qualification and adoption."
- "The source manifest should remain the authority, with monitoring as a derived view."

## What changed

- `source_monitor/gnomad_release_check.py` and `source_monitor/request_verifier.py`: `APPROVED_BASELINE = "4.1.1"`.
  The two remain independent declarations; `test_the_verifier_and_the_adapter_declare_the_SAME_baseline_and_kind`
  requires agreement. Any release newer than 4.1.1 (4.1.2, 4.2, 4.10, 5.0) still alerts, tested.
- The verifier's plan fingerprint now includes the approved baseline, so an outcome from before this approval and one
  after it are never compared as if under one plan. The fingerprint identifies CONFIGURATION only: it does not prove
  which code executed or which run produced a report.
- A monitor defect found while making this change, then a second one found in its fix by the owner's review: the
  runner's claim/witness cross-check was one-directional, and its forward direction matched SUBSTRINGS (witnesses
  4.1.2 and 4.1.20 with one claim for 4.1.20 exited 1, 4.1.2 unreported). One reconciliation now requires the claimed
  versions to EQUAL the independent witnesses as parsed identities, every claim to name the current baseline, and
  canonical claim text -- otherwise the run is refused (exit 2).
- `configs/data_manifest.yaml`, source `gnomad`: an approval note. `version: "v4.1 exomes"` is unchanged.

## What did NOT change -- the distinction this record exists to keep

Nothing was migrated, re-annotated, or relabelled. Measured on 2026-09-24, every gnomAD input of the PRODUCTION pipeline
is v4.1: the constraint table `gnomad.v4.1.constraint_metrics.tsv` (Dockerfile, preflight scripts, run logs), the exome
VCFs `gnomad.exomes.v4.1.sites.*`, and `scripts/gnomad_cloud_sync.py` (`release/4.1/`). This is a claim about PRODUCTION, not about everything the project has ever used: exploratory work has used 4.1.1 --
gnomAD's published 4.1.1 guidance in the 2026-08-09 constraint incident, and a 4.1.1 constraint table examined in the
2026-09-12 probe (gvc_gnomad411_probe_2026-09-12.txt: 113 columns, 221,898 rows). Those keep their actual version.
Existing 4.1 artifacts must not be relabelled 4.1.1. Acquisition, qualification and adoption of 4.1.1 -- per product -- are tracked in their own
GitHub issue, linked from issue #17's closing comment.

## A newer release was observed while this was being made

The live listing read by the source-monitor run of 2026-09-24 03:55 UTC (on `main` at 25645de; response SHA-256
`577c27b247c50be254d03e023205d7d8319e53beeb5152da9e0b56d016cd6122`, 327 bytes, one terminal page, 13 prefixes)
contains `release/4.1.2/`, absent from the listing measured 2026-09-14. 4.1.2 is genuinely newer than 4.1.1 and is
NOT approved. With this change on main the monitor therefore still reports "release 4.1.2 is newer than the approved
4.1.1" (exit 1, review required) -- correctly. That finding is a separate decision, not part of #17.

## Verification after this reaches main, and closing #17

GitHub -> Actions -> source-monitor -> Run workflow on `main`. Verify with `Check_SourceMonitorReport_2026-09-24.ps1`,
which checks three layers because a report cannot authenticate itself: EXECUTION from GitHub's own run and artifact
metadata (repository, workflow, branch, event, the run's commit tree equal to the approved tree, and the artifact digest
equal to the downloaded zip); CONFIGURATION (the 4.1.1 plan fingerprint -- configuration only); and OBSERVATION (an
independent replay of the retained captures whose newer releases must equal both the witnesses and the claims). A
qualified 4.1.2 finding is expected to remain and is reported as its own review item.

Do NOT close #17 until, per the owner's ruling of 2026-09-24: the approval is verified on the intended `main`; a
durable 4.1.2 constraint-qualification item exists; the production-adoption item is linked; each has an owner, a next
review date and acceptance conditions. Those items must not carry the generic `source-monitor-alert` label, which the
alert workflow uses to route its own comments.

## Outcome (2026-09-25)

- Merged: commit `da02b23c` via pull request #20 as `42ab3550` on `main` (tree `49504a98`, parents `0ea9f3a` and
  `da02b23c`); every pull-request check passed (lockfile, pytest 3.11 and 3.12, both drift monitors, Docker build smoke
  test), and CI passed again on `main`. It was validated beforehand in a disposable clone on the owner's machine (all
  13 stages; checkout preservation verified unchanged).
- Verified on `main`: source-monitor run 36095779494 (2026-09-25 04:46 UTC). The three-layer check passed 20 of 20:
  4.1.1 approval VERIFIED, observation COMPLETE, 4.1.2 outstanding as its own review item. A calibration run on the
  pre-approval run 35953468798 failed exactly the six checks predicted for the old code.
- Issue #17 closed as completed on 2026-09-25 at 06:17 UTC. Successors: #21 (gnomAD 4.1.2 constraint -- bounded
  qualification) and #22 (gnomAD constraint adoption for cohort v2), each owned by Monzia Moodie with a next review on
  2026-09-28, acceptance conditions, and no `source-monitor-alert` label.
- Correction: the commit message of `da02b23c` describes an earlier one-directional claim check, says "every gnomAD
  artifact in use is still v4.1" (true of production inputs only), and gives the suite as 6,938 (it is 6,963). The
  code and this record are correct; the correction is posted on pull request #20 and in docs/CHANGELOG.md.

## Open questions recorded, not decided here

The project's release grammar (shared by adapter and verifier, so their agreement test cannot see this) accepts
leading zeros (`04.1` and `4.01` parse as 4.1), treats `4.1.0` as newer than `4.1`, and accepts four-part versions.
Whether to canonicalise those is a policy change to what counts as newer, for a separate decision.

## Follow-up (separate change)

Per "monitoring as a derived view": a typed `approved_release` field in the manifest (`source_registry.py` refuses
unknown keys today), with the monitor's baseline derived from it rather than declared in code. And a parameterised
three-layer verifier in the repository, so future approvals are verified by versioned code rather than a script
delivered beside it (`Check_SourceMonitorReport_2026-09-24.ps1` hard-codes this approval's tree and plan).
