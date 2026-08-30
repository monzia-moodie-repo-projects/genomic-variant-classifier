# CORRECTION 2026-08-30 -- a tool that already knew

**Author: Monzia Moodie**
**Applies to:** `c8c3240` (`INCIDENT_2026-08-29_a-source-nobody-declared.md`)
**Status:** the corrected claims are recorded here; `c8c3240` is not amended.

---

## 0. What went wrong

The incident at `c8c3240` recorded three findings about GENCODE. Two stand. One
rests on a pattern that does not exist, and the incident was written without
running a tool that already reports the central fact.

---

## 1. REFUTED: the `mim2gene` / `omim` pattern

**The claim**, section 4 of the incident:

> GENCODE must be DECLARED before its product structure can be modelled,
> because declaring it forces the product question into the manifest's own
> schema -- where `mim2gene` versus `omim` already shows the established form
> for one publisher's differently-governed products.

**MEASURED 2026-08-30 at `c8c3240`:**

```
data/external/omim      exists True
    genemap2.txt    3,482,253 B
    mim2gene.txt    1,015,104 B
data/external/mim2gene  exists False
```

`mim2gene` and `omim` are **not two directories with different licences.** They
are two FILES IN ONE DIRECTORY, declared as two sources. The manifest declares
DIRECTORIES -- its per-source fields are `location`, `tier`, `class`, `aliases`,
`version`, `acquire`, `regenerate`, `sync`, `notes`, and nothing enumerates the
files inside.

So there is no established form for one publisher's differently-governed
products, and declaring GENCODE would give it ONE entry covering all five
artifacts. The product question would remain exactly where it was.

`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`. The standard, section 3,
prescribes `data/<location>/<name>/`; this pair does not follow it, and the
auditor confirms the consequence -- `mim2gene` reports `MISS` while its file
sits in `omim/`.

### What remains true

GENCODE still needs declaring: 636,522,106 bytes under `data/` outside a
registry that calls itself canonical for everything under `data/`. Only the
STATED MECHANISM was wrong. Declaring it does not answer the product question,
and `ARTIFACT-KEY-INSUFFICIENT-1` stays open on its own terms.

---

## 2. The incident duplicated a tool that already reports this

`scripts/maintenance/audit_data_tree.py`, run 2026-08-30 with
`--data-dir data --manifest configs/data_manifest.yaml`:

```
[orphans / naming hygiene]
  [warn] external/eve_smoke: ORPHAN in external/ (not in manifest)
  [warn] external/gencode:   ORPHAN in external/ (not in manifest)
  [warn] external/grch38:    ORPHAN in external/ (not in manifest)
VERDICT: 3 warning(s) -- aliases/orphans/naming/review above. data/ usable.
```

**Exactly the three orphans my probe found**, and better classified: it also
reports `processed/` and `raw/` subtrees as untracked BY DESIGN rather than as
gaps, which my probe did not distinguish.

`gencode` was never hidden. It has been one line of an existing report since
whenever it was acquired.

### `AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`

MEASURED across every tracked file: **ten name `audit_data_tree`, and not one
invokes it.** Every mention is documentation, a `.gitignore` comment, or the
script itself. `docs/runbooks/MIGRATE_DATA_JUNCTION_TO_LOCAL.md:112` says "Run
`audit_data_tree.py` at session start and before every run" -- an instruction
to a human, not an invocation.

This is the third instance of one shape:

```
preflight_data_guard.py   recorded of ITSELF that nothing called it
drift source kernel       zero production construction sites (DRIFT-SOURCE-
                          KERNEL-HAS-NO-PRODUCTION-CALLER-1, c8c3240)
audit_data_tree.py        ten mentions, zero invocations
```

The subsystem is not weak. It is **unwired**, and that is a different problem
with a different repair.

---

## 3. A REAL defect in the auditor

`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`.

```
ok    ensembl   public      public_redownloadable   0.0B   0f
ok    gtex      public      public_redownloadable   0.0B   0f
ok    tcga      controlled  irreplaceable           0.0B   0f
MISS  hgmd      controlled  irreplaceable           0.0B   0f
MISS  topmed    controlled  irreplaceable           0.0B   0f
```

`tcga` and `topmed` are both `controlled` and `irreplaceable`, both hold ZERO
BYTES, and one reports `ok` while the other reports `MISS`. The only difference
is whether an empty directory exists on disk.

An empty directory is not more usable than an absent one. A run that depends on
`tcga` fails identically either way, and only one of the two is flagged.

Not repaired here: whether `ok` should become `EMPTY`, or whether an empty
declared directory should be `MISS`, is a decision about what the verdict line
means -- and the verdict feeds an exit code that continuous integration may
one day consume.

---

## 4. And nothing is currently cloud-backed

```
cloud-backup (rclone -> G)        0.0B   syncable
offline-only (controlled)       341.1MB  encrypted/offline -- NOT cloud
regenerable (rebuild)            70.5GB  do not back up
```

Five sources carry `sync: true`. Four are `MISS` -- `gtex_gene_expression`,
`reactome_gene_pathways`, `rnaseq`, `rnaseq_gene_expression`. The fifth,
`reference`, is `public_redownloadable`, so it does not meet the standard's
must-back-up bar of `irreplaceable` or `regenerable_expensive`.

The rollup is CORRECT. The consequence is that zero bytes are cloud-backed
today, which is a fact about the estate rather than a defect in the tool.

`cosmic` and `omim` hold 341.1 MB of controlled, irreplaceable data whose
backup is explicitly offline-only. Whether that offline backup exists is
outside what any tool in this repository can see.

---

## 5. What is NOT claimed

That the auditor should be wired into the gate. It is read-only and fast, but
its verdict is a WARNING for orphans, and a gate that fails on three known
orphans would fail every run until they are declared -- a decision about
sequencing, not a measurement.

That `eve_smoke` or `grch38` should be declared or removed. `eve_smoke` is
16,023,084 bytes and `grch38` is 4,033,396,532 bytes; what they are for is
unmeasured.

That `mim2gene` should be merged into `omim`. They have DIFFERENT tiers in the
manifest -- `public` versus `controlled` -- and if that distinction is real
then the layout is wrong rather than the declaration.

---

## 6. Status

The refuted claim is corrected. Three findings are registered and OPEN:
`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`,
`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`,
`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`.

No file in the repository is changed by these measurements.
