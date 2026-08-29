# SESSION 2026-08-29 -- reading replaces building, five times

**Author: Monzia Moodie**
**Commits:** `482c0c9`, `02c13b4`, `54989dc`, `62d0a33`
**Ratchet:** 5682 -> 5682 -> 5682 -> 5682 -> 5690
**Preceding head:** `b67e30f`
**Ending head:** `62d0a33`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `482c0c9` | INCIDENT-ARTIFACT-IDENTITY | NEUTRAL | 5682 | 5667p/15s, 1126.9s |
| `02c13b4` | CORRECTION-REGISTRY-MISSED | NEUTRAL | 5682 | 5667p/15s, 1131.4s |
| `54989dc` | CORRECTION-PART-2 | NEUTRAL | 5682 | 5667p/15s, 1205.6s |
| `62d0a33` | ALIAS-MERGE-DIGEST | ADDITION +8 | 5682 -> 5690 | 5675p/15s, 1537.9s |

DRIFT-1 Phase 1B.4-A through 1B.4-C, and one repair that came out of them.

---

## 1. THE RESULT: Phase 1B.4 needs no new types

Phase 1B.4 was specified to build `ArtifactProductId`, `SourceAuthority`,
`SourceInventory` and `DerivedArtifactLineage`. MEASURED across 2026-08-28 and
2026-08-29, none of them should be built:

```
source registry            configs/data_manifest.yaml, 32 sources, 407 lines
layout convention          docs/standards/DATA_LAYOUT_STANDARD.md, 129 lines
readers                    five scripts under scripts/maintenance/
alias resolver             consolidate_aliases.py
manifest auditor           audit_data_tree.py, exit 2 on controlled+sync
generated projection       configs/rclone_data_filter.txt
tests binding the manifest TWO
```

The subsystem exists and is well made. What was missing is narrow: a digest
check where a size check did destructive work, sixteen sources `SourceName`
cannot express, and test coverage over 32 declarations that had none.

---

## 2. `482c0c9` -- the key installed that morning was already too coarse

Three measurements, all falsifying the committed identity model.

**`ARTIFACT-KEY-INSUFFICIENT-1`.** `SourceArtifactKey(source, artifact_kind)`
was installed at `cffc51f` after measurement proved `source` alone too coarse.
It is too coarse FOR THE SAME REASON. GENCODE release 50 publishes
`transcripts`, `pc_transcripts` and `lncRNA_transcripts`; all three collapse to
`GENCODE/sequence_fasta` and `SourceEvidenceManifest` refuses the duplicate.

THE CENSUS THAT MOTIVATED THE DESIGN CONTAINED THE COUNTEREXAMPLE. It reported
"GENCODE 3 kinds, 5 files". I read that line, used it to justify the key, and
did not ask why five files occupy three kinds.

**Fifteen collisions are THREE phenomena.** Several published products; 3,212
per-protein EVE files that are PARTITIONS of one product; and 18 ClinVar
parquets that are project-derived and attributed to a publisher because the
path contains its name. `primary_release` is not a KIND -- it partly encodes
PROVENANCE.

**`ARTIFACT-ORIGIN-UNMEASURABLE-FROM-CODE-1`.** Of 3,273 artifacts, FOUR have a
creator or consumer site in 1,037 tracked Python files. The invariant held --
zero promoted to `PUBLISHER_BYTES` by a filename -- and 3,263 origins are
simply unrecoverable.

**`CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`** and
**`CACHE-KEY-OPAQUE-AND-INCONSISTENT-1`.** Four of 41 cache keys embed
filesystem locations, one a TEMPORARY DIRECTORY that no longer exists. 37 are
opaque, and `eve_eve_lookup` both DUPLICATES identical results (444,367,755
bytes twice) and SEPARATES different ones (143,457 bytes, two digests).
450,324,943 bytes are held in duplicate, accounted for exactly across five
groups.

### How the cache finding was found

The probe's own test could not WRITE its fixture: a filename cannot contain a
path separator. That failure -- easy to dismiss as a harness quirk -- revealed
that the entries are DIRECTORY TREES ten levels deep, and that analysing the
filename would have reported ZERO located keys while the defect was the
directory structure itself.

---

## 3. `02c13b4` -- a registry existed, and the search missed it

`AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`.

The Phase 1B.2 authority search concluded "no source registry exists anywhere
in the repository". `configs/data_manifest.yaml` describes itself on its own
third line as the "Canonical registry of every data source under data/".

The search looked for PYTHON IDENTIFIERS -- `SOURCE_NAMES`, `SourceRegistry`,
`KNOWN_SOURCES` and four others -- across tracked `.py` files. A registry
expressed as YAML matched none of them and lay outside the file set.

MEASURED consequence. `SourceName`, installed at `cffc51f`:

| quantity | manifest | `SourceName` |
|---|---|---|
| declared sources | 32 | 18 |
| sources it cannot name | -- | **16** |
| members not declared anywhere | -- | 2 (`gencode`, `esm2`) |
| declared aliases | 8 | **0 accepted** |
| aliases I invented | -- | 26 |

Four of the sixteen are `irreplaceable` and constrained: `tcga` and `topmed`
are `controlled`, `rnaseq` and `validation_cohort` are `review`. A vocabulary
that cannot name `tcga` cannot express a manifest containing it.

---

## 4. `54989dc` -- two of those findings were wrong

`CORRECTION-PUBLISHED-BEFORE-THE-STANDARD-WAS-READ-1`.

`configs/data_manifest.yaml` cites `docs/standards/DATA_LAYOUT_STANDARD.md` on
its own line 5. I read the manifest, wrote the correction, committed it at
`02c13b4`, and read the standard afterwards. The standard answers two of that
correction's findings -- the same incomplete-search error the correction itself
describes.

**WITHDRAWN: `MANIFEST-TIER-VOCABULARY-INCOMPLETE-1`.** The standard declares
`tier: review` at line 106 as a deliberate fourth tier, and
`audit_data_tree.py:164` enforces it. What remains is a stale one-line comment.

**DOWNGRADED: `MANIFEST-LOCATION-CONTRADICTS-REGENERATE-OUTPUT-1`.** Lines
66-70 make a built artifact under `external/` a DOCUMENTED EXCEPTION, taken
because moving it would break the connectors that read it. Restated as
`GTEX-BUILT-ARTIFACT-EXISTS-AT-TWO-PATHS-1`: the exception explains the
location, not the 1,093,500 bytes existing twice.

**And the alias framing was backwards.** The standard, line 60: "Aliases are
forbidden: a source has exactly one canonical name. The manifest records known
aliases so the auditor can flag and guide migration." The eight aliases are
DIRECTORY NAMES PENDING REMOVAL, not spellings to accept. Refusing them is
defensible; I had reached that behaviour without understanding it and called it
a defect.

---

## 5. `62d0a33` -- a deletion stops trusting a size

`ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1`, and it is the only defect found in
two days that can DESTROY DATA.

`consolidate_aliases.py` folds an alias directory into its canonical name and
then REMOVES the alias directory. Collision detection compared `st_size` at
line 78; post-merge verification compared `st_size` at lines 122-124; line 127
deleted the directory.

REPRODUCED in a sandbox, two files named `scores.csv`, both exactly 612,501
bytes, different content:

```
ORIGINAL   exit 0   "merged + verified; removed .../spliceai_scores"
                    the alias file was DESTROYED
REPAIRED   exit 1   "[ABORT] 1 differing name-collision(s)"
                    the alias directory and its file survive
```

The script never overwrites -- `shutil.copy2` is skipped when the target exists
-- so it discarded the SOURCE instead. The loss is silent and the report says
"verified".

612,501 IS NOT ARBITRARY. The lineage census measured exactly that size for
`TPIS_HUMAN.csv` and `TSHB_HUMAN.csv` under `data/external/eve/`, with
different digests.

**Why it survived:** the script had NO TEST. Of the five scripts reading the
manifest, only the storage guard was bound by one.

**The repair:** `_digest` streams SHA-256 in one-megabyte chunks;
`_same_content` keeps a size PRE-CHECK -- different sizes cannot be equal
content -- then decides by digest.

### Ten boundaries sabotaged, ten detected, after four repairs to the TESTS

The first matrix left four undetected, and every one was a weakness in the
tests rather than the repair:

- a dry-run fixture that ABORTED before reaching the branch under test
- a structural check satisfied by the substring `hashlib` after the IMPORT was
  deleted, because `hashlib.sha256` remained in the body
- two guards genuinely unreachable from a single-process test

The two unreachable guards are KEPT and asserted structurally. One guards a
filesystem changing between planning and execution; the other a copy that
silently did not happen. Neither is an impossible state, so unlike the three
checks `suite_transition.py` deleted, they stay.

---

## 6. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Phase 1B.2 searched only `.py` and declared no registry exists | the manifest's own third line |
| 2 | Published a correction before reading the standard it depends on | the standard answered two of its findings |
| 3 | `.format()` on a block containing dict literals consumed their braces | `KeyError` |
| 4 | Referenced `doc.txt`, `commit.txt`, `ratchet.txt` before writing them | `FileNotFoundError`, twice |
| 5 | An installer anchor reconstructed from MEMORY | the real code is ONE print with an embedded newline, not two |
| 6 | A package-path check carried into a unit that tests a SCRIPT | the installer refused a correct payload |
| 7 | A label hard-coded `c77a1a9` while the baseline was `54989dc` | visible in the dry run |
| 8 | Quoted a byte count in prose without measuring it | 3,441 stated, 3,318 measured |
| 9 | Asserted a chain digest using a FILE digest | one attestation check failed |
| 10 | Wrote "the commit is COVERED" above output reading `1 NOT named` | the measurement was two lines up |

Errors 3 through 7 were caught by assertions or by the installer refusing.
Errors 8 through 10 were in PROSE, where nothing asserts.

**The artifacts have been correct throughout, because they compute rather than
recall. The sentences around them are where the errors live.**

---

## 7. Findings

### Closed
`ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1` (repaired at `62d0a33`).
`MANIFEST-TIER-VOCABULARY-INCOMPLETE-1` (withdrawn).

### Registered and OPEN
- `ARTIFACT-KEY-INSUFFICIENT-1`
- `ARTIFACT-ORIGIN-UNMEASURABLE-FROM-CODE-1`
- `CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`
- `CACHE-KEY-OPAQUE-AND-INCONSISTENT-1` -- 450,324,943 duplicated bytes
- `AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`
- `CORRECTION-PUBLISHED-BEFORE-THE-STANDARD-WAS-READ-1`
- `GTEX-BUILT-ARTIFACT-EXISTS-AT-TWO-PATHS-1`
- `PROBE-FETCH-CALL-NAME-IS-AMBIGUOUS-1` -- judging `get` by callee reported
  1,453 acquisition sites; judging by ARGUMENT reported 5, all of them
  literature APIs
- `PROBE-DISPLAY-LIMIT-USED-AS-A-DATA-LIMIT-1`
- `COMMIT-ID-EXTRACTION-CANNOT-CLASSIFY-1`
- `PROBE-DIRECTORY-LARGELY-UNTESTED-1` -- 57 of 58 tracked probes untested

---

## 8. Ending state

```
HEAD     62d0a33
ratchet  5690
gate     5675 passed, 15 skipped, 0 failed, 0 errors
counters ratchet, badge and roadmap all 5,690
```

Suite identity moved for the first time in five commits:
`17c32d1da8f78ecd` -> `14339e6e37abcb84`.

## 9. Next intended action

`SourceName` cannot name 16 of 32 declared sources. The repair is to replace an
invented vocabulary with a typed reader over `configs/data_manifest.yaml`. The
defect is LATENT -- nothing constructs a `SourceEvidenceManifest` from real data
-- which is the window in which to do it.

Before that, an authority search for existing readers, ACROSS EVERY LANGUAGE.
The five maintenance scripts already parse this file, and `StoragePolicy.load`
in `preflight_data_guard.py` is a typed reader over it that records
`policy_source`. That is the pattern to follow, and possibly the code to reuse.
