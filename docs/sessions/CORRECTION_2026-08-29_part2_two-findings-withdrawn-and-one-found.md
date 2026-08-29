# CORRECTION 2026-08-29 part 2 -- two findings withdrawn, and one found

**Author: Monzia Moodie**
**Applies to:** `02c13b4` (`CORRECTION_2026-08-29_a-registry-existed-and-the-search-missed-it.md`)
**Status:** the corrected claims are recorded here; `02c13b4` is not amended.

---

## 0. What went wrong

`02c13b4` recorded that a source registry existed and that my Phase 1B.2
authority search missed it because the search was scoped to Python files.

That finding stands. Two OTHER findings in the same document do not, and both
fail for the same reason the document itself describes: **I published before
reading a document the claims depend on.**

`configs/data_manifest.yaml` cites `docs/standards/DATA_LAYOUT_STANDARD.md` on
its own line 5. I read the manifest, wrote the correction, committed it at
`02c13b4`, and read the standard afterwards. The standard answers both claims.

`CORRECTION-PUBLISHED-BEFORE-THE-STANDARD-WAS-READ-1`.

---

## 1. WITHDRAWN: `MANIFEST-TIER-VOCABULARY-INCOMPLETE-1`

**The claim.** The manifest header declares three tiers -- `public | academic |
controlled` -- while `review` appears three times, on `rnaseq`,
`validation_cohort` and `rnaseq_gene_expression`. I recorded this as an
incomplete vocabulary.

**Why it is withdrawn.** `DATA_LAYOUT_STANDARD.md` line 106 declares it:

> `tier: review` marks sources whose access tier or leakage-independence must
> be confirmed before they are synced or used.

`review` is a deliberate fourth tier with a stated meaning, and it is enforced:
`audit_data_tree.py:164` warns when a `review` source is present AND marked
`sync: true`.

**What remains true.** The manifest's own header comment lists three tiers and
the standard lists four. That is a stale COMMENT, not an incomplete vocabulary
-- a documentation defect one line long, not a modelling gap.

---

## 2. DOWNGRADED: `MANIFEST-LOCATION-CONTRADICTS-REGENERATE-OUTPUT-1`

**The claim.** `gtex_gene_expression` declares `location: external` while its
regenerate command writes `--out data/processed/gtex/gtex_v11_gene_expression.parquet`,
and both files exist BYTE-IDENTICAL at 1,093,500 bytes, digest
`73985d43c41cc69f`. I recorded the declaration as contradicting itself.

**Why it is downgraded.** `DATA_LAYOUT_STANDARD.md` lines 66-70:

> Built artifacts whose paths are already referenced by code (e.g.
> `data/external/reactome_gene_pathways.parquet`) are kept where the code
> expects them and recorded in the manifest as `regenerable_expensive`; they
> are NOT moved (moving them would break connectors).

So a built artifact living under `external/` is a DOCUMENTED EXCEPTION, taken
deliberately because moving it would break the connectors that read it.

**What remains true, and it is still a defect.** The exception explains the
LOCATION. It does not explain TWO COPIES. 1,093,500 bytes exist twice, and the
standard's rule would keep ONE file at the code-referenced path. Whether the
`processed/` copy is a stale build output or a second live path is unmeasured.

Restated as `GTEX-BUILT-ARTIFACT-EXISTS-AT-TWO-PATHS-1`, severity reduced from
"declaration contradicts itself" to "one artifact, two locations, one of them
probably stale".

---

## 3. FOUND, and it is more serious than either withdrawal

`ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1`.

`scripts/maintenance/consolidate_aliases.py` folds alias directories into
canonical ones. It is careful: dry-run by default, re-confirms emptiness before
deleting, keeps the alias directory if verification fails, and its docstring
states "never overwrite".

Collision detection, at line 78:

```python
if tgt.exists() and tgt.stat().st_size != f.stat().st_size:
    collisions.append(str(rel))
```

Verification after merge, at lines 122-124, compares `st_size` again. Then line
127 removes the alias directory.

**EQUAL SIZE IS NOT EQUAL CONTENT, and this project has measured that.** The
lineage census of 2026-08-28 found THREE equal-size groups with different
digests, including two EVE files at exactly 612,501 bytes each:

```
data/external/eve/EVE_all_data/variant_files/TPIS_HUMAN.csv   465d9fd2eee342c8
data/external/eve/EVE_all_data/variant_files/TSHB_HUMAN.csv   2ef2b73abcadc062
```

Two files of equal size and different content take this path: the collision
check passes, `shutil.copy2` skips because the target exists, verification
confirms "same size", and the alias directory is DELETED. The alias file is
lost, silently, and the script does not overwrite -- it discards the source
instead.

The repair is `hashlib.sha256` on both sides of both comparisons. Merges are
rare, so the cost is bounded.

**`audit_data_tree.py` does not warn about this.** It reports an alias as
"migrate into `<canonical>/`" and never inspects contents, so nothing warns
before the destructive path runs.

---

## 4. What the standard settled that no probe of mine could

Reading `DATA_LAYOUT_STANDARD.md`, `consolidate_aliases.py` and
`audit_data_tree.py` answered Phase 1B.4-C completely:

```
source registry            configs/data_manifest.yaml, 32 sources
layout convention          DATA_LAYOUT_STANDARD.md, 129 lines
readers                    five scripts under scripts/maintenance/
alias resolver             consolidate_aliases.py
manifest auditor           audit_data_tree.py, exit 2 on controlled+sync
generated projection       configs/rclone_data_filter.txt
tests binding the manifest TWO
```

**Nothing in Phase 1B.4 needs building.** The subsystem exists and is well
made. What is missing is narrow: a digest check where a size check does
destructive work, sixteen sources `SourceName` cannot express, and test
coverage over 32 declarations that currently have none.

### And the alias semantics were backwards in `02c13b4`

That document says `resolve_source_name` "REFUSES every alias this project
actually uses". `DATA_LAYOUT_STANDARD.md` line 60 states:

> Aliases are forbidden: a source has exactly one canonical name. The manifest
> records known aliases so the auditor can flag and guide migration.

The eight aliases are DIRECTORY NAMES PENDING REMOVAL, not spellings to accept.
A resolver that accepted `hgmd_pro` would accept the very thing
`consolidate_aliases.py` exists to eliminate. Refusing them is defensible; I
had reached that behaviour without understanding it, and described it as a
defect.

**The measured defect in `SourceName` is unchanged: it cannot name 16 of the 32
declared sources, including `tcga` and `topmed`, both controlled and
irreplaceable.**

---

## 5. What is not claimed

That `consolidate_aliases.py` has ever lost data. No alias directory currently
on disk has been checked for an equal-size, different-content collision against
its canonical target. The hazard is demonstrated by construction and by the
census's three real equal-size groups; an actual loss is unmeasured.

That the `processed/` copy of the GTEx artifact is stale. Only that two copies
exist with identical bytes.

That `SourceName` should be deleted rather than rebuilt as a manifest reader.
That decision needs the consumers measured, and nothing constructs a
`SourceEvidenceManifest` from real data yet.

---

## 6. Why a second correction rather than an amendment

`CORRECTION_2026-08-29_a-registry-existed-and-the-search-missed-it.md` is
pinned by digest `aef3414b67e41b22` in the attestation for `02c13b4`. The
convention is unchanged: corrections sit beside records, never inside them --
and a correction is itself a record that can require one.
