# SESSION 2026-09-01 -- a coordinate the evidence required

**Author: Monzia Moodie**
**Commits:** `24bfb11`, `4bed1b8`
**Ratchet:** 5719 -> 5732
**Preceding head:** `260d1bc`
**Ending head:** `4bed1b8`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `24bfb11` | DECLARE-GENCODE | NEUTRAL | 5719 | 5704p/15s |
| `4bed1b8` | PRODUCT-COORDINATE | ADDITION +13 | 5719 -> 5732 | 5717p/15s, 1277.1s |

Items 2, 3 and 4 of the adopted priority order, closed.

---

## 1. `24bfb11` -- 636 megabytes acquire a declaration

`GENCODE-ACQUIRED-VALIDATED-UNDECLARED-1`, recorded at `c8c3240`. 636,522,106
bytes under `data/external/gencode/` sat outside the registry that calls itself
the "Canonical registry of every data source under data/", and since `fd6cd4e`
the data-tree gate reported it as an ORPHAN before every Run-17 preflight.

### The bytes were verified before the assertion

The publisher's own manifest records a SHA-256 per artifact, captured
2026-06-17. All five were re-verified against disk on 2026-08-30:

```
gencode.v50.annotation.gtf.gz          83FBA3E9B03F0B8C   124,527,720 B
gencode.v50.annotation.gff3.gz         2AAF245C91ED00E8   160,590,749 B
gencode.v50.transcripts.fa.gz          5A320F524D73B579   183,554,921 B
gencode.v50.pc_transcripts.fa.gz       ED3BCD295A39E97F   129,953,566 B
gencode.v50.lncRNA_transcripts.fa.gz   0F49F602B9536A15    37,879,655 B
```

Nothing had checked them in the seventy-four days since acquisition:
`validate_gencode_assets.py` resolves `GENOMIC_DATA_ROOT`, which IS SET to a
directory that does not exist. A declaration asserts something about bytes on
disk, so the bytes were established first.

### Every value is sourced, and three are labelled DERIVED

`location`, `version` and `acquire` are MEASURED. The acquire string is
recorded five times in `GENCODE_v50_manifest.json`, once per artifact,
byte-identical -- not invented, which matters because `SourceName` shipped with
twenty-six plausible aliases that appeared nowhere. `version` states GRCh38
explicitly because the same release directory also publishes a
`GRCh37_mapping/` subtree.

`tier`, `class` and `sync` are DERIVED, and the derivation is stated in the
installer so it can be rejected: the archive is anonymous, no `acquire` string
in the file carries a credential, and every other `public_redownloadable`
source is `sync: false`.

### Measured after patching

By PARSING both versions and differencing, not by reading a diff: only
`gencode` added, no existing entry altered, `version`/`gdrive`/`storage`
unchanged. `SourceRegistry` loads 33 sources and classifies it published with
`must_back_up` False. The auditor stops reporting it as an orphan.

---

## 2. `4bed1b8` -- three GENCODE products stop being one key

`ARTIFACT-KEY-INSUFFICIENT-1`, open since `482c0c9`. Of four measured collision
classes, THREE dissolved into axes that already exist:

| collision | files | resolved by |
|---|---|---|
| `ClinVar/vcf` | 2 | `CoordinateContext` -- they differ by ASSEMBLY |
| `ClinVar/primary_release` | 18 | `acquire`/`regenerate` -- project-derived |
| `EVE/csv` | 3,212 | a PARTITION axis, not a product axis |
| `GENCODE/sequence_fasta` | 3 | **nothing. The genuine missing dimension.** |

### The design, and what each choice prevents

**OPTIONAL AND LAST.** Eighteen construction sites pass two positional
arguments, and 73 of 74 tests passed before one was touched. The single failure
was the one literal `canonical_key` tuple assertion. A sabotage that makes the
default `"default"` now FAILS, so the rulings' prohibition is enforced rather
than intended.

**ABSENCE RENDERS AS `""` IN A FIXED THREE-TUPLE**, and `product=""` is REFUSED
at construction. A variable-length tuple would make a two-field and a
three-field identity structurally different -- the concatenation ambiguity
`RepresentationIdentity` was length-prefixed to prevent on 2026-08-28. Without
the refusal, `""` and `None` would produce the SAME canonical key while being
different values: two identities claiming one identity.

**THE DOMAIN IS NOW v4.** A product coordinate alters EQUALITY, so a v3 digest
and a v4 digest are incomparable. That is the mechanism by which a legacy
record which cannot say WHICH product it describes is REFUSED rather than
silently given `product="unknown"`.

### The transition

MEASURED by collecting both trees and differencing: 74 old, 87 new, 13 added,
**ZERO REMOVED**. Widening the key's equality moved no existing identity in the
whole repository suite -- the acceptance run reported `unchanged 5719`.

---

## 3. Ten boundaries sabotaged, ten detected -- after one repair

One was undetected: `of()` could DISCARD the product and no test noticed,
because `SourceArtifactKey.of(...)` was called ZERO times in the file. **I had
measured that when sizing the unit and did not act on it.**

Two tests close it, and the second catches a subtler mutation: coercing `""` to
`None` would bypass the validator rather than refusing it, collapsing the
absent/empty distinction the encoding depends on.

---

## 4. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Declared myself blocked without re-reading a file already in hand | it contained the acquire URL, the digests and the block format |
| 2 | A slice matched at index 0 and truncated an installer | restored from the template; redone with asserted-unique boundaries |
| 3 | Replacing a docstring header produced a stray `_OF_DOC` class attribute | grep for the leftover, then a measured re-slice |
| 4 | `if road is None:` survived a rename, directly above `MANIFEST_PAYLOAD` | stale-reference count, then reading the four lines |
| 5 | The pin stem transform kept a `_new` suffix rule the payloads no longer use | simulated every payload against its pin before delivery |
| 6 | Measured that `of()` was never called, and did not act on it | sabotage, one boundary later |
| 7 | Said "every prior gate today reported no warning count" without checking | the warnings' CONTENT proves they are pre-existing; the claim about counts was unverified |

Errors 2, 3 and 4 are one shape: **an edit anchored on something not measured
in the file being edited.** Each cost one iteration because an assertion or a
stale-reference count caught it.

---

## 5. Findings

### Closed
`GENCODE-ACQUIRED-VALIDATED-UNDECLARED-1` -- `24bfb11`.
`ARTIFACT-KEY-INSUFFICIENT-1` -- `4bed1b8`, narrowly, around three products.

### Registered
`RESOURCE-WARNING-FROM-UNCLOSED-READS-1`. MEASURED 2026-09-01 with `-W default`:
`tests/unit/test_backup_artifacts.py:80` does
`ast.parse(io.open(f, encoding="utf-8").read())` with no close, once per source
module scanned, and `scripts/train.py:76` leaves `logs/train.log` open through
`logging.basicConfig`. Also `tests/test_resolve_alleleless_ncbi.py` at 75, 119
and 122. Nothing breaks today; on Windows an unclosed handle can block a later
rename, which is what an installer's atomic replace performs.

The remaining warnings are `sklearn` `UndefinedMetricWarning` from
`test_no_curve_in_a_degenerate_cohort_mixes_valid_and_absent_entries` -- a test
whose purpose IS a degenerate cohort, so those are correct behaviour observed.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`,
`CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`,
`VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`,
`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`,
`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`,
`GTEX-BUILT-ARTIFACT-EXISTS-AT-TWO-PATHS-1`,
`CACHE-KEY-OPAQUE-AND-INCONSISTENT-1` (450,324,943 duplicated bytes),
`GATE-TIMING-NOISE-EXCEEDS-TREND-1` (observed, cause unknown).

---

## 6. Ending state

```
HEAD     4bed1b8
ratchet  5732
gate     5717 passed, 15 skipped, 0 failed, 0 errors, 33 warnings
suite    02535ddfc579feab -> 51fe4d149b0097bb
```

## 7. Next intended action

Item 5: Phase 1C's production `SourceEvidenceManifest` builder, closing
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`.

The rulings are specific about its shape. The manifest must be
EXECUTION-DERIVED, not disk-derived: instrumentation follows what a computation
actually opened, so a source the model did not use does NOT enter the run
evidence. A directory census is inventory; this is provenance, and conflating
them is easier now that the auditor exists.

Registry admission belongs at that construction boundary --
`registry.canonical_for(observed_source)` there, never inside
`SourceArtifactKey`, which must stay deterministic and environment-independent.
