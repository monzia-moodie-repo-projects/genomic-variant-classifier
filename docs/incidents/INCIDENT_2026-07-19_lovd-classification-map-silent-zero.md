# INCIDENT 2026-07-19 — LOVD classification map: a sentinel that means four things

**Status:** OPEN — measured and reconciled; remedy not yet designed.
**Severity:** Feature-integrity. Not a crash, not a test failure, not a blocker. The suite is
green and every gate passes.
**Related:** `docs/incidents/INCIDENT_2026-05-02_lovd-silent-zero.md` (closed) — this is a
*different* defect in the same connector, found while reconciling that incident's numbers.
**Evidence:** `docs/measurements/LOVD_CLASSMAP_2026-07-19.txt`.
**Tree:** `988c082`. Suite 1,968 collected / 1,961 passed / 7 skipped, Continuous Integration
green (#526).

LOVD = Leiden Open Variation Database.

---

## 1. How this was found

Not by a failing test. By reconciling two numbers that disagreed by a factor of fifteen:

- **5,553** — inner-join matches recorded at `INCIDENT_2026-05-02_lovd-silent-zero.md:124`,
  with a regression assertion at line 247 expecting **≥4,500 in train**.
- **369** — LOVD coverage recorded in the 2026-06-01 source-status audit.

Both figures were correct. They measure different quantities, and the gap between them is the
defect.

The instrument that produced the 5,553 — `diag_lovd_join.py` — had been sitting in
`stash@{0}` since **2026-05-08 00:57:46 −0400**, seventy-two days. It had never been committed,
did not exist on disk, and was cited **by filename** at `INCIDENT_2026-05-02:117`. The
incident's own exhibit was a dangling reference, recoverable only from a stash that any
re-clone would have destroyed silently.

## 2. What was measured, 2026-07-19

### 2A. The map covers ClinVar's vocabulary; the artifact speaks LOVD's

`_CLASSIFICATION_MAP` in `src/genomic_variant_classifier/data/lovd.py` holds **11 entries**,
parsed directly from source. Every one is a ClinVar clinical-significance term: `pathogenic`,
`likely benign`, `variant of uncertain significance`, `conflicting interpretations of
pathogenicity`, and so on. All 11 map to a nonzero ordinal 1–4.

The artifact `data/external/lovd/lovd_all_variants.parquet` holds **18,006 rows** across
**14 distinct `classification_raw` values**:

| classification_raw | rows | mapped |
|---|---:|---|
| `notClassified` | 7,435 | **no → 0** |
| `functionAffected` | 7,200 | **no → 0** |
| `functionNotAffected` | 1,508 | **no → 0** |
| `pathogenic` | 977 | yes → 4 |
| `functionProbablyAffected` | 264 | **no → 0** |
| `likely benign` | 208 | yes → 1 |
| `likely pathogenic` | 191 | yes → 3 |
| `benign` | 105 | yes → 1 |
| `unknown` | 77 | **no → 0** |
| `pathogenic (dominant)` | 26 | **no → 0** |
| `likely pathogenic (dominant)` | 8 | **no → 0** |
| `nan` | 3 | **no → 0** |
| `likely pathogenic (!)` | 2 | **no → 0** |
| `functionProbablyNotAffected` | 2 | **no → 0** |

**1,481 rows (8.2%) map to a nonzero ordinal. 16,525 rows (91.8%) are zeroed.**

### 2B. The reconciliation, exact

Joining `models/v1/clinvar_enriched.parquet` (1,700,687 rows) to the artifact on
`(chrom, pos, ref, alt)` — the same keys `LOVDConnector.annotate_dataframe` uses:

```
inner-join matches : 5,576
    ordinal 0      : 5,207     <-- reads as NOT IN LOVD
    ordinal 1      :    87
    ordinal 3      :    54
    ordinal 4      :   228
matched AND nonzero:   369     <-- EXACTLY the 2026-06-01 audit figure
matched BUT zero   : 5,207
```

**369 = 369.** The audit counted `lovd_variant_class > 0`; the incident counted join matches.
Both were right. No regression occurred. The join is healthy and has been throughout.

### 2C. A second, smaller discrepancy — 5,576 against 5,553

The measured join returns **23 more matches** than the incident recorded. The cause is
methodological: `diag_lovd_join.py` filters ClinVar to the ten LOVD genes *before* joining;
`lovd.py` does not. Those 23 variants match positionally but carry a `gene_symbol` outside
LOVD's ten genes — a gene-annotation disagreement between the two sources, or a missing symbol.

**5,576 is the figure the connector actually produces. 5,553 was measured through a filter the
production path never applies.** All 23 map to ordinal 0, so the 369 is unaffected. The
incident's line 124 should be read with that caveat.

## 3. The defect: `0` means four different things

Three independent code paths converge on the same value, and nothing downstream can separate
them:

1. **Variant absent from LOVD** — `merged[...].fillna(_DEFAULT_CLASS)` (`lovd.py:120-122`).
   Correct and intended.
2. **Variant PRESENT in LOVD, classification string not a key of the map** — the same
   `fillna`, because the map lookup produced nothing. **5,207 variants in the measured cohort.**
3. **Column absent entirely** — `variant_ensemble.py:803-804`:
   `feats["lovd_variant_class"] = df.get("lovd_variant_class", pd.Series([0] * len(df), ...))`.
   A missing connector fabricates a full column of zeros.
4. **Stub mode** — `lovd.py:70,102-105`: with no `--lovd-path`, every variant receives 0 and a
   warning is logged.

This is the defect class the project has now hit repeatedly: `PLACEHOLDER_BASE = "A"`
one-hot-encoding to confident adenine (Phase 3b), `X_seq` placeholder DataFrames satisfying a
signature (roadmap 6.28), and now a sentinel ordinal absorbing four distinct states. **A value
that means "absent" must not be reachable by data that is present.**

### 3A. Thirty-six pathogenic calls are being discarded

`pathogenic (dominant)` (26), `likely pathogenic (dominant)` (8) and `likely pathogenic (!)`
(2) are **pathogenic assertions recorded as "not in LOVD."** The map holds `pathogenic` → 4 and
`likely pathogenic` → 3, and drops the identical call the moment an inheritance annotation or a
curator's mark is appended.

This is a parsing failure, not a vocabulary gap, and it discards precisely the class of variant
the project exists to identify. **Whether it is a normalisation gap or a deliberate exclusion
is NOT yet established** — that depends on whether the lookup lowercases or strips before
mapping, which requires reading the connector body. Recorded here as unresolved rather than
assumed.

### 3B. Nine thousand functional classifications are on a different axis

`functionAffected` (7,200), `functionNotAffected` (1,508), `functionProbablyAffected` (264) and
`functionProbablyNotAffected` (2) total **8,974 rows** — half the artifact. These are
**assay results**, not clinical assertions. `functionAffected` says an experiment showed altered
protein function; `pathogenic` says a curator judged the variant disease-causing.

Folding them into a clinical-significance ordinal would conflate two axes and corrupt the
feature's meaning. Leaving them at 0 discards them. Neither is right, and the choice between a
separate feature and deliberate exclusion is a scope decision.

### 3C. Seven thousand explicit "unclassified" statements collapse into "absent"

`notClassified` (7,435), `unknown` (77) and `nan` (3) total **7,515 rows**. "Present in LOVD
but not classified" is information the current encoding cannot express at all. `nan` as a
literal three-character string indicates null values stringified before lookup.

## 4. Why no gate caught this

**The tests are plumbing tests, and they pass correctly.** Both ran and passed on 2026-07-19
with `-rs` showing no skips, in 6.60 seconds:

```
test_lovd_annotation_reaches_training_matrix          PASSED
test_lovd_annotation_silent_zero_when_path_omitted    PASSED
```

Their names state their scope: one asserts annotation *reaches* the feature matrix, the other
asserts the stub-mode zero when `--lovd-path` is omitted. **Neither is a coverage assertion**,
so 91.8% of the artifact can be zeroed with both green. That is not a failure of the tests; it
is an absence of a test for this property.

**The feature census guards variance, not coverage.** `variant_ensemble.py:562-563, 2176, 2206`
fail loud on features that are dead — constant. `lovd_variant_class` takes values 0, 1, 3 and 4,
so it has variance and passes cleanly, while being nonzero for **369 of 1,700,687 rows
(0.0217%)**. A feature that is 99.98% one value is not constant, and the guard cannot see it.

Same shape as the 0.5% razor recorded at `run_phase2_eval.py:455-465`: a gate measuring the
wrong quantity passes with room to spare.

*(Constant name for the record: `EXPECTED_TABULAR_FEATURE_COUNT`, `variant_ensemble.py:2208` —
not `EXPECTED_FEATURE_TABULAR_COUNT`.)*

## 5. What is NOT established

- **Coverage on the current cohort.** Everything above measures
  `models/v1/clinvar_enriched.parquet` (1,700,687 rows), the Run 9-era cohort the original
  incident used. The live cohort is `data/processed/clinvar_grch38_clean_seq.parquet`
  (4,399,089 rows). Coverage there is a separate measurement, not attempted.
- **Whether the lookup normalises.** Requires the connector body.
- **What the two passing tests assert in detail.** Requires the test body.
- **Whether the 23-variant gene-symbol disagreement indicates a data problem** or is benign.

## 6. Candidate directions — NOT decisions

`lovd_variant_class` is in `TABULAR_FEATURES` under a fail-loud count guard, so any change here
moves the feature contract. Scope is Monzia's call.

1. **Normalise parentheticals** so `pathogenic (dominant)` reaches 4. Narrow, unambiguous,
   recovers 36 pathogenic calls. Separable from everything else.
2. **Add a presence indicator** so `0` stops meaning both "absent" and "present but
   unclassified". Removes the sentinel collision at its root.
3. **Treat functional effect as its own feature**, or exclude it deliberately and in writing.
   Do not fold an assay axis into a clinical-significance ordinal.
4. **Fail loud on unmapped strings** rather than silently defaulting — the connector should
   refuse, or at minimum log every distinct unmapped value with its row count.
5. **Add a coverage assertion** to the suite, since variance guarding demonstrably cannot see
   this class of defect.

**Nothing here should be bundled.** Item 1 is a bug fix; items 2 and 3 change the feature
contract; item 4 changes failure behaviour; item 5 adds tests and moves the suite-size ratchet.

---

## 7. CONFIRMED BY A SECOND, CONNECTOR-FAITHFUL MEASUREMENT (2026-07-19)

Everything above was measured by `probe_lovd_classification_map_2026-07-19.py`, written BEFORE
the connector source was read. `src/genomic_variant_classifier/data/lovd.py` then showed two
differences between that probe and production:

| | connector | first probe |
|---|---|---|
| normalisation | `.str.lower().str.strip()` before the map lookup | raw strings |
| deduplication | `groupby(keys).max()` **before** joining | merged against the raw artifact |

Either could have moved a figure. The deduplication was the serious one: if any variant carried
more than one submitter row, the probe's "5,576" would have counted JOIN ROWS while the
connector annotates DISTINCT VARIANTS.

`probe_lovd_dedup_2026-07-19.py` remeasured using the connector's exact method. Evidence:
`docs/measurements/LOVD_DEDUP_2026-07-19.txt`.

```
normalisation : 1,481 nonzero WITHOUT lower/strip, 1,481 WITH, 0 rows disagree
duplicates    : 18,006 rows -> 18,006 DISTINCT keys, 0 keys with >1 submitter
left join     : 1,700,687 rows in, 1,700,687 out -- row count preserved
result        : 5,576 matched / 369 nonzero -- IDENTICAL to the first probe
```

**Every figure in sections 2 and 3 stands, now confirmed by the connector's own method.**

Two notes worth keeping:

**Normalisation is irrelevant to THIS artifact, not in general.** The four strings that map are
already lowercase and the camelCase values fail to match either way. An artifact with mixed
casing would diverge, and the earlier probe would have under-counted.

**The `groupby(...).max()` is a no-op here but is not decoration.** `annotate_dataframe` assigns
via `merged["lovd_variant_class"].fillna(...).astype(int).values`. A lookup carrying duplicate
keys would make the left join return MORE rows than the input frame, and `.values` would then be
the wrong length -- a silent misalignment or a raised exception. The row-count assertion above
is that invariant checked rather than assumed.

### 7A. NEW FINDING -- the ordinal has a hole where uncertainty belongs

The deduplicated lookup's distribution, measured across all 18,006 variants:

| ordinal | variants | meaning |
|---:|---:|---|
| 0 | 16,525 | reads as NOT IN LOVD |
| 1 | 313 | benign (105) + likely benign (208) |
| **2** | **0** | **variant of uncertain significance -- EMPTY** |
| 3 | 191 | likely pathogenic |
| 4 | 977 | pathogenic |

**Ordinal 2 is empty across the entire artifact.** Five of the eleven map keys target that tier
-- `uncertain significance`, `variant of uncertain significance`, `vus`, `conflicting`,
`conflicting interpretations of pathogenicity` -- and none of them occurs in LOVD even once.
LOVD expresses uncertainty as `notClassified` (7,435) and `unknown` (77), which the map does not
cover, so those variants land on 0.

This undercuts the encoding's stated rationale. The connector's own docstring says:

> This ordinal encoding preserves the clinical severity ranking and lets tree models exploit the
> natural ordering.

The realised encoding is `{0, 1, 3, 4}`. A tree splitting at 2 divides `benign` from
`likely pathogenic` on a boundary no data occupies, and the 7,515 variants that ARE uncertain
sit at 0 beside "not in LOVD" -- so the one tier the ordering was built to place in the middle
is the one tier that is absent, while its members are filed at the bottom as though missing.

This is a consequence of the vocabulary mismatch in section 2A, not a separate defect. It is
recorded because any remedy that adds the missing strings must decide where `notClassified`
belongs, and "2" is the answer the existing scale implies.

### 7B. Still unmeasured

Coverage on the current cohort, `data/processed/clinvar_grch38_clean_seq.parquet`
(4,399,089 rows). Every figure in this incident measures `models/v1/clinvar_enriched.parquet`
(1,700,687 rows), the Run 9-era cohort the original incident used.
