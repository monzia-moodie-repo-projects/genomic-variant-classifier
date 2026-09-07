# MEASUREMENT 2026-09-04 -- what the repository's own behaviour says about two undecided axes

**Author: Monzia Moodie**
**Measured at commit:** 70674c1
**Bears on:** `PARTITION-AXIS-UNDECIDED-1`, `PUBLISHER-VERSUS-PROJECT-AXIS-ABSENT-1`,
both registered 2026-09-04 against `ARTIFACT-KEY-INSUFFICIENT-1`.

---

## 0. Why this was measured

`MEASUREMENT_2026-09-04_artifact-key-one-of-three.md` established that
phenomenon A is resolved and that the finding survives on two phenomena:

```
B  partitioned members        AMBIGUOUS -- three EVE partitions as products and
                              the whole score set as one artifact are BOTH accepted
C  project-derived misattributed  INEXPRESSIBLE -- no field distinguishes a
                              publisher's bytes from this project's
```

`INCIDENT_2026-08-28_artifact-identity-and-cache-keys.md` forbids answering
either plausibly: *plausible is what produced the defect being recorded here.*

So neither is answered here. What is measured is **what the repository already
does**, which is evidence a decision can rest on rather than intuition.

---

## 1. Phenomenon B: the estate is CONSUMED as one artifact

```
data/external/eve/EVE_all_data/variant_files   3,211 files  10,654,071,064 B
data/external/eve_smoke                            1 file       16,023,084 B
```

**THE CONNECTOR TAKES ONE PATH.** Read from the parse tree of
`src/genomic_variant_classifier/data/eve.py`, 20,157 bytes, 490 lines:

```
def __init__(self, eve_path, config, entry_map_path)
def annotate_dataframe(self, df)
def fetch(self, variant_df)
def _get_lookup(self)
def _parse_csv_directory(self, directory)
def _parse_single_csv(self, csv_file, entry_map)
def _parse_merged_parquet(self, parquet_path)
def _annotate(self, variant_df, lookup)
```

`_parse_single_csv` is called BY `_parse_csv_directory`; it is not a
caller-facing entry point. `_parse_csv_directory` takes a DIRECTORY.

**NO CONSUMER SELECTS AN INDIVIDUAL PROTEIN.** MEASURED across all tracked
Python files by exact-case token: 34 files carry an EVE token -- 6 production,
10 test, 18 script. Every one passes either the path or the resulting
`eve_score` column. A parse-tree scan for `glob`, `iterdir` or `rglob` beside an
EVE path under `src/` returned NOTHING.

`monitoring/registry.py:95` names the source asset as a SINGLE MERGED PARQUET,
`data/raw/cache/eve_eve_lookup.parquet`, and `_parse_merged_parquet` reads it.
So the 3,211 files and the one parquet are TWO MATERIALISATIONS of one score
set, interchangeable from the caller's view.

**WHAT THIS IS EVIDENCE FOR, AND WHAT IT IS NOT.** It is evidence that minting
3,212 product identifiers would model a distinction NO CONSUMER MAKES. It is
NOT a decision: a schema may legitimately record structure the current code
ignores, and `ESM-2` and `EVE` are both silent-zero pending the HGVSp parser,
so present behaviour may understate intended behaviour.

**A CONTAMINATED CENSUS, CORRECTED.** The first scan used `'eve' in l.lower()`
and matched EVERY, SEVERAL, LEVEL, REVEL, SEVERITY and NEVER -- 246 files,
almost all noise, including `_lazy_agent.py` matching on `annotation`. That is
the same defect as counting 31 "invocations" of `preflight_data_guard` that
were all Markdown prose. The exact-case rerun gave 34.
`PROBE-GLOB-TOO-SHALLOW-1` and `PROBE-PATH-ASSUMED-1` were already registered
on 2026-08-26 for this class, and were not read until after the defect was
repeated.

---

## 2. `EVE-DECLARATION-CARRIES-NO-VERSION-OR-NOTES-1`

```
eve   location EXTERNAL   tier PUBLIC   version ''   notes ''
      acquire 'https://evemodel.org/ (Zenodo)'
```

`gencode` carries a version, a release identifier and roughly 900 characters of
notes recording five artifacts and an open finding. `eve` carries a version of
the empty string and notes of the empty string, while 10.65 gigabytes sit on
disk across 3,211 files.

`eve_smoke`, 16,023,084 bytes, is the LAST REMAINING ORPHAN under
`data/external/` after the grch38 consolidation, and
`tests/unit/test_data_tree_gate.py` names it as one of the three real orphans.

---

## 3. Phenomenon C: twenty-five files, not eighteen, and growing

MEASURED at `70674c1` by a pruned `os.walk` -- 5,446 directories in 0.5
seconds, with `.git`, `.venv312`, `renv`, `.mypy_cache`, `node_modules` and
`__pycache__` excluded BY NAME rather than filtered afterwards:

```
parquet files repository-wide                                630
whose PATH contains clinvar                                   25
the incident measured, 2026-08-28                             18
```

Reproducing the incident's own method -- path substring on `.parquet` --
because a different method would produce a different set and prove nothing
about the finding.

```
data\_drift_check\clinvar_clean_DRIVE.parquet                      142,269,185 B
data\_drift_check\clinvar_clean_REGRESSED_17col_2026-07-08.parquet 141,253,023 B
data\external\dbnsfp\dbnsfp_clinvar_index.parquet                   36,247,405 B
data\processed\_invalidated_2026-07-09\clinvar_grch38_alleleless_q     714,632 B
data\processed\_invalidated_2026-07-09\clinvar_grch38_clean_v3_ver 141,309,456 B
data\processed\clinvar_grch38.parquet                              142,021,534 B
data\processed\clinvar_grch38_clean.parquet                        142,269,185 B
data\processed\clinvar_grch38_clean_seq.parquet                    560,606,657 B
data\processed\clinvar_grch38_clean_v2_verified.parquet            142,025,553 B
data\processed\clinvar_grch38_clean_v3_verified.parquet            141,309,456 B
data\processed\clinvar_grch38_conflicts.parquet                          7,346 B
data\processed\clinvar_grch38_fresh.parquet                        140,815,095 B
data\processed\clinvar_grch38_noalleles.parquet                    117,552,128 B
data\processed\clinvar_grch38_pathfix.parquet                      141,978,247 B
data\processed\clinvar_grch38_structural.parquet                       758,618 B
data\processed\clinvar_smoke.parquet                                 3,543,042 B
data\processed\clinvar_smoke3000.parquet                               148,034 B
data\processed\clinvar_smoke3000_seq.parquet                           452,846 B
data\processed\clinvar_smoke_seq.parquet                               751,471 B
models\smoke_run16\clinvar_enriched.parquet                            302,775 B
models\smoke_run16b\clinvar_enriched.parquet                           302,775 B
models\v1\clinvar_enriched.parquet                                  63,651,595 B
notebooks\genomic_variant_classifier\data\raw\clinvar_BRCA1.parquet     17,764 B
outputs\probe_patch6b\clinvar_5k.parquet                               247,528 B
outputs\run16\clinvar_enriched.parquet                             223,304,719 B
```

Across TEN directories. **Not one is an NCBI publication.** ClinVar publishes
`.vcf.gz` and `.txt.gz`, and those three sit in `data/external/clinvar` and
`data/raw/clinvar`, enumerated separately and matching the declaration's
`acquire` field exactly:

```
data\external\clinvar   clinvar.vcf.gz            192,290,992 B
                        variant_summary.txt.gz    439,962,524 B
data\raw\clinvar        clinvar_GRCh38.vcf.gz     190,311,812 B
```

`data\external\dbnsfp\dbnsfp_clinvar_index.parquet` is the sharpest case: a
**dbNSFP-derived index** that a path-substring census attributes to ClinVar.

**NO DIRECTORY NAMED `primary_release` EXISTS.** The incident's
`ClinVar/primary_release` was a CENSUS CATEGORY, not a path -- it says so:
*they were attributed to ClinVar because the PATH contains the substring, and
called `primary_release` because the name ends in `.parquet`.*

`PROJECT-DERIVED-CLINVAR-PARQUETS-NOW-TWENTY-FIVE-1`. Thirty-nine per cent more
attributable-by-substring artifacts in ten days, and no field in
`SourceArtifactKey` -- exhaustively `source`, `artifact_kind`, `product` -- can
say any of them is this project's rather than NCBI's.

---

## 4. What these decide, and what remains Monzia's

They decide NOTHING about the schema. They establish:

**For B**, that the repository consumes EVE as one artifact in two
materialisations, so a partition axis would model a distinction no consumer
makes -- evidence against minting 3,212 identifiers, not proof that a schema
should not record them.

**For C**, that the misattribution has real and growing volume across ten
directories, including one artifact derived from a DIFFERENT publisher.

Both axes remain undecided, and `INCIDENT_2026-08-28`'s ruling stands: Phase 1C
must not persist a source manifest yet.

---

## 5. What is NOT claimed

That the 3,211 EVE files were digested. They were counted and sized. Whether
any are duplicates is unmeasured -- and the same incident records two EVE files
at exactly 612,501 bytes with DIFFERENT digests, so size proves nothing here.

That the 25 ClinVar-attributed parquets were classified by this measurement.
Their names are printed; the reading is a person's. If any IS a ClinVar
publication, this record is wrong about it.

That `eve_smoke` should be declared, folded or removed. It is the last orphan;
what to do with it is a decision, not a measurement.

That the twenty-five are all still WANTED. Several sit under
`_invalidated_2026-07-09` and `smoke_run16`, which suggests supersession --
suggests, not establishes.

That EVE's connector is LIVE. `INCIDENT_2026-04-17_esm2-hgvsp-parser.md`
records ESM-2 and EVE as silent-zero pending an HGVSp parser, so the consumer
shape measured here may be the shape of code that does not currently produce a
signal.
