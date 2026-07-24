# Session 2026-07-24 -- AlphaFold structural coverage quantified; LOVD acquisition planned

**Repository state throughout:** `main` at `715bcfa1bdd8320718c7fa9135834ce1c66d9a59`,
"feat(evaluation): typed contracts for Panel S0 expert identity", 2026-07-23T09:57:33-04:00.
**No commit was made during the session.** Every state check was performed against a fresh
full clone of the remote, never against recollection, and `git rev-list --count 715bcfa..HEAD`
returned 0 at the start of every turn.

**Acronyms on first use.** LOVD = Leiden Open Variation Database. VUS = variant of uncertain
significance. ACMG = American College of Medical Genetics and Genomics. GRCh37 / GRCh38 =
Genome Reference Consortium human build 37 / 38. UTC = Coordinated Universal Time.

---

## 1. What this session set out to do, and what it actually found

It began as item 3 of a handoff queue: push a prepared commit, then audit 328
`pandas.read_parquet` call sites, then investigate two unreachable data sources.

The call-site audit and the commit closed cleanly. The data-source investigation did not
stop where it was pointed. Following a stale version string in a monitoring registry led
into the production feature pipeline, and from there to a quantified defect affecting
**12.725 percent of the variant cohort**, of which **5.016 percent receives structural
feature values that are wrong rather than absent**.

---

## 2. Completed and pushed

**S0 Commit 2** -- `715bcfa`, 2026-07-23T09:57:33-04:00. Four files, +576 / -2. Suite-size
ratchet 2874 -> 2893; local run 2886 passed, 7 skipped in 716.30s; Continuous Integration
run #591 green on Python 3.11 and 3.12, 13m 37s.

The `Push image to GHCR` job showed a grey slashed icon and is **correctly skipped**, not
failed: `.github/workflows/ci.yml:558` gates it on `github.event_name == 'release'` and this
was a push event. Recorded so it is not re-raised.

---

## 3. Numeric errors found in inherited documents

Each was verified in a tool call, not read.

| Claim | Source | Actual |
| --- | --- | --- |
| "seven skips, four in test_mc_dropout_calibration.py" | handoff section 1 | **five** skipped tests from **four** class-level decorators; `TestUncertaintyErrorCorrelation` holds two methods |
| "328 pandas.read_parquet call sites" | handoff section 4 | **327** pre-fix, **326** post-fix. No counting rule at any commit yields 328 |
| "19 local assets missing" | handoff section 5 | **18** |
| "4 sources present, no local_path" then lists 6 | handoff section 5 | **6** |

The ratchet ledger itself was audited: **53 entries, zero arithmetic mismatches**, total
growth 1870 -> 2893 = 1023 exactly equal to the sum of stated deltas. It is clean, and it is
non-monotonic by design (entries at 1962 and 1967 record -4 and -1), which is correct for a
two-sided equality gate.

---

## 4. Item 1 -- the parquet call-site audit

Steps 1 and 2 of the five-step protocol are complete. 600 Python files scanned by
abstract-syntax-tree walk, zero parse failures, so the count is a total and not a bound.

| Bucket | Files | Sites | Share |
| --- | ---: | ---: | ---: |
| `A_script_entrypoint` | 153 | 243 | 74.5 % |
| `C_library_caller_dependent` | 32 | 49 | 15.0 % |
| `B_service_long_running` | 13 | 16 | 4.9 % |
| `A2_script_no_main_guard` | 7 | **15** | 4.6 % |
| `A3_src_module_with_main` | 2 | 3 | 0.9 % |
| **TOTAL** | **207** | **326** | |

The A2 bucket was not anticipated by the handoff: **15 reads execute at module import time**,
the shortest possible gap between reading a parquet and interpreter shutdown, which is the
condition the teardown-abort incident identified as dominant. Those are the first arms to add
to the diagnostic, ahead of the bulk 243.

Artifact: `docs/audits/evidence/2026-07-23/READ_PARQUET_SITES_2026-07-23.csv`.

Rates carried forward, each with its denominator: pandas read with immediate exit 27/5000 =
0.540 %; native `pq.read_table` 0/5000, exact 95 percent upper bound 3/5000 = 0.060 %;
`ARROW_IO_THREADS=1` 22/5000 = 0.440 %, so thread constraint does not suppress; real script
1/5000 = 0.020 %. Total child executions across both diagnostic rounds: 115,000.

---

## 5. Item 2 -- both unreachable sources root-caused

**AlphaFold, HTTP 404.** A compound defect entirely in this repository, neither cause
upstream. `monitoring/registry.py:138` stores `https://alphafold.ebi.ac.uk/api/prediction/`
-- a URL template with its required accession parameter removed -- and declares
`Check.HTTP_ETAG`, which `database_freshness_detector.py:65` implements as a HEAD request.
Measured 2026-07-23T17:50:09Z: parameterless GET and HEAD both 404; parameterised **HEAD 405
Method Not Allowed**; parameterised **GET 200**, 31,457 bytes. Correcting only the URL moves
the failure from 404 to 405. Both must be corrected together.

**LOVD, HTTP 403.** Not drift and not an outage. `scripts/build_lovd_index.py` lines 11-26
have documented since March 2026 that the database enforces a browser-validation anti-bot
challenge on all endpoints including the public interface. Measured 2026-07-23T17:50:09Z: all
four forms 403, including the landing page. The older `data_freshness_agent.py:306-313`
already handles 401/403 as an authentication skip and appends the gene parameter; the newer
registry-driven detector does neither. **This is a regression across a subsystem rewrite.**

The failure mode also changed between observations -- HTTP 400 on 2026-07-20, HTTP 403 on
2026-07-23 -- and nothing recorded it, because the detector persists `last_seen` only on
success. A change in how a source fails is itself drift and is currently invisible.

Further registry defects found: `gnomad` and `gnomad_constraint` share one probe URL, so
constraint-table drift is structurally undetectable; `agent_layer/config.py` is a **second,
divergent store of upstream URLs** that `data_freshness_agent.py:46` reads instead of the
registry, contradicting the registry's claim to be the single source of truth; and only
**7 of 24 sources (29.2 %) have any automated probe at all**, with 17 marked `manual_skip`,
which the report's "24 sources scanned" headline conceals.

---

## 6. The principal finding -- AlphaFold structural coverage

Full write-up: `docs/audits/AUDIT_2026-07-24_alphafold_structural_coverage.md`.
Defect record: `docs/INCIDENT_2026-07-23_protein_pipeline_alphafold_fetch.md` (revision 2).

**The ceiling, located rather than assumed.** Complete census of all 296 index accessions at
or above 2,400 residues, 2026-07-24T03:31:29Z. Longest with a canonical model **2,699**
(TENM3); shortest without **2,701** (ITPR2, CENPE). The window contains one integer, 2,700,
and the index holds no protein of that length, so the ceiling is 2,699 or 2,700 and nothing
else is consistent.

**The split is total.** All 81 accessions at or below 2,699 returned `CANONICAL_PRESENT`. All
215 at or above 2,701 did not: 102 isoform-only, 109 no model at all, 4 sequence drift. That
215 agrees exactly with the independent count of index accessions above 2,700 residues.

**The mechanism.** `pipelines/protein_pipeline.py:171` takes `data[0]` unconditionally. For
UBR4 (Q5T4S7, 5,183 residues) `data[0]` is `AF-Q5T4S7-6-F1`, a **212-residue** model sharing
**one residue** with the canonical sequence. Positions 2 through 212 receive predicted Local
Distance Difference Test scores, solvent accessibility and secondary-structure context from a
different sequence, and they look entirely valid.

**The impact, cross-checked across ten cohort files with a spread of 0.043 percentage points:**

| Group | Variants | Share of 4,399,089 |
| --- | ---: | ---: |
| wrong data, isoform substituted | 220,590 | 5.014 % |
| missing data, no model | 339,141 | 7.709 % |
| sequence drift | 55 | 0.001 % |
| **combined** | **559,786** | **12.725 %** |
| gene-weighted, for contrast | 215 / 20,190 | 1.065 % |

**The variant-weighted figure is 11.95 times the gene-weighted one.** On the review-tier <= 3
subset that Runs 15 and later actually train on -- 3,716,674 rows -- it is **12.544 percent**,
essentially unchanged.

Most affected: NF1 16,807, DMD 10,058, USH2A 9,377 on wrong data; TTN 39,316, BRCA2 21,166,
ATM 19,121, APC 16,675 on missing data. Three of the four most consulted cancer-predisposition
genes in the corpus have no AlphaFold structure at all through this path.

**Four further defects in the same function**, all recorded in the incident note: a hard-coded
`model_v4` cache filename that makes the cache un-invalidatable by any upstream release;
failure logging at `DEBUG` at lines 166, 181, 299 and 536, below the default threshold; a bare
`except Exception: pass` at lines 275-276; and zero test coverage. The 2026-07-02 session
fixed the identical version-hard-coding defect in `scripts/build_alphafold_parquet.py`, added
four tests, and recorded that zero fetch-path coverage was why it had survived -- and did not
search the repository for the second copy. **21 days.**

**Sequence drift is a separate defect.** Three cases characterised at 2026-07-24T03:35:17Z:
ARL16 (173 residues, AlphaFold sequence version 2006-09-19), APOLTP (3,320, 2004-07-05),
MUC3B (13,477, 2023-09-13), against a local index built 2026-06-25. **AlphaFold's snapshot is
the older one**, not ours -- which refutes the causal text my own census printed, and means
rebuilding the local index would not fix it.

---

## 7. Cohort file hygiene -- a finding that arrived sideways

Thirteen cohort parquet files inventoried. **Nine are measurably NOT CLEAN**, including two
whose names say otherwise:

| File | Rows | Null or empty alleles | Duplicate identifiers |
| --- | ---: | ---: | ---: |
| `clinvar_grch38_clean_v2_verified.parquet` | 4,420,180 | **21,091** | 4,217 |
| `clinvar_grch38_clean_v3_verified.parquet` | 4,400,192 | **1,103** | 0 |
| `clinvar_grch38_noalleles.parquet` | 4,420,180 | **4,420,177** | 513,428 |

4,420,180 - 4,399,089 = 21,091, exactly the quarantine count `docs/CHANGELOG.md` records for
the 2026-05-31 null-key leak remediation. Two files named "clean ... verified" would raise at
`real_data_prep.py:476` if used as training cohorts. The four `_structural` files are
correctly not clean -- they are the quarantine.

Measurably CLEAN: `clinvar_grch38_clean.parquet`, `clinvar_grch38_clean_seq.parquet`,
`cohort_fresh.parquet`, `cohort_stale.parquet`.

---

## 8. Line-ending pin failure -- 277 files

`git ls-files --eol` against the working tree: 234 files pinned `text eol=lf` hold carriage
returns, and **42 more hold BOTH styles in the same file**, plus `scripts/Run16_Monitor.ps1`
which is `mixed` under an `eol=crlf` pin. **277 violations.** A fresh Linux clone has zero.

Cause, dated: `.gitattributes` has been pinned progressively in six commits -- `794cc72`
2026-03-30, `ee2fa72` 2026-05-11, `6091988` 2026-06-17 (the large one), `c53d61d` 2026-06-17,
`0849da3` 2026-07-12, `9362f2c` 2026-07-18 -- and **none renormalised what was already on
disk**. Git applies the filter on checkout and on comparison, never retroactively, so
`git status` stays clean and the corruption is invisible by construction.

Functional impact checked rather than assumed: every `hashlib` call in `src/` hashes
in-memory data, not tracked text files, so there is **no cross-platform file-hash hazard**.
The harm is that a byte-stability guarantee the repository documents is not in force.

Also found: `scripts/discover_recount3_projects.R` and `scripts/install_recount3_deps.R` are
`mixed` under **no pin at all** -- `.R` has no `.gitattributes` rule. A gap, not a violation.

The remedy is `git rm --cached -r .` followed by `git reset --hard`, which is irreversible for
uncommitted tracked changes. **Not executed. Awaiting decision.**

---

## 9. LOVD acquisition planning

Plan: `docs/LOVD_ACQUISITION_PLAN_rev2_2026-07-24.md`. Revision 1 superseded to
`docs/superseded/`.

**Access policy governs everything.** Two internet-protocol-address bans in 2026, both from an
automated client looping the per-gene endpoint; administrator correspondence of 2026-03-30
records roughly 34 gigabytes of discarded egress from about a thousand calls, and that the
human-interface paths are server-rendered. Every tool built this session prints that it
performs no network access, and does not.

**Held, read from the parquet:** 18,006 rows, 10 genes -- APC, BRCA1, BRCA2, MLH1, MSH2, MSH6,
NF1, PTEN, RB1, TP53. `scripts/build_lovd_index.py:502` declares a default naming **ATM** and
**LDLR**, neither held, and omitting **NF1** and **RB1**, both held. Configuration drift.

**Four selection methods failed before one worked**, and all four are recorded in the plan so
nobody repeats them: single-axis ranking buried LDLR; two-axis ranking produced **disjoint**
Tier 2 lists with CHEK2 in neither; the Pareto frontier was **size 1** in Tier 1 and admitted
MUC17 and CCDC168 with **zero deletions**; floors at the median held gene qualified **2 of
209 and 0 of 21,167**, because the ten held genes sit near the 99.97th percentile.

**What worked: cumulative coverage.** Genes partition the cohort, so coverage is a plain sum.
Held 10 cover 106,561 of 4,399,089 variants = 2.422 percent. Doubling deletions takes 15
genes against 6 for uncertain variants -- deletions are the scarcest resource and the flattest
curve, which follows directly from the 2026-07-08 incident measuring the training cohort at
0.0521 percent deletions.

**Batch 1, eleven genes:** TTN, ATM, TSC2, PKD1, FBN1, NEB, PALB2, DMD, USH2A, CHEK2, LDLR.
Effect: uncertain 49,099 -> 106,978 (2.18x), variants 106,561 -> 240,737 (2.26x), deletions
15,516 -> 28,753 (1.85x), cohort coverage 2.422 -> 5.472 percent.

**Three genes reconciled and closed:** MUC3B, SSPOP and APOLTP appear in the census lost set
but have **zero ClinVar rows**. Not a gene-symbol join gap; they simply carry no clinical
submissions.

---

## 10. Errors I made this session, and how each was caught

Recorded because the pattern matters more than any single instance.

| Error | How it surfaced | Correction |
| --- | --- | --- |
| Endpoint probe v1 compared HEAD-vs-HEAD for AlphaFold and GET-vs-GET for LOVD, discarding a decisive 200 response | printed UNRESOLVED against its own captured evidence | v2 pairs methods consistently and models compound defects |
| Claimed the two random samples were independent | verified with `random.sample` | they are **nested**; the 60-draw is a strict prefix of the 300-draw and they must not be pooled |
| Sampled 50 accessions by sorted accession | all 50 returned one record; 45 shared the prefix A0A075B6 | zero informative trials; replaced with seeded random sampling and a power gate |
| Quoted the rule of three at n = 2 | 3/2 = 1.5 is not a probability | exact binomial bound 1 - 0.05^(1/n) throughout |
| Cross-check searched for `review_status`, top level only | every cohort reported "tier column: none present" and the script still **exited 0** | the column is `ReviewStatus`, and the nested field is `metadata.review_status`; the inventory walks nested fields and exits 1 on any unmeasured request |
| Ranking report truncated to `tier1[:200]`, `tier2[:200]` | a query for PCSK9 returned nothing, indistinguishable from absence | v2 stores all 21,386 genes and reports every queried gene as FOUND or NOT FOUND |
| Ranked on one axis, then two, then a frontier, then floors | each failed differently | replaced with a coverage curve that shows the trade-off instead of resolving it |
| Delivered a document containing an em dash and an ellipsis | own ASCII check at delivery time | fixed; the ellipsis was inside a truncated hash, now written in full |

---

## 11. Methods used, for reproducibility

**Exact binomial (Clopper-Pearson) intervals** via `scipy.stats.beta.ppf`, used wherever a
rate is quoted from a sample. For zero events the one-sided upper bound is 1 - alpha^(1/n);
the rule of three (3/n) is reported **only at n >= 30** and explicitly withheld below that.

**Power gating.** A measurement is reported as UNDERPOWERED, with no bound quoted, when fewer
than three informative trials were collected -- mirroring the evidence consolidator's rule
that suppression is not called when fewer than three events are expected.

**Census over sampling.** Where a population is small enough to enumerate (296 accessions at
or above 2,400 residues), it was enumerated. There is no sampling error in section 6.

**Conditioning on informative trials.** The isoform defect can only manifest when a response
carries more than one record, so its rate is reported over that subset with the denominator
stated, alongside the unconditional rate clearly labelled as not the defect rate.

---

## 12. Open items carried forward, in severity order

1. **`docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md`** -- CRITICAL, OPEN,
   affects Runs 15 through 17. The review-status join fails on 98.834 percent of deletions;
   retention is label-correlated (34.556 percent of pathogenic survive against 95.236 percent
   of likely-benign). Remedy already identified in its own section 2.7: `metadata.review_status`
   agrees with `ReviewStatus` on 3,974,573 of 3,974,573 populated rows with zero disagreements
   and rescues 178,563 deletions. **This leads everything on severity, and it is upstream of
   the deletion scarcity driving the entire LOVD case.**
2. **AlphaFold shared-resolution commit** -- fully specified in the incident note, ready to
   build: one shared resolution function, canonical-sequence matching, `None` on no match, the
   four DEBUG logs raised, the bare swallow replaced, and two guard tests.
3. **Cohort file hygiene** -- nine not-clean files, two dangerously named.
4. **Line-ending renormalisation** -- 277 files; remedy involves `git reset --hard`.
5. **`docs/CHANGELOG.md` defects** -- 25 days stale, 263 double-encoded sequences across 217
   lines, and five entries sharing one header of which two are byte-identical. See section 13.
6. **Registry rebuild** -- probe parameters, method compatibility, LOVD to `Check.MANUAL`,
   `gnomad_constraint` its own fingerprint, a fourth local status, failure-status persistence,
   and the duplicate URL store in `agent_layer/config.py` collapsed.
7. **Orphan installers** `install_ratchet_bump_2725_2026-07-21.py` and `..._2741_...` --
   suite sizes that never entered the ledger.
8. **Item 1 step 3** -- the diagnostic measurement campaign, awaiting a decision on which
   process shapes get arms.
9. **`docs/METRICS.md` audit** -- last touched 2026-06-13, 41 days, not audited since the
   R3a/R3b split.

---

## 13. A defect in the file this session was about to append to

Verified before writing anything into `docs/CHANGELOG.md`:

- **25 days stale.** Newest entry 2026-06-29; last commit `d1039df`, 2026-06-29. Sessions on
  07-02, 07-08, 07-11, 07-12, 07-18, 07-19, 07-20, 07-22 and 07-23 are documented under
  `docs/sessions/` and `docs/incidents/` but absent from the CHANGELOG, against a standing
  convention that it is updated every session.
- **263 double-encoded UTF-8 sequences across 217 lines.** Commit `ec33c5d` on 2026-06-25 is
  titled "docs: fix UTF-8 mojibake in 2026-06-25 changelog append" -- so this was fixed once,
  narrowly, and is still present elsewhere.
- **Five entries share the header `## 2026-06-05 -- Run 15 all-models smoke`**, of which the
  blocks at lines 205 and 251 are **byte-identical** (46 lines each). The other three differ,
  so this is not five copies of one thing.

The entry for this session is therefore delivered as a **guarded prepend** that touches
nothing else and introduces no non-ASCII characters. Repairing the existing 217 lines is a
separate decision and is **not** bundled with a session append.
