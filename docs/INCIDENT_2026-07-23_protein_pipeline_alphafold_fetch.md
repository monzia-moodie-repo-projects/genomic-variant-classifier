# Incident -- 2026-07-23 -- the AlphaFold structure fetch in the production protein pipeline

**Revision 2, 2026-07-23.** Revision 1 was written before the second endpoint probe
returned. Revision 2 adds the measured facts from that probe, corrects one statistical
claim, and adds the finding that changes this incident's character: **this is a
recurrence of a defect this project already diagnosed and fixed on 2026-07-02, in a
second copy of the same logic that the 2026-07-02 remediation never searched for.**

**Status:** OPEN. Root cause established. Two of five defects are proven from source
alone, one is proven by measurement, two require measurement before severity can be
stated. No code has been changed.

**Repository state:** `main` at `715bcfa1bdd8320718c7fa9135834ce1c66d9a59`,
"feat(evaluation): typed contracts for Panel S0 expert identity", 2026-07-23T09:57:33-04:00.

**Affected module:** `src/genomic_variant_classifier/pipelines/protein_pipeline.py`,
functions `_fetch_alphafold_cif` (lines 149-182) and `_fetch_active_sites` (lines 261-300).

---

## 1. How this was found

It was not found by looking for it. On 2026-07-23 the Item 2 data-source-freshness
investigation was tracing why the AlphaFold freshness probe returned HTTP 404. The
registry row at `src/genomic_variant_classifier/monitoring/registry.py:138` declares the
source version as the string `"v4"`. Checking whether that string was still true led to
`docs/ALPHAFOLD_SESSION_2026-07-02.md`, and from there into the production pipeline.

The freshness defect and the pipeline defect are unrelated in mechanism. They are related
only in that a stale version string in a monitoring registry was the visible end of a
thread whose other end was in production code.

---

## 2. Measured facts, 2026-07-23T17:50:09Z

From `probe_freshness_endpoints_v2_2026-07-23.py`, two accessions, live against the
European Bioinformatics Institute prediction application programming interface:

| Observation | P38398 (BRCA1_HUMAN) | P04637 (P53_HUMAN) |
| --- | --- | --- |
| `GET` status | 200 OK | 200 OK |
| response bytes | 31,457 | 18,911 |
| container | list | list |
| **records returned** | **8** | **9** |
| top-level keys per record | 44 | 44 |
| `latestVersion` | 6 | 6 |
| `modelCreatedDate` | 2025-08-01T00:00:00Z | 2025-08-01T00:00:00Z |
| `cifUrl` | `.../AF-P38398-F1-model_v6.cif` | `.../AF-P04637-F1-model_v6.cif` |
| model version served | **model_v6** | **model_v6** |
| `HEAD` status | 405 Method Not Allowed | not requested |

The keys include `uniprotSequence`, `sequenceChecksum`, `entryId`, `modelEntityId`,
`allVersions` and `isReferenceProteome`, so every field the correct implementation relies
on is present and the canonical-matching strategy is supported by the service.

**The registry's `"v4"` is stale.** AlphaFold DB serves version 6.

---

## 3. This is a recurrence, not a discovery

`docs/ALPHAFOLD_SESSION_2026-07-02.md`, section "Defect 1 -- stale AlphaFold DB download
URL (v4 -> v6)", records that on **2026-07-02** this project:

- observed every structure fetch returning 404 and `verify_alphafold_build.ps1` Stage 2
  aborting with exit code 4, "zero residue features extracted";
- traced it to `build_alphafold_parquet.py` hard-coding `.../AF-{acc}-F1-model_v4.cif`;
- confirmed live that P04637 version 4 returned 404 and version 6 returned 200, and that
  the prediction application programming interface reported `latestVersion=6`;
- fixed it by resolving `cifUrl` from the prediction endpoint and saving under the
  server's filename, via `Install_alphafold_url_fix.ps1`;
- **added four offline fetch-path unit tests, noting explicitly: "previously the fetch
  path had zero coverage -- which is how a stale URL passed 15/15."**

That remediation was correct and complete **for the module it touched**. It did not
search the repository for other copies of the same logic. The second copy, in
`pipelines/protein_pipeline.py`, still hard-codes `model_v4` today, **21 days later**,
and still has zero test coverage -- the very condition the 2026-07-02 note identified as
the reason the defect survived the first time.

The lesson is not "fix line 158." It is that a defect fixed in one of two copies is not
fixed, and that the fix should have ended with a repository-wide search for the pattern.

---

## 4. Blast radius

`_fetch_alphafold_cif` is production code on the feature-engineering path.

- `src/genomic_variant_classifier/data/real_data_prep.py:314` lists it as
  "14. ProteinStructurePipeline -- protein structure features (Phase 6.2)".
- `real_data_prep.py:1070` imports `ProteinStructurePipeline`; line 1085 instantiates it
  with `cache_dir=ac.protein_cache_dir`.
- `protein_pipeline.py:502`, inside `ProteinStructurePipeline._get_structure`, is the sole
  call site.

Downstream: the four Phase D structural features `alphafold_plddt`,
`solvent_accessibility`, `secondary_structure_context`, `dist_to_active_site`.

---

## 5. Defect 1 -- the cache can never be invalidated by an upstream release

**Severity: HIGH.** Revision 1 rated this MEDIUM and framed it as a provenance mislabel.
That was an under-statement. The mislabel is a symptom; the un-invalidatable cache is the
defect.

`protein_pipeline.py:158-160`:

```python
cache_file = cache_dir / f"AF-{accession}-F1-model_v4.cif"   # fixed literal
if cache_file.exists():
    return cache_file.read_text(encoding="utf-8", errors="replace")
```

The filename is a fixed literal, so the existence check at line 159 always looks for the
same name it wrote. **No upstream release can ever invalidate this cache.** Contrast
`scripts/build_alphafold_parquet.py`, which names files from the server's `cifUrl`: a new
release yields a new filename, the existence check misses, and a fresh download happens
automatically. Correct invalidation there is a free consequence of correct naming; here,
incorrect naming removes invalidation entirely.

Two distinct machine states follow, and they fail differently:

**State A -- warm cache predating version 6.** A file written before the version-6 release
is found forever. The pipeline serves **genuine version 4 coordinates indefinitely**, never
contacting the service, on a database that has since moved to version 6. Silent staleness.

**State B -- cold cache today.** Lines 163-178 call the prediction endpoint, read `cifUrl`,
download **version 6 bytes**, and write them to the hard-coded **version 4** filename:

```python
url = ALPHAFOLD_API.format(accession=accession)      # line 163
resp = requests.get(url, timeout=_REQUEST_TIMEOUT)   # line 164
data = resp.json()                                   # line 168
cif_url = data[0].get("cifUrl", "")                  # line 171
cif_resp = requests.get(cif_url, timeout=30)         # line 174
content = cif_resp.text                              # line 177
cache_file.write_text(content, encoding="utf-8")     # line 178
```

The download itself succeeds -- the URL comes from the service, so this path does **not**
404 the way `build_alphafold_parquet.py` did before 2026-07-02. Only the name is wrong.
Any manifest, audit or provenance record that reads the filename records a model version
the bytes do not have.

Neither state raises. Neither logs at a visible level. Both look normal.

### The repository holds two contradictory policies

| Module | Policy | Tested |
| --- | --- | --- |
| `scripts/build_alphafold_parquet.py::_resolve_cif_url` | filename from the server's `cifUrl` | yes -- `tests/unit/test_alphafold.py:329-343` |
| `pipelines/protein_pipeline.py::_fetch_alphafold_cif` | hard-coded `model_v4` | **no** |

`tests/unit/test_alphafold.py:342` states the correct policy in an assertion message:
`"must save under server version, not v4"`.

---

## 6. Defect 2 -- blind isoform selection, now known to be a one-in-eight choice

**Severity: PENDING MEASUREMENT. Potentially HIGH -- scientific correctness, not hygiene.**

`protein_pipeline.py:171` takes `data[0]` unconditionally. Revision 1 called this a risk.
The version-2 probe measured how large a choice that is: **the service returned 8 records
for P38398 and 9 for P04637.** `data[0]` is one of eight, and one of nine.

`scripts/build_alphafold_parquet.py::_resolve_cif_url` explains why it matters, in this
repository's own words:

> Select the record whose UniProt sequence EXACTLY equals our canonical index sequence.
> AlphaFold DB returns one record per isoform (entryId AF-{acc}-{N}-F1) whose residue
> numbering follows THAT isoform; attaching an isoform structure to canonical
> `protein_pos` would silently mis-number features. If no record matches the canonical
> sequence (giants over the AFDB length ceiling, isoform-only entries), return None so the
> gene is a documented coverage miss -- never a partial/isoform substitute. Verified on
> AARS1/ABCB1 (base) and DYST/SYNE1 (None).

If `data[0]` is an isoform rather than the canonical entry, then `alphafold_plddt`,
`solvent_accessibility`, `secondary_structure_context` and `dist_to_active_site` are read
at residue indices belonging to a different numbering. The values are wrong and look
entirely normal.

### What the two observations do and do not establish

In both probed accessions `data[0]` carried `modelEntityId` `AF-P38398-F1` and
`AF-P04637-F1` -- the canonical `-F1` form, not an `-N-F1` isoform form. So in these two
cases `data[0]` was canonical.

**That is not evidence of anything.** Revision 1 proposed the rule-of-three bound; at
n = 2 the rule of three yields 3/2 = 1.5, which is not a probability, and quoting it would
have been the same class of error as the corrected over-claim recorded in the 2026-07-23
teardown-abort incident. The correct statistic for zero events in n trials is the exact
one-sided binomial bound:

    1 - 0.05**(1/2) = 0.7764

**With zero mismatches in two accessions, the mismatch rate is bounded above by 77.6
percent.** The observation constrains nothing. The rule of three becomes usable only at
large n; at n = 30 it gives 10.0 percent, at n = 300 it gives 1.0 percent.

Section B of `audit_alphafold_fetch_2026-07-23.py` performs this measurement against the
cohort's own UniProt index and reports the count with its denominator.

---

## 7. Defect 3 -- fetch failures are logged at DEBUG and are therefore invisible

**Severity: MEDIUM. It is why defects 1 and 2 could run in production unnoticed.**

Every failure path in the AlphaFold and UniProt fetch functions logs at `DEBUG`:

| Line | Condition | Level |
| --- | --- | --- |
| 166 | prediction endpoint returned a non-2xx status | `logger.debug` |
| 181 | any exception during the fetch | `logger.debug` |
| 299 | UniProt feature fetch failed | `logger.debug` |
| 536 | structure fetch failed for a gene | `logger.debug` |

`DEBUG` is below the default threshold, so in a normal run these produce no output at all.
A pipeline that silently fails to fetch any structure emits identical logs to one that
fetches every structure successfully; the only visible difference is the resulting feature
values, which default to sentinels. This directly violates the project's standing
principle that nothing fails silently. It is also the mechanism by which the 2026-07-02
"zero residue features extracted" condition could arise with no error visible until a
gate caught it downstream.

---

## 8. Defect 4 -- `_fetch_active_sites` has the same cache problem and a bare swallow

**Severity: MEDIUM.**

`protein_pipeline.py:270-276`:

```python
cache_file = cache_dir / f"uniprot_features_{accession}.json"
if cache_file.exists():
    try:
        data = json.loads(cache_file.read_text())
        return data.get("active_sites", [])
    except Exception:
        pass
```

Two problems. First, the same never-invalidated cache: once written, the file is returned
forever with no freshness check, and UniProt feature annotations are revised over time.
`dist_to_active_site` is computed from this. Second, `except Exception: pass` swallows a
corrupt cache with no log at any level. The fall-through to a network fetch is a reasonable
recovery, but the silence is not: a systematically corrupt cache would be indistinguishable
from a warm one.

---

## 9. Defect 5 -- the function has no test coverage

**Severity: MEDIUM. It is the reason all of the above survived.**

`_fetch_alphafold_cif` appears exactly twice in the repository: its definition at line 149
and its call site at line 502. No test file references it. The 2026-07-02 session note
already identified zero fetch-path coverage as the root reason a stale URL survived
fifteen green tests, added four tests to close it in `build_alphafold_parquet.py`, and did
not extend the same reasoning to the second copy.

---

## 10. Remediation direction -- ground up, not patchwork

Changing `model_v4` to a variable on line 158 would leave two divergent copies of the same
logic, one of which would drift again, and would fix none of defects 2 through 5.

1. **Promote one implementation.** Move the correct resolution logic out of
   `scripts/build_alphafold_parquet.py` into a single library function, so scripts and
   pipelines share one code path. It must return both the resolved `cifUrl` and the
   server-derived filename, and must match on canonical sequence with `None` on no match.
2. **Rewire `_fetch_alphafold_cif`** to call it, inheriting server-version naming, correct
   cache invalidation, and canonical selection together.
3. **Move the existing assertions onto the shared function** so both callers are covered by
   `tests/unit/test_alphafold.py:329-343` rather than only one.
4. **Raise the failure logs** from `DEBUG` to `WARNING` at lines 166, 181, 299 and 536, and
   replace the bare `except Exception: pass` at line 275 with a logged branch.
5. **Add a syntax-tree guard test** forbidding any hard-coded `model_v[0-9]+` string literal
   outside `tests/fixtures/`, modelled on `tests/unit/test_rnaseq_ablation_native_read.py`,
   which walks the tree rather than grepping. This pins the class, not the instance.
6. **Add a guard test asserting that the AlphaFold prediction response is treated as a list
   and that record selection is by canonical-sequence match**, so `data[0]` cannot return.
7. **Decide what happens to already-cached `*-model_v4.cif` files.** Nothing is deleted.
   The options are to leave them with the ambiguity recorded, or to re-derive each file's
   true version from its own contents and rename with a recorded mapping. This is Monzia's
   decision, informed by section A of the audit, not an implementation detail.
8. **Correct the registry's `"v4"` to the measured `latestVersion`**, and make the AlphaFold
   freshness fingerprint `latestVersion` plus `modelCreatedDate` rather than a hash of the
   response body. Both are semantically meaningful and both were measured on 2026-07-23.

Steps 3, 5 and 6 add tests and each needs its own suite-size ratchet accounting. Step 8
belongs to the separate registry rebuild and should not be smuggled into this fix.

---

## 11. Open questions

| # | Question | Status | How it is answered |
| --- | --- | --- | --- |
| 1 | Which model version does AlphaFold DB serve? | **ANSWERED 2026-07-23**: `model_v6`, `latestVersion = 6`, `modelCreatedDate = 2025-08-01T00:00:00Z`, consistent across two accessions. | probe version 2 |
| 2 | How often is `data[0]` not the canonical record? | **OPEN.** Bounded above by 77.6 percent, which is no bound. | `audit_alphafold_fetch_2026-07-23.py` section B |
| 3 | How many cached `*-model_v4.cif` files exist, and what are their true versions? | **OPEN.** | section A |
| 4 | Which cached files were written after the version-6 release, and are therefore version 6 bytes under a version 4 name? | **OPEN.** | section C |
| 5 | Were any shipped Phase D feature values produced through this path, on which machine, with which cache state? | **OPEN.** Requires the run-artifact trace, not yet designed. | to be scoped after sections A-C |

Question 5 determines whether this is a forward-looking correctness fix or also a
retrospective data-provenance correction. It must not be answered by assumption.

---

## 12. What this incident says about process

The 2026-07-02 remediation was well executed and well documented, and it still left the
defect live in production for 21 days. The gap was not diligence. The gap was that the fix
ended at the module boundary instead of at a repository-wide search for the pattern, and
that the guard added was a set of behavioural tests on one function rather than a
structural guard on the pattern.

The same shape appears elsewhere in this repository on the same date: the 2026-07-23
teardown-abort fix corrected one `pandas.read_parquet` call site out of 327 and explicitly
declined to generalise without measurement -- the right call there, because exposure was
unmeasured and the blast radius was 326 further sites. The distinction worth recording is
that a *behavioural* fix should be scoped by measured exposure, while a *pattern* fix
should be scoped by a repository-wide search. Hard-coding a version literal is a pattern.
