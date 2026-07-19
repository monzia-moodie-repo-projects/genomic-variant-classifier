# SESSION 2026-07-19 — repository identity, tag drift, a 72-day stash, and the LOVD map

**Tree at start:** `988c082` · **at end:** `4cfa6a2` · Continuous Integration #528 GREEN.
**Suite:** 1,968 collected / 1,961 passed / 7 skipped, ratchet enforced on every run.
**Evidence:** `docs/measurements/REPO_COMPARISON_2026-07-18.txt`,
`REPO_IDENTITY_2026-07-18.txt`, `LOVD_CLASSMAP_2026-07-19.txt`, `LOVD_DEDUP_2026-07-19.txt`;
`docs/incidents/INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md`.

LOVD = Leiden Open Variation Database.

This session began with a screenshot and produced no feature work. It found one live data
defect, one repository-integrity defect, one piece of unversioned work that had survived
seventy-two days by accident, and a sequence of errors of my own — one of which would have taken
the live project read-only, and one of which briefly degraded it. The errors are recorded in
section 6 at the same level of detail as the findings, because in this session they were the
findings.

---

## 1. WHAT STARTED IT

A screenshot of `github.com/monzia-moodie/genomic-variant-classifier`, filename dated
**2026-02-07**: Private, **5 commits**, a README describing *"an ensemble of 3 models"*,
*"~10K-100K variant dataset (not billions)"*, and *"Scikit-learn, XGBoost, TensorFlow"*.

Against the live project — 13 base models, 95 features, 4,399,089 variants, PyTorch, no
TensorFlow anywhere — that read as a second, stale repository sharing a name. **It was not.**

## 2. ONE REPOSITORY, TWO ADDRESSES

`git clone` of the old address returned figures no five-commit prototype could produce:

```
remote: Enumerating objects: 6444    Receiving objects: 20.23 MiB
HEAD 988c082   905 commits   branches incl. run9a-prep
```

`988c082` was pushed to `origin/main` earlier the same evening.

**`compare_repos_2026-07-18.py`** — content comparison across every reachable commit in both,
by blob SHA-1 so a moved file counts as a rename rather than a loss:

```
OLD distinct blobs 2,653   LIVE 2,654   shared 2,653 of 2,653 (100.0%)   UNIQUE to OLD: 0
OLD HEAD paths 1,216       LIVE 1,216   in OLD only: 0
```

**`verify_repo_identity_2026-07-18.py`** settled it. Ten advertised references on each address,
byte-for-byte identical, `main/HEAD = 988c0823b98c` on both. Then the decisive test — GitHub
resolves a renamed or transferred repository to its current name:

```
via gh command-line interface : unavailable -- gh: Bad credentials (HTTP 401)
via https api.github.com      : monzia-moodie-repo-projects/genomic-variant-classifier
```

**The old address is a permanent redirect to the live repository.** The February page and
today's tree are the same repository before and after a transfer to the organisation account.
There is no second repository, no stale snapshot, and nothing to archive.

Two facts fell out of that output that nobody was looking for:

- **The repository is PUBLIC now.** The *unauthenticated* application-programming-interface
  request succeeded; a private repository returns 404 to an anonymous caller. The February
  screenshot carried a **Private** badge, so visibility changed at some point in between.
- **The GitHub command-line interface token is dead** — `Bad credentials (HTTP 401)`. Nothing
  depended on it here because the fallback answered. `gh auth login` when convenient.

## 3. TAG DRIFT — THE REMOTE WAS WRONG, AND I MADE THE LOCAL COPY WRONG TOO

Chasing a two-commit discrepancy (`rev-list --count --all` reading 907 locally against 905 in a
fresh clone) exposed something unrelated to the count:

| tag | local target | remote target |
|---|---|---|
| `run9a-baseline` | `61c1b2b` — **ancestor of main** | `93be9ed` — **not** an ancestor |
| `v2.0.0` | `c6c02c1` — **ancestor of main** | `0eb40ce` — **not** an ancestor |
| `v4.0.0` | `72d9d13` — **ancestor of main** | `fd5c05b` — **not** an ancestor |

The pairs share **identical author dates to the second** and identical subjects
(`2026-04-30 21:44:43 -0400`, `2026-03-25 23:05:16 -0400`) — the signature of a history rewrite,
which preserves author metadata while re-parenting every commit. Consistent with the transfer in
section 2.

**Content was fully preserved.** All three `tree-identical=True`; `git diff --stat` empty for
both annotated pairs. This was pure re-parenting; nothing was lost.

The defect: the rewrite was force-pushed for `main` but the re-created tags were pushed
**without** `--force`, so GitHub silently kept the pre-rewrite ones. The working copy held the
repaired state; the remote never received it. `git checkout v2.0.0` in a fresh clone landed on a
commit outside `main`'s history.

**Then I made it worse.** On a misreading (section 6.4) I instructed
`git fetch --tags --force --prune-tags`, which overwrote three correct local tags with three
defective remote ones. Recoverable only because a snapshot had been taken first —
`C:\Users\monzi\Downloads\local_tags_before_2026-07-18.txt`. **Annotated tags have no reflog.**

Repair, then verification from a fresh clone rather than from the machine that did the repair:

```
update-ref refs/tags/run9a-baseline 078fb765...   (annotated objects, messages intact)
push --force  a8858c6...078fb76 | b56fba8...c14387a | fd5c05b...72d9d13

fresh clone:  run9a-baseline -> 61c1b2b ancestor-of-main=True
              v2.0.0         -> c6c02c1 ancestor-of-main=True   "v2.0.0 - Phase 6-8 complete"
              v4.0.0         -> 72d9d13 ancestor-of-main=True
              rev-list --count --all = 904
```

The count question closed by arithmetic: **904 references + 3 stash commits = 907**, and the
905 that a clone reported had been counting the three orphans.

## 4. A 72-DAY-OLD STASH HOLDING THE ONLY COPY OF A CITED FILE

```
stash@{0}  7e4b5375  2026-05-08 00:57:46 -0400
           "On main: stash diag_lovd_join.py for migration pre-flight"
           parents: 6075960 (HEAD) + e1c337e (index) + 4c5d1a9 (untracked)
```

`git stash show --stat` returned **empty** — no tracked changes at all. The stash was one
untracked file, which is why it had a third parent.

```
git log --oneline -- "*diag_lovd_join.py"      ->  (empty)  never in any commit
Get-ChildItem -Recurse -Filter diag_lovd_join.py ->  (empty)  not on disk
```

**The stash was the only copy in existence** — invisible to `git status` and to Continuous
Integration, never pushed, and destroyed silently by any re-clone. And
`INCIDENT_2026-05-02_lovd-silent-zero.md:117` cites it **by filename**, quoting its output at
lines 121–124. A 19,684-byte incident record rested on an exhibit that had never been
versioned.

Recovered non-destructively with `git restore --source='refs/stash^3'`, which leaves the stash
intact — the right choice while it remained the sole copy. Filed as `scripts/diag_lovd_join.py`
in `4cfa6a2`.

## 5. THE LOVD FINDING

Full record: `docs/incidents/INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md`.
Summary only here.

Two figures disagreed by a factor of fifteen — **5,553** join matches in the 2026-05-02
incident, **369** coverage in the 2026-06-01 source audit. Both were correct; they measure
different quantities, and the gap between them is a defect.

`_CLASSIFICATION_MAP` holds 11 entries, **every one a ClinVar clinical-significance term**. The
artifact's vocabulary is largely LOVD's functional-effect vocabulary:

```
18,006 rows / 14 distinct classification strings / 18,006 DISTINCT variant keys
   1,481 map to a nonzero ordinal   (8.2%)
  16,525 are silently zeroed        (91.8%)
```

Joined to the Run 9-era cohort (1,700,687 rows): **5,576 matched, 369 nonzero, 5,207
matched-but-zero.** The 369 reconciles the audit figure **exactly**.

`0` now carries four meanings: absent from LOVD; present but unmapped; column absent entirely
(`variant_ensemble.py:803-804` fabricates zeros via `df.get`); and stub mode. Same defect class
as `PLACEHOLDER_BASE = "A"` encoding to confident adenine, and as the `X_seq` placeholder frames.

Three details worth carrying:

- **36 pathogenic calls are discarded** — `pathogenic (dominant)` (26),
  `likely pathogenic (dominant)` (8), `likely pathogenic (!)` (2). The connector lowercases and
  strips but does not normalise parentheticals, so this is a parsing bug, not a decision.
- **Ordinal 2 is empty across all 18,006 variants.** Five of eleven map keys target the
  uncertainty tier and none occurs even once; LOVD says `notClassified` (7,435) and `unknown`
  (77). The realised encoding is `{0, 1, 3, 4}`, which undercuts the connector docstring's
  stated rationale about letting tree models exploit the ordering.
- **No gate could have caught this.** The two LOVD tests are plumbing tests and pass correctly
  in 6.60 seconds with no skips; they guard the 2026-05-02 root cause (`AnnotationConfig`
  constructed without `lovd_path`), not coverage. The feature census guards **zero variance**,
  and this feature has values 0/1/3/4 — variance, while being nonzero for 0.0217% of rows.

## 6. DEFECTS IN MY OWN WORK THIS SESSION

Thirteen. Recorded at full detail because several were more instructive than the findings.

**6.1 — I read a five-month-old screenshot as current state.** Filename `2026-02-07`, treated
as today. Everything in section 2 followed from that. The same class of error as the stale
`WindowAttachment` todo list flagged one session earlier.

**6.2 — I instructed archiving at the old address.** Settings → Danger Zone → Archive. Had it
been followed before verification, the live project would have become read-only: no pushes, no
Continuous Integration. Reversible, but not something to discover by doing it.

**6.3 — I verified remote against remote and called it verification.** `ls-remote` on both
addresses returned ten identical references and I read that as the whole picture. The comparison
that mattered — **local references against remote references** — is the one I never ran, and it
is where the tag defect lived.

**6.4 — I said the local tags pointed into dead history.** The opposite: they were ancestors of
`main` and were the correct ones. On that misreading I instructed `fetch --tags --force`, which
degraded the working copy (section 3).

**6.5 — I predicted `remote prune` would change the commit count.** It cannot: `ci/widen-test-scope`
merged into `main` at `0996dec`, so its commits stay reachable regardless of any branch label.
Pruning was correct hygiene and irrelevant to the count.

**6.6 — I predicted `fetch --all --prune` would close the gap.** The references were already
identical; the fetch found nothing and the count did not move.

**6.7 — `-ErrorAction SilentlyContinue` swallowed a real failure.** `Copy-Item` of a file that
was never in Downloads reported nothing, `git add` then failed on the missing pathspec, and
because `git add` aborts entirely when any pathspec fails, `docs/` went unstaged too. A silent
failure, written by me, into a project whose first principle is that nothing fails silently.

**6.8 — `$t` and `$T` are the same variable.** PowerShell variable names are case-insensitive. A
`foreach ($t in ...)` loop destroyed the clone path held in `$T`, surfacing as
`fatal: cannot change to 'v4.0.0'`. Recorded as PowerShell hygiene item 15.

**6.9 — `Select-String -Recurse` does not exist.** Already in my notes; written anyway.

**6.10 — I claimed 5,553 "reproduced exactly."** It did — but through `diag_lovd_join.py`, which
filters ClinVar to the ten LOVD genes before joining. The connector does not. The
connector-faithful figure is **5,576**; the 23-variant difference is that gene pre-filter.

**6.11 — My first LOVD probe neither lowercased nor deduplicated.** Both differ from production.
Both turned out to change nothing — measured, not assumed, in `LOVD_DEDUP_2026-07-19.txt` — but
they were right by luck rather than by method.

**6.12 — I suggested the LOVD regression guard might be inert.** Both tests ran and passed with
`-rs` showing no skips. The guard is sound; it simply guards something else.

**6.13 — I mis-stated the feature-count constant** as `EXPECTED_FEATURE_TABULAR_COUNT`. It is
`EXPECTED_TABULAR_FEATURE_COUNT` (`variant_ensemble.py:2208`), which is why an earlier grep
returned nothing.

**The pattern.** Every one of these was caught by reading source or by running a control — none
by re-reasoning. Three tools built this session to check other people's claims each produced a
claim of its own that needed checking. And a tool can pass every one of its controls while
answering the wrong question, if its inputs rest on a premise nobody checked: `compare_repos`
returned "SAFE TO ARCHIVE", which was true, useless, and nearly harmful, because it compared the
live repository against itself.

## 7. COMMITS

```
4cfa6a2  docs(lovd): the classification map zeroes 91.8% of the artifact;
         recover diag_lovd_join.py from a 72-day stash     6 files, 648 insertions   CI #528 GREEN
988c082  (session start)
```

Also pushed, outside the commit graph: three tag references force-updated to their correct
targets (section 3).

## 8. OPEN, CARRIED FORWARD

- **Part 3 of the hybrid** — make `X_seq` optional in `VariantEnsemble.fit`/`evaluate`/
  `predict_proba`, failing loudly when `cnn_1d` is active without sequence. Designed in
  `docs/status/SEQ_BRANCH_FINDINGS_2026-07-18.md`; the `y`-positional question must be settled
  against the 85 real call sites first. **This is the next substantive engineering work.**
- **LOVD remedies** — five candidates, deliberately unbundled, in the incident's section 6.
  Scope decision required: parenthetical normalisation is a bug fix; a presence indicator and a
  functional-effect feature change the feature contract.
- **LOVD coverage on the current cohort** — every figure measures the Run 9-era
  `models/v1/clinvar_enriched.parquet` (1,700,687 rows). The live cohort is
  `data/processed/clinvar_grch38_clean_seq.parquet` (4,399,089 rows). Unmeasured.
- `_att_tune` calibration gate (`train.py:477`) — mask logged, never gated.
- `--seq-windows` resolver — one flag, two contracts; 6.29a's remedy needs re-deriving.
- `v4.0.0` is a lightweight tag on a Dockerfile build fix, and no `v3.0.0` exists.
- `run9a-prep` local branch is fully merged into `main`.
- The `skipif` guard in `test_lovd_annotation_reaches_training_matrix.py` is dead weight; its
  own docstring says to remove it once the imports stabilise, and they have.
- `git gc` is now safe — the three tag orphans are unreferenced.
- Session-record gaps: **2026-05-08** (the day the stash was created), 07-13, 07-14, 07-15.
- Standing documentation debts: living metrics glossary, per-model algorithm comparison.
