# SESSION 2026-08-07 — DOCKERCOPY-1: the image did not contain a module the service imports

**Base: `e8de6a6`. Result: `2ccad69`, pushed. Continuous Integration: success
on all seven jobs.**

**Ratchet 4449 → 4462 (+13). Armed full suite 4456 passed, 6 skipped, 0 failed
in 13m29s; 4462 collected. Skip surface unchanged at 6.**

---

## 1. What broke

`4d334f9` added `api/attribution.py`, which imports
`genomic_variant_classifier.monitoring.model_registry` at module level. The
Dockerfile's `api` stage copies `api/`, `models/`, `utils/` and the package
`__init__.py` — and **not** `monitoring/`.

Inside the image that import raised `ModuleNotFoundError`, gunicorn never
bound, the container exited, and the smoke step failed sixteen seconds later
with `curl` returning nothing. `4d334f9` and `e8de6a6` were both red on
`origin/main`.

## 2. Nothing in the repository could have caught it

The import-resolution gate installed with REGISTRY-1 runs against the **full
source tree**, where the import resolves perfectly. So does the test suite —
4,449 tests passed on both Python versions on the Linux runner while the image
was broken.

Only the **image** has the narrower file surface. And the only thing that
exercised the image was:

```
curl --fail --silent http://localhost:8000/health | grep -q '"status"'
```

which passes on any response containing that literal. It caught this **only
because the container died**. A service that started and lied would have gone
green — and that step had been the sole verification of the container for the
life of the project.

The author enumerated the consumers of `api.main` **in Python**. The
container's file surface is a consumer expressed in a Dockerfile, and no Python
search finds it.

## 3. The repair is the gate, not the missing line

`tests/unit/test_docker_image_covers_the_api.py` walks the static import graph
from `api/main.py` and asserts every reachable first-party module lives under a
path the `api` stage copies. It was proved:

| scenario | result |
|---|---|
| the Dockerfile as it stood | **red**, naming `monitoring.model_registry` |
| with the single-file `COPY` added | **green** |
| a near-miss `COPY` of `monitor/` | **red** |
| the `api` stage renamed | **red, loudly** |

The last row matters most: a lazier parser finds no `COPY` instructions after a
rename and reports nothing missing — a check that disarms itself when the thing
it checks moves.

The parser also joins Dockerfile line continuations, because reading `COPY a \`
line by line gives the right answer for one source path and the wrong one for
two. **Correct by luck is not correct.**

**One file, not the directory.** `monitoring/` also holds the drift detector,
the ClinVar tracker and the data-source registry, none of which inference
touches. It has no `__init__.py` — an implicit namespace package — so no module
body executes and the single file suffices. That is what line 94 of the
Dockerfile already asked for: *"only what inference needs"*.

## 4. The smoke step now tests something

It asserts the honest contract for an image that ships no artifact, and dumps
`docker logs` on failure. The container's `/health` body, printed for the first
time in this project's history:

```json
{"status":"degraded","live":true,"ready":false,"model_loaded":false,
 "model_attributed":false,"gnomad_index_loaded":false,
 "gene_counts_loaded":false,"uptime_seconds":13.8}
```

Every field as designed. Asserting `"ok"` would have demanded a lie; asserting
a field name accepted anything.

## 5. Two stale claims removed from surfaces nothing checked

The image `LABEL` published a validation area-under-the-curve figure into
**every image ever built**, and pointed `org.opencontainers.image.source` at a
repository that is not this one. `scripts/run_benchmark.py` claimed its results
were directly comparable to a published baseline that nothing publishes any
more.

**Neither figure is repeated in the replacement text.** BASELINE-1 holds the
number, because which experiment produced it is unestablished — this repository
cited it as a Run-8 holdout result, as a Phase-4 validation result, and against
a cohort size drawn from a third run.

`connector_1kgp.py` is **deliberately untouched**. Its citation sits in a
measured comparison whose arithmetic is internally consistent; rewriting it
would falsify a record and deleting it would lose the motivation for the 1000
Genomes Project connector.

## 6. Four refusals, every one a post-check rather than the repair

Nothing was written until the fifth run. The twenty-odd lines of actual repair
were correct from the first attempt; what failed each time was a check.

1. A text search for `0.9847` **satisfied by the comment explaining its
   removal**.
2. The same defect in a second file, fixed one file at a time rather than as a
   class.
3. The same defect in a third file — which finally produced the
   generalisation.
4. And then the **over-correction**: broadening `grep -q '"status"'` to bare
   `grep -q`, which refused a correct edit because two lockfile steps use the
   idiom legitimately.

The generalisation is a **stage-0 self-check running one forbidden list in both
directions** — the replacements must not reproduce it, the edited files must
not retain it — plus precision about *what* is forbidden rather than cleverness
about detecting it. Proved by poisoning an edit and watching it refuse, and by
running the file check against a workflow containing an unrelated `grep`.

The deeper repair was not a smarter check but **removing the string from the
files entirely**: the register holds the number, and the comment points at it.
A number written in two places is wrong in at least one eventually, and a
comment repeating it is a second place.

## 7. Observations recorded, not acted on

**The "duplicate" Continuous Integration alert was not a duplicate.** Two
`workflow_run` events fifteen minutes apart, both labelled `e8de6a6`, turned
out to be the completions of `4d334f9` and `e8de6a6` — a `workflow_run`-
triggered run carries the **default branch's head at the time it runs**, not
the head of the run that triggered it. Raised as unexplained, measured, and
explained as correct behaviour.

**`.github/workflows/data_freshness.yml.pre_scipy.bak`**, 902 bytes. Actions
only parses `.yml`/`.yaml`, so it does not run, but `.gitignore:262` matches
`*.bak_*` and this is `.pre_scipy.bak`, which does not match. A stale backup
beside a live workflow is the same hazard that justified removing the five
README patchers.

**`git add` warned that `Dockerfile`'s working copy will become CRLF.** The
committed blob is LF, but several installers refuse on carriage returns, and
this will surface eventually as a mysterious refusal.

## 8. Acceptance

| | |
|---|---|
| base | `e8de6a6` |
| result | `2ccad69` |
| diff | 6 files, 455 insertions, 7 deletions |
| targeted | 28 passed (13 gate + 15 workflow), 0 failed |
| unarmed full suite | 4456 passed, 6 skipped, 0 failed, 33 warnings, 17m29s |
| coupled tests after the bump | 100 passed in 10.81s |
| armed full suite | 4456 passed, 6 skipped, 0 failed, 33 warnings, 13m29s |
| collected | 4462, measured |
| Continuous Integration | **success, all seven jobs, including the smoke test** |

Insertions reconcile: 290 + 1 + 83 = 374 accounted for, leaving 81 across the
three patched files; deletions 2, leaving 5; and 81 + 5 = 86 = 38 + 32 + 16.

## 9. Next

**Commit B (GATE-1 / REGISTRY-1c)** — the workflow still calls `prod.auroc`,
which the registry deliberately does not provide, masked only because it exits
3 at the absent-registry guard first. `0.9842` is deleted rather than typed;
`0.97` and `0.002` are preserved as pending justification; Continuous
Integration becomes an adapter consuming `validate_production_declaration`; and
**PIPELINE-1's four call sites** are repaired, including the
`tuple(new_pipe.base_models)` shipped in `372cea1`.

Then **Commit C (`SealedEvaluation`)** after a field-by-field source census,
**DRIFT-1 with README-1**, **OP-1 step 5** against STEP K, **OP-2**, and
**RETRAIN-GATE** last.

Fifty-four follow-ups are open.
