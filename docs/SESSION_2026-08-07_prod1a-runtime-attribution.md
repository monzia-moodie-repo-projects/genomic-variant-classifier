# SESSION 2026-08-07 — PROD-1 Commit A: the service reports what it is serving

**Base: `63e5da0`. Result: `4d334f9`, pushed.**

**Ratchet 4417 → 4449 (+32). Armed full suite 4443 passed, 6 skipped, 0 failed
in 14m39s; 4449 collected. Skip surface unchanged at 6.**

The web service stops publishing constants nobody maintained and starts
reporting the identity of the artifact it actually loaded. It publishes **no
metric at all**, and that is the scientific point rather than an omission.

---

## 1. What was wrong

`api/main.py` carried five provenance constants written in `ae1853b` on
**2026-03-25**, under a comment reading *"update after each training run"*.
They were never updated — through Runs 9 to 16.

`HOLDOUT_AUROC = 0.9847` fused a **Run-8, sixty-four-feature** figure with
**154,404**, the validation split size of the **Runs 10–14** cohort. Two
measurements, two eras, one line. The same digits are Run 15's unseen-gene
**F1**, so a reader reconciling them lands on a third quantity.

`/info` published all of it regardless of which artifact was loaded — indeed
regardless of whether any artifact was loaded — and **four of the five were
pinned by literal in `test_api.py`**. The suite defended them. That is why they
survived four and a half months.

The endpoint also described a *"LightGBM / XGBoost / GBM / RF / LR ensemble"* —
five models against a roster of thirteen — and *"1.2 M tier-2 ClinVar
variants"* against Run 15's 1.49 M. The OpenAPI description served at `/docs`
asserted the stale AUROC and the 154,404-row cohort to every reader.

**The sharpest statement of the defect:** `/info` set `model_version =
MODEL_VERSION`, the module constant `"phase2-v1"`, while
`PipelineMetadata.model_version` travels *inside the artifact* and defaults to
`"phase2"`. Load a pipeline whose metadata says `run15` and `/info` still
reports `phase2-v1`. **The endpoint did not read the thing it claimed to
describe.**

## 2. What replaced it

Nothing hand-maintained. Model identity is derived from the digest of the bytes
actually loaded, resolved against `deployments/registry.v1.json`.

`API_VERSION = "2.0.0"` covers the software contract only. `PIPELINE_VERSION`
was **retired rather than narrowed**, because one symbol was serving as both
the OpenAPI version and prediction provenance — and that conflation is why
`"1.0.0"` sat unexamined.

### Four orthogonal axes

| vocabulary | question |
|---|---|
| `ArtifactResolutionStatus` | can these bytes be identified? |
| `DeploymentAlignment` | are they what the registry **declares**? |
| `RosterAlignment` | does the executable roster match, given a declared serving projection? |
| `EvaluationApplicabilityStatus` | may a metric measured on that record be shown as evidence for **this** artifact? |

Collapsing any two would recreate the drift with better types. A registered
**shadow** artifact served by accident must not look healthy merely because its
digest resolves — which is precisely why resolution and alignment are separate.

`NO_MODEL_LOADED` and `NO_ARTIFACT_IDENTITY` are also deliberately distinct. A
pipeline object injected straight into the process is loaded and usable and
simply has no provenance; calling that "no artifact" would say something false
about the model rather than something true about its identity.

## 3. No metric is published, in any state

`InferencePipeline.from_variant_ensemble` excludes `cnn_1d`, which needs a
FASTA context window unavailable over the web path. **The deployable artifact
is a twelve-model projection of a thirteen-model trained ensemble.**

A metric measured on the record therefore does not automatically describe these
bytes. **Resolving a digest authorises identity, not evidence.** A sealed
evaluation naming this artifact digest *and* this served-roster fingerprint is
what will authorise publication, and that is Commit C.

A four-way parametrised test asserts that no binding, in any state, ever
reports `APPLICABLE`. A structural test asserts `RuntimeModelBinding` carries
no field containing `auroc`, `metric` or `score` — because a future convenience
property is exactly how `record.auroc` would return.

## 4. The projection is declared, not inferred

`ServingProjection` records which models a derived serving artifact omits and
why. An intentional omission is `SERVING_SUBSET`; an **undeclared** one is
`UNKNOWN`; anything further missing is `INCONSISTENT`.

Without it, a missing CatBoost and a missing `cnn_1d` look identical at
runtime — and this project has already lost a model silently once: the imodelsx
Kolmogorov-Arnold Network was absent from every Continuous Integration run
until a repair was written for it.

`schema_version` stays at **1**. An absent projection is a legitimate state,
not an older file, so there is no migration and all 46 existing registry tests
pass unchanged.

## 5. Liveness and readiness

A process that loaded bytes it cannot identify is **alive and not ready**: an
orchestrator should stop routing inference traffic to it rather than restart
it. For a clinical-facing service, *"I loaded a model but cannot establish what
it is"* must not be operationally green.

The existing `client` fixture injects a pipeline object directly, so it
constructs exactly that world — and the suite asserted it was `"ok"` until
today. `HEALTHSEM-1` tracks splitting the endpoint into `/health/live` and
`/health/ready`.

## 6. The artifact is measured around the load

A digest taken only **before** describes bytes that may have been replaced
during the load; one taken only **after** describes bytes that may not be what
was deserialised. `load_pipeline_with_identity` takes both, compares them, and
raises `ArtifactChangedDuringLoadError` on disagreement — and that exception is
caught **separately** from the generic one, so the pipeline is set to `None`
rather than served.

Exactly two measurements: registry resolution reuses the second rather than
reading a large artifact a third time.

## 7. Author defects, and where each was caught

Three in the code, one in the check written to catch the first, and one in the
sabotage matrix's target. **Not one was caught by review.**

1. **`ModelAttributionResponse` referenced in `/info` and never imported** —
   `NameError` on every call to that endpoint. Four tests failed. No import
   check and no collection sees it: the import-resolution gate covers
   `from X import Y`, and this is a name with no import at all. F821's
   territory, and **F821-1 is open**.
2. **`InferencePipeline.save` is `joblib.dump`, and `MagicMock` will not
   pickle** — so the three tests built to drive the real serialisation path
   never ran at all. The fixture was written specifically to exercise
   serialisation and was then handed something unserialisable.
3. **The undefined-name check written to catch defect 1 collected only the
   OUTER function's parameters**, so a `lambda f:` left `f` looking undefined.
   It reported that against `main.py` and **refused a correct fix**.

Defect 3 is the instructive one. Its self-test passed because **the author
chose three cases and omitted lambdas and nested definitions** — the two
commonest binding forms in Python outside assignment. An instrument built to
stop conclusions that agree with their author did exactly that, in its own
proof.

The proof now runs **eight** cases, two chosen specifically because they would
catch that blind spot, and one that **fails if the checker is weakened** to
make the others pass. That last property is what the three-case version lacked.

**And the sabotage matrix found a fifth:** a test named
`test_a_registered_shadow_artifact_is_not_production` actually constructed *no
production declared*, so the branch a mutation targeted never executed. The
test's **name** claimed more than the test checked. The world it should have
built is now
`test_an_artifact_registered_but_not_the_declared_production_differs`.

## 8. Acceptance

| | |
|---|---|
| base | `63e5da0` |
| result | `4d334f9` |
| diff | 8 files, 1478 insertions, 69 deletions |
| targeted | 172 passed, 0 failed, 0 errors |
| unarmed full suite | 4443 passed, 6 skipped, 0 failed, 33 warnings, 18m34s |
| coupled tests after the bump | 100 passed in 16.87s |
| armed full suite | 4443 passed, 6 skipped, 0 failed, 33 warnings, 14m39s, gate silent |
| collected | 4449, measured; predicted 4449 |
| sabotage | 12 mutations, 12 detected, 0 undetected |
| anchors | 20 defined, 20 exact matches, first attempt |

Every number in the diff reconciles: insertions 1 + 407 + 583 + 96 = 1,087
accounted for, leaving 391 across the four patched files; deletions 2, leaving
67; and 391 + 67 = 458 = 183 + 105 + 87 + 83.

The 33 warnings are identical in composition to the pre-change suite — all
pre-existing scikit-learn degenerate-cohort warnings. None comes from the new
code.

## 9. Next

**Commit B (GATE-1 / REGISTRY-1c)** — the workflow still calls `prod.auroc`,
which the registry deliberately does not provide, masked only because it exits
3 at the absent-registry guard first. The repair extends the registry's
existing promotion policy and exposes `validate_production_declaration`, so
Continuous Integration becomes an adapter rather than a second place that knows
AUROC arithmetic. `0.9842` is deleted rather than typed: it is copied
arithmetic from a stale constant, not a policy. `0.97` and `0.002` are
preserved as **pending justification**, because the architecture can be valid
before the numbers are endorsed.

**Commit C (`SealedEvaluation`)** — only after a field-by-field source census.
`0.9847` and `154,404` came from different runs, and that is exactly the
failure a hand-composed record reproduces.

Then **DRIFT-1 with README-1**, **OP-1 step 5** against STEP K, **OP-2**, and
**RETRAIN-GATE** last.

Fifty-three follow-ups are open.
