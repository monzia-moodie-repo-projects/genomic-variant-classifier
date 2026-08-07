# SESSION 2026-08-07 — REGISTRY-1: a model registry, and the gate that would have caught it

**Base: `5298e90`. Result: `372cea1`, pushed.**

**Ratchet 4353 → 4417 (+64). Armed full suite 4411 passed, 6 skipped, 0 failed
in 12m43s; 4417 collected. Skip surface unchanged at 6.**

A class referenced at four call sites and defined nowhere; an
import-resolution gate that makes that condition impossible to reintroduce
silently; and a fail-closed boundary so that repairing the import does not arm
four unresolved scientific defects.

Found while chasing a stale constant in the REST application programming
interface. The audit that produced it is recorded in the canonical
specification commit `5298e90`.

---

## 1. What was wrong

`ModelRegistry` was imported at `continual_trainer.py:127` and `:266`, and in
`drift_monitor.yml:614-626`, and defined **nowhere**. Established three
independent ways:

| instrument | result |
|---|---|
| direct execution | `ImportError: cannot import name 'ModelRegistry'` |
| `git log --all -S "class ModelRegistry" -- src/ scripts/` | **empty** — never written, not deleted |
| a 527-name import census | the **only** unimportable name in the codebase |

It survived because both Python imports are **function-local**, so module
collection never executes them, and `continual_trainer.py` has no test coverage
across 410 lines. The adaptive-retraining and model-promotion chain could not
run and nothing said so.

`models/registry.json` has never existed — `.gitignore:75` ignores `/models/`
wholesale — so `drift_monitor.yml`'s guard exits 3 before reaching the
`ImportError`. A correct diagnosis of the symptom that conceals the cause.

## 2. What the call sites had already specified

`continual_trainer.py:386-401` calls `register(model_path=…, metrics=…,
data_manifest={clinvar_release, n_train, n_features, reclassified},
feature_names=…, drift_report=…)` and then `promote(record.version, "shadow")`.

**That is the artifact-identity chain**, written into a call site and never
implemented. `ModelRecord` completes something the project already asked for
rather than inventing architecture.

## 3. What was built

`src/genomic_variant_classifier/monitoring/model_registry.py`, 717 lines. A
**new module**, not an addition to `monitoring/registry.py` — which is a
data-source registry of `Category`, `Check`, `Source`, `Verdict`, `REGISTRY`
and five accessors over external corpora. The call sites specified an interface
worth preserving; they did not establish that their module placement was wise.
Grafting deployment state onto a data-source registry would resolve an
`ImportError` by creating a semantic junk drawer, so the imports were corrected
instead.

**No bare scalar, ever.** `ModelRecord` has no `auroc` property, not even as a
convenience, and `test_no_bare_auroc_property_exists_on_a_record` asserts the
absence. A metric detached from its protocol is what made 0.9988 unseen-gene
look comparable with 0.9984 ordinary test. Callers write
`record.evaluation.metrics["auroc"]`, keeping the number in the same expression
as `record.evaluation.protocol`.

**Six refusals on production promotion**, each individually tested: candidate
not in shadow, non-durable artifact uniform resource identifier, roster
mismatch, metric absent, below the absolute minimum, and **evaluation-protocol
mismatch**.

**Identity is lineage plus content.** `ArtifactIdentity.measure` reads the
digest from disk; a caller cannot supply one. `record_id` is
`run_id + "-" + sha256[:12]`, so a re-export under the same version label is a
different record.

**The roster is enumerated, never counted**, with an order-independent
fingerprint. **Promotion history is append-only**, so *what was production
before this, and when did it move* stays answerable — which a mutable
`record.stage = "production"` destroys.

**A missing registry is an error, not an empty one.** `create_if_missing=True`
is required to declare one. "No declaration exists" and "a declaration declares
nothing" are different statements, and conflating them is the drift monitor's
green-lie shape refused at the type level.

## 4. `deployments/`, a new namespace

```
models/          artifacts        large, machine-local, gitignored
data/reference/  reference data   scientific, committed
deployments/     control plane    small reviewable declarations
```

`deployments/README.md` records what a committed registry **can** claim — the
structural coherence of a declaration — and what it **cannot**: the health of a
running deployment. Closing that gap needs the serving environment to attest
that the artifact digest it loaded equals the declared one. That is DEPLOY-1's
work and is not done.

An empty `records` list is therefore honest: **no deployment is declared.**

## 5. Repairing the import must not arm retraining

A missing class is a terrible safety mechanism, and removing it without
replacing the boundary would be worse. `AdaptiveRetrainingInputs` turns four
measured findings into constructor preconditions:

- **LSIF-1** — `lsif.fit(X_ref=current_pipe._prepare(pd.DataFrame(X_train_new)),
  X_new=X_train_new.to_numpy())` gives the two density roles **the same rows in
  two different feature representations**, one through the production
  pipeline's column selection and scaling and one raw, with basis centres drawn
  from the raw side. The declared ratio p_new/p_old has no reference population
  and is not identified. Line 328 then takes `sqrt(lsif × ewc)`, so half the
  adaptive weighting signal is that quantity.
- **ROSTER-1** — the retraining subprocess passes `--skip-nn --skip-svm`, so
  the intervention is "new data + adaptation + architecture change" and any
  shadow-versus-production movement is confounded.
- **EVALPROV-1** — `X_val_new`, the new release's **validation** split, is
  registered as `holdout_auroc` while the module contract promises "evaluate on
  canonical holdout set". Measured: `run_phase2_eval.py` has **zero**
  occurrences of `calibrat`, no `eval_set` and no `early_stop`, and stacking
  uses gene-disjoint inner cross-validation on train only — so within that
  script this is a provenance defect, not leakage. **Unmeasured residual:**
  whether `VariantEnsemble.fit` uses an internal validation split for
  gradient-boosting early stopping.
- **EWCSEL-1** — `best_score_` is set **nowhere** in `src/`, so
  `max(..., key=getattr(m, "best_score_", 0.0))` compares an all-equal keyspace
  and returns whichever base model comes first in dictionary iteration order.

**PIPELINE-1** is named in the refusal too: `InferencePipeline` has no
`_prepare`, which `continual_trainer.py:299` calls, so the line raises
`AttributeError` before any of the above is reached. It is also the class of
defect an import gate **cannot** see, and the gate's docstring says so rather
than implying a coverage it lacks.

**EVALPROV-1 is now visible by construction.** The register call builds
`EvaluationProtocol(split_kind="new_release_validation", …)`, so the code states
what the split actually is, and a promotion against a production record under a
different protocol is refused by the registry rather than silently performed.

## 6. The gate, and the three attempts it took

`tests/unit/test_import_resolution_gate.py`, 390 lines, 18 collected cases.

**It does not reimplement import resolution.** The first attempt used
`hasattr(module, name)` and reported **eleven** working submodule imports as
broken — `hasattr(email, "message")` is `False` while `from email import
message` succeeds. It now **executes** the statement in a child interpreter and
lets Python answer.

**A child interpreter**, because importing this package in-process applies the
imodelsx Kolmogorov-Arnold Network repair and initialises PyTorch. A structural
check must not reshape the interpreter the rest of the suite runs in.

**It was observed failing before it was trusted.** The installer ran it against
the live tree before any edit and refused unless it reported exactly one
unresolved name — `ModelRegistry`, from two sites — then ran it again after the
edits and refused unless it reported zero. Red then green, against the real
repository, with the tree never left red.

**It complements, and does not duplicate, `tests/smoke_test_imports.py`.** That
file walks MODULES with `pkgutil.walk_packages` and an `onerror` collector. It
could never catch `from …registry import ModelRegistry`, because
`monitoring.registry` imports perfectly. It is also **never collected** —
SMOKE-1.

## 7. DOWNLOADSHADOW-1, found by the gate refusing

The gate's first run against the live tree reported **sixteen**
`AttributeError: module 'catalogue' has no attribute 'create'` failures from
inside `thinc`. The repository was entirely fine.

`C:\Users\monzi\Downloads\catalogue.py` shadows the installed `catalogue`
distribution, and the installer runs from that folder, so `sys.path[0]` was
Downloads. Downloads also holds `metrics.py`, `registry.py`,
`kernels_final.py`, `test_population_wiring.py`, `test_metric_catalogue.py` and
`test_brier_decomposition.py`. **Every installer this project uses runs from
there**; this was the first one that imports the package.

The suite is unaffected — pytest runs from the repository root with
`testpaths = ["tests"]`.

The child now takes an **explicit import path** with the caller's own directory
removed, and every entry is resolved to **absolute** form first, because an
empty entry means the *parent's* working directory to the parent and the
*child's* to the child.

## 8. Author defects, and where each was caught

Twenty-seven across the session; seven belong to this piece of work, and not
one was caught by review.

1. `hasattr(package, submodule)` — eleven phantom failures. Caught because one
   flagged file was one I had just read in full.
2. Expecting **two** gate failures when `resolve_in_child` deduplicates by
   `(module, name)`; the two call sites collapse to one. Caught by the
   installer refusing.
3. A namespace-package hypothesis for the `catalogue` failure. Explicitly
   flagged as needing measurement, and **falsified** by it.
4. `sys.path` entries crossing a process boundary: `""` means different
   directories to parent and child. Caught by a shadow test still failing after
   the "fix".
5. An anchor transcribed from my own reconstruction rather than from the
   source — a `notes=` argument wrapped where the source has it on one line.
   The simulation **confirmed** the error because I wrote both sides of it.
   Caught by the installer refusing.
6. A `\U` escape in a non-raw docstring. Caught by `ast.parse` before delivery.
7. Arithmetic: 46 + 18 = 64, not 65. The installer's own `NEXT` hint also
   printed a stale "46 + 16 = 62 / predicted 4415" it had never been updated
   from.

The pattern is one pattern: **a conclusion that agreed with its author for the
wrong reason.** Defect 5 is the sharpest instance the session produced — an
instrument built specifically to catch that failure, failing at it, because its
author supplied both the claim and the evidence.

## 9. Acceptance

| | |
|---|---|
| base | `5298e90` |
| result | `372cea1` |
| diff | 8 files, 1847 insertions, 24 deletions |
| targeted | 64 passed in 20.47s |
| unarmed full suite | 4411 passed, 6 skipped, 0 failed, 33 warnings, 16m01s |
| armed suite **before** the bump | **ERROR — the ratchet gate fired** |
| coupled tests after the bump | 100 passed in 10.78s |
| armed full suite after the bump | 4411 passed, 6 skipped, 0 failed, 33 warnings, 12m43s, gate silent |
| collected | 4417, measured; predicted 4417 by two static counts |
| sabotage | 16 mutations, 16 detected, 0 undetected |
| import pairs discovered | 635 before, 638 after |

Running the armed suite **before** the bump was not planned, and it is the most
useful thing in the table: the ratchet gate was observed firing —
*"expected 4353, actually collected 4417, 64 MORE test(s) than expected"* — and
then observed silent. The step-3c rule discharged by demonstration.

The 33 warnings are identical in composition to the pre-change suite: all
pre-existing scikit-learn degenerate-cohort warnings. None comes from the new
modules.

## 10. Next

**PROD-1 and GATE-1** are now unblocked; both were downstream of an absent
identity chain. **DRIFT-1 with README-1.** Then **OP-1 step 5** against STEP K,
then **OP-2**.

**RETRAIN-GATE** — the third commit of the REGISTRY-1 sequence, arming
retraining once LSIF-1, ROSTER-1, EVALPROV-1, EWCSEL-1 and PIPELINE-1 each have
an answer — is not started and should not be until they do.

Forty-nine follow-ups are open.
