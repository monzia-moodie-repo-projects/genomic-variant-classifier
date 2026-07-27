# SESSION 2026-07-27 -- MetricResult moves to the vocabulary layer

**Branch point:** `origin/main` at `accc41d`, ratchet 3169, tree clean.
Roadmap Tier 1 item 6, commit 1 of 3. The registry is NOT in this commit.

---

## 1. WHAT MOVED, AND WHY IT WAS IN THE WRONG PLACE

`MetricResult` was defined at `clustering_metrics.py:176` -- inside a 1,326-line
PANEL module -- and imported by two others:

```
representation_geometry.py:84   from .clustering_metrics import MetricResult
norm_angle_probe.py:58          from .clustering_metrics import MetricResult
```

Three modules already spoke it and only one was about clustering, so it was
already a SHARED result contract living inside a single panel. Worse, its
`__post_init__` depends on `MetricStatus`, which lives in `capabilities.py`: the
dependency ran UPWARD, from the vocabulary layer into a panel.

A metric registry importing `clustering_metrics.py` merely to obtain a result
type would bind the orchestrator to one panel. So the vocabulary moves first.

THE PRECEDENT IS EXACT. `BootstrapUnit` received this relocation for this reason,
and `test_there_is_exactly_one_metric_status_class` already pins the identity
guarantee for the status enum: *two enums sharing a name is the divergence
problem removed in b8275a0, where the legacy evaluator was DELETED rather than
wrapped because two evaluation contracts in one codebase invite drift.*

---

## 2. THE ONE DESIGN DECISION, MEASURED RATHER THAN ARGUED

`capabilities.py` imported only `dataclass`, `Enum` and `Optional` -- pure
standard library. `MetricResult.__post_init__` calls `np.isfinite`. So the move
either adds numpy to the vocabulary layer, or swaps in `math.isfinite`.

MEASURED, on every input the type admits:

```
input                not np.isfinite(v)   not math.isfinite(v)   equivalent
python float/int     False                False                  yes
NaN / infinity       True                 True                   yes
numpy float64/32/int False                False                  yes
0-d array            False                False                  yes
bool                 False                False                  yes
None / str           TypeError            TypeError              yes
ONE-element array    False  (ACCEPTED)    TypeError              NO
TWO-element array    ValueError           TypeError              NO
```

They agree on every scalar and differ only on arrays, where numpy SILENTLY
ACCEPTS a one-element array as finite. Since this commit's acceptance criterion
is that behaviour does not change, `np.isfinite` was kept verbatim and numpy was
added to `capabilities.py`.

That is permitted: the contract enforced by
`test_evaluator_phase5.py::test_module_imports_without_sklearn` blocks
`sklearn` and `sklearn.*` only, and the test file imports numpy itself. Nothing
asserts that `capabilities.py` is standard-library-only, and its docstring makes
no such claim. The new test file states the no-scikit-learn guarantee directly
for `capabilities.py` rather than only through `evaluator.py`, because that
module is what the registry will import next.

---

## 3. TWO DEFECTS OF MY OWN, BOTH CAUGHT BY TESTS

### 3.1 The first extraction silently deleted a function

The boundary was taken as "from `MetricResult` to the next `@dataclass`", which
was `EstimatedMetric`. Between them sat `def aggregate(...)`, a top-level
function. It was deleted.

`tests/unit/test_clustering_metrics.py` caught it with
`ImportError: cannot import name 'aggregate'`, and a name-set diff against
`origin/main` confirmed exactly two losses where one was intended. The boundary
is now "from `MetricResult` to `def aggregate(`", asserted by extracting exactly
two top-level constructs, and a test pins the neighbours so the mistake cannot
recur.

### 3.2 `git checkout` destroyed the uncommitted relocation

Cleaning up a sabotage with `git checkout <file>` restores from the INDEX, which
held `origin/main`. The relocation was uncommitted, so it was wiped -- leaving
`clustering_metrics.py` expecting a re-export that no longer existed and the
package unable to import.

Detected immediately by an identity probe. Rebuilt from `git show origin/main:`
and, from that point, every sabotage cleanup used a FILE COPY taken beforehand.

**Standing lesson: never use `git checkout` to undo a sabotage while the real
work is uncommitted.**

---

## 4. A GUARD THAT COULD NOT FAIL, FOUND BY SABOTAGE

Four sabotages were run against the new tests. Three fired. The fourth did not:

```
1. a second MetricResult class            -> 2 tests FAILED   correct
2. capabilities importing a panel         -> 1 test  FAILED   correct
3. a registry module appearing            -> 1 test  FAILED   correct
4. np.isfinite swapped for math.isfinite  -> PASSED           *** guard is inert ***
```

The guard used `np.float64`, where the two implementations AGREE. It tested the
equivalent case, not the differing one, so it could not detect the swap it was
written to detect. Rewritten around the one discriminating input -- a one-element
array -- it now fails with `TypeError` at `capabilities.py:244`.

This is the fifth instance in three days of the same defect class: a check that
passes for the wrong reason. It was found only because the sabotage was actually
run.

---

## 5. VERIFIED

```
identity      clustering_metrics.MetricResult IS capabilities.MetricResult
              representation_geometry and norm_angle_probe resolve the SAME object
              defined in genomic_variant_classifier.evaluation.capabilities
neighbours    aggregate, EstimatedMetric, ClusteringPopulationAccounting intact
              name-set diff versus origin/main: exactly one loss, MetricResult itself
behaviour     all four construction invariants still fire
no-sklearn    contract green, asserted directly for capabilities.py in a subprocess
3.10 floor    imports AND constructs, not merely parses
suites        530 passed across fourteen affected files
sabotage      4 of 4 guards now fail when the thing they guard is broken
hygiene       LF only, no byte-order mark, pure ASCII
```

Ratchet 3169 -> 3185 (+16).

---

## 6. WHAT THIS COMMIT DELIBERATELY DOES NOT DO

No registry. No metric behaviour change. `metrics.evaluate()` untouched. A test
asserts `evaluation/registry.py` does NOT exist, to be deleted by the commit that
introduces it -- so a later registry regression can never be mistaken for a
vocabulary regression.

Next: commit 2, the declarative registry. Applicability is evaluated BEFORE the
kernel is invoked, so an inapplicable metric is never computed and a finite but
scientifically unsupported value -- the single-class Brier and expected
calibration error case -- can be refused rather than reported.

---

*Written 2026-07-27.*
