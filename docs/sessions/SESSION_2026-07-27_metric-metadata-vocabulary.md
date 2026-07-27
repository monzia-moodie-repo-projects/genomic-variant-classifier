# SESSION 2026-07-27 -- the controlled metadata vocabulary

**Branch point:** `origin/main` at `a6df4ef`, ratchet 3227, tree clean.
Roadmap Tier 1 item 6, commit 2b of 3 -- the foundation commit 3 depends on.

---

## 1. WHY A CONTROLLED VOCABULARY

Free-form metadata keys drift. Without one canonical spelling,
`population_scope`, `populationScope`, `population` and `scope` can all appear
and each looks correct in isolation. That is the same wording drift that let one
word name two estimands in the 2026-07-25 P6 audit, and that let "explicit
conflicts preserved" name a count of withheld-label states.

`MetricMetadataKey` is now the single source of spelling for seven keys:
POPULATION_SCOPE, CERTIFICATION_ELIGIBLE, CERTIFICATION_BLOCKED_BY,
N_OBSERVATIONS, N_CLASSES_OBSERVED, N_CLUSTERS, METRIC_NAME.

### A correction to the recommendation

The design called for `class MetricMetadataKey(StrEnum)`. `StrEnum` arrived in
Python 3.11; pyproject declares `requires-python = ">=3.10"`, and
`test_no_module_uses_strenum_which_would_break_the_declared_python_floor` guards
it. Implemented as `class MetricMetadataKey(str, Enum)`, the pattern MetricStatus,
BootstrapUnit, MetricInput and CapabilityState all use.

### The property that makes it non-breaking, verified not assumed

Measured on Python 3.10.20:

```
hash(member) == hash("population_scope")      True
{"population_scope": 1}[member]               1
{member: 1}["population_scope"]               1
json.dumps({member: 1})                       {"population_scope": 1}
```

So the enum is the canonical spelling in code while serialized artifacts keep
plain string keys and every existing reader of `metadata["population_scope"]`
keeps working.

---

## 2. ACCESSORS, NOT CONSTRUCTOR FIELDS -- A MEASURED DECISION

Six read-only properties on `MetricResult`: `population_scope`,
`certification_eligible`, `n_observations`, `n_classes_observed`, `n_clusters`,
`metric_name`. Each returns None when the key is absent or wrongly typed.

THE MEASUREMENT THAT DECIDED IT. On 2026-07-27 there were **53 MetricResult
construction sites** -- 39 in `src/`, 14 in `tests/` -- and **35 of the 39** are
in `representation_geometry.py` and `norm_angle_probe.py`, using POSITIONAL
arguments such as `MetricResult(mean, MetricStatus.OK)`.

Those are mathematical probes over embedding spaces: effective rank, anisotropy,
angular concentration, hubness. "Population scope" has no epidemiological meaning
for a spectral effective-rank measurement. Making the fields mandatory would put
ceremonial values exactly where they cannot be checked, which is a WEAKER
contract.

**TWO SEMANTIC FAMILIES.** Family A is cohort evaluation -- AUROC, Brier,
expected calibration error, coverage -- which naturally carries population scope,
support counts and certification. Family B is representation probes, which
naturally carries matrix shape, embedding dimension and partition, but not
population scope in any epidemiological sense.

`MetricResult` therefore stays a GENERIC result contract. The registry requires
and fills these keys; probes do not, and the accessors return None. A stronger
domain-specific contract (`EvaluationMetricResult`) can be layered later without
touching any of the 53 sites.

---

## 3. THE GUARDS FOUND ONE REAL DEFECT AND TWO INSTRUCTIVE FALSE POSITIVES

### 3.1 REAL, mine

`test_the_registry_uses_the_enum_rather_than_string_literals` failed: registry.py
still used `"n_classes_observed"` as a string literal at two sites, inside the
applicability predicates written earlier the same day. The enum only prevents
drift if the registry actually uses it. Fixed; the test now passes and would fail
again on any regression.

### 3.2 FALSE POSITIVE -- `prediction_artifacts.py` "scope"

A COLUMN in a calibration-breakdown table (`{"scope": "global", "bucket": i}`),
not result metadata. Unrelated to population scope.

### 3.3 FALSE POSITIVE, and the important one -- `representation_geometry.py:209`
`"n_rows"`

This IS genuine `MetricResult` metadata:
`MetricResult(..., {"n_zero_norm": n_zero, "n_rows": int(x.shape[0])})`.

But it means rows of an EMBEDDING MATRIX, not observations in a cohort. It is
Family B's word for a different quantity, not a misspelling of `n_observations`.
The first forbidden list would have forced a rename that made the vocabulary
WRONG. Narrowed to same-meaning spellings only -- camelCase and true synonyms --
with both findings recorded in the test file as the reason.

That is the two-family distinction appearing in live code, and it is why the
distinction is preserved rather than erased.

---

## 4. VERIFIED

```
20 new tests in tests/unit/test_metric_metadata_vocabulary.py
331 passed across ten affected files
3.10 floor: MetricMetadataKey imports and the accessors work
positional constructor still works; 35 probe sites untouched
representation_geometry and norm_angle_probe still resolve the SAME MetricResult
```

Ratchet 3227 -> 3247 (+20).

---

## 5. WHAT REMAINS -- commit 3, the integration

Not started. See HANDOFF_2026-07-27_metric-stack-commit3.md.

---

*Written 2026-07-27.*
