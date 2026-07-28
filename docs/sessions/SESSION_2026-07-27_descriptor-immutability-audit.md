# SESSION 2026-07-27 — the descriptor immutability audit (commit 2b-3)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `2c4aa9e`, ratchet 3415
**Roadmap position:** Tier 1 item 6, commit 2b-3 — the last commit before commit 3
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. Why this commit exists

Commit 2b-2 made descriptors the semantic authority of the metric stack: they
carry the classification, the threshold provenance, the calibration provenance
and the parameter schema. Two things about that authority were asserted but never
proved.

**It was never proved that evaluation leaves descriptors alone.** A descriptor is
frozen in type, but `parameters` is reached through a mapping and a
`ThresholdParameters` can be edited through `object.__setattr__`. An in-place
edit during one evaluation would silently change what every LATER evaluation
means, and ordinary numerical tests never notice it, because the run that did the
mutating still produces the right answer.

**It was never proved that the acceptance oracle came from the tree it claims
to.** `tests/fixtures/registry_snapshot_2b1.json` is the frozen 2b-1 baseline
against which 2b-2 proved that nothing moved. A stale fixture is a visible
failure. A silently REGENERATED one is not: it becomes a photograph of the thing
it was supposed to be checking, and every identity comparison passes for the one
reason that guarantees nothing.

---

## 2. The snapshot is now self-validating

Six header fields were added:

    snapshot_version            "2b-1"
    captured_from_commit        "683b514"
    registry_schema_at_capture  1
    n_metrics                   6
    n_results                   48
    note                        why the header exists and what it guards

The decisive assertion is that `registry_schema_at_capture` must NOT equal the
current `REGISTRY_SCHEMA_VERSION`. The snapshot was captured under schema 1;
2b-2 raised it to 2. If those ever agree, the fixture was regenerated on the
current tree.

**The header was added WITHOUT regenerating a single recorded result.** The
digest of the `fixtures` block is `09713c3ee9279f5b8d4fafe3d5e953ef` before and
after, and it is the same digest the file carried when commit 2b-2 installed it.

---

## 3. The descriptor immutability audit

Fingerprint every descriptor, run every metric over five cohorts chosen to
exercise every result path — normal, degenerate confusion margin, single class,
non-probability input, non-finite probabilities — then fingerprint again and
compare.

The fingerprint covers name, required inputs, result kind, display name,
description, cluster requirement, output kind, report inclusion, the whole
parameter mapping, the threshold declaration, and the OBJECT IDENTITY of the
kernel and the applicability predicate. Identity matters: 2b-2's guarantee is
that one `ThresholdParameters` instance is shared by the mapping, the kernel and
the predicate, and a swap that preserved every value would defeat a value-only
comparison.

---

## 4. Two guards that could not fail, both caught by sabotage

Recorded because in both cases the test was written to prevent a defect and was
structurally incapable of detecting it.

**The guard-the-guard built its comparison as a dict literal.** It formed
`{**base, "field": other}`, which ADDS the key even when `_descriptor_fingerprint`
has stopped emitting it, so the inequality held whether or not the field was
covered. Removing three fields from the fingerprint left the suite green.
Rewritten to compare the fingerprints of two REAL descriptors differing in
exactly one field, which exercises the function instead of simulating it.

**The probe built a fresh lambda on every call.** `id(function)` therefore
differed on every construction and every fingerprint comparison was trivially
unequal — the guard passed whichever field the fingerprint had stopped covering.
The probe now holds one shared function and one shared predicate, so the only
difference between two probes is the field under test.

---

## 5. THE SEPARABILITY PRINCIPLE, now codified

Three times in this series a test intended to prevent a defect was incapable of
observing it: the calibration interval convention, the duplicate calibration
aggregation, and now the immutability fingerprint. Rediscovering a principle
three times is the point at which it stops being a lesson and becomes a rule.

> **Every regression fixture targeting an algorithmic distinction shall first
> demonstrate that the injected defect changes observable behaviour.**

In practice a fixture opens by proving it can fail — asserting the superseded
implementation and the current one disagree on this cohort — before asserting
which one is correct. Agreement that could not have been disagreement is not
evidence.

---

## 6. Verification

### 6.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list: 40, all sandbox dependency gaps. No test was lost; four were added.

### 6.2 Sabotage matrix

Eight breaks applied, **eight detected, zero undetected**.

| break | detected |
|---|---|
| B1 a kernel mutates descriptor parameters mid-run | yes |
| B2 the fingerprint stops covering parameters | yes |
| B2b the fingerprint stops covering the threshold declaration | yes |
| B2c the fingerprint stops covering result kind | yes |
| B3 the fixture is REGENERATED on the current tree | yes |
| B4 the fixture is emptied | yes |
| B5 the fixture claims metrics it never recorded | yes |
| B6 one recorded value is tampered with | yes |

B1 and B3 are the two failure modes this commit exists to catch.

---

## 7. A process finding about delivery hygiene

While preparing this commit, the scratch output directory was found to hold
NEWER copies of two 2b-2 files than the ones delivered and installed: the test
module and the snapshot fixture. Work had continued after the package was cut and
had overwritten them.

Nothing installed was affected. The installer hashes every payload file at run
time rather than trusting its name, and all nine matched at install. The two
fixture versions were compared field by field: 384 fields, ZERO differences,
identical `fixtures` digest. The divergence was purely additive — header fields
and four tests — and is exactly the content of this commit.

The finding is recorded because the only reason it was benign is that the
installer verifies rather than trusts. **A delivered payload must be treated as
immutable once cut**: a scratch directory that keeps mutating is
indistinguishable, by inspection, from one that was tampered with.

---

## 8. Files

    tests/fixtures/registry_snapshot_2b1.json           self-describing header, results untouched
    tests/unit/test_registry_vocabulary_completion.py   58 -> 62 tests

Ratchet 3415 -> 3419 (+4), measured by `pytest --collect-only`.

---

## 9. Next

Commit 3, the last of Tier 1 item 6: the registry becomes the only computation
path, flat report fields become derived views through one centralised projection,
schema version 3 lands, `result_kind` joins the serialised surface, the canonical
UNDEFINED semantics replace the evaluator's scikit-learn `0.0` through a NAMED
legacy projection, and `test_registry_kernel_is_called_once_per_metric` uses
counting wrappers so a second computation path fails rather than passing quietly.
Carried item (o), the evaluator abstract-syntax-tree guard, lands there too.

---

*Written 2026-07-27.*
