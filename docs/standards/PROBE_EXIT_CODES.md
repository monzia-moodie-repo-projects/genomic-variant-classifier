# Probe exit-code standard

**Effective 2026-07-24.** Every measurement probe under `scripts/probe_*.py` uses
these codes and nothing else. A probe must reference this document rather than
define its own semantics, because a code that means one thing in one tool and
another elsewhere is worse than no code at all: the operator, or the
orchestration layer, acts on it.

## The codes

| Code | Meaning | The probe ran? | Blocks work? |
| ---: | --- | --- | --- |
| **0** | Completed. No finding. | yes | no |
| **1** | Completed. **Blocking finding.** | yes | **yes** |
| **2** | **Execution or contract failure.** Nothing was measured. | no | **yes** |
| **3** | Completed. Non-blocking finding, recorded. | yes | no |
| **4** | **Insufficient support.** Ran; required inputs absent. | yes | **yes**, until inputs exist |

## Why 2 and 4 are different, and why both differ from 1

**2 is "the tool could not run correctly."** The repository does not look the way
the probe requires, a source contract could not be read, an artifact could not be
opened. Nothing was measured and **no verdict is issued**. An absence of
measurement must never be reported as a clean measurement.

**4 is "the tool ran correctly and the inputs cannot answer the question."** The
probe worked; the data was not there. `probe_cohort_schema_census.py` returns 4
when no cohort artifact carries every column the measurement requires. That is
not a finding about the data -- it is the absence of data to have a finding about.

**1 is "the tool ran, the inputs were sufficient, and the answer is bad."**

Collapsing these loses the distinction between a broken tool, a missing input,
and a real defect, which are three different repairs.

## Why 3 exists

On 2026-07-24 `probe_label_column_terms.py`'s predecessor exited 1 on any
divergence, including one confined to a column production never labels from. Its
own closing text said the blocking condition was the labelling column
specifically. **An operator following exit codes would have halted for a
non-blocker.** A fail-loud guard that fails for a false reason trains the operator
to ignore it -- the lesson recorded from the Run-16 preflight gate on 2026-07-20.
Code 3 exists so a finding worth recording can be recorded without being obeyed.

## Collision warning: pytest also exits 4

`pytest` returns 4 for a command-line usage error, and this project's suite-size
ratchet surfaces its failure through that path. Orchestration that reads both a
probe's exit code and pytest's **must not treat the two as one signal**. Where
both are invoked in one script, capture and label them separately.

## Requirement on orchestration

Any layer consuming these codes must preserve four outcomes, not two:

    blocking finding | non-blocking finding | execution failure | insufficient support

Treating every non-zero code as equivalent failure defeats the purpose of the
standard and reintroduces the defect it was written to remove.

## Current conformance

| Probe | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| `probe_label_column_terms.py` | no divergence | labelling column diverges | contract failure | divergence outside the labelling column | -- |
| `probe_cohort_schema_census.py` | an artifact supports the measurement | -- | contract failure | -- | no artifact carries every required column |
| `probe_cohort_delta_forensics.py` | every finding benign | unaccounted rows, unmatched status, or sequence failures | contract failure | -- | -- |

A dash means the probe has no condition of that kind. It does not mean the code
is unavailable to it.
