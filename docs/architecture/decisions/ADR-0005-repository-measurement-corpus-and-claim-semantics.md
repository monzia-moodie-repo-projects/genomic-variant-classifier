# ADR-0005 -- Repository measurement corpus and claim semantics

**Author: Monzia Moodie**
**Status:** accepted
**Date:** 2026-09-05
**Authority:** normative
**Domains:** data_schema
**Measured at commit:** c18a1df

---

## Context

ADR-0001 named four machine-readable artifact types and the epistemic role each
carries:

```
Observation    what was measured                 (census attestations)
Finding        classification of observations    (the defect register)
Decision       normative ruling                  (these records)
Attestation    that an operation executed, and its outcome
```

MEASURED 2026-09-05 at `c18a1df`: two of the four have a typed owner.
`Attestation` has `transactions/install_attestation.py`, schema version 3, with
eighteen attestations under `records/attestations/installations/`. `Decision`
has this directory and `tests/unit/test_adr_contract.py`.

`Observation` and `Finding` have none. ADR-0001 also specifies, at its own
lines 152 to 165, a preservation manifest for the ruling sequence; `git
ls-files` for `*decision_*` returns zero files. Three declared obligations,
zero implementations, in one accepted record.

That is not three oversights. It is the shape ADR-0001 itself names at line
181 -- **semantic compression** -- one level up: a role declared in prose and
never given a typed owner is carried by whatever artifact is convenient, which
in practice has been narrative session records.

The cost is measurable. On 2026-09-04 and 2026-09-05 a sequence of repository
inspections produced conclusions that exceeded what their evidence licensed:

```
a scan for `import audit_data_tree` returned ZERO while the gate was
demonstrably invoked, because the wiring loads by path through
importlib.util.spec_from_file_location; the same scan counted 31
"invocations" of preflight_data_guard, every one a line of Markdown prose

a false-positive rate calibrated on EIGHT markdown documents was applied to
1,637 tracked files, yielding 2,408 noise tokens

a `git grep` pathspec that matched NO FILE exited 0 with no output, and the
silence was read as "zero matches in an existing file"

fifteen findings were called "coherent" on the basis of narrative heading
sequence, which establishes narrative consistency and not current state
```

Each inspection was individually reasonable. Each conclusion drawn from it was
not. The common defect is that no artifact recorded what population was
inspected, how completely, or what the result licensed.

## Decision

Repository inspections shall declare their corpus, enumeration semantics,
analysis completeness, evidentiary scope, and -- where applicable -- the
predicate they adjudicate. Repository measurements do not by themselves own
project, finding, carried-item, architecture, or scientific state.

The typed owner is `src/genomic_variant_classifier/repository_measurement/`.
It is the implementation of the `Observation` role ADR-0001 declared on
2026-08-21, not a new architectural layer.

Its responsibility is exactly three things: declare what repository population
was inspected, describe how completely it was inspected, and state what the
resulting evidence licenses.

It does not own finding lifecycle, finding status, carried-item lifecycle,
project state, decision-record status, scientific-data provenance,
durable-record placement, retention, publication, transaction semantics, Git
orchestration, or the analysis mechanisms themselves.

### The governing rule

```
measurement evidence is not state authority
```

A measurement may claim no more authority than its corpus, method,
completeness and evidence jointly establish.

### The non-equivalences this record encodes

These are failure boundaries, not stylistic preferences. Every one has a dated
instance in this repository's records.

```
filesystem membership     !=  Git identity
Git identity              !=  package membership
package membership        !=  runtime reachability

member count              !=  member identity

grep absence              !=  runtime absence

descriptive measurement   !=  predicate verdict

historical statement      !=  current state

date consistency          !=  atomic publication

file existence            !=  production use

finding mention           !=  finding status

measurement evidence      !=  status authority
```

### Boundaries with existing owners

```
provenance/               scientific and data lineage
repository_records/       durable repository-record classification and placement
repository_measurement/   epistemic semantics of repository inspections
transactions/             repository mutation and publication mechanics
state/                    runtime application state
evaluation/               scientific and model evaluation
```

`repository_measurement` is near-leaf infrastructure. It depends on the Python
standard library and nothing else, and it must never import `provenance`,
`repository_records`, `transactions`, `state`, `evaluation` or `models`. A
diagnostic layer that cannot load when the application is broken is a
diagnostic layer that is absent when it is most needed.

### Transport, not import

A standalone instrument analysing this repository must not import the checkout
it is measuring. Otherwise the thing under measurement becomes a runtime
dependency of the measuring instrument.

Instruments therefore emit a versioned wire schema constructed from ordinary
dictionaries, and repository-side code parses and validates it strictly.
Unknown keys are an error rather than tolerated entropy.

## Consequences

- The `Observation` role acquires the typed owner ADR-0001 declared for it.
- `Finding` remains without one, deliberately. Whether the finding namespace
  has one lifecycle is unmeasured; 886 identifiers are named by records and
  absent from any register, and a monolithic register that falsely normalises
  heterogeneous objects would be worse than narrative records. That question
  is deferred until the namespace has been inventoried and classified.
- ADR-0001's decision-sequence preservation manifest remains unimplemented and
  is recorded here as an open obligation of that record, not of this one.
- No universal state registry is introduced. No probe is relocated. No script
  taxonomy changes.
- A measurement that adjudicates nothing carries no verdict. A descriptive
  census must not fabricate a passing verdict in order to look complete, and
  `NOT_JUDGED` is not `PASS`.
- Historical extrema are not predictive intervals. Eleven gate durations
  between 842.9 and 1355.5 seconds justify the statement "eleven observations,
  minimum 842.9 seconds, maximum 1355.5 seconds, no validated predictive model
  exists" and justify no interval claim.

## What this record does not decide

Which artifact is authoritative for any given claim. That is ADR-0001's
lattice, and this record does not extend it.

Whether an instrument is authoritative. The population of authoritative probes
is not defined by any existing class, directory, naming rule or manifest, so a
universal compliance claim over it would quantify over a population nobody has
enumerated.

Whether the probes that currently emit these reports should become durable
repository tooling. They remain external instruments; the contract is what is
productised, not the analysis logic.
