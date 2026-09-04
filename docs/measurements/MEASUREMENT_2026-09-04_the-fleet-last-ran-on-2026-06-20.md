# MEASUREMENT 2026-09-04 -- the fleet last ran on 2026-06-20, and the gate said OK

**Author: Monzia Moodie**
**Measured at:** `05f7868`
**Status:** MEASUREMENT ONLY. Nothing is built.

---

## 0. Why this exists

The session record at `05f7868` names the agent-liveness measurement as the
next action, and gives the reason it had not been taken: the agent-architecture
document proposes `AGENT-FLEET-STALE-1` on a sixty-two-day figure measured
2026-08-21. A finding whose content is a number that moves is stale the day it
is filed.

`AGENT-LIVENESS-SEMANTICS-1` is registered as a conceptual defect about what
"active" means. This measures what the repository's own gate actually reports,
and what it reports is worse than dormancy.

The tool is `scripts/check_agents_active.py`, 15,047 bytes, sha256
`6f40830e736d459ae76f70e0a7a929dcad671080dd9828768807b3726a9b9afb`. It was READ
in full before it was run, and established from its parse tree rather than its
docstring: ZERO write calls, seven standard-library imports, no `subprocess`,
no network, and no import of the project package -- deliberate, so that a
broken agent import is REPORTED rather than fatal.

---

## 1. The fleet ran once, on 2026-06-20

Twenty-two agents, `registered=22, scheduled=22`. Every one carries
`agent_runs` telemetry, and every timestamp falls inside one window:

```
earliest   2026-06-20T02:30:20.131343+00:00
latest     2026-06-20T03:37:19.861777+00:00
span       4,019.7 seconds
```

That is ONE orchestrator execution, not a scatter of independent runs. Nothing
since.

The tool's default window is 30 days. Every agent measured between 76.08 and
76.13 days, so all 22 read STALE. Those day figures are DISPLAY ONLY and will
be wrong tomorrow; the durable statement is the timestamp.

`AGENT-FLEET-NO-TELEMETRY-SINCE-2026-06-20T033719Z-1`.

The parser reads registrations from the `self._agent_registry = {...}` dict
literal. Against an incremental registry it would return EMPTY and every
scheduled agent would read `MISSING_IMPL` -- verified against a fixture before
the run. The header printing `registered=22` proves the live orchestrator uses
the literal form, so the report is not silently vacuous.

---

## 2. The gate reported OK, and the control proves the flag decides

Same fleet, same instant, one flag apart:

```
default invocation   exit 0   OK: 0 dormant/problem agent(s)
--strict             exit 1   PROBLEM: 22 dormant/problem agent(s)
```

`STALE` is a hard failure ONLY under `--strict`. So the standing rule -- no
agent may be dormant -- is not enforced by the default invocation, and every
preflight that ran this check without `--strict` recorded a pass it had not
earned.

ADR-0004 section B3 names this family in a different subsystem: "the liveness
gate whose default invocation could not fail". Here the gate CAN fail, and the
`--strict` run proves it on this exact fleet, so the tool is sound and the
threshold is not binding. A check passing is not evidence of correctness until
it has been observed to fail; this one was.

`AGENT-LIVENESS-GATE-REPORTS-OK-ON-A-WHOLLY-STALE-FLEET-1`.

---

## 3. Five agents last recorded `skipped`

Last telemetry status, all 22 accounted for:

```
ok  11   skipped  5   plan  1   agent_ops_scan  1   poll_and_flag  1
data_readiness_gate  1   registry_freshness_scan  1   ewc_lifecycle  1
```

The five are `FinOpsAdvisorAgent`, `InterpretabilityAgent`,
`LiteratureScoutAgent`, `ModelInsightsAgent`, `ProvisioningAgent`.

`assess()` treats only `status == "error"` as `ERRORED`. Every other string --
including `skipped` -- reads as evidence of activity indistinguishable from a
real run. That is the same semantic gap `DRY_RUN_ONLY` exists to close, one
level along: the tool already refuses to call a dry-run section write ACTIVE,
and its own docstring says a section timestamp alone "never reads as ACTIVE --
that is exactly how the agents looked dormant". A `skipped` telemetry row is
the same claim wearing the authoritative channel's clothes.

`ProvisioningAgent` is the sharpest case. The roadmap required it be REGISTERED
AND SCHEDULED in a pipeline; both are now measured true. Its only telemetry is
`skipped`, so it has never provisioned anything.

`AGENT-TELEMETRY-SKIPPED-READS-AS-A-RUN-1`.

---

## 4. The SharedState read was the right file

The tool distinguishes the orchestrator SharedState at
`src/genomic_variant_classifier/agent_layer/agent_state.json` from the flat
`literature_scout.*` store at `data/agent_state.json`, and warns loudly when
handed the second, because all agents would then read `NEVER_RUN`.

The table branch prints every warning it holds. NO warning line appeared, so
the warnings list was empty and the file carried genuine SharedState sections.
The absence is evidence here, not silence: the code path that would have
printed a warning is the same one that printed the table.

---

## 5. What this does NOT decide

**Whether the default window should be 30 days, or whether `--strict` should
be the default.** Both are governance choices. What is measured is that they
give opposite verdicts on the same evidence.

**Whether `skipped` should be a distinct status.** Adding one changes the
record shape the tool emits under `--json`, and a new record shape needs a
typed owner in the same unit -- ADR-0004 section G. It is a unit, not a patch.

**Whether the fleet should be RUN.** A real orchestrator execution writes
telemetry and would change `agent_state.json`. That is a production action with
its own preflight, not something to do because a measurement looked bad.

**Where the raw probe outputs belong.** They are machine evidence, and
`records/measurements/` is declared in `repository_records/roles.py` and
currently holds nothing. Writing there would be the first write to a new record
family, which ADR-0004 section H requires be measured first and section G
requires ship with a typed owner, a validator and negative controls in the same
unit. `EVIDENCE-DISPOSITION-INCONSISTENT-1` is already open over exactly this,
and its ruling is that artifacts are classified INDIVIDUALLY rather than moved
because a directory now exists. So the outputs stay outside the repository and
this record cites the tool's digest and the values it produced.

---

## 6. Status

Three findings registered, none closed. No file in the repository is changed by
this record beyond its own creation and the changelog entry that accompanies
it.

The measurement that was owed since 2026-08-21 is taken, and stated as a
timestamp rather than an age.
