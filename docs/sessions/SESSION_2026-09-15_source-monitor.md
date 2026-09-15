# SESSION 2026-09-14 to 2026-09-15 -- a monitoring layer that cannot fail cannot report

Predecessor at the start: `39f8e94d4bc965c3cf12836010a8b4a1dae96072`
Head at the end:          `326492636c006cc6ef1f72c9fc7354f365c9c3bc`

Five units installed. Twelve defects closed in the last of them, every one
found by probing rather than by a passing suite.

## 0. What this session did

    G2  08f9435b -> 222d425f  the preparation driver stops hiding whether
                              its matrices are scaled
    S2  222d425f -> 39f8e94d  predict_single stops promising medians it
                              never supplies
    V   39f8e94d -> 5cb1a330  the version monitor gains the gnomAD target
                              its docstring promised
    Q   5cb1a330 -> 82b31110  a source-monitoring subsystem that fails
                              when it should
    R2  82b31110 -> 32649263  the retained evidence is finally checked by
                              something independent

Suite: 6528 -> 6528 -> 6528 -> 6539 -> 6616 -> 6658.
Every count MEASURED at the live repository inside a rolled-back transaction.

## 1. The question that started it

On 2026-09-12 Monzia observed that gnomAD -- the Genome Aggregation Database --
had released version 4.1.1, and asked whether an agent in this project should
have reported it.

MEASURED, that day and the next: `VersionMonitorAgent` existed, was
registered, was structurally operational, and RAN on 2026-06-20 -- eighty-two
days after the 2026-03-30 release -- recording status `ok`.

It could not have detected the release. Its module docstring promised
"3. gnomAD -- watch for v4.2+ constraint metrics column changes" and `run()`
dispatched SEVEN targets, none of them gnomAD. `run()` dispatches by
hand-written sequence with no registry, so a documented target can simply never
be written into it.

## 2. Every instrument reported success

MEASURED 2026-09-14 at `5cb1a330`, across the monitoring layer:

    check_agents_active.py   22 of 22 agents STALE at 84.59 days -> "OK", exit 0
    audit_agent_operational  structure only; never asks whether anything ran
    VersionMonitorAgent      status "ok" was a LITERAL, set unconditionally
    run_pipeline             printed "[OK]" beside action=error
    run_data_freshness.py    returned 0 whether or not a change was detected
    _record_run_telemetry    ignored result["status"], so degraded became ok

Four independent layers, each letting a broken agent through. And a fifth:
`VersionMonitorAgent` appears in NO workflow. Its only caller is
`scripts/run_adaptation.py`, itself unscheduled. `orchestrator.py:78` defines
the one scheduled pipeline, `"database_monitor"`, as
`["DatabaseFreshnessMonitorAgent"]` and nothing else.

So the agent ran only when a human ran it.

## 3. Unit V -- the gnomAD target, and why the obvious pattern was wrong

`_check_alphamissense` polls ONE STABLE URL and compares an ETag. Copying it
would have produced a target STRUCTURALLY INCAPABLE of detecting a gnomAD
release: gnomAD publishes each release at a NEW PATH, so the v4.1 object was
never modified when v4.1.1 appeared and a HEAD watch would report unchanged
forever.

The target enumerates release directories instead. MEASURED against the live
endpoint: twelve prefixes, one of them `release/v4.0/` carrying a `v` that a
strict numeric parse raises on.

Eleven bound tests. `status` is now DERIVED -- it was a literal `"ok"`, and
`check_agents_active.py` reported that constant back as the agent's health.

The apply REFUSED first: `SuiteTransitionError: an ADDITION transition must
name what it adds`. This installer was derived from the D lineage, whose units
are all NEUTRAL. UNIT N INSTALLED THAT GUARD FOUR COMMITS EARLIER AND IT
REFUSED ITS AUTHOR.

## 4. Unit Q -- a subsystem under a distinct package

MEASURED by `git ls-tree`: `monitoring/` holds sixteen files and NO
`__init__.py`. It is an implicit namespace package while `drift/` inside it is
a regular package. Creating `monitoring/__init__.py` would CONVERT it, changing
import semantics for all sixteen -- a behavioural change disguised as a new
file.

MEASURED by reading `registry.py`: it already declares `Category`, `Check`,
`Verdict`, `Source`, `REGISTRY` over sources including `_ALPHAMISSENSE_GCS` and
`_CLINVAR_FTP`.

    monitoring/registry.py  answers  "is this source stale?"
    source_monitor/         answers  "did the check run, and can its
                                      evidence be qualified?"

Two verdict vocabularies in one package, unrelated, is the structure that
produced an eight-target agent carrying a seven-target docstring.

### Three refusals before it landed

**The working tree.** An untracked `var/monitor/findings.sqlite3`, left by the
live run requested earlier that day. The runner's default store was
repository-relative -- `LITERATURE-STATE-CWD-RELATIVE-1` reproduced in a new
subsystem AFTER reading the test file that documents it.

**An identity mismatch**, +77 observed against +70 declared. SEVEN identities
come from `tests/unit/test_workflow_action_pins.py`, which parametrises over
EVERY workflow -- so adding one workflow adds identities to a file this unit
does not touch. A COUNT of +77 could not have distinguished that from seven
missing tests of my own.

**The acceptance gate.** Three pin tests failed: I pinned SHAs from published
release tags instead of `EXPECTED_PINS`, the repository's authority. One was
the trap that file documents BY NAME -- `actions/upload-artifact` v4.4.3 is
older than v5, and "upload-artifact v5 is STILL node20. v6 is the first node24
release." All three correct SHAs were in `data_freshness.yml`, which I had read
IN FULL that morning and used as the model for this very workflow.

## 5. Unit R2 -- twelve defects, none found by a passing suite

    1. a TypeError on the default store path. EVERY one of the seventy tests
       passed --store EXPLICITLY, so the default was covered only in isolation.
    2. retained evidence nobody checked. request_url, response_sha256 and
       response_bytes existed and nothing read them.
    3. a first-failure inventory presented as complete: three simultaneous
       defects produced ONE finding.
    4. a capture sequence read for labelling and never validated: captures
       numbered [1, 3] verified CLEAN -- a page retained and then LOST.
    5. a finding naming an attempt that NEVER BEGAN. SQLite disables foreign
       keys BY DEFAULT, per connection.
    6. a lossy key coercion: {1: "x"} accepted, recovered as {"1": "x"}.
    7. an unrestricted URL scheme: file:///etc/passwd was ACCEPTED, so a
       misconfigured endpoint made the heartbeat READ FROM DISK and report a
       delivery.
    8. a corrupted query string: "https://h/tok?x=1" became
       "https://h/tok?x=1/fail", a URL the operator never wrote.
    9. a duplicated policy accepted by supervise(): ("a","a") with one result
       returned EXIT 0 -- the obligation set shrank while reporting success.
   10. plan findings flattened to one reason: a LOST PAGE recorded in the
       durable store as a QUERY MISMATCH.
   11. SIX codes the profile forbade its own producers from emitting. Fixing
       #10 made the producer gate fire; I added the one code and declared it
       closed, and a DERIVED enumeration found five more.
   12. a registered check no test ever ran, found by a public-surface census.

The suite passed at 70, 81, 93, 97, 106, 107, 110 and 112 tests -- each time
over at least one of these.

### The independence that matters

`request_verifier.py` declares the endpoint, the field mask and both budgets
ITSELF rather than importing them from the adapter. A verifier that derives its
expectation from the subject cannot detect a subject that changed: reading the
mask from the adapter would have APPROVED the `fields=prefixes` request that
made truncation undetectable, because it would have been comparing the adapter
to itself.

Two tests assert the independent declarations AGREE, making drift a failure
rather than a silence.

### A gap is not truncation

The shared catalog gained `EVIDENCE_CAPTURE_SEQUENCE_INVALID`. Reusing
`TRAVERSAL_TRUNCATED` would conflate a DECLARED LIMIT with a SILENT LOSS.
Inventing the code inside the verifier would be worse: the assessor classifies
an unrecognised reason as a CONTRACT failure, so a private code would surface
as a contract violation every time it fired.

### Eight rebuilds

This unit was rebuilt EIGHT times. Every rebuild was justified and none reached
the repository until the eighth. A correction that is never installed is not a
correction, and the loop itself became the problem: probe, repin, present, wait,
probe again.

The stopping criterion was stated so it could be checked rather than felt:
every public name in all seven modules is referenced by a test. It is a FLOOR,
not a ceiling -- a name referenced once is not a name exercised thoroughly.

## 6. The live run at 32649263

    exit 1  (qualified; review required)
    store               AppData\Local\GenomicVariantClassifier\source_monitor\
    plan_verification   []
    finding             release 4.1.1 is newer than the approved 4.1
    capture 1           305 B  sha256 1fdd37bad8eaa741...  accepted
    query               ...&fields=kind,prefixes,nextPageToken
    heartbeat           no heartbeat endpoint configured

The store path proves `resolve_runtime_paths()` executed -- every earlier run
took the fallback branch, and a bare `except Exception` would have hidden a
failure. The live request matched the independently held plan.

## 7. Suspicions measured and RETRACTED

- `preprocessor` versus `preprocessor_` looked like a fatal mismatch.
  `__init__` assigns `self.preprocessor_ = preprocessor`; the trailing
  underscore is the scikit-learn fitted-state convention.
- the token-chain state after a FATAL exit looked stale. It is conservative and
  correct: a continuation page carrying no token is still refused.

Both cost one command. Asserting either would have been a fabricated defect in
working code.

## 8. Findings

### Open

- `external_heartbeat_check_does_not_exist` -- the workflow carries
  `GVC_HEARTBEAT_URL` as a secret and the runner signals execution, but the
  EXTERNAL CHECK (period 7 days, grace 2 days, start signal enabled) has not
  been created and the secret has not been set. A grace period longer than the
  interval between runs cannot detect a single missed run.
- `version_monitor_has_no_delivery` -- `VersionMonitorAgent` itself still logs
  alerts at INFO into a state file.
- `watch_targets_are_a_hand_written_sequence` -- `run()` dispatches by name and
  three tests silence by name. Target nine will be lost the way target three
  was.
- `ensemble_persistence_intermittent` -- `test_per_model_checkpoints_written`
  failed once with `ValueError: concurrent send_bytes() calls are not
  supported` from a joblib pool teardown at `variant_ensemble.py:2948`, and
  passed twice at the same commit. One failure and two passes is not a
  diagnosis; the mechanism is unread.

### Ending state

    HEAD           326492636c006cc6ef1f72c9fc7354f365c9c3bc
    suite          6658 collected; 6643 passed, 15 skipped, 0 failed
    working tree   clean, untracked included

## 9. Remaining uncertainty

- Whether anyone reads the alarm. An exit code is visible only to someone
  looking, and a red weekly run in a repository nobody watches is as silent as
  a green one.
- Whether the twelve-prefix traversal is a point-in-time snapshot. Google Cloud
  Storage documents that objects created in an already-traversed portion of the
  namespace can be missed; a validated terminal traversal establishes
  COMPLETION OF THAT TRAVERSAL, nothing more.
- Whether a newer release PREFIX means a usable product. "A newer directory
  exists" and "a newer compatible constraint dataset is published and usable"
  require different evidence.
