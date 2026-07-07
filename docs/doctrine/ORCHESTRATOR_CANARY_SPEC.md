# ORCHESTRATOR SELF-CANARY SPECIFICATION
### How the GenAssoc agent layer verifies that its own guards still work — before it trusts them
**Version 1.0 — 2026-07-06.** Companion to ANTI_DRIFT_DOCTRINE.md (practice 3.3).

---

## 0. What this is and what it honestly cannot be

The agent layer exists to catch drift in data, models, and infrastructure. But the agents are
themselves code that can go stale (practice 3.2/§2 of the doctrine). The **canary** is the
startup self-test that answers one question loudly: *"Do my own guards still fire?"* — before a
single real decision is trusted.

**Honest limit (do not oversell):** the canary is itself in the trusted base and cannot fully
validate itself (§2 regress). It does NOT make the orchestrator drift-proof. What it does is
convert *silent staleness of a guard* into a *loud startup failure* — the single highest-leverage
move available. The regress still terminates at the human cadence (§3.5). This spec is written to
be maximally useful within that honest ceiling, not to claim past it.

---

## 1. The canary's contract

On orchestrator startup, BEFORE any agent processes real data, run `canary()`. It must:
- exercise every agent on **synthetic fixtures** engineered to trip each assertion,
- verify each agent's **contract** against the **live source** it guards (not a cached copy),
- confirm each agent **fails loud** on bad input (a guard that no longer rejects bad data is a
  broken guard, even if it passes good data),
- emit a single `CANARY: PASS/FAIL` with per-check detail and, on any FAIL, HALT the orchestrator
  with a named remedy. No real run proceeds on a red canary.

Idempotent, fast (seconds, synthetic data), and itself version-stamped.

---

## 2. Per-agent canary checks (GenAssoc's four agents)

For each agent, the canary runs THREE kinds of check. The third is the one most systems omit and
the one that actually guards the guard.

### 2.1 DataFreshnessAgent
- **Positive:** hand it a fresh synthetic manifest → expects PASS.
- **Negative (guard-fires):** hand it a manifest with a stale timestamp / missing file / wrong
  row-count → the agent MUST reject it. If it passes, the freshness guard has broken → CANARY FAIL.
- **Contract-vs-live (drift):** the agent's notion of "the expected schema/columns" must be
  DERIVED from the current contract file at runtime and compared to the live data source's actual
  schema — not a hardcoded column list. Canary asserts: (a) the agent reads the contract at
  runtime, (b) contract version matches the declared current, (c) a deliberately mutated schema is
  detected. This is where the doctrine's "derive, don't store" (3.1) is enforced ON the agent.

### 2.2 TrainingLifecycleAgent
- **Positive:** synthetic "healthy run" state → PASS.
- **Negative:** synthetic states for each known bad condition (checkpoint missing at T+45,
  base-model count wrong, rc!=0, degenerate metric) → agent MUST flag each. A silently-passed bad
  state = broken guard = CANARY FAIL.
- **Contract-vs-live:** the expected model roster count and artifact set are read from config at
  runtime, not hardcoded; canary trips a wrong-count fixture.

### 2.3 InterpretabilityAgent
- **Positive:** synthetic feature-importance + predictions → PASS.
- **Negative:** inject a leakage signature (a feature perfectly correlated with the label), an
  all-constant feature, and a degenerate (single-value) prediction vector → agent MUST flag each.
  These are the exact conditions it exists to catch; if a fixture-planted leak slips through, the
  leakage guard has drifted → CANARY FAIL.
- **Contract-vs-live:** leakage thresholds / flagged-feature lists read from config at runtime.

### 2.4 LiteratureScoutAgent (and any agent touching external APIs/models)
- **Positive:** a mocked well-formed source response → PASS.
- **Negative:** a malformed / schema-changed API response → agent MUST fail loud, not silently
  return empty (an external API changing shape is drift; swallowing it is the trap).
- **Contract-vs-live (the reflexive check):** if the agent loads any model with
  `trust_remote_code=True` or hits an external API, the canary does a **clean-cache / live-probe**
  load to confirm the remote contract still matches — this is exactly the NT-stale-cache class
  (Appendix G). A load that reuses a stale cached module does NOT count; the canary clears the
  relevant cache first (or uses a pinned vendored copy and asserts the pin).

---

## 3. The self-check (guarding the guard, as far as is possible)

Beyond exercising each agent, the canary runs a **meta-check** on itself and the assertion
machinery:
- **Assertion liveness:** for each critical assertion in the codebase (feature-count guard,
  fail-loud data verifier, GNN degeneracy check, split-integrity check), the canary feeds the
  precise input that SHOULD trip it and confirms it DOES. An assertion that has been accidentally
  weakened (e.g. an `assert` downgraded to a `warn`, a threshold loosened, a broad `except` added
  above it) is caught here. This is the concrete implementation of "do my own checks still fire?"
- **Trusted-base inventory check:** the canary reads the trusted-base inventory (container digest,
  lockfile hash, NT revision, vendored-file hashes, CLI version) and compares to the running
  environment. Any mismatch, or any item past its staleness budget, is flagged loud.
- **Swallowed-exception scan:** a static check (grep-grade is fine) for `except: pass` / broad
  excepts around critical steps, failing the canary if a new one appears near a guarded boundary.

The self-check cannot prove the canary itself is correct (§2). Its honest value: it catches the
*common, mechanical* ways guards rot (downgraded asserts, loosened thresholds, swallowed errors,
stale trusted-base items), which is where real regressions actually occur.

---

## 4. Output contract (fail-loud, dated, actionable)

```
=== ORCHESTRATOR CANARY  <UTC>  (canary vX, trusted-base hash ....) ===
[PASS] DataFreshnessAgent: positive / negative-fires / contract@vN matches live schema
[FAIL] InterpretabilityAgent: planted leakage NOT flagged -> leakage guard drifted
       remedy: review leakage_audit thresholds; a fixture leak with r=1.0 passed
[PASS] assertion-liveness: 7/7 critical assertions fired on trip-fixtures
[FAIL] trusted-base: NT revision f34324c... past 30d budget AND no vendored modeling_esm.py
       remedy: vendor NT modeling file (RUN_BOOTSTRAP_DOCTRINE §3) or re-validate on clean cache
------------------------------------------------------------------
CANARY: FAIL (2 of 9) -- orchestrator HALTED. No run proceeds. Fix remedies above.
```
Green → orchestrator proceeds. Red → HALT, no real data touched, every failure named with a remedy
and a date.

---

## 5. Implementation notes (concrete, GenAssoc)

- Put fixtures in `tests/canary/` — synthetic parquets + states engineered to trip each assertion.
  These ARE the negative tests; reuse the existing synthetic-data test scaffolding.
- Wire `canary()` as the first call in the orchestrator entrypoint, gated so a red canary raises
  before any agent runs on real data.
- Run the canary ALSO in CI on every commit (catches guard-rot at PR time, not just run time) and
  at every cloud-run start (via `vm_bootstrap_run.sh` → add a `canary` phase once implemented).
- Version-stamp the canary and record its trusted-base hash in each run's provenance, so a run's
  artifacts carry proof of which guard-set validated them.
- **Staleness budgets to declare now:** lockfile (monthly clean-build re-validate), NT
  revision/vendored file (re-check on any transformers change), vastai CLI (before each provision),
  data schema contracts (per data refresh). The canary enforces these budgets as flags.

---

## 6. What this buys, honestly

It does not make the orchestrator immune to drift — nothing can (§2). It makes the orchestrator
**fail loud at startup when its own guards have rotted or its trusted base has moved**, instead of
silently passing drifted data into a run. Combined with `vm_bootstrap_run.sh` (environment gate),
the fail-loud data verifier (data gate), and the human re-validation cadence (the honest bottom),
it closes the largest remaining silent-failure surface: *the guards themselves going stale
unnoticed.* That is the ceiling, and it is worth building to.
