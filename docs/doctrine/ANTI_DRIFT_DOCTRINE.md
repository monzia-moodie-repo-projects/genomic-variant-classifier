# ANTI-DRIFT ENGINEERING DOCTRINE
### A portable standard for building systems that stay correct as the software universe moves
**Version 1.0 — 2026-07-06 — reusable across projects; GenAssoc appendix at the end.**

---

## 0. The premise, stated honestly

Everything outside your own committed code changes without asking you: cloud CLIs, base
images, pip dependencies, remote model code, external data schemas, API contracts, and the
AI models you build with (which are themselves versioned and updated). **Drift is the
default state of the software universe, not an incident.** A system is only as reliable as
its behavior *when* — not *if* — its dependencies move underneath it.

This doctrine is how a project stays correct under continuous drift. It is deliberately
honest about a hard limit (§2) rather than promising an impossible guarantee.

---

## 1. The one principle everything follows from

> **Derive truth from ground truth at runtime; never trust a value because it was true when
> the code was written.**

A hardcoded schema, a hardcoded version, a hardcoded path, a hardcoded CLI flag — each is a
timestamp of a belief that has been decaying since the moment it was typed. Replace every
such stored belief with a *runtime derivation from the authoritative source*:

| Stored belief (drifts) | Runtime derivation (drift-resistant) |
|---|---|
| `EXPECTED_COLS = [...]` hardcoded | read the current schema, compare to a declared *contract* |
| `pip install pandas==2.3.3` in a script | read the pin from the lockfile at runtime |
| "the CLI flag is `--foo`" | query `tool --help` / the CLI's own schema |
| "the remote has file X" | list the remote and check |
| "the model loads" | actually load it, on a clean cache |
| "my agent's assertions still fire" | run a self-canary that tries to trip them |

The code that *derives* can still go stale — but it fails loud when the source it reads has
moved in a shape it doesn't understand, instead of silently proceeding on a stale constant.

---

## 2. The irreducible limit (why "fully self-validating" is a lie to avoid)

**You cannot build a finite system that completely validates its own correctness.** Whatever
checks the checkers can itself be stale; the regress does not bottom out inside the system.
This is the same structural fact behind Gödel's incompleteness, the halting problem, and why
a bootstrapping compiler needs a trusted seed binary it cannot itself verify. Any component
claiming to "exhaustively" guard against drift is *itself* the most dangerous unguarded
component, precisely because it invites you to stop looking.

So the goal is NOT immunity. The goal is:

> **Minimize the trusted base that cannot self-check; re-derive everything outside it at
> runtime; fail loud the instant a check can't confirm its own premises; and terminate the
> regress at deliberate, dated, human-set re-validation checkpoints — not at a magic layer.**

The regress ends at a human decision on a schedule, honestly labeled as such. That is the
ceiling. It is a high ceiling, and pretending it's higher is how silent failures get in.

---

## 3. The five practices (how the principle becomes a system)

### 3.1 Minimal, explicit, pinned trusted base
List — literally, in a file — everything the system trusts without re-deriving: the container
digest, the lockfile, pinned model revisions, pinned CLI versions, vendored third-party code,
and the human. **If it's in the trusted base, it must be (a) pinned, (b) inventoried, and (c)
on a re-validation cadence.** The NT-stale-cache bug (GenAssoc appendix) is the canonical
failure: an untracked file was silently in the trusted base and nobody had inventoried it.
The fix for anything discovered in the trusted base but not inventoried is to *pull it into
version control* (vendor it), shrinking the untracked surface toward zero.

### 3.2 Contracts + assertions at every boundary, failing loud
Every interface — data → agent, agent → agent, external source → pipeline — declares an
explicit *contract* (schema, dtype, range, count, invariant). Every boundary *asserts* the
contract and HALTS on violation. Never a silent default, never a swallowed exception. A guard
like `assert n_features == DECLARED_FEATURE_COUNT` converts an invisible drift into a loud,
located stop. Audit for swallowed exceptions (`except: pass`, broad try/except that masks a
failed sub-step) — these are where drift hides. (GenAssoc's GNN "degenerate" verifier exists
because a GNN failure was once swallowed by a broad except; that pattern is the enemy.)

### 3.3 The self-canary (the closest thing to guarding the guards)
At startup, the system runs a **canary**: synthetic-data end-to-end exercises of each
component that DELIBERATELY tries to trip every assertion, plus a "do my own checks still
fire?" self-test. A checker that has silently broken is caught by the canary *before* it
passes bad data downstream. This does not escape the §2 regress — the canary itself is in the
trusted base and on a cadence — but it converts *silent staleness of the guards* into a *loud
startup failure*, which is the single highest-leverage move available. See the companion
`ORCHESTRATOR_CANARY_SPEC.md`.

### 3.4 Provenance + dated staleness budgets
Every artifact and every automated decision carries: *derived from {source} at {time} against
{contract version}*. Every trusted-base item carries a **staleness budget** (e.g. "re-verify
the CLI syntax if > 14 days"; "re-validate the lockfile against a clean build monthly"). When
an item exceeds its budget, it is FLAGGED for re-derivation, loudly, rather than used silently.
This makes staleness *visible and dated* instead of latent — you can see what's overdue.

### 3.5 Deliberate human re-validation cadence
The regress terminates at humans on a schedule. Define, in writing, WHO re-validates WHAT and
HOW OFTEN: the container rebuild, the lockfile against a clean environment, the pinned model
revisions against a clean cache, the CLI syntax against current docs, the canary itself. This
is the honest bottom of the stack. It is cheap relative to a destroyed run.

---

## 4. The failure-mode ledger (institutional memory)

Every project keeps a living table: *observed failure → root cause → permanent guard → which
practice (3.1–3.5) it belongs to*. This is not bureaucracy; it is how the *class* of a failure
gets recognized so the next instance is anticipated, not re-discovered. The meta-signature to
watch for: several "surprises" that are all really "a thing outside our repo changed since we
last looked." When you see that pattern, the fix is never another one-off patch — it's a
practice from §3 applied to the whole class.

---

## 5. Applying this to building WITH an AI (Claude) — the reflexive case

The AI you build with is itself versioned and drifting; its knowledge has a cutoff; its tools
change under it (this very project watched a CLI deprecate a command mid-session). Therefore:
- **The AI must treat its own knowledge as a stored belief subject to §1** — re-derive current
  facts (tool syntax, versions, APIs) from ground truth (docs, `--help`, the lockfile) rather
  than from training. "I recall the flag is X" is a stale constant; check it.
- **After doing research, extend "this drifts" to the WHOLE class**, not just the item
  researched. Researching a CLI's syntax but not questioning the base image, the pip stack, or
  a model revision is the exact partial-application failure to avoid.
- **Prefer fail-loud verification over confident assertion.** An AI's fluency makes stale
  claims sound authoritative; the antidote is to gate every consequential claim behind a
  runtime check the human can see.

---

## APPENDIX G — GenAssoc-specific instantiation

- **Trusted base inventory (pin + cadence):** container/base image digest; `requirements.lock`;
  the NT model revision AND vendored `modeling_esm.py` (§3 of RUN_BOOTSTRAP_DOCTRINE); the
  vastai CLI version; `rclone.conf`; the Drive data layout. Each gets a staleness budget.
- **The NT stale-cache bug is the archetype for 3.1:** an untracked `modeling_esm.py` in the HF
  modules cache was silently in the trusted base. Fix = vendor it into VCS.
- **The imodelsx dependency-drift is the archetype for 3.2/3.4:** installing it silently moved
  pandas 2.3→3.0 and transformers 5.8→5.13 over the pinned stack. Guard = the bootstrap re-pins
  from the lockfile after every install that can drift, and asserts the result.
- **Existing GenAssoc assets that already embody this doctrine:** the `EXPECTED_TABULAR_FEATURE_COUNT`
  guard (3.2), the fail-loud data-pull verifier (3.2/3.4), the GNN degeneracy verifier (3.2), the
  agent layer's DataFreshnessAgent (should be upgraded to derive schemas at runtime per 3.1), and
  now `vm_bootstrap_run.sh` (3.1/3.2) + the canary spec (3.3).
- **Cadence to set:** re-verify vastai CLI syntax before each provision; re-validate the lockfile
  against a clean container monthly; re-run the orchestrator canary at every run start; re-check
  NT loads on a CLEAN modules cache whenever transformers or NT revision changes.
