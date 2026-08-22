# CORRECTION 2026-08-22 -- a neutral transition was not verified

**Author: Monzia Moodie**
**Applies to:** `e1a5297` (ADR-0003) and `ba9060d` (second session record)
**Placed beside those records, not inside them.**

---

## What the records claim

Both commit messages state, under their own summary:

```
Suite transition: NEUTRAL. Verified inside the transaction, not assumed.
```

**Both commits are correct.** This note does not retract them. What it records is
that the check they cite could not have established what they claim, and that the
claim is therefore stronger than the evidence behind it.

---

## What was actually verified

The installers for both units performed exactly two checks inside the
transaction:

```python
after = collect(repo, interp, env)
if after != EXPECTED_COLLECT:                 # 1. the count is what we expected
    raise Refused(...)
bare = [...]                                   # 2. the ratchet equals the count
if int(bare[0].strip()) != after:
    raise Refused(...)
```

Neither compares node identity.

ADR-0003 defines a neutral transition as `expected_added_nodeids` and
`expected_removed_nodeids` both empty. The installers verified neither. A
transition that removed one test and added another satisfies both checks:

```
before   {test_a, test_b, test_c}     count 3
after    {test_a, test_b, test_d}     count 3
```

Equal counts. Different suite. **Accepted.**

That is the same shape as the acceptance line that recorded `0 passed, 0
skipped, 0 failed` because it was rendered before the gate ran: a true-looking
record produced by a check that could not have established it. It has moved from
a ratchet into a commit message.

---

## What the two commits actually did

Measured 2026-08-22 from git, after the defect was found.

| commit | write set | entries participating in collection |
|---|---|---|
| `e1a5297` | 1 path: `docs/architecture/decisions/ADR-0003-*.md` | **0** |
| `ba9060d` | 2 paths: `docs/CHANGELOG.md`, `docs/sessions/SESSION_2026-08-22_*.md` | **0** |

If a write set contains no file that participates in collection, the collected
identity set cannot change. Collection participates in `tests/**`, any
`conftest.py`, pytest configuration, and `src/**` only where imported at
collection time.

### The premise was examined, not assumed

That argument fails if any test evaluates a parametrization over external
content at import time. So every non-literal parametrization and every
module-level filesystem read across `tests/` was enumerated: **98** and **2**
respectively. Of those, **two** mention a documentation file, both in
`tests/unit/test_changelog_encoding.py`:

```python
@pytest.mark.parametrize('path', [CHANGELOG, ROADMAP],
                         ids=['CHANGELOG.md', 'ROADMAP.md'])
```

The identifiers are **literal strings**. The node identities are
`...[CHANGELOG.md]` and `...[ROADMAP.md]` regardless of file content, and that
module performs no module-level filesystem read -- it holds paths, not content.
Editing the changelog cannot change a single node identity.

**Verdict: both transitions were genuinely neutral in identity as well as in
count.**

---

## PROBE-OVERREFUSAL-1

The probe that examined the premise reported `PREMISE DOES NOT HOLD` and
`NOT PROVABLE` for both commits.

That verdict is wrong, and the fault is mine. Its filter matched *"this
expression mentions the changelog"* rather than *"this parametrization derives
identities from content"*. It failed closed, which is correct behaviour for a
checker, but an over-broad refusal believed without examination is as misleading
as a vacuous acceptance -- in the opposite direction.

The distinction now recorded:

> A checker that fails closed is right to refuse and wrong to be believed
> without examination. A refusal is a claim, and a claim is checkable.

This is the mirror image of the defects this project keeps finding -- the
vacuous detritus iterator, the liveness gate whose default invocation could not
fail, the alert never observed to alert. Those accept because they cannot
reject. This one rejects because it cannot discriminate.

---

## Why the finding stands anyway

`SUITE-NEUTRAL-IDENTITY-1` is **not** retracted by the proof above.

The installer's check was insufficient regardless of whether its conclusion
happened to be true. A correct answer from an invalid method is the defect, not
an exception to it. The 2026-08-21 correction note reached the same conclusion
about a changelog prepend whose true claim was nearly established by comparing
two different decodings of the same bytes.

---

## What changed as a result

`a60f18f` installed `SuiteTransition` into the package: one typed owner, twelve
guards, every one of them proven detectable by a sabotage matrix. Four
installers had each carried a private notion of "neutral"; two were wrong.

The primitive refuses the pathological case by name rather than by number:

```
ADDED IDENTITIES ARE NOT THE DECLARED SET.
  observed but not declared: ['t.py::test_d']
  declared but not observed: []
  a count of +0 cannot distinguish these.
```

`88e844e` then gave the attestation a version that changes when its shape
changes, so the identity digests this evidence rests on can be recorded without
corrupting the format that records them.

---

## Status

`e1a5297` and `ba9060d` stand as committed. Their content is correct, their
transitions were genuinely neutral, and the repository is unaffected.

No file is changed by this note. It is a record of a claim that outran its
check, and of a refusal that outran its evidence.
