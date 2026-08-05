# ERRATUM REC-1 — three inconsistencies in the CERT-1 records

**Author: Monzia Moodie**
**Written 2026-08-05, against `HEAD = origin/main = 390cd65`.**

**Applies to:** the CERT-1 roadmap delta in `docs/ROADMAP.md` and
`docs/SESSION_2026-08-05_cert1-population-required.md`, both committed in
`390cd65`.

**Neither record is edited.** The C2-1 precedent holds: a correction belongs
beside a record rather than inside it. These are narrative documents rather than
measurement reports, but the reason is the same — a reader auditing `390cd65`
should see what it said, and see the correction as a separate act with its own
date.

---

## 1. The refusal count contradicts itself within one delta

The delta's title reads *"six refusals of my own"*. Its section 4 tabulates
**five miscounts plus one composed-string search**, and section 7 then records a
**seventh** — the follow-up register's own count.

**The correct figure is seven**, and the informative form distinguishes their
origin:

> **Six CERT-1 installer refusals, plus one documentation-register refusal.**

The seventh is different in kind: the first six were gates refusing a write to
the repository, while the seventh was a check I wrote against my own delta
catching a miscount in its prose. That distinction is worth preserving rather
than folding into a single tally.

---

## 2. The session document's follow-up count is stale

It closes with *"Twenty follow-ups carried; C2-1 remains the newest."*

The roadmap delta enumerates **twenty-one**, and the enumeration is correct —
verified by listing them individually rather than counting by eye, which is how
the error was found in the first place.

The precise statement:

> Twenty follow-ups entered the CERT-1 work; **C2-1 raised the carried total to
> twenty-one** before session close.

---

## 3. The session header states a base that reads as a result

```
Base: HEAD = origin/main = 58bf4e1
```

Historically correct — that was the starting point — but a reader arriving at the
document after `390cd65` landed can mistake it for the resulting head, and
conclude the record is stale when it is describing its own beginning.

Both belong:

```
Base:   58bf4e1
Result: 390cd65
```

**This applies to every session document in the sequence**, not only this one.
Each states a base and none states a result, so each is open to the same
misreading. Correcting them retroactively would edit committed records; the
convention changes from the next document onward.

---

## 4. A stronger form of the working-method finding

The CERT-1 delta concluded:

> derive the expectation from what is measured

That is necessary and **not sufficient**. A derived expectation can still measure
the wrong thing. The post-check it praised —

```python
before.count("population=None") - 3
```

— was better than a guessed constant and remained a **lexical count** standing in
for a **semantic question**: *does any code construct an OK outcome with no
population?* That is an abstract-syntax-tree question about constructor calls,
and I answered it by counting a string that happens to appear in them.

The complete rule:

> **State the intended invariant, then measure it at the highest semantic level
> available.**

| question | the right instrument |
|---|---|
| does any code construct an OK outcome without a population? | abstract syntax tree, or a typed construction test |
| does every enum member have prose? | set equality |
| does every result carry every runtime support key? | compare against `ctx.support()` |
| is a name importable? | import it |
| are two runtime strings equal? | compare the values, not the sources |
| did the assertions change? | abstract-syntax-tree comparison, including `pytest.raises` |

Six of today's seven refusals used a **textual proxy for a structural fact**. That
is the finding, and it is sharper than the one the delta recorded.

---

## 5. REG-REASON-1 — a new follow-up

CERT-1's third vocabulary gate reads `_certification_eligibility`'s **source** to
confirm it still returns `"unattributed_population"`.

**That is a drift detector, not an architecture.** It synchronises two
independently represented facts by inspecting code, and I presented it as settled
when it is provisional.

The stronger endpoint: have the registry return

```python
OperatingPointCertificationBlocker.UNATTRIBUTED_POPULATION
```

rather than the raw string. Then there is no second spelling to synchronise and
the gate becomes unnecessary.

**Not folded into step 3c.** Changing the registry's reason type widens a
contract that `MetricResult.reason` and every artifact reader depend on, and step
3c is a narrow extraction. Recorded as **REG-REASON-1**, raising the carried
total to **twenty-two**.

---

## 6. Why an erratum rather than an edit

Section 1 is an internal contradiction — a title disagreeing with its own body —
which is a defect in the record rather than a superseded measurement. It is the
strongest case for editing in place that this sequence has produced.

It is still not edited. A reader auditing `390cd65` should see the delta as it
was written, and see this correction as a separate act with its own date and its
own reasoning. **A record that silently improves is a record whose history cannot
be reconstructed** — and the whole argument for preserving R1's incomplete
prediction, and REG-1's wrong rationales, rests on that.
