# SESSION 2026-08-12 -- the PhyloP connector: four repairs to one source

**Author: Monzia Moodie**

**Commits:** `01a4345`, `33c8dae`, `935dfec`, `cc350b9` (four)
**Ratchet:** 4605 -> 4685, with one deliberate decrease
**Preceding head:** `082e8ee`

---

## What this session was

Four consecutive repairs to the PhyloP conservation-score connector, each one
uncovering the next. The unit names state the principles rather than the
mechanics, because in every case the mechanical defect was a symptom of a
principle the code had silently abandoned.

| commit | unit | files | lines |
|---|---|---|---|
| `01a4345` | PHYLOP-SOURCE-OWNERSHIP-1 | 6 | +595 -280 |
| `33c8dae` | PHYLOP-INGEST-INTEGRITY-1 | 5 | +843 -38 |
| `935dfec` | PHYLOP-CACHE-INTEGRITY-1 | 5 | +743 -21 |
| `cc350b9` | PHYLOPPERF-1 | 6 | +590 -10 |

All figures quoted from `git show --stat`, not reconstructed.

---

## 1. `01a4345` -- a connector may not redefine canonical evidence

**PHYLOP-SOURCE-OWNERSHIP-1**, and paired with **PHYLOPTEST-DUP-1**.

The ratchet entry for this unit is the only **decrease** in the whole
progression:

```
# 2026-08-12 -- 4622 -> 4613 (-9). PHYLOP-SOURCE-OWNERSHIP-1 and PHYLOPTEST-DUP-1.
```

Nine duplicate tests were removed. A suite that only ever grows will accumulate
redundant coverage that looks like rigour and is not, so a negative step is a
healthy signal rather than a regression -- provided it is recorded as
deliberate, which the ratchet comment does.

---

## 2. `33c8dae` -- a source must be trustworthy before it is fast

**PHYLOP-INGEST-INTEGRITY-1.** +843 lines, the second-largest single unit of
the three days.

The ordering in the title is the whole argument: performance work on an ingest
path whose integrity is unestablished optimises the delivery of unverified
data.

---

## 3. `935dfec` -- a cache is a claim about a source

**PHYLOP-CACHE-INTEGRITY-1.**

A cache asserts that its contents correspond to something upstream. If the
correspondence is not verified, the cache is an unfalsifiable claim, and every
consumer inherits it.

---

## 4. `cc350b9` -- the lookup engine A1 built the abstraction for

**PHYLOPPERF-1.** `DictPhyloPBackend.lookup_many` ran a Python-level loop:
roughly 4.4 million interpreter dispatches per annotation pass. A1 had declared
that implementation transitional and built the abstraction to replace it.

**The repair.** A new module holding a `pandas.Series` with a `(chrom, pos)`
`MultiIndex`, using `Series.reindex` -- vectorised, row identity preserved by
construction, `NaN` for absent loci. Duplicates refused at construction via
`MultiIndex.has_duplicates`.

**Measured: 2.4x faster** -- 0.387 s to 0.165 s on 200,000 queries against a
500,000-locus index. Measured, not estimated.

### Four installer defects the gates caught

1. `r"^(?i)chr"` raises on Python 3.11 and later. pandas 3.0.2 swallowed the
   exception; pandas 2.3.3 propagated it, failing 13 tests. Fixed with
   `case=False`.
2. The idempotence check was a string search that matched four prose mentions
   of `PHYLOPPERF-1` in comments. Fixed to count `ast.Call` nodes.
3. `-SkipFullSuite` tested `$rc -ge 2`, but pytest returns 1 on test failures --
   so a failing run left a partial install. Fixed to `-ge 1`.
4. A1's own test asserted `isinstance(backend, DictPhyloPBackend)`, which
   blocked its declared successor. Rewritten to assert the protocol.

Defect 2 is the recurring shape: **a checker that fires on prose describing its
own rule.** Defect 4 is the shape where a test encodes an implementation rather
than a contract, and thereby forbids the improvement it was written to enable.

---

## What carried forward

The connector's remaining items -- `BIGWIG-IDENTITY-1`, `PHYLOP-AGREEMENT-1`,
`PHYLOP-ADMISSION-1`, `PHYLOP-RECONCILE-1`, `DBNSFP-VERSION-1` -- were open at
the close of this session and are not addressed by these four commits.

`PHYLOP-QUERY-INTEGRITY-1` was in progress and landed on 2026-08-13; its
ratchet entry is dated 2026-08-12 because the work was done before local
midnight while the commit was written after it. That divergence is expected and
is noted here so a future reader comparing the ratchet to `git log` does not
conclude one is wrong.
